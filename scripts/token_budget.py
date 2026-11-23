"""Vision-token accounting for the four target models.

WHY THIS FILE EXISTS
--------------------
The paper wants "equal visual budget across input organisations". That is only
meaningful if the token count is computed the way each model's own processor
computes it. The three architectures do it in three incompatible ways:

  moondream2    fixed resize to 378x378, patch 14      -> 27*27+1 = 730, NOT controllable
  InternVL2.5   dynamic tiling, tile 448, pixel-shuffle 4x -> 256 tok/tile, DISCRETE and
                tile count is chosen to MATCH THE INPUT ASPECT RATIO
  Qwen2.5-VL    native resolution, patch 14 + merge 2  -> ceil(h/28)*ceil(w/28), CONTINUOUS

Consequence (measured, see report()): for a square input the reachable budgets are
  moondream2  730 (only value)
  InternVL    257 (1 tile) or 1025 (2x2 tiles) -- 513/769 need a 2:1 / 3:1 input
  Qwen2.5-VL  729 (27x27 cells) up to 784/1024 ...
so an identical budget does not exist. Two further facts fall out of this and matter
for the paper:

  (a) InternVL's own tiling already performs a 2x2 split on square images, which is
      exactly our C4 "quad-tile" condition. On that model C1 and C4 may be washed out.
  (b) This is a testable hypothesis, not a nuisance: the effect of input organisation
      should shrink as the model's internal token allocation becomes more adaptive
      (fixed grid -> dynamic tiling -> native resolution).
"""
import math

REF_BUDGET = 730
MAX_ASPECT_DISTORTION = 1.6  # accept a tiling grid up to 1.6x away from the input aspect

MOONDREAM_SIDE = 378
MOONDREAM_PATCH = 14

INTERNVL_TILE = 448
INTERNVL_TOKENS_PER_TILE = 256
INTERNVL_MAX_TILES = 12

QWEN_PATCH = 14
QWEN_MERGE = 2
QWEN_CELL = QWEN_PATCH * QWEN_MERGE  # 28  -> Qwen2.5-VL

# Qwen3-VL 改了视觉 patch：patch_size 16（不是 14）、spatial_merge_size 2
# => 每 32x32 像素 1 token。同样 729 token 需要 864x864（27x27 格），不是 756x756。
QWEN3_PATCH = 16
QWEN3_MERGE = 2
QWEN3_CELL = QWEN3_PATCH * QWEN3_MERGE  # 32


def moondream_tokens():
    return (MOONDREAM_SIDE // MOONDREAM_PATCH) ** 2 + 1


def internvl_tokens(n_tiles):
    return n_tiles * INTERNVL_TOKENS_PER_TILE + 1


def internvl_candidates(aspect):
    out = []
    for cols in range(1, INTERNVL_MAX_TILES + 1):
        for rows in range(1, INTERNVL_MAX_TILES + 1):
            n = cols * rows
            if n > INTERNVL_MAX_TILES:
                continue
            dist = abs(math.log((cols / rows) / aspect))
            out.append((n, cols, rows, dist))
    return out


def internvl_best(aspect=1.0, budget=REF_BUDGET):
    """Tile grid closest to the budget among grids that do not distort the aspect too much."""
    cand = [c for c in internvl_candidates(aspect) if c[3] <= math.log(MAX_ASPECT_DISTORTION)]
    cand = cand or internvl_candidates(aspect)
    return min(cand, key=lambda c: (abs(internvl_tokens(c[0]) - budget), c[3]))


def qwen_tokens(w, h, cell=QWEN_CELL):
    return math.ceil(h / cell) * math.ceil(w / cell)


def qwen_size_for_budget(budget=REF_BUDGET, aspect=1.0, cell=QWEN_CELL):
    """Largest (w, h) with Qwen token count <= budget, at roughly the wanted aspect."""
    best = (cell, cell)
    for cols in range(1, 64):
        for rows in range(1, 64):
            t = cols * rows
            if t > budget:
                continue
            if abs(math.log((cols / rows) / aspect)) > math.log(1.08):
                continue
            best = max(best, (cols * cell, rows * cell), key=lambda wh: wh[0] * wh[1])
    return best


def plan(aspect=1.0, budget=REF_BUDGET):
    """Reachable configuration for every model at the given aspect / reference budget."""
    n, cols, rows, _ = internvl_best(aspect, budget)
    qw, qh = qwen_size_for_budget(budget, aspect)
    q3w, q3h = qwen_size_for_budget(budget, aspect, QWEN3_CELL)
    return {
        "moondream2": {"input": (MOONDREAM_SIDE, MOONDREAM_SIDE), "tokens": moondream_tokens(),
                       "note": "fixed by architecture"},
        "internvl2_5_2b": {"input": (cols * INTERNVL_TILE, rows * INTERNVL_TILE),
                           "tokens": internvl_tokens(n), "note": f"{cols}x{rows} tiles"},
        "internvl2_5_8b": {"input": (cols * INTERNVL_TILE, rows * INTERNVL_TILE),
                           "tokens": internvl_tokens(n), "note": f"{cols}x{rows} tiles"},
        "internvl3_5_8b": {"input": (cols * INTERNVL_TILE, rows * INTERNVL_TILE),
                           "tokens": internvl_tokens(n), "note": f"{cols}x{rows} tiles"},
        # same checkpoint via its own processor: 2x2 tiles PLUS a thumbnail -> 5 tiles.
        # Kept as a separate key so the equal-budget grid and the native-usage result
        # are never mixed in one table.
        "internvl3_5_8b_native": {"input": (cols * INTERNVL_TILE, rows * INTERNVL_TILE),
                                  "tokens": INTERNVL_TOKENS_PER_TILE * (n + 1),
                                  "note": f"{cols}x{rows} tiles + thumbnail"},
        "qwen2_5_vl_3b": {"input": (qw, qh), "tokens": qwen_tokens(qw, qh), "note": "native res"},
        "qwen2_5_vl_7b": {"input": (qw, qh), "tokens": qwen_tokens(qw, qh), "note": "native res"},
        "qwen3_vl_4b": {"input": (q3w, q3h), "tokens": qwen_tokens(q3w, q3h, QWEN3_CELL),
                        "note": "native res, 32px cell"},
        "qwen3_vl_8b": {"input": (q3w, q3h), "tokens": qwen_tokens(q3w, q3h, QWEN3_CELL),
                        "note": "native res, 32px cell"},
    }


def report(aspect=1.0):
    p = plan(aspect)
    rows = []
    for name, d in p.items():
        t = d["tokens"]
        rows.append((name, d["input"], t, (t - REF_BUDGET) / REF_BUDGET, d["note"]))
    return rows


if __name__ == "__main__":
    for aspect, tag in ((1.0, "square 1:1"), (2.0, "wide 2:1"), (0.5, "tall 1:2")):
        print(f"\n=== aspect {tag} ===")
        print(f"{'model':<18}{'input':<12}{'tokens':>8}{'vs ref':>9}   note")
        print("-" * 62)
        for name, (w, h), t, dev, note in report(aspect):
            print(f"{name:<18}{f'{w}x{h}':<12}{t:>8}{dev:>8.1%}   {note}")
    print("\nNOTE: identical budgets are unreachable across these three architectures.")
    print("      Report measured tokens in the paper; treat budget as a covariate, not a control.")
