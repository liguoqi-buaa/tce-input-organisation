"""Constrained two-way logit scoring, to replace free-form generation.

WHY
Free-form generation collapsed to a single class on both models (moondream2 -> all
MALIGNANT, Qwen3B -> all BENIGN), which erased every between-configuration difference.
The models were not reading the image; they were following a language prior.

Scoring forces a decision between exactly two continuations and returns a probability,
so the primary metric becomes AUROC -- threshold-free, insensitive to class priors and
to threshold choice. That is the only way to test whether input organisation changes
the models' *ranking* ability when their absolute accuracy is near chance.

This script first validates the method on Qwen (which exposes logits cleanly). Once
AUROC is shown to depart from 0.5, the same protocol is extended to InternVL and
moondream2.
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
import input_org as io
import token_budget as tb

ROOT = os.environ.get("TCE_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BV = os.environ.get("TCE_MODELS", os.path.join(ROOT, "models"))
MALIGNANT = {"MEL", "SCC", "BCC"}
PROMPT = ("This is a photograph of a skin lesion taken with a smartphone. "
          "Is the lesion malignant (skin cancer) or benign? "
          "Answer: the lesion is")
QWEN = {
    "qwen2_5_vl_3b": BV + "/Qwen--Qwen2.5-VL-3B-Instruct/66285546d2b821cf421d4f5eb2576359d3770cd3",
    "qwen2_5_vl_7b": BV + "/Qwen--Qwen2.5-VL-7B-Instruct/cc594898137f460bfe9f0759e9844b3ce807cfb5",
    # Qwen3-VL: patch 16 x merge 2 -> 32-pixel token cell, so 729 tokens need 864x864
    "qwen3_vl_4b": ROOT + "/models/Qwen--Qwen3-VL-4B-Instruct",
    "qwen3_vl_8b": ROOT + "/models/Qwen--Qwen3-VL-8B-Instruct",
}
INTERNVL = {
    "internvl2_5_2b": BV + "/OpenGVLab--InternVL2_5-2B/573169ee54df216786bb9a189e9a32a060a008cf",
    "internvl2_5_8b": BV + "/OpenGVLab--InternVL2_5-8B/e9e4c0dc1db56bfab10458671519b7fa3dd29463",
}

# InternVL3.5 ships in HF-native form, so it needs a different call path from the
# GitHub-format InternVL2.5 above. Three facts were verified on 2026-10-28 *without*
# weights, because each of them would silently corrupt the equal-budget design:
#   * the image placeholder must be <IMG_CONTEXT>; passing "<image>" raises
#     "Number of image placeholders in the prompt does not match the number of images";
#   * one 448 tile costs 256 tokens (32x32 patches, downsample 0.5);
#   * a single 896x896 image is tiled 2x2 BY THE PROCESSOR **and** gets a thumbnail
#     appended -> 1280 tokens. Equal budget therefore requires slicing by hand:
#     4x448 -> 1024 tokens (+1) = the same 1025 as InternVL2.5.
INTERNVL3 = {
    "internvl3_5_8b": ROOT + "/models/OpenGVLab--InternVL3_5-8B-HF",
    # Native-usage control. The processor slices an 896x896 canvas into 2x2 AND appends
    # a thumbnail, costing 1280 tokens instead of our hand-sliced 1025. Measured on dev:
    # 0.776 vs 0.686 AUROC, so the equal-budget constraint costs this model ~0.09 and
    # the two paths must be reported separately rather than averaged.
    "internvl3_5_8b_native": ROOT + "/models/OpenGVLab--InternVL3_5-8B-HF",
}


def first_token_id(tok, word):
    """Token id of ' word' (leading space), falling back to the bare word."""
    for s in (" " + word, word):
        ids = tok.encode(s, add_special_tokens=False)
        if ids:
            return ids[0]
    raise ValueError(word)


def auroc(y_true, y_score):
    """Mann-Whitney U formulation, ties averaged. y_true: 1 = malignant."""
    pairs = sorted(zip(y_score, y_true))
    n = len(pairs)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    pos = [r for r, (_, y) in zip(ranks, pairs) if y == 1]
    n_pos, n_neg = len(pos), n - len(pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    u = sum(pos) - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)



def load_processor(path):
    """Load a processor with the SLOW image processor whenever one exists.

    Why this exists: from transformers 4.52 the fast image processor becomes the
    default even when the checkpoint saved a slow one, and the two produce
    different pixel tensors (measured: max|diff| = 0.030 on Qwen2.5-VL) even
    though the token grid is identical. Every number in this project was produced
    with the slow processor under 4.52.4; pinning use_fast=False reproduces that
    tensor bit-exactly (verified, max|diff| = 0.000000), so results stay
    comparable across environments. Qwen3-VL ships no slow processor, so we fall
    back to the default and record which path was taken.
    """
    from transformers import AutoProcessor
    try:
        proc = AutoProcessor.from_pretrained(path, use_fast=False)
        if getattr(proc, "image_processor", None) is not None and \
                "Fast" in type(proc.image_processor).__name__:
            raise ValueError("fast processor returned despite use_fast=False")
        return proc, "slow"
    except Exception as e:
        proc = AutoProcessor.from_pretrained(path)
        print(f"[load_processor] slow unavailable for {path}: {type(e).__name__}; "
              f"using {type(proc.image_processor).__name__}")
        return proc, "fast"


class QwenScorer:
    """Qwen2.5-VL and Qwen3-VL. They differ in two ways that matter here:
    the visual patch (14 vs 16 px, i.e. a 28 vs 32 px token cell, so the canvas that
    hits a given budget changes) and the tokenizer (Qwen3 vocabulary). Both are read
    from the plan and from the checkpoint rather than hard-coded."""

    def __init__(self, key):
        import transformers
        p = QWEN[key]
        if not os.path.isdir(p):
            raise SystemExit(f"[wait] {key}: {p} not downloaded yet")
        Cls = (transformers.Qwen3VLForConditionalGeneration if key.startswith("qwen3")
               else transformers.Qwen2_5_VLForConditionalGeneration)
        kw = dict(device_map="cuda", local_files_only=True)
        try:  # transformers >= 4.56 renamed torch_dtype -> dtype
            self.m = Cls.from_pretrained(p, dtype=torch.bfloat16, **kw).eval()
        except TypeError:
            self.m = Cls.from_pretrained(p, torch_dtype=torch.bfloat16, **kw).eval()
        self.proc, self.proc_kind = load_processor(p)
        side = tb.plan(1.0)[key]["input"][0]
        self.budget_px = side * side
        self.cell = tb.QWEN3_CELL if key.startswith("qwen3") else tb.QWEN_CELL
        self.id_mal = first_token_id(self.proc.tokenizer, "malignant")
        self.id_ben = first_token_id(self.proc.tokenizer, "benign")
        print(f"[{key}] proc={self.proc_kind} canvas={side}x{side} "
              f"budget_px={self.budget_px} cell={self.cell}")

    def score(self, imgs):
        content = [{"type": "image"}] * len(imgs) + [{"type": "text", "text": PROMPT}]
        txt = self.proc.apply_chat_template([{"role": "user", "content": content}],
                                            tokenize=False, add_generation_prompt=True)
        try:
            inp = self.proc(text=[txt], images=imgs, return_tensors="pt",
                            min_pixels=self.budget_px, max_pixels=self.budget_px).to("cuda")
        except TypeError:
            # processor without min/max_pixels support: canvases are already exactly
            # budget_px, so the token grid is identical without them
            inp = self.proc(text=[txt], images=imgs, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = self.m(**inp)
        last = out.logits[0, -1].float()
        pair = torch.stack([last[self.id_mal], last[self.id_ben]])
        return F.softmax(pair, 0)[0].item()

    def close(self):
        del self.m
        torch.cuda.empty_cache()


class InternVLScorer:
    """InternVL's chat() is a thin wrapper over generate(); we call generate directly
    with output_scores so the first-token distribution is available."""

    def __init__(self, key="internvl2_5_2b"):
        from transformers import AutoModel, AutoTokenizer, GenerationConfig
        from transformers.generation.utils import GenerationMixin
        p = INTERNVL[key]
        self.m = AutoModel.from_pretrained(p, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                           trust_remote_code=True).eval().cuda()
        lm = self.m.language_model
        if not hasattr(lm, "generate"):
            lm.__class__ = type("PatchedLM", (type(lm), GenerationMixin), {})
        if getattr(lm, "generation_config", None) is None:
            lm.generation_config = GenerationConfig.from_model_config(lm.config)

        # --- compatibility patch for transformers >= 4.57 (added 2026-11-20) ---
        # 4.57 replaced the legacy past_key_values tuple with a Cache object, but the
        # model's remote code still does past_key_values[0][0].shape[2]. On the first
        # decoding step the Cache is empty, so that indexing yields None and generation
        # dies with "'NoneType' object has no attribute 'shape'". We score with
        # max_new_tokens=1, so no KV cache is needed at all: drop whatever cache is
        # passed and let the model run one full forward pass over the prompt.
        # This is a call-site patch rather than editing the cached remote code, because
        # transformers re-materialises that file on load and silently reverts edits.
        _orig_prepare = lm.prepare_inputs_for_generation

        # The wrapper must repeat the original parameter names verbatim: transformers
        # validates model_kwargs against this signature and rejects any argument it
        # cannot find there, so **kwargs alone is not enough.
        def _prepare_without_cache(input_ids, past_key_values=None, attention_mask=None,
                                   inputs_embeds=None, **kw):
            return _orig_prepare(input_ids, past_key_values=None,
                                 attention_mask=attention_mask,
                                 inputs_embeds=inputs_embeds, **kw)

        lm.prepare_inputs_for_generation = _prepare_without_cache
        self._cache_patched = True

        self.tok = AutoTokenizer.from_pretrained(p, trust_remote_code=True, use_fast=False)
        mod = sys.modules[type(self.m).__module__]
        self.get_conv_template = mod.get_conv_template
        self.m.img_context_token_id = self.tok.convert_tokens_to_ids("<IMG_CONTEXT>")
        # no HF processor here: tiles are resized by torchvision, so there is no
        # fast/slow ambiguity to record -- but say so explicitly rather than leaving
        # the field absent and looking like an oversight.
        self.proc_kind = "torchvision.Resize(448,BICUBIC) manual tiles"
        self.id_mal = first_token_id(self.tok, "malignant")
        self.id_ben = first_token_id(self.tok, "benign")
        import torchvision.transforms as T
        from torchvision.transforms import InterpolationMode
        self.tf = T.Compose([
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def _tiles(self, img):
        w, h = img.size
        if w <= 448 and h <= 448:
            return self.tf(img).unsqueeze(0)
        cols, rows = max(1, round(w / 448)), max(1, round(h / 448))
        out = []
        for r in range(rows):
            for c in range(cols):
                box = (c * w // cols, r * h // rows, (c + 1) * w // cols, (r + 1) * h // rows)
                out.append(self.tf(img.crop(box)))
        return torch.stack(out, 0)

    def score(self, imgs):
        pvs = [self._tiles(im) for im in imgs]
        npl = [p.shape[0] for p in pvs]
        pv = torch.cat(pvs, 0).to(torch.bfloat16).cuda()
        q = "\n".join(["<image>"] * len(imgs)) + "\n" + PROMPT
        for n in npl:
            q = q.replace("<image>", "<img>" + "<IMG_CONTEXT>" * self.m.num_image_token * n + "</img>", 1)
        tpl = self.get_conv_template(self.m.template)
        tpl.system_message = self.m.system_message
        tpl.append_message(tpl.roles[0], q)
        tpl.append_message(tpl.roles[1], None)
        inputs = self.tok(tpl.get_prompt(), return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = self.m.generate(pixel_values=pv, **inputs, max_new_tokens=1,
                                  output_scores=True, return_dict_in_generate=True, do_sample=False)
        last = out.scores[0][0].float()
        return F.softmax(torch.stack([last[self.id_mal], last[self.id_ben]]), 0)[0].item()

    def close(self):
        del self.m
        torch.cuda.empty_cache()



class InternVL3Scorer:
    """InternVL3.5 in HF-native form (InternVLForConditionalGeneration + InternVLProcessor).

    Differs from the GitHub-format InternVLScorer in two ways, both asserted here
    rather than discovered halfway through a run:
      * placeholders are <IMG_CONTEXT>;
      * the processor tiles large canvases on its own and appends a thumbnail, which
        would quietly raise the budget (896x896 -> 1280 tokens instead of 1025).
        We slice into 448 tiles ourselves and check the resulting image-token count.
    """

    TILE = 448

    def __init__(self, key="internvl3_5_8b", with_thumb=False):
        import transformers
        p = INTERNVL3[key]
        if not os.path.isdir(p) or not os.path.exists(os.path.join(p, "config.json")):
            raise SystemExit(f"[wait] {key}: {p} not downloaded yet")
        Cls = transformers.InternVLForConditionalGeneration
        kw = dict(device_map="cuda", local_files_only=True)
        try:
            self.m = Cls.from_pretrained(p, dtype=torch.bfloat16, **kw).eval()
        except TypeError:
            self.m = Cls.from_pretrained(p, torch_dtype=torch.bfloat16, **kw).eval()
        self.proc = transformers.AutoProcessor.from_pretrained(p)
        self.native = key.endswith("_native")
        # --thumb: the manual path gets the global view back as one extra 448 tile.
        # Purpose is the attribution test in 01_plan/主线锁定_C: manual 1,281 vs
        # native 1,280 isolates "thumbnail present" from "budget larger".
        self.with_thumb = bool(with_thumb) and not self.native
        self.proc_kind = type(self.proc.image_processor).__name__
        self.image_token_id = int(getattr(self.m.config, "image_token_id", 151671))
        self.seq_len = int(getattr(self.m.config, "image_seq_length", 256))
        self.id_mal = first_token_id(self.proc.tokenizer, "malignant")
        self.id_ben = first_token_id(self.proc.tokenizer, "benign")
        print(f"[{key}] native-HF | tile={self.TILE} tok/tile={self.seq_len} "
              f"| malignant={self.id_mal} benign={self.id_ben}")

    def _tiles(self, img):
        if self.native:
            # hand the canvas over whole and let the processor decide; it will tile and
            # append a thumbnail. Must not pre-slice here.
            return [img]
        w, h = img.size
        cols, rows = max(1, round(w / self.TILE)), max(1, round(h / self.TILE))
        if cols * rows != 4:
            raise ValueError(f"canvas {w}x{h} -> {cols}x{rows} tiles; expected 2x2 "
                             "(the equal-budget grid assumes a 896x896 canvas)")
        return [img.crop((c * w // cols, r * h // rows,
                          (c + 1) * w // cols, (r + 1) * h // rows))
                for r in range(rows) for c in range(cols)]

    def _tiles_thumb(self, img):
        """4 tiles as above, plus one 448 thumbnail = the global view the manual
        path otherwise throws away. 5 tiles x 256 = 1,281 tokens."""
        from PIL import Image
        out = self._tiles(img)
        out.append(img.convert("RGB").resize((self.TILE, self.TILE), Image.BICUBIC))
        return out

    def score(self, imgs):
        tiles = []
        for im in imgs:
            tiles.extend(self._tiles_thumb(im) if self.with_thumb else self._tiles(im))
        content = [{"type": "image"}] * len(tiles) + [{"type": "text", "text": PROMPT}]
        text = self.proc.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True)
        inp = self.proc(images=tiles, text=text, return_tensors="pt").to("cuda")
        got = int((inp["input_ids"] == self.image_token_id).sum())
        if self.native:
            # processor-chosen, so record what actually happened rather than assume
            if got % self.seq_len or got == 0:
                raise RuntimeError(f"native path gave a non-multiple image token "
                                   f"count: {got}")
        else:
            want = len(tiles) * self.seq_len
            if got != want:
                raise RuntimeError(f"image tokens {got} != expected {want} "
                                   f"({len(tiles)} tiles x {self.seq_len})")
        self.last_n_tokens = got
        with torch.no_grad():
            out = self.m(**inp)
        last = out.logits[0, -1].float()
        pair = torch.stack([last[self.id_mal], last[self.id_ben]])
        p_mal = float(F.softmax(pair, dim=0)[0])
        del inp, out
        return p_mal

    def close(self):
        del self.m
        torch.cuda.empty_cache()


def build_scorer(key, with_thumb=False):
    if key.startswith("qwen"):
        return QwenScorer(key)
    if key.startswith("internvl3"):
        return InternVL3Scorer(key, with_thumb=with_thumb)
    if key.startswith("internvl"):
        return InternVLScorer(key)
    raise ValueError("no logit scorer for " + key)



def _sha256_of(paths):
    h = hashlib.sha256()
    for pth in paths:
        if os.path.exists(pth):
            with open(pth, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
    return h.hexdigest()


def dump_run_meta(outdir, key, sc, args, rows, sizes):
    """Provenance record that makes a run citable.

    Two fields exist because they have already caused near-misses here:
      * proc_kind -- the fast and slow image processors give identical token grids but
        different pixel tensors (measured max|diff| 0.016-0.030 on Qwen2.5/3-VL);
      * n_tokens_expected -- InternVL tiles a large canvas by itself and appends a
        thumbnail (896x896 -> 1280 instead of 1025), so the budget must be stated.
    A run missing either field must not be quoted in the paper.
    """
    import transformers
    mpath = getattr(sc.m, "name_or_path", "") or getattr(sc.m.config, "_name_or_path", "")
    meta = {
        "tag": args.tag,
        "split": args.split,
        "model": key,
        "model_path": mpath,
        "weights_fingerprint_sha256": _sha256_of(
            [os.path.join(mpath, f) for f in ("config.json", "tokenizer_config.json")]),
        "proc_kind": getattr(sc, "proc_kind", "n/a"),
        "canvas_side": int(sizes[key]),
        "n_tokens_expected": tb.plan(1.0).get(key, {}).get("tokens", "n/a"),
        "n_rows_per_config": len(rows),
        "configs": (args.configs.split(",") if getattr(args, "configs", "")
                    else list(io.CONFIGS)),
        "with_thumb": bool(getattr(args, "thumb", False)),
        "prompt": PROMPT,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": sys.version.split()[0],
        "code_sha256": _sha256_of(sorted(glob.glob(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "*.py")))),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    pth = os.path.join(outdir, key + ".meta.json")
    with open(pth, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    print("  meta ->", os.path.basename(pth))

def run(args):
    sizes = {n: d["input"][0] for n, d in tb.plan(1.0).items()}
    cfgs = args.configs.split(",") if getattr(args, "configs", "") else list(io.CONFIGS)
    with open(os.path.join(ROOT, "data/splits/" + args.split + ".csv")) as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(ROOT, "runs", args.tag + "_" + stamp)
    os.makedirs(outdir, exist_ok=True)
    print("scoring", args.models, "split", args.split, "n", len(rows), "->", outdir)

    for key in args.models.split(","):
        sc = build_scorer(key, with_thumb=getattr(args, "thumb", False))
        path = os.path.join(outdir, key + ".csv")
        fh = open(path, "w", newline="")
        w = csv.writer(fh)
        w.writerow(["model", "config", "img_id", "patient_id", "truth", "p_malignant",
                    "n_img", "n_tokens", "latency_s"])
        dump_run_meta(outdir, key, sc, args, rows, sizes)
        t0 = time.time()
        n = 0
        try:
            for cfg in cfgs:
                for r in rows:
                    img = io.load(r["img_id"])
                    imgs = io.build(img, cfg, sizes[key])
                    t = time.time()
                    p = sc.score(imgs)
                    if key.startswith("qwen"):
                        ntok = tb.qwen_tokens(*imgs[0].size, getattr(sc, "cell", tb.QWEN_CELL)) * len(imgs)
                    else:
                        tiles = sum(max(1, round(i.size[0] / 448)) * max(1, round(i.size[1] / 448))
                                    for i in imgs)
                        ntok = getattr(sc, "last_n_tokens", None) or tb.internvl_tokens(tiles)
                    w.writerow([key, cfg, r["img_id"], r["patient_id"],
                                1 if r["diagnostic"] in MALIGNANT else 0,
                                round(p, 6), len(imgs), ntok,
                                round(time.time() - t, 3)])
                    n += 1
                    if n % 100 == 0:
                        print("  ", cfg, n, "rows", int(time.time() - t0), "s")
                    fh.flush()
        finally:
            fh.close()
            sc.close()

        by_cfg = {}
        with open(path) as f:
            for r in csv.DictReader(f):
                by_cfg.setdefault(r["config"], []).append((int(r["truth"]), float(r["p_malignant"])))
        print("\n== " + key + " AUROC by configuration ==")
        for cfg in cfgs:
            ys = by_cfg.get(cfg, [])
            if not ys:
                continue
            a = auroc([y for y, _ in ys], [s for _, s in ys])
            mean_p = sum(s for _, s in ys) / len(ys)
            print(f"  {cfg:<15} AUROC={a:.3f}  mean p(malignant)={mean_p:.3f}  n={len(ys)}")
        print()
    print("OUTDIR", outdir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen2_5_vl_3b,internvl2_5_2b")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="logit")
    ap.add_argument("--configs", default="",
                    help="comma list; default = all six in input_org.CONFIGS")
    ap.add_argument("--thumb", action="store_true",
                    help="manual path only: append one 448 thumbnail (attribution test)")
    run(ap.parse_args())
