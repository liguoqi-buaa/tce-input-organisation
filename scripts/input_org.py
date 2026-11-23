"""Six input organisations for a fixed visual budget, plus the equal-budget check.

WHAT "EQUAL BUDGET" MEANS HERE
The claim is *within-model* equal budget, not across models. Inside one model all six
organisations are rendered onto the same square canvas, so they all consume the same
number of vision tokens; the absolute number differs per architecture because the
three tokenisers are not commensurable (730 / 1025 / 729). Cross-model statements are
therefore made about the *effect size* of a change, never about absolute accuracy.

THE CENTRE ASSUMPTION
PAD-UFES-20 ships no bounding boxes. C2/C3/C5 assume the lesion is roughly centred,
which matches how a phone is held when photographing a lesion. C1 makes no such
assumption, so (C2 - C1) is an upper bound on what any centre-crop can buy, and a
large negative value would falsify the assumption outright.
"""
import os
import numpy as np
from PIL import Image, ImageOps

ROOT = os.environ.get("TCE_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMG_DIRS = [os.path.join(ROOT, "data/raw/PAD-UFES-20/all_images", f"imgs_part_{i}") for i in (1, 2, 3)]
IMG_DIRS += [os.path.join(ROOT, "data/raw/HAM10000/archive", d)
             for d in ("HAM10000_images_part_1", "HAM10000_images_part_2")]

CONFIGS = ("C1_full", "C2_center75", "C3_center50", "C4_quad", "C5_multiscale", "C6_multi")


def find_image(img_id):
    for d in IMG_DIRS:
        p = os.path.join(d, img_id)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(img_id)


def load(img_id):
    return Image.open(find_image(img_id)).convert("RGB")


def _fit(img, size):
    """Letterbox: preserve aspect, pad with black. No distortion."""
    return ImageOps.pad(img.convert("RGB"), size, color=(0, 0, 0), method=Image.LANCZOS)


def _cover(img, size):
    """Fill the box, cropping the overflow. No distortion, loses margins."""
    return ImageOps.fit(img.convert("RGB"), size, method=Image.LANCZOS)


def _center_crop(img, frac):
    w, h = img.size
    side = max(8, int(min(w, h) * frac))
    cx, cy = w // 2, h // 2
    half = side // 2
    box = (max(0, cx - half), max(0, cy - half), min(w, cx + half), min(h, cy + half))
    return img.crop(box)


def build(img, config, size):
    """Render one input organisation onto a size x size canvas.

    Returns a list of PIL images: one image for C1-C5, two for C6 (native multi-image).
    """
    s = size
    if config == "C1_full":
        return [_fit(img, (s, s))]
    if config == "C2_center75":
        return [_fit(_center_crop(img, 0.75), (s, s))]
    if config == "C3_center50":
        return [_fit(_center_crop(img, 0.50), (s, s))]
    if config == "C4_quad":
        w, h = img.size
        half_s = (s // 2, s // 2)
        boxes = [(0, 0, w // 2, h // 2), (w // 2, 0, w, h // 2),
                 (0, h // 2, w // 2, h), (w // 2, h // 2, w, h)]
        canvas = Image.new("RGB", (s, s), (0, 0, 0))
        for i, b in enumerate(boxes):
            # _fit (letterbox), NOT _cover (crop-to-fill). Using _cover here cropped
            # 25% of the width on the 4:3 inputs this study uses and raised local
            # density by the aspect ratio (1.33x), which made C4 a confounded
            # comparison and made "same pixels" false. With _fit every quadrant keeps
            # all of its content, each rendered at the same sampling rate as C1, so
            # C4 differs from C1 only in how the pixels are arranged on the canvas.
            tile = _fit(img.crop(b), half_s)
            canvas.paste(tile, ((i % 2) * half_s[0], (i // 2) * half_s[1]))
        return [canvas]
    if config == "C5_multiscale":
        canvas = Image.new("RGB", (s, s), (0, 0, 0))
        left = _fit(img, (s // 2, s))
        right = _fit(_center_crop(img, 0.50), (s - s // 2, s))
        canvas.paste(left, (0, 0))
        canvas.paste(right, (s // 2, 0))
        return [canvas]
    if config == "C6_multi":
        # Deliberately NOT equal budget: each image is charged separately by the
        # tokeniser, so this condition costs 2x. It measures whether the native
        # multi-image interface is worth its price, and is analysed separately
        # from the equal-budget grid C1-C5.
        return [_fit(img, (s, s)), _fit(_center_crop(img, 0.50), (s, s))]
    raise ValueError(config)


def build_by_model(img, config, model_targets):
    """Render every model-specific version of one config from a single source image."""
    return {name: build(img, config, size) for name, size in model_targets.items()}


if __name__ == "__main__":
    import csv, sys
    sys.path.insert(0, os.path.join(ROOT, "code"))
    import token_budget as tb

    sizes = {n: tb.plan(1.0)[n]["input"][0] for n in tb.plan(1.0)}
    with open(os.path.join(ROOT, "data/splits/dev.csv")) as f:
        rows = list(csv.DictReader(f))[:6]
    img = load(rows[0]["img_id"])
    print(f"source image: {rows[0]['img_id']}  {img.size}\n")
    print(f"{'config':<16}" + "".join(f"{m:>22}" for m in sizes))
    print("-" * 104)
    for cfg in CONFIGS:
        line = f"{cfg:<16}"
        for m, s in sizes.items():
            n_img = len(build(img, cfg, s))
            tok = tb.moondream_tokens() if m == "moondream2" else (
                tb.internvl_tokens(tb.internvl_best(1.0)[0]) if m.startswith("internvl")
                else tb.qwen_tokens(s, s))
            tot = tok * n_img
            line += f"{f'{n_img}pix x {tok} = {tot}':>22}"
        print(line)
