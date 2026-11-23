# Interface-Dependent Measurement of Input-Organisation Effects in Frozen Vision-Language Models for Skin Lesion Triage

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

<!-- ↑ 拿到 Zenodo DOI 后，把两处 XXXXXXX 换成真实 DOI 编号 -->

Per-sample model outputs and analysis code for the paper of the same title, submitted to
*IEEE Transactions on Consumer Electronics* (Special Section: Extended CiSoc Conference
Paper - ICLT 2026).

Everything reported in the paper can be recomputed from this repository. No model weights
are needed and no inference is re-run.


## What the paper shows

A frozen vision-language model consumes a fixed number of visual tokens, and that number is
set by the processor rather than by the developer. Applied work nonetheless treats the
arrangement of the image inside that budget as a knob worth tuning. We test that assumption
on seven frozen arms, six input organisations and 1,627 smartphone images.

The short version:

- Rearranging an image into an equal-cost tiling is worth at most **0.014 AUROC** and reaches
  significance on only one of the seven arms.
- Doubling the budget is worth **-0.001** on average.
- Cropping and multi-scale composition do move the readout, but their **sign reverses across
  arms** (+0.047 against -0.081 for cropping; +0.078 against -0.057 for composition).
- The reversal belongs to the **delivery interface**, not the image: one checkpoint served
  through two tokenisation paths moves from **0.687 to 0.791** AUROC while the same
  multi-scale condition stays at **0.766 against 0.767**.


## Reproducing Table I

Three commands, numpy only:

```bash
pip install -r requirements.txt
python scripts/reproduce_table1.py
```

Expected output (all 42 cells match the paper at three decimals):

```
Arm                             C1      C2      C3      C4      C5      C6
--------------------------------------------------------------------------
InternVL2.5-2B               0.617   0.634   0.665   0.618   0.638   0.596
InternVL2.5-8B               0.716   0.724   0.724   0.705   0.766   0.736
Qwen2.5-VL-3B                0.617   0.602   0.534   0.613   0.627   0.607
Qwen2.5-VL-7B                0.658   0.638   0.611   0.669   0.683   0.652
InternVL3.5-8B (manual)      0.687   0.688   0.651   0.681   0.766   0.709
InternVL3.5-8B (native)      0.791   0.801   0.789   0.787   0.767   0.778
Qwen3-VL-8B                  0.738   0.749   0.731   0.724   0.681   0.739
```


## What is here

```
predictions/     per-sample malignancy probabilities, one CSV per condition set
scripts/         renderer, scorer, token accounting, Table I rebuild
requirements.txt
LICENSE          MIT (code)
LICENSE-DATA     CC BY 4.0 (the CSV files)
CITATION.cff     metadata used by Zenodo to mint the DOI
```

### predictions/

| File | Rows | Content |
|---|---|---|
| `internvl2_5_2b.csv` | 9,762 | InternVL2.5-2B, six organisations |
| `internvl2_5_8b.csv` | 9,762 | InternVL2.5-8B, six organisations |
| `qwen2_5_vl_3b.csv` | 9,762 | Qwen2.5-VL-3B, six organisations |
| `qwen2_5_vl_7b.csv` | 9,762 | Qwen2.5-VL-7B, six organisations |
| `internvl3_5_8b_manual.csv` | 9,762 | InternVL3.5-8B, hand-stitched tiles |
| `internvl3_5_8b_native.csv` | 9,762 | InternVL3.5-8B, native processor |
| `qwen3_vl_8b.csv` | 9,762 | Qwen3-VL-8B, six organisations |
| `internvl3_5_8b_thumbnail.csv` | 1,627 | manual path plus downscaled thumbnail (Section V-C) |
| `ham_6organisations.csv` | 4,800 | dermoscopic tier, 800 images x 6 organisations (Section V-D) |
| `preproc_smartphone_colour.csv` | 6,508 | four colour conditions, smartphone tier (Section V-E) |
| `preproc_dermoscopy_colour.csv` | 3,200 | four colour conditions, dermoscopic tier (Section V-E) |

### Columns

Organisation and dermoscopic files:

| Column | Meaning |
|---|---|
| `model` | arm identifier |
| `config` | `C1_full`, `C2_center75`, `C3_center50`, `C4_quad`, `C5_multiscale`, `C6_multi` |
| `img_id` | dataset image identifier |
| `patient_id` | dataset patient identifier |
| `truth` | 1 = malignant, 0 = benign |
| `p_malignant` | predicted probability from constrained scoring |
| `n_img` | images the arm actually received |
| `n_tokens` | observed count of `image_token_id` entries |
| `latency_s` | wall-clock seconds for that forward pass |

Colour files carry `preproc` (`P0_raw`, `P1_clahe`, `P2_grayworld`, `P3_both`) instead of
`config`, plus a `crop_ok` flag.

No names, file paths or hospital identifiers appear in any file. The patient and image
identifiers belong to the public datasets and are not re-identifiable codes.


## Scripts

| Script | Purpose | Dependencies |
|---|---|---|
| `reproduce_table1.py` | rebuilds Table I from `predictions/` | numpy |
| `input_org.py` | renders the six input organisations | Pillow |
| `score_logits.py` | constrained two-way logit scoring | torch, transformers |
| `token_budget.py` | token accounting per processor family | none |

**`input_org.py` is the operational definition of the treatment.** C1 letterboxes the image
into a square canvas; C4 splits that canvas into four quadrants and letterboxes each quadrant
into a quarter tile, so it changes layout and local density while preserving the pixels. This
distinction matters: an earlier version cropped each quadrant instead of letterboxing it, which
silently changed the local density by a factor of 1.33 on 4:3 inputs. The code here is the
version used for every number in the paper.

**`score_logits.py`** exists because free-form generation from these models collapses onto a
language prior (one model answered MALIGNANT for everything, another BENIGN for everything),
which erased every between-condition difference. All reported results come from constrained
next-token logits over the two class names, normalised in isolation.

Both scripts read `TCE_ROOT` (and `TCE_MODELS` for `score_logits.py`) from the environment and
default to the repository directory. No absolute paths are hard-coded.


## Datasets

| Dataset | Source | Licence |
|---|---|---|
| PAD-UFES-20 | https://github.com/labcin-ufes/PAD-UFES-20 | CC BY 4.0 |
| HAM10000 | ISIC Archive | open |

Model weights are not redistributed. The seven checkpoints (InternVL2.5-2B/8B,
InternVL3.5-8B, Qwen2.5-VL-3B/7B, Qwen3-VL-8B) are available from their original releases.
Every number in the paper can be checked from the stored predictions without re-running
inference.


## Citation

If you use these files, please cite the paper:

```bibtex
@article{yin2026interface,
  title   = {Interface-Dependent Measurement of Input-Organisation Effects in Frozen
             Vision-Language Models for Skin Lesion Triage},
  author  = {Yin, Zihang and Li, Guoqi and Liu, Yang and Huang, Sai and Zhu, Wenwei
             and Song, Cuihao},
  journal = {IEEE Transactions on Consumer Electronics},
  year    = {2026},
  note    = {Submitted}
}
```

The archived snapshot carries its own DOI; see the badge at the top of this file and
`CITATION.cff`.


## License

Code is released under the MIT licence (`LICENSE`). The CSV files in `predictions/` are
released under CC BY 4.0 (`LICENSE-DATA`).
