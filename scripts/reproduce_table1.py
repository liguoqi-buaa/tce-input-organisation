#!/usr/bin/env python3
"""Rebuild Table I from the per-sample prediction files.

Reads predictions/<arm>.csv for the seven model arms, computes AUROC for each
(arm, input organisation) cell, and prints the 42-entry table. No model weights
are needed: everything is derived from the stored per-image probabilities.

Usage:
    python scripts/reproduce_table1.py

Requires numpy only.
"""

import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PRED = os.environ.get("TCE_DATA", os.path.join(os.path.dirname(HERE), "predictions"))

ARMS = [
    ("internvl2_5_2b.csv",        "InternVL2.5-2B"),
    ("internvl2_5_8b.csv",        "InternVL2.5-8B"),
    ("qwen2_5_vl_3b.csv",         "Qwen2.5-VL-3B"),
    ("qwen2_5_vl_7b.csv",         "Qwen2.5-VL-7B"),
    ("internvl3_5_8b_manual.csv", "InternVL3.5-8B (manual)"),
    ("internvl3_5_8b_native.csv", "InternVL3.5-8B (native)"),
    ("qwen3_vl_8b.csv",           "Qwen3-VL-8B"),
]

CFGS = ["C1_full", "C2_center75", "C3_center50", "C4_quad", "C5_multiscale", "C6_multi"]


def auroc(y, p):
    """Rank-based AUROC with correct handling of tied scores."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ps, ys = p[order], y[order]
    ranks = np.empty(len(p), dtype=float)
    i = 0
    while i < len(ps):
        j = i
        while j + 1 < len(ps) and ps[j + 1] == ps[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    return (ranks[ys == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def load(path):
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.setdefault(r["config"], {})[r["img_id"]] = (
                int(r["truth"]), float(r["p_malignant"]))
    return rows


def main():
    print("Table I rebuilt from per-sample predictions")
    print("data directory: %s" % PRED)
    print()

    header = "%-26s" % "Arm" + "".join("%8s" % c.split("_")[0] for c in CFGS)
    print(header)
    print("-" * len(header))

    ok_rows = 0
    for fname, label in ARMS:
        path = os.path.join(PRED, fname)
        if not os.path.isfile(path):
            print("%-26s  MISSING %s" % (label, fname))
            continue
        rows = load(path)
        cells = []
        for c in CFGS:
            if c not in rows:
                cells.append(float("nan"))
                continue
            d = rows[c]
            ids = sorted(d)
            y = np.array([d[i][0] for i in ids])
            p = np.array([d[i][1] for i in ids])
            cells.append(auroc(y, p))
        if any(np.isnan(cells)):
            print("%-26s  INCOMPLETE: %s" % (label, fname))
            continue
        best = max(range(len(cells)), key=lambda k: cells[k])
        line = "%-26s" % label + "".join("%8.3f" % v for v in cells)
        print(line + "   (max %s)" % CFGS[best])
        ok_rows += 1

    print()
    print("%d of 7 arms reported; 1,627 test images per arm, six organisations each." % ok_rows)
    print("Intervals in the paper are patient-level paired bootstrap with Holm")
    print("correction across the five non-reference organisations within each arm.")
    return 0 if ok_rows == len(ARMS) else 1


if __name__ == "__main__":
    sys.exit(main())
