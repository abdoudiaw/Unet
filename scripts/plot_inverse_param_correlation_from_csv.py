# Copyright 2025-2026 Oak Ridge National Laboratory
# @authors: Abdourahmane (Abdou) Diaw - diawa@ornl.gov
#
# SPDX-License-Identifier: MIT

#!/usr/bin/env python3
import argparse
import csv
import math
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


def _read_rows(csv_path):
    with open(csv_path, "r", newline="") as f:
        return list(csv.DictReader(f))


def _infer_params(fieldnames):
    out = []
    for k in fieldnames:
        if k.endswith("_true"):
            p = k[:-5]
            if f"{p}_rec" in fieldnames:
                out.append(p)
    return out


def _to_float_array(rows, key):
    vals = []
    for r in rows:
        v = r.get(key, "")
        if v is None or str(v).strip() == "":
            vals.append(np.nan)
        else:
            vals.append(float(v))
    return np.asarray(vals, dtype=float)


def main():
    ap = argparse.ArgumentParser(description="Plot inverse parameter true-vs-recovered from existing CSV.")
    ap.add_argument("--csv", required=True, help="Path to inverse_cycle_metrics.csv")
    ap.add_argument("--out", default="outputs/inverse_param_correlation.png")
    ap.add_argument("--fontsize", type=float, default=18.0)
    ap.add_argument("--tick-fontsize", type=float, default=14.0)
    ap.add_argument("--marker-size", type=float, default=42.0)
    ap.add_argument("--dpi", type=int, default=220)
    args = ap.parse_args()

    rows = _read_rows(args.csv)
    if not rows:
        raise RuntimeError(f"No rows in CSV: {args.csv}")
    params = _infer_params(rows[0].keys())
    if not params:
        raise RuntimeError("Could not find *_true/*_rec parameter columns in CSV.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": args.fontsize,
            "axes.labelsize": args.fontsize,
            "xtick.labelsize": args.tick_fontsize,
            "ytick.labelsize": args.tick_fontsize,
        }
    )

    nP = len(params)
    ncols = min(3, nP)
    nrows = int(math.ceil(nP / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for k, name in enumerate(params):
        r = k // ncols
        c = k % ncols
        ax = axes[r, c]
        t = _to_float_array(rows, f"{name}_true")
        p = _to_float_array(rows, f"{name}_rec")
        v = np.isfinite(t) & np.isfinite(p)
        t = t[v]
        p = p[v]
        if t.size == 0:
            ax.axis("off")
            continue
        ax.scatter(t, p, s=args.marker_size, alpha=0.85)
        lo = min(float(np.min(t)), float(np.min(p)))
        hi = max(float(np.max(t)), float(np.max(p)))
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.0)
        pear = float(np.corrcoef(t, p)[0, 1]) if len(t) > 2 else np.nan
        spear = float(spearmanr(t, p).correlation) if len(t) > 2 else np.nan
        ax.set_xlabel(f"True {name}")
        ax.set_ylabel(f"Recovered {name}")
        ax.text(
            0.03, 0.97, f"r={pear:.3f}\n$\\rho$={spear:.3f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=args.tick_fontsize
        )
        ax.grid(alpha=0.25)

    for k in range(nP, nrows * ncols):
        r = k // ncols
        c = k % ncols
        axes[r, c].axis("off")

    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
