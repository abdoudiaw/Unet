#!/usr/bin/env python3
"""Re-generate inverse param correlation plots from existing CSV results."""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

PARAM_LATEX = {
    "Gamma_D2": r"$\Gamma_{\mathrm{D}_2}$",
    "Ptot_W": r"$P_{\mathrm{tot}}$ [W]",
    "n_core": r"$n_{\mathrm{core}}$",
    "dna": r"$D_\perp$",
    "hci": r"$\chi_i$",
    "log10_thr": r"$\log_{10}(\mathrm{throughput})$",
    "ratio_nc": r"$n_{\mathrm{core}} / \mathrm{throughput}$",
}

PHYS_KEYS = ["Gamma_D2", "Ptot_W", "n_core", "dna", "hci"]
TRANS_KEYS = ["log10_thr", "Ptot_W", "ratio_nc", "dna", "hci"]


def _latex(name):
    return PARAM_LATEX.get(name, name)


def make_corr_plot(keys, true_cols, pred_cols, rows, save_path, title,
                   fontsize=13, tick_fontsize=10, marker_size=38):
    nP = len(keys)
    ncols = min(3, nP)
    nrows = int(np.ceil(nP / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)
    for k, name in enumerate(keys):
        r, c = k // ncols, k % ncols
        ax = axes[r, c]
        t = np.array([float(row[true_cols[k]]) for row in rows])
        p = np.array([float(row[pred_cols[k]]) for row in rows])
        ax.scatter(t, p, s=marker_size, alpha=0.85)
        lo, hi = min(np.min(t), np.min(p)), max(np.max(t), np.max(p))
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.0)
        pear = float(np.corrcoef(t, p)[0, 1]) if len(t) > 2 else np.nan
        spear = float(spearmanr(t, p).correlation) if len(t) > 2 else np.nan
        label = _latex(name)
        ax.set_xlabel(f"True {label}")
        ax.set_ylabel(f"Predicted {label}")
        ax.text(0.03, 0.97, f"r={pear:.3f}\n$\\rho$={spear:.3f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=tick_fontsize)
        ax.grid(alpha=0.25)
    for k in range(nP, nrows * ncols):
        axes[k // ncols, k % ncols].axis("off")
    fig.suptitle(title, fontsize=fontsize + 2)
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="inverse_cycle_metrics.csv")
    ap.add_argument("--out-dir", default=None, help="Output directory (default: same as csv)")
    ap.add_argument("--prefix", default="", help="Filename prefix for output plots")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.dirname(args.csv) or "."

    with open(args.csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"Loaded {len(rows)} rows from {args.csv}")
    cols = list(rows[0].keys())

    # Detect column naming: old uses _rec, new uses _pred
    suffix = "_rec" if f"{PHYS_KEYS[0]}_rec" in cols else "_pred"

    # Physical-space plot
    true_cols_phys = [f"{k}_true" for k in PHYS_KEYS]
    pred_cols_phys = [f"{k}{suffix}" for k in PHYS_KEYS]
    if all(c in cols for c in true_cols_phys + pred_cols_phys):
        make_corr_plot(
            PHYS_KEYS, true_cols_phys, pred_cols_phys, rows,
            os.path.join(out_dir, f"{args.prefix}param_correlation_physical.png"),
            "Physical space",
        )

    # Transformed-space plot: compute from physical values
    # log10_thr = log10(Gamma_D2 + n_core), ratio_nc = n_core / (Gamma_D2 + n_core)
    has_phys = all(f"{k}_true" in cols and f"{k}{suffix}" in cols
                   for k in ["Gamma_D2", "n_core"])
    if has_phys:
        for row in rows:
            g_true = float(row["Gamma_D2_true"])
            n_true = float(row["n_core_true"])
            g_pred = float(row[f"Gamma_D2{suffix}"])
            n_pred = float(row[f"n_core{suffix}"])
            thr_true = g_true + n_true
            thr_pred = g_pred + n_pred
            row["log10_thr_true"] = np.log10(max(thr_true, 1e-30))
            row[f"log10_thr{suffix}"] = np.log10(max(thr_pred, 1e-30))
            row["ratio_nc_true"] = n_true / max(thr_true, 1e-30)
            row[f"ratio_nc{suffix}"] = n_pred / max(thr_pred, 1e-30)

        true_cols_trans = [f"{k}_true" for k in TRANS_KEYS]
        pred_cols_trans = [f"{k}{suffix}" for k in TRANS_KEYS]
        make_corr_plot(
            TRANS_KEYS, true_cols_trans, pred_cols_trans, rows,
            os.path.join(out_dir, f"{args.prefix}param_correlation_transformed.png"),
            "Transformed space",
        )
    else:
        print("Cannot compute transformed-space plot: missing Gamma_D2/n_core columns")


if __name__ == "__main__":
    main()
