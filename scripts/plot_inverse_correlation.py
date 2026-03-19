#!/usr/bin/env python3
"""Re-generate inverse param correlation plots from existing CSV results."""

import argparse
import csv
import os

import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

PARAM_LATEX = {
    "Gamma_D2": r"$\Gamma_{\mathrm{D}_2}$ [at/s]",
    "Ptot_W": r"$P_{\mathrm{tot}}$ [MW]",
    "Gamma_core": r"$\Gamma_{\mathrm{core}}$ [at/s]",
    "dna": r"$D_\perp$ [m$^2$/s]",
    "hci": r"$\chi_i$ [m$^2$/s]",
    "Gamma_t": r"$\Gamma_t$ [at/s]",
    "ratio_nc": r"$\Gamma_{\mathrm{core}}/\Gamma_t$",
}

def _latex(name):
    return PARAM_LATEX.get(name, name)


PHYS_KEYS  = ["Gamma_D2", "Ptot_W", "Gamma_core", "dna", "hci"]
TRANS_KEYS = ["Gamma_t", "Ptot_W", "ratio_nc", "dna", "hci"]


def make_corr_plot(keys, true_cols, pred_cols, rows, save_path, title,
                   fontsize=13, tick_fontsize=10, marker_size=38):

    fontsize = 14
    tick_fontsize = 14

    # 2x3 layout: 5 params + 1 summary slot
    nrows, ncols = 2, 3

    plt.rcParams.update(
        {
            "font.size": fontsize,
            "axes.labelsize": fontsize,
            "axes.titlesize": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "legend.fontsize": fontsize,
            "axes.linewidth": 1.2,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "xtick.major.size": 6,
            "ytick.major.size": 6,
        }
    )

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5 * ncols, 4 * nrows),
        constrained_layout=True
    )
    axes = np.asarray(axes).reshape(nrows, ncols)

    # store correlations for the summary panel
    all_rho = []  # list of (name, pearson_r, spearman_rho)

    # ---- Per-parameter scatter panels (up to 5)
    for k, name in enumerate(keys):
        if k >= 5:
            break

        r, c = k // ncols, k % ncols
        ax = axes[r, c]

        # robust parse (skip bad rows quietly)
        t_list, p_list = [], []
        for row in rows:
            try:
                tv = float(row[true_cols[k]])
                pv = float(row[pred_cols[k]])
            except Exception:
                continue
            if np.isfinite(tv) and np.isfinite(pv):
                t_list.append(tv)
                p_list.append(pv)

        t = np.asarray(t_list, dtype=float)
        p = np.asarray(p_list, dtype=float)

        if t.size == 0:
            ax.axis("off")
            ax.text(0.5, 0.5, f"No data for {_latex(name)}",
                    ha="center", va="center", fontsize=fontsize)
            all_rho.append((name, np.nan, np.nan))
            continue

        ax.scatter(t, p, s=marker_size, alpha=0.85)

        # 1:1 line
        lo = float(min(t.min(), p.min()))
        hi = float(max(t.max(), p.max()))
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.2)

        pear = float(np.corrcoef(t, p)[0, 1]) if t.size > 2 else np.nan
        spear = float(spearmanr(t, p).correlation) if t.size > 2 else np.nan
        all_rho.append((name, pear, spear))

        label = _latex(name)
        ax.set_xlabel(f"True {label}")
        ax.set_ylabel(f"Pred. {label}")

        ax.text(
            0.03, 0.97, f"r={pear:.3f}\n$\\rho$={spear:.3f}",
            transform=ax.transAxes,
            va="top", ha="left",
            fontsize=tick_fontsize
        )

        ax.grid(True, linestyle="--", alpha=0.3)
        ax.minorticks_on()
        ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)

    # ---- 6th slot: correlation summary inside the 2x3 grid
    sum_ax = axes[1, 2]  # bottom-right

    all_rho = all_rho[:min(5, len(all_rho))]
    names = [n for (n, _, __) in all_rho]
    pears = np.array([r for (_, r, __) in all_rho], dtype=float)
    spears = np.array([rho for (_, __, rho) in all_rho], dtype=float)
    x = np.arange(len(names))

    if np.all(~np.isfinite(spears)):
        sum_ax.axis("off")
        sum_ax.text(0.5, 0.5, "No correlations",
                    ha="center", va="center", fontsize=fontsize)
    else:
        sum_ax.bar(x, spears, alpha=0.85)
        sum_ax.set_ylabel(r"Spearman $\rho$")
#        sum_ax.set_xlabel("Parameter")
        sum_ax.set_xticks(x)
        def _latex_symbol(name):
            s = _latex(name)
            if "[" in s:
                s = s.split("[")[0].strip()
            return s

        sum_ax.set_xticklabels([_latex_symbol(n) for n in names], rotation=0)
        sum_ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        sum_ax.minorticks_on()
        sum_ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)

        for i, (pr, sr) in enumerate(zip(pears, spears)):
            if np.isfinite(sr):
                txt = f"r={pr:.3f}" if np.isfinite(pr) else "r=nan"
                sum_ax.text(i, sr, txt, ha="center", va="bottom", fontsize=tick_fontsize)

        sum_ax.set_ylim(.0, 1.05)

    # ---- Turn off any unused axes besides the summary slot
    used_slots = min(len(keys), 5)
    for k in range(used_slots, nrows * ncols):
        rr, cc = k // ncols, k % ncols
        if (rr, cc) == (1, 2):
            continue
        axes[rr, cc].axis("off")

    fig.suptitle(title, fontsize=fontsize + 2)
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.show()
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

    # Transformed-space plot (linear):
    # Gamma_t = Gamma_D2 + Gamma_core
    # ratio_nc = Gamma_core / Gamma_t
    gc_key = "Gamma_core" if f"Gamma_core_true" in cols else "n_core"
    has_phys = all(f"{k}_true" in cols and f"{k}{suffix}" in cols
                   for k in ["Gamma_D2", gc_key])
    if has_phys:
        for row in rows:
            g_true = float(row["Gamma_D2_true"])
            n_true = float(row[f"{gc_key}_true"])
            g_pred = float(row[f"Gamma_D2{suffix}"])
            n_pred = float(row[f"{gc_key}{suffix}"])

            gt_true = g_true + n_true
            gt_pred = g_pred + n_pred

            row["Gamma_t_true"] = gt_true
            row[f"Gamma_t{suffix}"] = gt_pred

            row["ratio_nc_true"] = n_true / max(gt_true, 1e-30)
            row[f"ratio_nc{suffix}"] = n_pred / max(gt_pred, 1e-30)

        true_cols_trans = [f"{k}_true" for k in TRANS_KEYS]
        pred_cols_trans = [f"{k}{suffix}" for k in TRANS_KEYS]
        make_corr_plot(
            TRANS_KEYS, true_cols_trans, pred_cols_trans, rows,
            os.path.join(out_dir, f"{args.prefix}param_correlation_transformed.png"),
            "Transformed space",
        )
    else:
        print("Cannot compute transformed-space plot: missing Gamma_D2/Gamma_core columns")


if __name__ == "__main__":
    main()
