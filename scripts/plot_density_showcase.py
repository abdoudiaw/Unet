#!/usr/bin/env python3
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from plot_paper_evaluation_mesh import (
    add_mesh_panel,
    center_crop_2d,
    load_mesh_polygons,
    load_npz_all,
    split_indices,
)
from solps_ai.predict import load_checkpoint, predict_fields, scale_params
from solps_ai.utils import pick_device


def main():
    ap = argparse.ArgumentParser(
        description="Create a side-by-side density plot: Predicted ne vs Ground Truth ne."
    )
    ap.add_argument("--npz", required=True, help="Dataset npz path.")
    ap.add_argument("--ckpt", required=True, help="Model checkpoint path.")
    ap.add_argument("--base-dir", required=True, help="SOLPS run_* base dir for mesh polygons.")
    ap.add_argument("--run-name", default=None, help="Optional run_* name for mesh.")
    ap.add_argument("--out", default="outputs/ne_showcase.png", help="Output PNG path.")
    ap.add_argument("--split", type=float, default=0.85, help="Train/val split for val index selection.")
    ap.add_argument("--seed", type=int, default=42, help="Seed used for train/val split.")
    ap.add_argument("--k", type=int, default=0, help="k-th validation sample to plot.")
    ap.add_argument("--log-eps", type=float, default=1e-12, help="Floor for log10(ne).")
    ap.add_argument(
        "--percent-floor-frac",
        type=float,
        default=0.02,
        help="Floor for percent denominator as frac * P95(|truth|).",
    )
    ap.add_argument("--dpi", type=int, default=400)
    args = ap.parse_args()

    outdir = os.path.dirname(args.out)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    device = pick_device()
    model, norm, (p_mu, p_std) = load_checkpoint(args.ckpt, device)
    Y, y_keys_data, M, P, _ = load_npz_all(args.npz)
    _, val_idx = split_indices(Y.shape[0], split=args.split, seed=args.seed)
    if len(val_idx) == 0:
        raise RuntimeError("No validation samples available.")

    if "ne" not in y_keys_data:
        raise KeyError("'ne' not found in dataset y_keys.")
    ne_data_idx = y_keys_data.index("ne")

    if hasattr(norm, "y_keys"):
        y_keys_pred = [str(k) for k in norm.y_keys]
    else:
        y_keys_pred = [y_keys_data[0]]
    if "ne" not in y_keys_pred:
        raise KeyError("'ne' not found in checkpoint output channels.")
    ne_pred_idx = y_keys_pred.index("ne")

    run_name, grid, polys = load_mesh_polygons(args.base_dir, args.run_name)
    mesh_hw = grid.shape[:2]

    kk = int(np.clip(args.k, 0, len(val_idx) - 1))
    gidx = int(val_idx[kk])

    y_true_full = Y[gidx, ne_data_idx]
    m_full = M[gidx]
    p_in = scale_params(P[gidx], p_mu, p_std)
    y_pred_full = predict_fields(model, norm, m_full, p_in, device=device, as_numpy=True)[ne_pred_idx]

    y_true = center_crop_2d(y_true_full, mesh_hw)
    y_pred = center_crop_2d(y_pred_full, mesh_hw)
    m = center_crop_2d(m_full, mesh_hw)

    yt = np.where(m > 0.5, y_true, np.nan)
    yp = np.where(m > 0.5, y_pred, np.nan)
    yt_plot = np.log10(np.maximum(yt, args.log_eps))
    yp_plot = np.log10(np.maximum(yp, args.log_eps))

    all_vals = np.concatenate([yt_plot[np.isfinite(yt_plot)], yp_plot[np.isfinite(yp_plot)]])
    vmin = float(np.nanpercentile(all_vals, 1))
    vmax = float(np.nanpercentile(all_vals, 99))

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
        }
    )
    fig, (ax0, ax1, ax2) = plt.subplots(
        1, 3,
        figsize=(12.0, 4.0),
        dpi=args.dpi,
        sharey=True,
        gridspec_kw={"wspace": 0.0},
        constrained_layout=False,  # <-- add this explicitly
    )

    add_mesh_panel(
        fig, ax0, polys, yp_plot, r"Pred $n_e$ ($m^{-3}$)", cmap="plasma",
        vmin=vmin, vmax=vmax, show_xlabel=True, show_ylabel=True, add_colorbar=False
    )
    add_mesh_panel(
        fig, ax1, polys, yt_plot, r"True $n_e$ ($m^{-3}$)", cmap="plasma",
        vmin=vmin, vmax=vmax, show_xlabel=True, show_ylabel=False, add_colorbar=False
    )
#    err_denom_floor = args.percent_floor_frac * float(np.nanpercentile(np.abs(yt[np.isfinite(yt)]), 95))
#    err_denom = np.maximum(np.abs(yt), max(args.log_eps, err_denom_floor))
#    err_plot = 100.0 * np.abs(yp - yt) / err_denom
#    err_plot = np.where(np.isfinite(err_plot), err_plot, np.nan)
#    err_vmax = float(np.nanpercentile(err_plot[np.isfinite(err_plot)], 95))
    # Absolute error in ne
    abs_err = np.log10(np.abs(yp - yt))

    # Robust vmax for plotting
    abs_vmax = float(np.nanpercentile(abs_err[np.isfinite(abs_err)], 95))

#    valid = np.isfinite(yt) & np.isfinite(yp)
#    if np.any(valid):
#        ad = np.abs(yp[valid] - yt[valid])
#        pd = err_plot[valid]
#        print(f"[verify] exact_equal={np.array_equal(yp[valid], yt[valid])}")
#        print(f"[verify] allclose(rtol=1e-6,atol=1e-12)={np.allclose(yp[valid], yt[valid], rtol=1e-6, atol=1e-12)}")
#        print(
#            "[verify] abs(ne): mean={:.3e}, p90={:.3e}, max={:.3e}".format(
#                float(np.mean(ad)), float(np.percentile(ad, 90)), float(np.max(ad))
#            )
#        )
#        print(
#            "[verify] %err: mean={:.2f}%, p90={:.2f}%, max={:.2f}%".format(
#                float(np.mean(pd)), float(np.percentile(pd, 90)), float(np.max(pd))
#            )
#        )
#    add_mesh_panel(
#        fig, ax2, polys, err_plot, "Percent Error (%)", cmap="magma",
#        vmin=0.0, vmax=err_vmax, show_xlabel=True, show_ylabel=False, add_colorbar=False
#    )

    add_mesh_panel(
        fig, ax2, polys, abs_err, r"Abs. error $n_e$ ($m^{-3}$)", cmap="magma",
        vmin=0.0, vmax=abs_vmax, show_xlabel=True, show_ylabel=False, add_colorbar=False
    )



    # Force identical spatial limits on all panels so axes framing is truly comparable.
    r_all = grid[:, :, :, 0].reshape(-1)
    z_all = grid[:, :, :, 1].reshape(-1)
    rmin, rmax = float(np.nanmin(r_all)), float(np.nanmax(r_all))
    zmin, zmax = float(np.nanmin(z_all)), float(np.nanmax(z_all))
    rpad = 0.02 * (rmax - rmin)
    zpad = 0.02 * (zmax - zmin)
    print(rpad, rmin, rmax)
    for ax in (ax0, ax1, ax2):
#        ax.set_xlim(rmin - rpad, rmax + rpad)
        ax.set_ylim(zmin - zpad, zmax + zpad)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(1.0, 2.5)

    # after setting xlim/ylim/aspect
    ax0.set_anchor("E")  # push plot to the right edge of its cell
    ax1.set_anchor("C")  # keep center plot centered
    ax2.set_anchor("W")  # push plot to the left edge of its cell


    # Horizontal inset colorbars placed inside the core void for clean presentation.
    cbax0 = inset_axes(ax0, width="42%", height="4%", loc="center")
    cbax1 = inset_axes(ax1, width="42%", height="4%", loc="center")
    cbax2 = inset_axes(ax2, width="42%", height="4%", loc="center")
    cbar0 = fig.colorbar(ax0.collections[0], cax=cbax0, orientation="horizontal")
    cbar0.set_label(r"$\log_{10}(n_e)$", fontsize=11)
    cbar1 = fig.colorbar(ax1.collections[0], cax=cbax1, orientation="horizontal")
    cbar1.set_label(r"$\log_{10}(n_e)$", fontsize=11)
    cbar2 = fig.colorbar(ax2.collections[0], cax=cbax2, orientation="horizontal")
    cbar2.set_label(r"$\log_{10}(\Delta n_e)$", fontsize=11)


    for ax in (ax0, ax1, ax2):
        ax.minorticks_on()
        ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
        ax.grid(True, linestyle="--", alpha=0.3)

    # Keep only left y-axis labels/ticks.
    ax1.set_ylabel("")
    ax2.set_ylabel("")
    ax1.tick_params(labelleft=False)
    ax2.tick_params(labelleft=False)

    # Hard enforce no horizontal gap between panels.
#    fig.subplots_adjust(wspace=0.0)
    fig.subplots_adjust(wspace=0.0, hspace=0.0)

    # Deterministic export: minimize whitespace and keep panel spacing stable.
    fig.canvas.draw()
#    fig.tight_layout(pad=0.0)
#    fig.subplots_adjust(wspace=0.0)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", pad_inches=0.0)
#    fig.savefig(args.out, dpi=300, bbox_inches="tight", pad_inches=0.0)
    print(f"Saved {args.out}")
#    plt.show()

if __name__ == "__main__":
    main()
