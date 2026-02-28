# Copyright 2025-2026 Oak Ridge National Laboratory
# @authors: Abdourahmane (Abdou) Diaw - diawa@ornl.gov
#
# SPDX-License-Identifier: MIT

#!/usr/bin/env python3
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker as mticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from plot_paper_evaluation_mesh import (
    add_mesh_panel,
    center_crop_2d,
    load_mesh_polygons,
    load_npz_all,
    split_indices,
)
from solpex.predict import load_checkpoint, predict_fields, scale_params
from solpex.utils import pick_device


DISPLAY_NAME = {
    "Te": r"$T_e\;(\mathrm{eV})$",
    "Ti": r"$T_i\;(\mathrm{eV})$",
    "ne": r"$n_e\;(\mathrm{m}^{-3})$",
    "ni": r"$n_i\;(\mathrm{m}^{-3})$",
    "ua": r"$u_a\;(\mathrm{m/s})$",
}


def _parse_fields(s):
    out = [k.strip() for k in str(s).split(",") if k.strip()]
    if not out:
        raise ValueError("No fields provided. Use --fields Te,Ti,ne,ni,ua")
    return out


def _style_inset_colorbar(cb):
    vmin, vmax = cb.mappable.get_clim()
    if np.isfinite(vmin) and np.isfinite(vmax):
        vmid = 0.5 * (vmin + vmax)
        cb.set_ticks([vmin, vmid, vmax])
    cb.formatter = mticker.FuncFormatter(_pow10_label)
    cb.update_ticks()
    cb.ax.tick_params(labelsize=6, length=1, pad=0.5)


def _pow10_label(x, _pos):
    if not np.isfinite(x):
        return ""
    if x == 0:
        return "0"
    ax = abs(float(x))
    exp = int(np.floor(np.log10(ax)))
    mant = x / (10 ** exp)
    if np.isclose(abs(mant), 1.0, rtol=1e-7, atol=1e-12):
        sign = "-" if x < 0 else ""
        return rf"${sign}10^{{{exp}}}$"
    return rf"${mant:.0g} \cdot 10^{{{exp}}}$"


def main():
    ap = argparse.ArgumentParser(
        description="Create mesh showcase for multiple plasma fields: rows are Truth/Pred/Abs Error."
    )
    ap.add_argument("--npz", required=True, help="Dataset npz path.")
    ap.add_argument("--ckpt", required=True, help="Model checkpoint path.")
    ap.add_argument("--base-dir", required=True, help="SOLPS run_* base dir for mesh polygons.")
    ap.add_argument("--run-name", default=None, help="Optional run_* name for mesh.")
    ap.add_argument("--out", default="outputs/plasma_showcase.png", help="Output PNG path.")
    ap.add_argument("--split", type=float, default=0.85, help="Train/val split for val index selection.")
    ap.add_argument("--seed", type=int, default=42, help="Seed used for train/val split.")
    ap.add_argument("--k", type=int, default=0, help="k-th validation sample to plot.")
    ap.add_argument("--fields", default="Te,Ti,ne,ni,ua", help="Comma-separated plasma fields to plot.")
    ap.add_argument("--dpi", type=int, default=400)
    ap.add_argument("--show", action="store_true", help="Show figure interactively after saving.")
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

    fields = _parse_fields(args.fields)
    missing_in_data = [k for k in fields if k not in y_keys_data]
    if missing_in_data:
        raise KeyError(f"Dataset missing field(s): {missing_in_data}; available={y_keys_data}")
    data_idx = {k: y_keys_data.index(k) for k in fields}

    if hasattr(norm, "y_keys"):
        y_keys_pred = [str(k) for k in norm.y_keys]
    else:
        y_keys_pred = [y_keys_data[0]] if y_keys_data else []
    missing_in_pred = [k for k in fields if k not in y_keys_pred]
    if missing_in_pred:
        raise KeyError(f"Checkpoint missing field(s): {missing_in_pred}; available={y_keys_pred}")
    pred_idx = {k: y_keys_pred.index(k) for k in fields}

    run_name, grid, polys = load_mesh_polygons(args.base_dir, args.run_name)
    mesh_hw = grid.shape[:2]
    r_all = grid[:, :, :, 0].reshape(-1)
    z_all = grid[:, :, :, 1].reshape(-1)
    rmin, rmax = float(np.nanmin(r_all)), float(np.nanmax(r_all))
    zmin, zmax = float(np.nanmin(z_all)), float(np.nanmax(z_all))
    mesh_aspect = (rmax - rmin) / max(zmax - zmin, 1e-12)

    kk = int(np.clip(args.k, 0, len(val_idx) - 1))
    gidx = int(val_idx[kk])

    m_full = M[gidx]
    p_in = scale_params(P[gidx], p_mu, p_std)
    m = center_crop_2d(m_full, mesh_hw)
    y_pred_all = predict_fields(model, norm, m_full, p_in, device=device, as_numpy=True)

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    n_fields = len(fields)
    cell_h = 2.2
    cell_w = max(1.2, cell_h * mesh_aspect)
    fig, axes = plt.subplots(
        3, n_fields,
        figsize=(cell_w * n_fields, cell_h * 3.0),
        dpi=args.dpi,
        squeeze=False,
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.075, right=0.997, bottom=0.07, top=0.965, wspace=0.005, hspace=0.01)

    for cc, key in enumerate(fields):
        y_true = center_crop_2d(Y[gidx, data_idx[key]], mesh_hw)
        y_pred = center_crop_2d(y_pred_all[pred_idx[key]], mesh_hw)
        yt = np.where(m > 0.5, y_true, np.nan)
        yp = np.where(m > 0.5, y_pred, np.nan)
        ae = np.abs(yp - yt)

        finite_tp = np.isfinite(yt) | np.isfinite(yp)
        if np.any(finite_tp):
            vals_tp = np.concatenate([yt[np.isfinite(yt)], yp[np.isfinite(yp)]])
            vmin = float(np.nanpercentile(vals_tp, 1))
            vmax = float(np.nanpercentile(vals_tp, 99))
        else:
            vmin, vmax = 0.0, 1.0

        finite_ae = np.isfinite(ae)
        if np.any(finite_ae):
            evmax = float(np.nanpercentile(ae[finite_ae], 95))
        else:
            evmax = 1.0

        show_y = (cc == 0)
        label = DISPLAY_NAME.get(key, key)
        add_mesh_panel(
            fig, axes[0, cc], polys, yt, f"{label}", cmap="inferno",
            vmin=vmin, vmax=vmax, show_xlabel=False, show_ylabel=show_y, add_colorbar=False
        )
        add_mesh_panel(
            fig, axes[1, cc], polys, yp, "", cmap="inferno",
            vmin=vmin, vmax=vmax, show_xlabel=False, show_ylabel=show_y, add_colorbar=False
        )
        add_mesh_panel(
            fig, axes[2, cc], polys, ae, "", cmap="magma",
            vmin=0.0, vmax=evmax, show_xlabel=False, show_ylabel=show_y, add_colorbar=False
        )
        for rr in range(3):
            cax = inset_axes(axes[rr, cc], width="3.5%", height="40%", loc="center right", borderpad=4.7)
            cb = fig.colorbar(axes[rr, cc].collections[0], cax=cax, orientation="vertical")
            _style_inset_colorbar(cb)
            axes[rr, cc].set_xlim(rmin, rmax)
            axes[rr, cc].set_ylim(zmin, zmax)
            axes[rr, cc].set_aspect("equal", adjustable="box")
            axes[rr, cc].set_anchor("C")

    axes[0, 0].set_ylabel("Truth")
    axes[1, 0].set_ylabel("Pred")
    axes[2, 0].set_ylabel("Abs Error")
    for rr in range(3):
        for cc in range(1, n_fields):
            axes[rr, cc].set_ylabel("")
            axes[rr, cc].tick_params(labelleft=False)
    for rr in range(3):
        for cc in range(n_fields):
            for sp in axes[rr, cc].spines.values():
                sp.set_visible(False)
            axes[rr, cc].set_frame_on(False)
            axes[rr, cc].tick_params(
                axis="both",
                which="both",
                labelbottom=False,
                labelleft=False,
                length=0,
            )

#    fig.supxlabel("R (m)", fontsize=10, y=0.02)
#    fig.supylabel("Z (m)", fontsize=10, x=0.02)
#    fig.supylabel("Z (m)", fontsize=10, x=0.02)

    fig.canvas.draw()
    fig.savefig(args.out, dpi=args.dpi)
    print(f"Saved {args.out}")
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
