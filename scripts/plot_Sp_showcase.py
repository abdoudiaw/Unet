# Copyright 2025-2026 Oak Ridge National Laboratory
# @authors: Abdourahmane (Abdou) Diaw - diawa@ornl.gov
#
# SPDX-License-Identifier: MIT

#!/usr/bin/env python3
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import ticker as mticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from plot_paper_evaluation_mesh import (
    add_mesh_panel,
    center_crop_2d,
    load_mesh_polygons,
    load_npz_all,
    split_indices,
)
from solpex.data import (
    MaskedLinearStandardizer,
    MaskedLogStandardizer,
    MaskedSymLogStandardizer,
    MultiChannelNormalizer,
)
from solpex.predict import load_checkpoint, predict_fields, scale_params
from solpex.utils import pick_device


DISPLAY_NAME = {
    "Sp": r"$S_p\;(\mathrm{s}^{-1})$",
    "Qp": r"$Q_p\;(\mathrm{W})$",
    "Qe": r"$Q_e\;(\mathrm{W})$",
    "Qi": r"$Q_i\;(\mathrm{W})$",
    "Sm": r"$S_m\;(\mathrm{N}/\mathrm{m})$",
}


def _parse_fields(s):
    out = [k.strip() for k in str(s).split(",") if k.strip()]
    if not out:
        raise ValueError("No fields provided. Use --fields Sp,Qe,Qi,Sm")
    return out


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


def _style_inset_colorbar(cb):
    vmin, vmax = cb.mappable.get_clim()
    if np.isfinite(vmin) and np.isfinite(vmax):
        vmid = 0.5 * (vmin + vmax)
        cb.set_ticks([vmin, vmid, vmax])
    cb.formatter = mticker.FuncFormatter(_pow10_label)
    cb.update_ticks()
    cb.ax.tick_params(labelsize=6, length=1, pad=0.5)


def _norm_from_ckpt(norm_ckpt):
    if isinstance(norm_ckpt, (tuple, list)) and len(norm_ckpt) == 3:
        mu, sigma, eps = norm_ckpt
        norm = MaskedLogStandardizer(eps=float(eps))
        norm.mu, norm.sigma = mu, sigma
        return norm

    if not isinstance(norm_ckpt, dict):
        raise ValueError("Unsupported norm checkpoint format.")

    kind = norm_ckpt.get("kind", "")
    if kind == "MaskedLogStandardizer":
        norm = MaskedLogStandardizer(eps=float(norm_ckpt.get("eps", 1.0)))
    elif kind == "MaskedLinearStandardizer":
        norm = MaskedLinearStandardizer(eps=float(norm_ckpt.get("eps", 1e-12)))
    elif kind == "MaskedSymLogStandardizer":
        norm = MaskedSymLogStandardizer(
            scale=float(norm_ckpt.get("scale", 1.0)),
            eps=float(norm_ckpt.get("eps", 1e-12)),
        )
    elif kind == "MultiChannelNormalizer":
        y_keys = norm_ckpt.get("y_keys", [])
        norms_pack = norm_ckpt.get("norms", {})
        norms_by_name = {k: _norm_from_ckpt(v) for k, v in norms_pack.items()}
        return MultiChannelNormalizer(y_keys=y_keys, norms_by_name=norms_by_name)
    else:
        raise ValueError(f"Unsupported normalizer kind: {kind!r}")

    mu = norm_ckpt.get("mu")
    sigma = norm_ckpt.get("sigma")
    if mu is not None and not torch.is_tensor(mu):
        mu = torch.tensor(float(mu), dtype=torch.float32)
    if sigma is not None and not torch.is_tensor(sigma):
        sigma = torch.tensor(float(sigma), dtype=torch.float32)
    norm.mu = mu
    norm.sigma = sigma
    return norm


def build_model_input(mask2d, y_full, p_raw, *, x_norm, include_params, p_mu, p_std, in_idx):
    m = mask2d.astype(np.float32, copy=False)
    m_t = torch.from_numpy(m[None, None]).float()

    yin = np.asarray(y_full[in_idx], dtype=np.float32)
    yin_t = torch.from_numpy(yin[None]).float()
    yin_n = x_norm.transform(yin_t, m_t).squeeze(0)

    channels = [torch.from_numpy(m[None]).float(), yin_n]
    if include_params and p_raw.size > 0:
        p_scaled = p_raw.astype(np.float32, copy=True)
        if p_mu is not None and p_std is not None:
            p_scaled = (p_scaled - p_mu) / p_std
        p_t = torch.from_numpy(p_scaled).float()
        H, W = m.shape
        channels.append(p_t.view(-1, 1, 1).expand(-1, H, W))

    return torch.cat(channels, dim=0)


@torch.no_grad()
def predict_sources(model, y_norm, x):
    device = next(model.parameters()).device
    xb = x.unsqueeze(0).to(device)
    z = model(xb)
    m = xb[:, :1]
    y = y_norm.inverse(z, m).squeeze(0).detach().cpu().numpy().astype(np.float32)
    return y


def main():
    ap = argparse.ArgumentParser(
        description="Create source-field showcase with rows Truth/Pred/Abs Error."
    )
    ap.add_argument("--npz", required=True, help="Dataset npz path (must contain Y,y_keys,mask,params/X).")
    ap.add_argument("--ckpt", required=True, help="Checkpoint path (source-from-plasma model).")
    ap.add_argument("--base-dir", required=True, help="SOLPS run_* base dir for mesh polygons.")
    ap.add_argument("--run-name", default=None, help="Optional run_* name for mesh.")
    ap.add_argument("--out", default="outputs/source_showcase.png", help="Output PNG path.")
    ap.add_argument("--split", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=0, help="k-th validation sample to plot.")
    ap.add_argument("--fields", default="Sp,Qe,Qi,Sm", help="Comma-separated source fields to plot.")
    ap.add_argument("--dpi", type=int, default=400)
    ap.add_argument("--show", action="store_true", help="Show figure interactively after saving.")
    args = ap.parse_args()

    outdir = os.path.dirname(args.out)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    device = pick_device()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model, y_norm, (p_mu, p_std) = load_checkpoint(args.ckpt, device)
    p_mu = None if p_mu is None else np.asarray(p_mu, dtype=np.float32)
    p_std = None if p_std is None else np.asarray(p_std, dtype=np.float32)
    source_mode = ("x_norm" in ck)
    x_norm = _norm_from_ckpt(ck["x_norm"]) if source_mode else None
    in_keys = [str(k) for k in ck.get("input_keys", [])] if source_mode else []
    out_keys = [str(k) for k in ck.get("output_keys", [])]
    include_params = bool(ck.get("include_params", True)) if source_mode else False
    if source_mode and not in_keys:
        raise RuntimeError("Checkpoint missing input_keys; cannot map plasma inputs.")

    Y, y_keys_data, M, P, _ = load_npz_all(args.npz)
    data_map = {k: i for i, k in enumerate(y_keys_data)}
    if source_mode:
        miss_in = [k for k in in_keys if k not in data_map]
        if miss_in:
            raise KeyError(f"Dataset missing input channels required by ckpt: {miss_in}")
        in_idx = [data_map[k] for k in in_keys]
    else:
        in_idx = []

    fields = _parse_fields(args.fields)
    missing_data = [k for k in fields if k not in data_map]
    if missing_data:
        raise KeyError(f"Dataset missing source field(s): {missing_data}; available={y_keys_data}")

    if hasattr(y_norm, "y_keys"):
        y_keys_pred = [str(k) for k in y_norm.y_keys]
    else:
        y_keys_pred = list(out_keys)
    missing_pred = [k for k in fields if k not in y_keys_pred]
    if missing_pred:
        raise KeyError(f"Checkpoint missing source field(s): {missing_pred}; available={y_keys_pred}")

    true_idx = {k: data_map[k] for k in fields}
    pred_idx = {k: y_keys_pred.index(k) for k in fields}

    _, val_idx = split_indices(Y.shape[0], split=args.split, seed=args.seed)
    if len(val_idx) == 0:
        raise RuntimeError("No validation samples available.")
    kk = int(np.clip(args.k, 0, len(val_idx) - 1))
    gidx = int(val_idx[kk])

    run_name, grid, polys = load_mesh_polygons(args.base_dir, args.run_name)
    mesh_hw = grid.shape[:2]
    r_all = grid[:, :, :, 0].reshape(-1)
    z_all = grid[:, :, :, 1].reshape(-1)
    rmin, rmax = float(np.nanmin(r_all)), float(np.nanmax(r_all))
    zmin, zmax = float(np.nanmin(z_all)), float(np.nanmax(z_all))
    mesh_aspect = (rmax - rmin) / max(zmax - zmin, 1e-12)

    m_full = M[gidx]
    m = center_crop_2d(m_full, mesh_hw)
    if source_mode:
        x = build_model_input(
            mask2d=M[gidx],
            y_full=Y[gidx],
            p_raw=P[gidx],
            x_norm=x_norm,
            include_params=include_params,
            p_mu=p_mu,
            p_std=p_std,
            in_idx=in_idx,
        )
        y_pred_all = predict_sources(model, y_norm, x)
    else:
        p_in = scale_params(P[gidx], p_mu, p_std)
        y_pred_all = predict_fields(model, y_norm, m_full, p_in, device=device, as_numpy=True)

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
        y_true = center_crop_2d(Y[gidx, true_idx[key]], mesh_hw)
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
        evmax = float(np.nanpercentile(ae[finite_ae], 95)) if np.any(finite_ae) else 1.0

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

    fig.canvas.draw()
    fig.savefig(args.out, dpi=args.dpi)
    print(f"Saved {args.out}")
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
