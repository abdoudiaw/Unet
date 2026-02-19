#!/usr/bin/env python3
import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from plot_paper_evaluation_mesh import (
    add_mesh_panel,
    center_crop_2d,
    load_mesh_polygons,
    load_npz_all,
    split_indices,
)
from solps_ai.predict import load_checkpoint
from solps_ai.utils import pick_device
from solps_ai.data import (
    MaskedLinearStandardizer,
    MaskedLogStandardizer,
    MaskedSymLogStandardizer,
    MultiChannelNormalizer,
)

# ----------------------------
# Helper: reconstruct x_norm from checkpoint
# ----------------------------
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
    """
    Build x = [mask, normalized_plasma_inputs, (optional) params expanded]
    Returns torch.FloatTensor with shape (Cin_total, H, W)
    """
    m = mask2d.astype(np.float32, copy=False)
    m_t = torch.from_numpy(m[None, None]).float()  # (1,1,H,W)

    yin = np.asarray(y_full[in_idx], dtype=np.float32)  # (Cin,H,W)
    yin_t = torch.from_numpy(yin[None]).float()         # (1,Cin,H,W)
    yin_n = x_norm.transform(yin_t, m_t).squeeze(0)     # (Cin,H,W)

    channels = [torch.from_numpy(m[None]).float(), yin_n]  # mask is (1,H,W)

    if include_params and p_raw.size > 0:
        p_scaled = p_raw.astype(np.float32, copy=True)
        if p_mu is not None and p_std is not None:
            p_scaled = (p_scaled - p_mu) / p_std
        p_t = torch.from_numpy(p_scaled).float()
        H, W = m.shape
        channels.append(p_t.view(-1, 1, 1).expand(-1, H, W))

    x = torch.cat(channels, dim=0)
    return x


@torch.no_grad()
def predict_sources(model, y_norm, x):
    device = next(model.parameters()).device
    xb = x.unsqueeze(0).to(device)  # (1,C,H,W)
    z = model(xb)                   # (1,Cout,H,W) in normalized space
    m = xb[:, :1]                   # (1,1,H,W)
    y = y_norm.inverse(z, m).squeeze(0).detach().cpu().numpy().astype(np.float32)
    return y  # (Cout,H,W)


def main():
    ap = argparse.ArgumentParser(
        description="Sp showcase on SOLPS mesh: Pred vs True vs Percent Error (plasma(+params)->sources model)."
    )
    ap.add_argument("--npz", required=True, help="Dataset npz path (must contain Y, y_keys, mask, params/X).")
    ap.add_argument("--ckpt", required=True, help="Checkpoint path (e.g., outputs/source_from_plasma.pt).")
    ap.add_argument("--base-dir", required=True, help="SOLPS run_* base dir for mesh polygons.")
    ap.add_argument("--run-name", default=None, help="Optional run_* name for mesh.")
    ap.add_argument("--out", default="outputs/Sp_showcase.png", help="Output PNG path.")
    ap.add_argument("--split", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=0, help="k-th validation sample to plot.")
    ap.add_argument("--log-eps", type=float, default=1e-12, help="Floor for log10(Sp).")
    ap.add_argument(
        "--percent-floor-frac",
        type=float,
        default=0.02,
        help="Robust percent denom floor as frac * P95(|truth|).",
    )
    ap.add_argument("--dpi", type=int, default=400)
    args = ap.parse_args()

    outdir = os.path.dirname(args.out)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    device = pick_device()
    print("Device:", device)

    # model + y_norm + param scaler tuple
    model, y_norm, (p_mu, p_std) = load_checkpoint(args.ckpt, device)
    p_mu = None if p_mu is None else np.asarray(p_mu, dtype=np.float32)
    p_std = None if p_std is None else np.asarray(p_std, dtype=np.float32)

    # Need x_norm + input_keys/output_keys from full ckpt dict
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if "x_norm" not in ck:
        raise KeyError("Checkpoint missing x_norm. Retrain/saved with run_source_from_plasma_pipeline.py style.")
    x_norm = _norm_from_ckpt(ck["x_norm"])
    in_keys = [str(k) for k in ck.get("input_keys", [])]
    out_keys = [str(k) for k in ck.get("output_keys", [])]
    include_params = bool(ck.get("include_params", True))

    # Data
    Y, y_keys_data, M, P, _ = load_npz_all(args.npz)
    data_map = {k: i for i, k in enumerate(y_keys_data)}

    if not in_keys:
        raise RuntimeError("Checkpoint missing input_keys; cannot map plasma inputs.")
    if "Sp" not in data_map:
        raise KeyError("'Sp' not found in dataset y_keys.")
    if "Sp" not in out_keys and (hasattr(y_norm, "y_keys") and "Sp" not in list(y_norm.y_keys)):
        raise KeyError("'Sp' not found in checkpoint outputs.")

    miss_in = [k for k in in_keys if k not in data_map]
    if miss_in:
        raise KeyError(f"Dataset missing input channels required by ckpt: {miss_in}")
    in_idx = [data_map[k] for k in in_keys]

    # Which output channel index is Sp?
    if hasattr(y_norm, "y_keys"):
        y_keys_pred = [str(k) for k in y_norm.y_keys]
    else:
        y_keys_pred = list(out_keys)
    sp_pred_idx = y_keys_pred.index("Sp")
    sp_true_idx = data_map["Sp"]

    # Pick validation sample
    _, val_idx = split_indices(Y.shape[0], split=args.split, seed=args.seed)
    if len(val_idx) == 0:
        raise RuntimeError("No validation samples available (check split/seed).")
    kk = int(np.clip(args.k, 0, len(val_idx) - 1))
    gidx = int(val_idx[kk])
    print(f"Using val sample: k={kk} -> global_idx={gidx}")

    # Mesh polygons
    run_name, grid, polys = load_mesh_polygons(args.base_dir, args.run_name)
    mesh_hw = grid.shape[:2]

    # Predict
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
    y_pred_full = y_pred_all[sp_pred_idx]     # (H,W) maybe training grid
    y_true_full = Y[gidx, sp_true_idx]
    m_full = M[gidx]

    # Crop to mesh
    y_true = center_crop_2d(y_true_full, mesh_hw)
    y_pred = center_crop_2d(y_pred_full, mesh_hw)
    m = center_crop_2d(m_full, mesh_hw)

    yt = np.where(m > 0.5, y_true, np.nan)
    yp = np.where(m > 0.5, y_pred, np.nan)
#
#    # Log display (Sp is positive; floor it)
#    yt_plot = np.log10(np.maximum(yt, args.log_eps))
#    yp_plot = np.log10(np.maximum(yp, args.log_eps))
#
#    all_vals = np.concatenate([yt_plot[np.isfinite(yt_plot)], yp_plot[np.isfinite(yp_plot)]])
#    vmin = float(np.nanpercentile(all_vals, 1))
#    vmax = float(np.nanpercentile(all_vals, 99))


    # Linear display
    yt_plot = yt*1e-22
    yp_plot = yp*1e-22

    all_vals = np.concatenate([yt_plot[np.isfinite(yt_plot)], yp_plot[np.isfinite(yp_plot)]])
    vmin = float(np.nanpercentile(all_vals, 1))
    vmax = float(np.nanpercentile(all_vals, 99))

    # Robust percent error
    truth_valid = yt[np.isfinite(yt)]
    p95 = float(np.nanpercentile(np.abs(truth_valid), 95)) if truth_valid.size else 1.0
    err_floor = args.percent_floor_frac * p95
    denom = np.maximum(np.abs(yt), max(args.log_eps, err_floor))
    err_plot = 100.0 * np.abs(yp - yt) / denom
    err_plot = np.where(np.isfinite(err_plot), err_plot, np.nan)
    err_vmax = float(np.nanpercentile(err_plot[np.isfinite(err_plot)], 95)) if np.any(np.isfinite(err_plot)) else 1.0
    abs_err = np.abs(yp - yt) * 1e-22  # same scaling as your plot


    # Plot styling similar to your density showcase
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
        constrained_layout=False,
    )

    add_mesh_panel(
        fig, ax1, polys, yp_plot, r"Pred: $S_p\ (10^{22}\,\mathrm{s}^{-1})$", cmap="plasma",
        vmin=vmin, vmax=vmax, show_xlabel=True, show_ylabel=False, add_colorbar=False
    )
    add_mesh_panel(
        fig, ax0, polys, yt_plot, r"True: $S_p\ (10^{22}\,\mathrm{s}^{-1})$", cmap="plasma",
        vmin=vmin, vmax=vmax, show_xlabel=True, show_ylabel=True, add_colorbar=False
    )


    add_mesh_panel(
        fig, ax2, polys, abs_err,
        r"Abs. error $S_p\ (10^{22}\,\mathrm{s}^{-1})$",
        cmap="magma",
        vmin=0.0,
        vmax=np.nanpercentile(abs_err[np.isfinite(abs_err)], 95),
        show_xlabel=True, show_ylabel=False, add_colorbar=False
    )


    # Match framing
    r_all = grid[:, :, :, 0].reshape(-1)
    z_all = grid[:, :, :, 1].reshape(-1)
    rmin, rmax = float(np.nanmin(r_all)), float(np.nanmax(r_all))
    zmin, zmax = float(np.nanmin(z_all)), float(np.nanmax(z_all))
    rpad = 0.02 * (rmax - rmin)
    zpad = 0.02 * (zmax - zmin)
    for ax in (ax0, ax1, ax2):
        ax.set_ylim(zmin - zpad, zmax + zpad)
        ax.set_xlim(rmin - rpad, rmax + rpad)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(1.0, 2.5)
        
    ax0.set_anchor("E")
    ax1.set_anchor("C")
    ax2.set_anchor("W")

    # Inset colorbars
    cbax0 = inset_axes(ax0, width="42%", height="4%", loc="center")
    cbax1 = inset_axes(ax1, width="42%", height="4%", loc="center")
    cbax2 = inset_axes(ax2, width="42%", height="4%", loc="center")
    cbar0 = fig.colorbar(ax0.collections[0], cax=cbax0, orientation="horizontal")
#    cbar0.set_label(r"$\log_{10}(S_p)$", fontsize=11)
    cbar0.set_label(r"$S_p$", fontsize=11)

    cbar1 = fig.colorbar(ax1.collections[0], cax=cbax1, orientation="horizontal")
    cbar1.set_label(r"$S_p$", fontsize=11)
#    cbar1.set_label(r"$\log_{10}(S_p)$", fontsize=11)
    cbar2 = fig.colorbar(ax2.collections[0], cax=cbax2, orientation="horizontal")
#    cbar2.set_label("abs. error", fontsize= 11)
    cbar2.set_label(r"$\Delta S_p$", fontsize=11)
    
    for ax in (ax0, ax1, ax2):
        ax.minorticks_on()
        ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
        ax.grid(True, linestyle="--", alpha=0.3)

    ax1.set_ylabel("")
    ax2.set_ylabel("")
    ax1.tick_params(labelleft=False)
    ax2.tick_params(labelleft=False)

    fig.subplots_adjust(wspace=0.0, hspace=0.0)
    fig.canvas.draw()
#    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", pad_inches=0.0)
    fig.savefig(args.out, dpi=400, bbox_inches="tight")
    print(f"Saved {args.out}")
    plt.show()

if __name__ == "__main__":
    main()
