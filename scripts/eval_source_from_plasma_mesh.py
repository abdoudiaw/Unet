import argparse
import csv
import json
import os

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import numpy as np
from scipy.stats import spearmanr
import torch

import quixote
from quixote import SolpsData

from solps_ai.predict import load_checkpoint
from solps_ai.utils import pick_device
from solps_ai.data import (
    MaskedLinearStandardizer,
    MaskedLogStandardizer,
    MaskedSymLogStandardizer,
    MultiChannelNormalizer,
)

UNITS_BY_KEY = {
    "Te": "eV",
    "Ti": "eV",
    "ne": "m^-3",
    "ni": "m^-3",
    "ua": "m/s",
    "Sp": "m^-3 s^-1",
    "Qp": "W/m^3",
    "Qe": "W/m^3",
    "Qi": "W/m^3",
    "Sm": "N/m^3",
}
SIGNED_FIELDS = {"ua", "Qp", "Qe", "Qi", "Sm"}
SPARSE_POSITIVE_FIELDS = {"Sp"}
LOG_DISPLAY_FIELDS = {"Sp"}


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


def split_indices(N, split=0.85, seed=42):
    idx = np.arange(N)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    cut = int(split * N)
    return idx[:cut], idx[cut:]


def load_npz_all(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    if "Y" not in d.files or "y_keys" not in d.files:
        raise KeyError("Dataset must contain Y and y_keys.")
    Y = d["Y"].astype(np.float32)
    y_keys = [str(k) for k in d["y_keys"]]
    if "mask" in d.files:
        m = d["mask"]
        if m.ndim == 2:
            m = np.repeat(m[None, :, :], Y.shape[0], axis=0)
        m = (m > 0.5).astype(np.float32)
    else:
        m = np.ones((Y.shape[0], Y.shape[2], Y.shape[3]), dtype=np.float32)
    if "params" in d.files:
        p = d["params"].astype(np.float32)
    elif "X" in d.files:
        p = d["X"].astype(np.float32)
    else:
        p = np.zeros((Y.shape[0], 0), dtype=np.float32)
    return Y, y_keys, m, p


def pick_reference_run(base_dir):
    runs = sorted(
        d for d in os.listdir(base_dir)
        if d.startswith("run_") and os.path.isdir(os.path.join(base_dir, d))
    )
    if not runs:
        raise RuntimeError(f"No run_* folders found under {base_dir}")
    for rn in runs:
        if os.path.exists(os.path.join(base_dir, rn, "params.json")):
            return rn
    return runs[0]


def load_mesh_polygons(base_dir, run_name=None):
    rn = run_name if run_name is not None else pick_reference_run(base_dir)
    shot = SolpsData(os.path.join(quixote.module_path(), os.path.join(base_dir, rn)))
    grid = np.asarray(shot.grid, dtype=np.float32)
    H, W = grid.shape[:2]
    polys = [Polygon(grid[i, j], closed=True) for i in range(H) for j in range(W)]
    return rn, grid, polys


def center_crop_2d(a, shape_hw):
    H0, W0 = shape_hw
    H, W = a.shape
    if (H, W) == (H0, W0):
        return a
    if H < H0 or W < W0:
        raise ValueError(f"Cannot crop from {(H,W)} to larger {(H0,W0)}")
    top = (H - H0) // 2
    left = (W - W0) // 2
    return a[top:top + H0, left:left + W0]


def pick_field_error_mode(requested_mode, y_key):
    if requested_mode != "auto":
        return requested_mode
    if y_key in SIGNED_FIELDS or y_key in SPARSE_POSITIVE_FIELDS:
        return "scaled_abs"
    return "percent_robust"


def pick_field_log_display(requested_mode, y_key):
    if requested_mode == "on":
        return True
    if requested_mode == "off":
        return False
    return y_key in LOG_DISPLAY_FIELDS


def compute_error_map(y_true, y_pred, mask, mode="abs", eps=1e-3, scale=None, rel_floor=0.02):
    if mode == "abs":
        e = np.abs(y_pred - y_true)
    elif mode == "percent":
        denom = np.maximum(np.abs(y_true), eps)
        e = 100.0 * np.abs(y_pred - y_true) / denom
    elif mode == "percent_robust":
        scl = eps if scale is None else max(float(scale), eps)
        denom = np.maximum(np.abs(y_true), rel_floor * scl)
        denom = np.maximum(denom, eps)
        e = 100.0 * np.abs(y_pred - y_true) / denom
    elif mode == "scaled_abs":
        scl = eps if scale is None else max(float(scale), eps)
        e = 100.0 * np.abs(y_pred - y_true) / scl
    elif mode == "smape":
        denom = np.abs(y_true) + np.abs(y_pred) + eps
        e = 200.0 * np.abs(y_pred - y_true) / denom
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return np.where(mask > 0.5, e, np.nan)


def apply_error_sign(err, y_true, y_pred, sign_mode):
    if sign_mode == "absolute":
        return err
    if sign_mode == "signed":
        return np.sign(y_pred - y_true) * err
    raise ValueError(f"Unknown error sign mode: {sign_mode}")


def add_mesh_panel(fig, ax, polys, values2d, title, cmap, vmin=None, vmax=None):
    vals = values2d.reshape(-1)
    mvals = np.ma.masked_invalid(vals)
    pc = PatchCollection(polys, cmap=cmap, edgecolor="none", linewidths=0.0)
    pc.set_array(mvals)
    if vmin is not None and vmax is not None:
        pc.set_clim(vmin, vmax)
    ax.add_collection(pc)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title(title)
    fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.03)


def build_model_input(mask2d, y_in_full, p_raw, x_norm, include_params, p_mu, p_std, in_idx):
    m = mask2d.astype(np.float32, copy=False)
    m_t = torch.from_numpy(m[None, None]).float()

    y_in = y_in_full[in_idx]
    y_in = np.asarray(y_in, dtype=np.float32)
    y_in_t = torch.from_numpy(y_in[None]).float()
    y_in_n = x_norm.transform(y_in_t, m_t).squeeze(0)  # (Cin,H,W)

    channels = [torch.from_numpy(m[None]).float(), y_in_n]
    p_scaled = p_raw.copy()
    if include_params and p_scaled.size > 0:
        if p_mu is not None and p_std is not None:
            p_scaled = (p_scaled - p_mu) / p_std
        p_t = torch.from_numpy(p_scaled).float()
        H, W = m.shape
        channels.append(p_t.view(-1, 1, 1).expand(-1, H, W))
    x = torch.cat(channels, dim=0)
    return x


def predict_one(model, y_norm, x):
    device = next(model.parameters()).device
    xb = x.unsqueeze(0).to(device)
    with torch.no_grad():
        z = model(xb)
        # mask is x[0] channel by construction
        m = xb[:, :1]
        y = y_norm.inverse(z, m).squeeze(0).detach().cpu().numpy().astype(np.float32)
    return y


def compute_metrics(y_true, y_pred, m):
    v = m > 0.5
    t = y_true[v].astype(np.float64, copy=False)
    p = y_pred[v].astype(np.float64, copy=False)
    if t.size == 0:
        return {"mae": np.nan, "rmse": np.nan, "pearson": np.nan, "spearman": np.nan}
    e = p - t
    mae = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(np.square(e))))
    pear = float(np.corrcoef(t, p)[0, 1]) if t.size > 2 else np.nan
    spear = float(spearmanr(t, p).correlation) if t.size > 2 else np.nan
    return {"mae": mae, "rmse": rmse, "pearson": pear, "spearman": spear}


def plot_triplet(path, polys, y_true, y_pred, mask, y_key, run_name, sample_idx, error_mode, error_sign, error_eps, error_rel_floor, log_display, log_display_eps):
    yt = np.where(mask > 0.5, y_true, np.nan)
    yp = np.where(mask > 0.5, y_pred, np.nan)
    v = mask > 0.5
    vscale = float(np.nanpercentile(np.abs(y_true[v]), 95)) if np.any(v) else error_eps
    err = compute_error_map(y_true, y_pred, mask, mode=error_mode, eps=error_eps, scale=vscale, rel_floor=error_rel_floor)
    err = apply_error_sign(err, y_true, y_pred, error_sign)

    if log_display:
        yt_plot = np.log10(np.maximum(yt, log_display_eps))
        yp_plot = np.log10(np.maximum(yp, log_display_eps))
        t0 = "Truth [log10]"
        t1 = "Prediction [log10]"
    else:
        yt_plot = yt
        yp_plot = yp
        t0 = "Truth"
        t1 = "Prediction"

    vmin = float(np.nanpercentile(yt_plot, 1))
    vmax = float(np.nanpercentile(yt_plot, 99))
    if error_sign == "signed":
        emax = float(np.nanpercentile(np.abs(err), 95))
        emin = -emax
        ecmap = "coolwarm"
    else:
        emin = 0.0
        emax = float(np.nanpercentile(err, 95))
        ecmap = "magma"
    err_title = {"abs": "Abs Error", "percent": "Percent Error [%]", "percent_robust": "Percent Error Robust [%]", "scaled_abs": "Scaled Abs Error [%]", "smape": "sMAPE [%]"}[error_mode]
    if error_sign == "signed":
        err_title = f"Signed {err_title}"

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    add_mesh_panel(fig, axes[0], polys, yt_plot, t0, cmap="inferno", vmin=vmin, vmax=vmax)
    add_mesh_panel(fig, axes[1], polys, yp_plot, t1, cmap="inferno", vmin=vmin, vmax=vmax)
    add_mesh_panel(fig, axes[2], polys, err, err_title, cmap=ecmap, vmin=emin, vmax=emax)
    fig.suptitle(f"{y_key} source-mesh-eval | run={run_name} | val_idx={sample_idx}")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--ckpt", required=True, help="outputs/source_from_plasma.pt")
    ap.add_argument("--base-dir", required=True, help="SOLPS run_* base for mesh polygons")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--all-fields", action="store_true")
    ap.add_argument("--y-key", default="Sp")
    ap.add_argument("--outdir", default="outputs/source_eval_mesh")
    ap.add_argument("--split", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-examples", type=int, default=3)
    ap.add_argument("--error-mode", choices=["auto", "abs", "percent", "percent_robust", "scaled_abs", "smape"], default="abs")
    ap.add_argument("--error-sign", choices=["absolute", "signed"], default="absolute")
    ap.add_argument("--error-eps", type=float, default=1e-3)
    ap.add_argument("--error-rel-floor", type=float, default=0.02)
    ap.add_argument("--log-display", choices=["auto", "off", "on"], default="auto")
    ap.add_argument("--log-display-eps", type=float, default=1e-12)
    ap.add_argument("--paper-grid", action="store_true")
    ap.add_argument("--paper-grid-k", type=int, default=0)
    ap.add_argument("--paper-grid-path", default=None)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = pick_device()
    print("Device:", device)

    model, y_norm, (p_mu, p_std) = load_checkpoint(args.ckpt, device)
    p_mu = None if p_mu is None else np.asarray(p_mu, dtype=np.float32)
    p_std = None if p_std is None else np.asarray(p_std, dtype=np.float32)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if "x_norm" not in ck:
        raise KeyError("Checkpoint missing x_norm. Retrain with run_source_from_plasma_pipeline.py")
    x_norm = _norm_from_ckpt(ck["x_norm"])
    in_keys = [str(k) for k in ck.get("input_keys", [])]
    out_keys = [str(k) for k in ck.get("output_keys", [])]
    include_params = bool(ck.get("include_params", True))

    Y, y_keys_all, M, P = load_npz_all(args.npz)
    data_map = {k: i for i, k in enumerate(y_keys_all)}
    miss_in = [k for k in in_keys if k not in data_map]
    miss_out = [k for k in out_keys if k not in data_map]
    if miss_in or miss_out:
        raise KeyError(f"Dataset missing channels. in_missing={miss_in}, out_missing={miss_out}")
    in_idx = [data_map[k] for k in in_keys]
    out_idx = {k: data_map[k] for k in out_keys}

    _, val_idx = split_indices(Y.shape[0], split=args.split, seed=args.seed)
    if args.all_fields:
        fields = list(out_keys)
    else:
        fields = [args.y_key]
    for k in fields:
        if k not in out_idx:
            raise KeyError(f"Requested y-key {k!r} not in output_keys={out_keys}")

    run_name, grid, polys = load_mesh_polygons(args.base_dir, args.run_name)
    mesh_hw = grid.shape[:2]

    summary = []
    for k in fields:
        print(f"[field] {k}")
        out_k = os.path.join(args.outdir, k) if len(fields) > 1 else args.outdir
        os.makedirs(out_k, exist_ok=True)
        y_idx = out_idx[k]
        mode_eff = pick_field_error_mode(args.error_mode, k)
        log_eff = pick_field_log_display(args.log_display, k)

        maes, rmses, pears, spears = [], [], [], []
        for j, gidx in enumerate(val_idx):
            x = build_model_input(
                mask2d=M[gidx], y_in_full=Y[gidx], p_raw=P[gidx],
                x_norm=x_norm, include_params=include_params, p_mu=p_mu, p_std=p_std,
                in_idx=in_idx,
            )
            y_pred_all = predict_one(model, y_norm, x)
            y_true = center_crop_2d(Y[gidx, y_idx], mesh_hw)
            m = center_crop_2d(M[gidx], mesh_hw)
            pred_ch = out_keys.index(k)
            y_pred = center_crop_2d(y_pred_all[pred_ch], mesh_hw)
            met = compute_metrics(y_true, y_pred, m)
            maes.append(met["mae"]); rmses.append(met["rmse"])
            pears.append(met["pearson"]); spears.append(met["spearman"])
            if j < args.n_examples:
                plot_triplet(
                    path=os.path.join(out_k, f"mesh_example_{j:02d}.png"),
                    polys=polys, y_true=y_true, y_pred=y_pred, mask=m, y_key=k,
                    run_name=run_name, sample_idx=int(gidx), error_mode=mode_eff,
                    error_sign=args.error_sign, error_eps=args.error_eps, error_rel_floor=args.error_rel_floor,
                    log_display=log_eff, log_display_eps=args.log_display_eps,
                )

        row = {
            "y_key": k,
            "unit": UNITS_BY_KEY.get(k, ""),
            "error_mode_used": mode_eff,
            "error_sign_used": args.error_sign,
            "display_scale_used": "log10" if log_eff else "linear",
            "global_mae": float(np.nanmean(np.asarray(maes, dtype=float))),
            "global_rmse": float(np.nanmean(np.asarray(rmses, dtype=float))),
            "mean_sample_pearson": float(np.nanmean(np.asarray(pears, dtype=float))),
            "mean_sample_spearman": float(np.nanmean(np.asarray(spears, dtype=float))),
        }
        with open(os.path.join(out_k, "metrics.json"), "w") as f:
            json.dump(row, f, indent=2)
        summary.append(row)

    cols = [
        "y_key", "unit", "global_mae", "global_rmse", "mean_sample_pearson", "mean_sample_spearman",
        "error_mode_used", "error_sign_used", "display_scale_used",
    ]
    with open(os.path.join(args.outdir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in summary:
            w.writerow({c: r.get(c) for c in cols})
    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved:", os.path.join(args.outdir, "summary.csv"))

    if args.paper_grid:
        kk = int(np.clip(args.paper_grid_k, 0, len(val_idx) - 1))
        gidx = int(val_idx[kk])
        grid_path = args.paper_grid_path or os.path.join(args.outdir, "paper_grid_sources.png")
        nrows = len(fields)
        fig, axes = plt.subplots(nrows, 3, figsize=(14, max(3.0 * nrows, 4.5)), constrained_layout=True)
        if nrows == 1:
            axes = np.array([axes])
        x = build_model_input(
            mask2d=M[gidx], y_in_full=Y[gidx], p_raw=P[gidx],
            x_norm=x_norm, include_params=include_params, p_mu=p_mu, p_std=p_std,
            in_idx=in_idx,
        )
        y_pred_all = predict_one(model, y_norm, x)
        for r, k in enumerate(fields):
            y_idx = out_idx[k]
            pred_ch = out_keys.index(k)
            y_true = center_crop_2d(Y[gidx, y_idx], mesh_hw)
            m = center_crop_2d(M[gidx], mesh_hw)
            y_pred = center_crop_2d(y_pred_all[pred_ch], mesh_hw)
            mode_eff = pick_field_error_mode(args.error_mode, k)
            log_eff = pick_field_log_display(args.log_display, k)
            v = m > 0.5
            yt = np.where(v, y_true, np.nan)
            yp = np.where(v, y_pred, np.nan)
            vscale = float(np.nanpercentile(np.abs(y_true[v]), 95)) if np.any(v) else args.error_eps
            err = compute_error_map(y_true, y_pred, m, mode=mode_eff, eps=args.error_eps, scale=vscale, rel_floor=args.error_rel_floor)
            err = apply_error_sign(err, y_true, y_pred, args.error_sign)
            if log_eff:
                yt_p = np.log10(np.maximum(yt, args.log_display_eps))
                yp_p = np.log10(np.maximum(yp, args.log_display_eps))
                t0 = f"Truth [log10] | {k} [{UNITS_BY_KEY.get(k, '')}]"
                t1 = f"Pred [log10] | {k} [{UNITS_BY_KEY.get(k, '')}]"
            else:
                yt_p = yt
                yp_p = yp
                t0 = f"Truth | {k} [{UNITS_BY_KEY.get(k, '')}]"
                t1 = f"Pred | {k} [{UNITS_BY_KEY.get(k, '')}]"
            vmin = float(np.nanpercentile(yt_p, 1))
            vmax = float(np.nanpercentile(yt_p, 99))
            if args.error_sign == "signed":
                emax = float(np.nanpercentile(np.abs(err), 95)); emin = -emax; ecmap = "coolwarm"
            else:
                emax = float(np.nanpercentile(err, 95)); emin = 0.0; ecmap = "magma"
            err_lbl = {"abs": "Abs Error", "percent": "Percent Error [%]", "percent_robust": "Percent Error Robust [%]", "scaled_abs": "Scaled Abs Error [%]", "smape": "sMAPE [%]"}[mode_eff]
            if args.error_sign == "signed":
                err_lbl = f"Signed {err_lbl}"
            add_mesh_panel(fig, axes[r, 0], polys, yt_p, t0, "inferno", vmin=vmin, vmax=vmax)
            add_mesh_panel(fig, axes[r, 1], polys, yp_p, t1, "inferno", vmin=vmin, vmax=vmax)
            add_mesh_panel(fig, axes[r, 2], polys, err, f"{err_lbl} | {k}", ecmap, vmin=emin, vmax=emax)
        fig.suptitle(f"Plasma->Sources Mesh Evaluation | run={run_name} | val_idx={gidx}", fontsize=14)
        fig.savefig(grid_path, dpi=240, bbox_inches="tight")
        plt.close(fig)
        print("Saved:", grid_path)

    print("Saved diagnostics to:", args.outdir)


if __name__ == "__main__":
    main()
