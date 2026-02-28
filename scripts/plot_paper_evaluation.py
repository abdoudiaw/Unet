# Copyright 2025-2026 Oak Ridge National Laboratory
# @authors: Abdourahmane (Abdou) Diaw - diawa@ornl.gov
#
# SPDX-License-Identifier: MIT

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

import torch

from solpex.predict import load_checkpoint, predict_fields, scale_params
from solpex.utils import pick_device


def load_npz_field(npz_path, y_key="Te"):
    d = np.load(npz_path, allow_pickle=True)

    if "Y" in d.files:
        y_keys = [str(k) for k in d["y_keys"]]
        if y_key not in y_keys:
            raise KeyError(f"{y_key!r} not in y_keys={y_keys}")
        k = y_keys.index(y_key)
        y = d["Y"][:, k].astype(np.float32)
    elif y_key == "Te" and "Te" in d.files:
        y = d["Te"].astype(np.float32)
    else:
        raise KeyError(f"Could not find field {y_key!r} in dataset.")

    if "mask" in d.files:
        mask = d["mask"]
        if mask.ndim == 2:
            mask = np.repeat(mask[None, :, :], y.shape[0], axis=0)
        mask = (mask > 0.5).astype(np.float32)
    else:
        mask = np.ones_like(y, dtype=np.float32)

    if "params" in d.files:
        params = d["params"].astype(np.float32)
    elif "X" in d.files:
        params = d["X"].astype(np.float32)
    else:
        params = np.zeros((y.shape[0], 0), dtype=np.float32)

    if "param_keys" in d.files:
        param_keys = [str(k) for k in d["param_keys"]]
    elif "x_keys" in d.files:
        param_keys = [str(k) for k in d["x_keys"]]
    else:
        param_keys = [f"p{i}" for i in range(params.shape[1])]

    R2d, Z2d = None, None
    if "Rg" in d.files and "Zg" in d.files:
        R2d = d["Rg"].astype(np.float32)
        Z2d = d["Zg"].astype(np.float32)
    elif "R2d" in d.files and "Z2d" in d.files:
        R2d = d["R2d"].astype(np.float32)
        Z2d = d["Z2d"].astype(np.float32)

    return y, mask, params, param_keys, R2d, Z2d


def split_indices(N, split=0.85, seed=42):
    idx = np.arange(N)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    cut = int(split * N)
    return idx[:cut], idx[cut:]


def compute_metrics(y_true, y_pred, m):
    v = m > 0.5
    t = y_true[v]
    p = y_pred[v]
    if t.size == 0:
        return {"mae": np.nan, "rmse": np.nan, "pearson": np.nan, "spearman": np.nan}
    e = p - t
    mae = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(e * e)))
    if t.size > 2:
        pearson = float(np.corrcoef(t, p)[0, 1])
        spearman = float(spearmanr(t, p).correlation)
    else:
        pearson = np.nan
        spearman = np.nan
    return {"mae": mae, "rmse": rmse, "pearson": pearson, "spearman": spearman}


def compute_log_metrics(y_true, y_pred, m, eps=1e-12):
    v = m > 0.5
    t = y_true[v]
    p = y_pred[v]
    if t.size == 0:
        return {"log_mae": np.nan, "log_rmse": np.nan, "log_pearson": np.nan, "log_spearman": np.nan}
    t = np.maximum(t, eps)
    p = np.maximum(p, eps)
    lt = np.log10(t)
    lp = np.log10(p)
    e = lp - lt
    log_mae = float(np.mean(np.abs(e)))
    log_rmse = float(np.sqrt(np.mean(e * e)))
    if lt.size > 2:
        log_pearson = float(np.corrcoef(lt, lp)[0, 1])
        log_spearman = float(spearmanr(lt, lp).correlation)
    else:
        log_pearson = np.nan
        log_spearman = np.nan
    return {
        "log_mae": log_mae,
        "log_rmse": log_rmse,
        "log_pearson": log_pearson,
        "log_spearman": log_spearman,
    }


def compute_error_map(y_true, y_pred, mask, mode="abs", eps=1e-3):
    v = mask > 0.5
    if mode == "abs":
        err = np.abs(y_pred - y_true)
    elif mode == "percent":
        denom = np.maximum(np.abs(y_true), eps)
        err = 100.0 * np.abs(y_pred - y_true) / denom
    elif mode == "smape":
        denom = np.abs(y_true) + np.abs(y_pred) + eps
        err = 200.0 * np.abs(y_pred - y_true) / denom
    else:
        raise ValueError(f"Unknown error mode: {mode}")
    return np.where(v, err, np.nan)


def plot_example_maps(save_path, y_true, y_pred, mask, title, R2d=None, Z2d=None, error_mode="abs"):
    v = mask > 0.5
    yt = np.where(v, y_true, np.nan)
    yp = np.where(v, y_pred, np.nan)
    err = compute_error_map(y_true, y_pred, mask, mode=error_mode)

    vmin = np.nanpercentile(yt, 1)
    vmax = np.nanpercentile(yt, 99)
    emax = np.nanpercentile(err, 99)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    err_label = {"abs": "Abs Error", "percent": "Percent Error [%]", "smape": "sMAPE [%]"}[error_mode]
    labels = ["Truth", "Prediction", err_label]
    arrays = [yt, yp, err]
    cmaps = ["inferno", "inferno", "magma"]
    lims = [(vmin, vmax), (vmin, vmax), (0.0, emax)]

    for ax, lbl, arr, cmap, lim in zip(axes, labels, arrays, cmaps, lims):
        if R2d is not None and Z2d is not None and R2d.shape == arr.shape and Z2d.shape == arr.shape:
            im = ax.pcolormesh(R2d, Z2d, arr, shading="auto", cmap=cmap, vmin=lim[0], vmax=lim[1])
            ax.set_xlabel("R [m]")
            ax.set_ylabel("Z [m]")
            ax.set_aspect("equal")
        else:
            im = ax.imshow(arr, origin="lower", cmap=cmap, vmin=lim[0], vmax=lim[1], aspect="auto")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
        ax.set_title(lbl)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    fig.suptitle(title)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=None, help="Path to dataset .npz")
    ap.add_argument("--npz_path", default=None, help="Alias for --npz")
    ap.add_argument("--ckpt", required=True, help="Path to model checkpoint .pt")
    ap.add_argument("--y-key", default="Te")
    ap.add_argument("--outdir", default="outputs/paper_eval")
    ap.add_argument("--split", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-examples", type=int, default=3)
    ap.add_argument("--scatter-points", type=int, default=50000)
    ap.add_argument("--sweep-param", type=str, default=None, help="Parameter name for 1D sensitivity")
    ap.add_argument("--error-mode", choices=["abs", "percent", "smape"], default="abs")
    ap.add_argument("--error-eps", type=float, default=1e-3)
    ap.add_argument("--log-metrics", action="store_true", help="Also compute log10-space metrics for positive fields.")
    ap.add_argument("--log-metrics-eps", type=float, default=1e-12)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    device = pick_device()
    print("Device:", device)
    npz_path = args.npz if args.npz is not None else args.npz_path
    if npz_path is None:
        raise ValueError("Provide --npz (or --npz_path).")

    model, norm, (p_mu, p_std) = load_checkpoint(args.ckpt, device)
    ch_idx = 0
    if hasattr(norm, "y_keys"):
        norm_keys = [str(k) for k in norm.y_keys]
        if args.y_key not in norm_keys:
            raise KeyError(f"--y-key {args.y_key!r} not found in checkpoint y_keys={norm_keys}")
        ch_idx = norm_keys.index(args.y_key)

    y_all, m_all, p_all, p_keys, R2d, Z2d = load_npz_field(npz_path, y_key=args.y_key)
    _, val_idx = split_indices(y_all.shape[0], split=args.split, seed=args.seed)

    y_true_cat = []
    y_pred_cat = []
    abs_err_cat = []
    pct_err_cat = []
    log_true_cat = []
    log_pred_cat = []
    sample_metrics = []

    for j, gidx in enumerate(val_idx):
        y_true = y_all[gidx]
        mask = m_all[gidx]
        p_raw = p_all[gidx]
        p_in = scale_params(p_raw, p_mu, p_std)
        y_pred_all = predict_fields(model, norm, mask, p_in, device=device, as_numpy=True)
        if y_pred_all.ndim != 3:
            raise RuntimeError(f"Expected predicted fields with shape (C,H,W), got {y_pred_all.shape}")
        if ch_idx >= y_pred_all.shape[0]:
            raise RuntimeError(f"Requested channel idx={ch_idx}, but model returned C={y_pred_all.shape[0]}")
        y_pred = y_pred_all[ch_idx]

        metrics = compute_metrics(y_true, y_pred, mask)
        sample_metrics.append(metrics)

        v = mask > 0.5
        y_true_cat.append(y_true[v])
        y_pred_cat.append(y_pred[v])
        abs_err_cat.append(np.abs(y_pred[v] - y_true[v]))
        pct_map = compute_error_map(y_true, y_pred, mask, mode="percent", eps=args.error_eps)
        pct_err_cat.append(pct_map[v])
        if args.log_metrics:
            pos = np.maximum(y_true[v], args.log_metrics_eps)
            ppos = np.maximum(y_pred[v], args.log_metrics_eps)
            log_true_cat.append(np.log10(pos))
            log_pred_cat.append(np.log10(ppos))

        if j < args.n_examples:
            plot_example_maps(
                save_path=os.path.join(args.outdir, f"example_map_{j:02d}.png"),
                y_true=y_true,
                y_pred=y_pred,
                mask=mask,
                title=f"{args.y_key} example idx={int(gidx)}",
                R2d=R2d,
                Z2d=Z2d,
                error_mode=args.error_mode,
            )

    y_true_cat = np.concatenate(y_true_cat)
    y_pred_cat = np.concatenate(y_pred_cat)
    abs_err_cat = np.concatenate(abs_err_cat)
    pct_err_cat = np.concatenate(pct_err_cat)
    if args.log_metrics:
        log_true_cat = np.concatenate(log_true_cat)
        log_pred_cat = np.concatenate(log_pred_cat)

    rng = np.random.default_rng(args.seed)
    take = min(args.scatter_points, y_true_cat.size)
    ii = rng.choice(y_true_cat.size, size=take, replace=False)

    pearson_global = float(np.corrcoef(y_true_cat, y_pred_cat)[0, 1])
    spearman_global = float(spearmanr(y_true_cat, y_pred_cat).correlation)
    mae_global = float(np.mean(np.abs(y_pred_cat - y_true_cat)))
    rmse_global = float(np.sqrt(np.mean((y_pred_cat - y_true_cat) ** 2)))
    p90_abs_err = float(np.percentile(abs_err_cat, 90))
    p95_abs_err = float(np.percentile(abs_err_cat, 95))
    p90_pct_err = float(np.percentile(pct_err_cat, 90))
    p95_pct_err = float(np.percentile(pct_err_cat, 95))

    metrics_out = {
        "y_key": args.y_key,
        "n_val_samples": int(len(val_idx)),
        "n_valid_points": int(y_true_cat.size),
        "global_mae": mae_global,
        "global_rmse": rmse_global,
        "global_pearson": pearson_global,
        "global_spearman": spearman_global,
        "p90_abs_error": p90_abs_err,
        "p95_abs_error": p95_abs_err,
        "p90_percent_error": p90_pct_err,
        "p95_percent_error": p95_pct_err,
        "mean_sample_pearson": float(np.nanmean([m["pearson"] for m in sample_metrics])),
        "mean_sample_spearman": float(np.nanmean([m["spearman"] for m in sample_metrics])),
    }
    if args.log_metrics:
        log_global = compute_log_metrics(y_true_cat, y_pred_cat, np.ones_like(y_true_cat, dtype=np.float32), eps=args.log_metrics_eps)
        metrics_out.update(log_global)
        take_log = min(args.scatter_points, log_true_cat.size)
        ii_log = rng.choice(log_true_cat.size, size=take_log, replace=False)
        fig = plt.figure(figsize=(6, 5))
        plt.hexbin(log_true_cat[ii_log], log_pred_cat[ii_log], gridsize=80, bins="log", mincnt=1)
        lo = float(np.percentile(log_true_cat[ii_log], 1))
        hi = float(np.percentile(log_true_cat[ii_log], 99))
        plt.plot([lo, hi], [lo, hi], "w--", linewidth=1.5, label="y=x")
        plt.xlabel(f"log10 True {args.y_key}")
        plt.ylabel(f"log10 Predicted {args.y_key}")
        plt.title(
            f"log10 Scatter (Pearson={metrics_out['log_pearson']:.4f}, "
            f"Spearman={metrics_out['log_spearman']:.4f})"
        )
        plt.colorbar(label="log10(count)")
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "scatter_log10_true_vs_pred.png"), dpi=220)
        plt.close(fig)
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(metrics_out, f, indent=2)

    fig = plt.figure(figsize=(6, 5))
    plt.hexbin(y_true_cat[ii], y_pred_cat[ii], gridsize=80, bins="log", mincnt=1)
    lo = float(np.percentile(y_true_cat[ii], 1))
    hi = float(np.percentile(y_true_cat[ii], 99))
    plt.plot([lo, hi], [lo, hi], "w--", linewidth=1.5, label="y=x")
    plt.xlabel(f"True {args.y_key}")
    plt.ylabel(f"Predicted {args.y_key}")
    plt.title(f"Scatter (Pearson={pearson_global:.4f}, Spearman={spearman_global:.4f})")
    plt.colorbar(label="log10(count)")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "scatter_true_vs_pred.png"), dpi=220)
    plt.close(fig)

    err_hist = abs_err_cat if args.error_mode == "abs" else (
        compute_error_map(
            y_true_cat, y_pred_cat, np.ones_like(y_true_cat, dtype=np.float32),
            mode=args.error_mode, eps=args.error_eps
        )
    )
    hist_label = f"|Error| ({args.y_key})" if args.error_mode == "abs" else (
        "Percent Error [%]" if args.error_mode == "percent" else "sMAPE [%]"
    )
    hist_name = "abs_error_hist.png" if args.error_mode == "abs" else f"{args.error_mode}_error_hist.png"
    fig = plt.figure(figsize=(6, 4))
    plt.hist(err_hist[np.isfinite(err_hist)], bins=80, alpha=0.9)
    plt.xlabel(hist_label)
    plt.ylabel("Count")
    if args.error_mode == "abs":
        plt.title(f"Abs Error Histogram (P90={p90_abs_err:.3g}, P95={p95_abs_err:.3g})")
    else:
        plt.title(f"{hist_label} Histogram (P90={p90_pct_err:.3g}, P95={p95_pct_err:.3g})")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, hist_name), dpi=220)
    plt.close(fig)

    fig = plt.figure(figsize=(6, 4))
    sample_pearsons = np.array([m["pearson"] for m in sample_metrics], dtype=float)
    plt.hist(sample_pearsons[np.isfinite(sample_pearsons)], bins=30, alpha=0.9)
    plt.xlabel("Per-sample Pearson")
    plt.ylabel("Count")
    plt.title("Distribution of Per-Sample Correlation")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "per_sample_pearson_hist.png"), dpi=220)
    plt.close(fig)

    if p_all.shape[1] > 0:
        sweep_name = args.sweep_param if args.sweep_param is not None else p_keys[0]
        if sweep_name in p_keys:
            k = p_keys.index(sweep_name)
            p_med = np.median(p_all, axis=0).astype(np.float32)
            p_lo = float(np.percentile(p_all[:, k], 5))
            p_hi = float(np.percentile(p_all[:, k], 95))
            xs = np.linspace(p_lo, p_hi, 25).astype(np.float32)
            mask_ref = m_all[val_idx[0]]
            ys = []
            for xv in xs:
                p_cur = p_med.copy()
                p_cur[k] = xv
                p_in = scale_params(p_cur, p_mu, p_std)
                y_all_pred = predict_fields(model, norm, mask_ref, p_in, device=device, as_numpy=True)
                y1 = y_all_pred[ch_idx]
                ys.append(float(np.nanmean(np.where(mask_ref > 0.5, y1, np.nan))))
            ys = np.array(ys)

            fig = plt.figure(figsize=(6, 4))
            plt.plot(xs, ys, "-o", markersize=3)
            plt.xlabel(sweep_name)
            plt.ylabel(f"Mean predicted {args.y_key} (masked)")
            plt.title("1D Parameter Response (model sanity)")
            plt.tight_layout()
            plt.savefig(os.path.join(args.outdir, f"sweep_{sweep_name}.png"), dpi=220)
            plt.close(fig)

    print("Saved plots/metrics to:", args.outdir)
    print(json.dumps(metrics_out, indent=2))


if __name__ == "__main__":
    main()
