# Copyright 2025-2026 Oak Ridge National Laboratory
# @authors: Abdourahmane (Abdou) Diaw - diawa@ornl.gov
#
# SPDX-License-Identifier: MIT

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

from solpex.predict import load_checkpoint
from solpex.utils import pick_device

POSITIVE_KEYS = {"Te", "Ti", "ne", "ni", "Sp"}

def split_indices(N, split=0.85, seed=42):
    idx = np.arange(N)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    cut = int(split * N)
    return idx[:cut], idx[cut:]


def load_dataset(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    if "Y" in d.files:
        Y = d["Y"].astype(np.float32)  # (N,C,H,W)
        y_keys = [str(k) for k in d["y_keys"]]
    elif "Te" in d.files:
        Y = d["Te"][:, None, :, :].astype(np.float32)
        y_keys = ["Te"]
    else:
        raise KeyError("Dataset must contain Y (+y_keys) or Te.")

    if "mask" in d.files:
        M = d["mask"]
        if M.ndim == 2:
            M = np.repeat(M[None, :, :], Y.shape[0], axis=0)
        M = (M > 0.5).astype(np.float32)  # (N,H,W)
    else:
        M = np.ones((Y.shape[0], Y.shape[2], Y.shape[3]), dtype=np.float32)

    if "params" in d.files:
        P = d["params"].astype(np.float32)
    elif "X" in d.files:
        P = d["X"].astype(np.float32)
    else:
        raise KeyError("Dataset must contain params or X.")

    if "param_keys" in d.files:
        p_keys = [str(k) for k in d["param_keys"]]
    elif "x_keys" in d.files:
        p_keys = [str(k) for k in d["x_keys"]]
    else:
        p_keys = [f"p{i}" for i in range(P.shape[1])]
    return Y, M, P, y_keys, p_keys


def build_x_from_scaled_params(mask_b11, p_scaled_bP):
    # mask_b11: (B,1,H,W), p_scaled_bP: (B,P)
    B, _, H, W = mask_b11.shape
    P = p_scaled_bP.shape[1]
    pch = p_scaled_bP.view(B, P, 1, 1).expand(B, P, H, W)
    return torch.cat([mask_b11, pch], dim=1)

def parse_fields_arg(fields_arg, y_keys):
    if fields_arg is None or fields_arg.strip() == "":
        return list(range(len(y_keys))), list(y_keys)
    req = [k.strip() for k in fields_arg.split(",") if k.strip()]
    idx = []
    for k in req:
        if k not in y_keys:
            raise KeyError(f"Requested field {k!r} not in dataset y_keys={y_keys}")
        idx.append(y_keys.index(k))
    return idx, req

def parse_channel_weights_arg(weights_arg, y_keys, default=1.0):
    w = {k: float(default) for k in y_keys}
    if weights_arg is None or weights_arg.strip() == "":
        return w
    parts = [p.strip() for p in weights_arg.split(",") if p.strip()]
    for p in parts:
        if ":" not in p:
            raise ValueError(f"Bad channel weight token {p!r}. Use name:value format.")
        k, v = p.split(":", 1)
        k = k.strip()
        if k not in w:
            raise KeyError(f"Weight key {k!r} not in y_keys={y_keys}")
        w[k] = float(v.strip())
    return w

def weighted_masked_mse(pred, target, mask, channel_weights):
    # pred/target: (1,C,H,W), mask: (1,1,H,W), channel_weights: (C,)
    if mask.shape[1] == 1 and pred.shape[1] != 1:
        m = mask.expand_as(pred)
    else:
        m = mask
    w = channel_weights.view(1, -1, 1, 1).to(pred.device, pred.dtype)
    diff2 = (pred - target) ** 2
    num = (diff2 * m * w).sum()
    den = (m * w).sum().clamp_min(1e-8)
    return num / den


def masked_mse(pred, target, mask):
    if mask.shape[1] == 1 and pred.shape[1] != 1:
        mask = mask.expand_as(pred)
    diff2 = (pred - target) ** 2
    return (diff2 * mask).sum() / mask.sum().clamp_min(1e-8)


def masked_mae(pred, target, mask):
    if mask.shape[1] == 1 and pred.shape[1] != 1:
        mask = mask.expand_as(pred)
    diff = (pred - target).abs()
    return (diff * mask).sum() / mask.sum().clamp_min(1e-8)

def masked_stats_np(y_pred, y_true, mask, eps=1e-12):
    # y_pred/y_true: (C,H,W), mask: (H,W) 0/1
    m = (mask > 0.5)
    if not np.any(m):
        return np.nan, np.nan
    # float64 to avoid overflow on large channels
    d = (y_pred.astype(np.float64) - y_true.astype(np.float64))
    d = d[:, m]
    mae = float(np.mean(np.abs(d)))
    rmse = float(np.sqrt(np.mean(d * d)))
    return mae, rmse

def masked_log_stats_np(y_pred, y_true, mask, eps=1e-12):
    m = (mask > 0.5)
    if not np.any(m):
        return np.nan, np.nan
    yt = np.maximum(y_true.astype(np.float64)[:, m], eps)
    yp = np.maximum(y_pred.astype(np.float64)[:, m], eps)
    d = np.log10(yp) - np.log10(yt)
    mae = float(np.mean(np.abs(d)))
    rmse = float(np.sqrt(np.mean(d * d)))
    return mae, rmse


@torch.no_grad()
def forward_from_scaled(model, norm, mask_2d, p_scaled_1d, device):
    m = torch.from_numpy(mask_2d).float().unsqueeze(0).unsqueeze(0).to(device)
    p = torch.from_numpy(p_scaled_1d).float().unsqueeze(0).to(device)
    x = build_x_from_scaled_params(m, p)
    y_norm = model(x)
    y_phys = norm.inverse(y_norm, m)
    return y_norm, y_phys, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-csv", default="outputs/inverse_cycle_metrics.csv")
    ap.add_argument("--out-plot", default="outputs/inverse_param_correlation.png")
    ap.add_argument("--out-param-corr-csv", default="outputs/inverse_param_correlation.csv")
    ap.add_argument("--split", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-cases", type=int, default=12)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=5e-2)
    ap.add_argument("--init", choices=["zero", "mean", "noisy_true"], default="mean")
    ap.add_argument("--noise-std", type=float, default=0.15, help="Used only for noisy_true init.")
    ap.add_argument("--fields", type=str, default="", help="Comma-separated field subset for inverse loss, e.g. Te,Ti,ne")
    ap.add_argument("--channel-weights", type=str, default="", help="Comma-separated name:weight, e.g. Te:1,Ti:1,ne:0.5,Sp:0.1")
    ap.add_argument("--n-restarts", type=int, default=1, help="Random restarts per case; best fit is kept.")
    ap.add_argument("--reg", type=float, default=1e-4, help="L2 regularization on scaled params.")
    ap.add_argument("--clip-scaled", type=float, default=4.0, help="Clamp recovered scaled params to [-v, v].")
    ap.add_argument("--log-eps", type=float, default=1e-12)
    ap.add_argument("--print-every", type=int, default=50)
    ap.add_argument("--plot-fontsize", type=float, default=16.0, help="Font size for inverse correlation plot.")
    ap.add_argument("--plot-tick-fontsize", type=float, default=13.0, help="Tick font size for inverse correlation plot.")
    ap.add_argument("--plot-marker-size", type=float, default=34.0, help="Scatter marker size for inverse correlation plot.")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_plot) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_param_corr_csv) or ".", exist_ok=True)

    device = pick_device()
    print("Device:", device)

    model, norm, (p_mu, p_std) = load_checkpoint(args.ckpt, device)
    if p_mu is None or p_std is None:
        raise RuntimeError("Checkpoint missing param scaler (param_mu/param_std).")
    p_mu = np.asarray(p_mu, dtype=np.float32)
    p_std = np.asarray(p_std, dtype=np.float32)

    Y, M, P_raw, y_keys, p_keys = load_dataset(args.npz)
    field_idx, field_names = parse_fields_arg(args.fields, y_keys)
    cw_map = parse_channel_weights_arg(args.channel_weights, y_keys, default=1.0)
    cw = torch.tensor([cw_map[k] for k in y_keys], dtype=torch.float32, device=device)
    cw_sel = cw[field_idx]
    print(f"Inverse loss fields: {field_names}")
    print(f"Channel weights (all): {cw_map}")
    _, val_idx = split_indices(Y.shape[0], split=args.split, seed=args.seed)
    n_cases = min(args.n_cases, len(val_idx))
    rng = np.random.default_rng(args.seed)
    picks = rng.choice(val_idx, size=n_cases, replace=False)

    rows = []
    for case_i, gidx in enumerate(picks):
        y_true_np = Y[gidx]             # (C,H,W) physical
        m_np = M[gidx]                  # (H,W)
        p_true_raw = P_raw[gidx]        # (P,)
        p_true_scaled = (p_true_raw - p_mu) / p_std

        y_true_t = torch.from_numpy(y_true_np).float().unsqueeze(0).to(device)  # (1,C,H,W)
        m_t = torch.from_numpy(m_np).float().unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W)
        y_true_norm = norm.transform(y_true_t, m_t)
        y_true_norm_sel = y_true_norm[:, field_idx]

        if args.init == "zero":
            p0 = np.zeros_like(p_true_scaled)
        elif args.init == "mean":
            p0 = np.zeros_like(p_true_scaled)  # mean of scaled params is ~0
        else:
            p0 = p_true_scaled + args.noise_std * rng.standard_normal(size=p_true_scaled.shape).astype(np.float32)

        best = None
        n_restarts = max(int(args.n_restarts), 1)
        for r in range(n_restarts):
            if args.init == "zero":
                p0r = np.zeros_like(p_true_scaled)
            elif args.init == "mean":
                p0r = np.zeros_like(p_true_scaled)
                if r > 0:
                    p0r = p0r + args.noise_std * rng.standard_normal(size=p0r.shape).astype(np.float32)
            else:
                p0r = p_true_scaled + args.noise_std * rng.standard_normal(size=p_true_scaled.shape).astype(np.float32)

            p_var = torch.nn.Parameter(
                torch.tensor(p0r[None, :], dtype=torch.float32, device=device)
            )  # (1,P), leaf parameter
            opt = torch.optim.Adam([p_var], lr=args.lr)

            for step in range(args.steps):
                x = build_x_from_scaled_params(m_t, p_var)
                y_pred_norm = model(x)
                y_pred_norm_sel = y_pred_norm[:, field_idx]
                loss_fit = weighted_masked_mse(y_pred_norm_sel, y_true_norm_sel, m_t, cw_sel)
                loss_reg = args.reg * (p_var ** 2).mean()
                loss = loss_fit + loss_reg
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                if args.clip_scaled is not None and args.clip_scaled > 0:
                    with torch.no_grad():
                        p_var.clamp_(-float(args.clip_scaled), float(args.clip_scaled))

                if (step % max(args.print_every, 1) == 0) or step == args.steps - 1:
                    print(
                        f"[case {case_i+1}/{n_cases} idx={int(gidx)} r={r+1}/{n_restarts}] "
                        f"step {step:04d} loss={float(loss.item()):.4e} fit={float(loss_fit.item()):.4e}"
                    )

            with torch.no_grad():
                x = build_x_from_scaled_params(m_t, p_var)
                y_pred_norm = model(x)
                fit_val = float(weighted_masked_mse(y_pred_norm[:, field_idx], y_true_norm_sel, m_t, cw_sel).item())
            cand = {
                "fit": fit_val,
                "p_scaled": p_var.detach().cpu().numpy()[0].copy(),
            }
            if best is None or cand["fit"] < best["fit"]:
                best = cand

        p_rec_scaled = best["p_scaled"]
        p_rec_raw = p_rec_scaled * p_std + p_mu

        # forward cycle with recovered params
        y_rec_norm, y_rec_phys, _ = forward_from_scaled(model, norm, m_np, p_rec_scaled, device)
        inv_fit_mse_norm = float(weighted_masked_mse(y_rec_norm[:, field_idx], y_true_norm_sel, m_t, cw_sel).item())
        cyc_mae_norm = float(masked_mae(y_rec_norm[:, field_idx], y_true_norm_sel, m_t).item())

        y_rec_phys_np = y_rec_phys.detach().cpu().numpy()[0]
        y_true_phys_np = y_true_t.detach().cpu().numpy()[0]
        cyc_mae_phys, cyc_rmse_phys = masked_stats_np(y_rec_phys_np, y_true_phys_np, m_np, eps=args.log_eps)
        cyc_log_mae, cyc_log_rmse = masked_log_stats_np(y_rec_phys_np, y_true_phys_np, m_np, eps=args.log_eps)

        p_abs = np.abs(p_rec_raw - p_true_raw)
        p_rel = np.abs((p_rec_raw - p_true_raw) / np.maximum(np.abs(p_true_raw), 1e-12))

        row = {
            "global_idx": int(gidx),
            "inverse_fields": ",".join(field_names),
            "best_restart_fit": float(best["fit"]),
            "inv_fit_mse_norm": inv_fit_mse_norm,
            "cycle_mae_norm": cyc_mae_norm,
            "cycle_mae_phys": cyc_mae_phys,
            "cycle_rmse_phys": cyc_rmse_phys,
            "cycle_log_mae": cyc_log_mae,
            "cycle_log_rmse": cyc_log_rmse,
            "param_mae_abs_mean": float(np.mean(p_abs)),
            "param_mre_mean": float(np.mean(p_rel)),
        }
        # Per-channel cycle stats for interpretability
        for c, yk in enumerate(y_keys):
            yc_pred = y_rec_phys_np[c:c+1]
            yc_true = y_true_phys_np[c:c+1]
            mae_c, rmse_c = masked_stats_np(yc_pred, yc_true, m_np, eps=args.log_eps)
            row[f"{yk}_cycle_mae_phys"] = mae_c
            row[f"{yk}_cycle_rmse_phys"] = rmse_c
            if yk in POSITIVE_KEYS:
                lmae_c, lrmse_c = masked_log_stats_np(yc_pred, yc_true, m_np, eps=args.log_eps)
                row[f"{yk}_cycle_log_mae"] = lmae_c
                row[f"{yk}_cycle_log_rmse"] = lrmse_c
        for k, name in enumerate(p_keys):
            row[f"{name}_true"] = float(p_true_raw[k])
            row[f"{name}_rec"] = float(p_rec_raw[k])
            row[f"{name}_abs_err"] = float(p_abs[k])
            row[f"{name}_rel_err"] = float(p_rel[k])
        rows.append(row)

    # Save CSV
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Parameter true-vs-recovered correlation plots
    nP = len(p_keys)
    ncols = min(3, nP)
    nrows = int(np.ceil(nP / ncols))
    plt.rcParams.update(
        {
            "font.size": args.plot_fontsize,
            "axes.labelsize": args.plot_fontsize,
            "xtick.labelsize": args.plot_tick_fontsize,
            "ytick.labelsize": args.plot_tick_fontsize,
        }
    )
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)
    for k, name in enumerate(p_keys):
        r = k // ncols
        c = k % ncols
        ax = axes[r, c]
        t = np.array([row[f"{name}_true"] for row in rows], dtype=float)
        p = np.array([row[f"{name}_rec"] for row in rows], dtype=float)
        ax.scatter(t, p, s=args.plot_marker_size, alpha=0.85)
        lo = min(np.min(t), np.min(p))
        hi = max(np.max(t), np.max(p))
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.0)
        pear = float(np.corrcoef(t, p)[0, 1]) if len(t) > 2 else np.nan
        spear = float(spearmanr(t, p).correlation) if len(t) > 2 else np.nan
        ax.set_xlabel(f"True {name}")
        ax.set_ylabel(f"Recovered {name}")
        ax.text(
            0.03, 0.97, f"r={pear:.3f}\n$\\rho$={spear:.3f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=args.plot_tick_fontsize
        )
        ax.grid(alpha=0.25)
    # hide empty subplots
    for k in range(nP, nrows * ncols):
        r = k // ncols
        c = k % ncols
        axes[r, c].axis("off")
    fig.savefig(args.out_plot, dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Per-parameter correlation table
    corr_rows = []
    for name in p_keys:
        t = np.array([row[f"{name}_true"] for row in rows], dtype=float)
        p = np.array([row[f"{name}_rec"] for row in rows], dtype=float)
        ae = np.abs(p - t)
        re = ae / np.maximum(np.abs(t), 1e-12)
        pear = float(np.corrcoef(t, p)[0, 1]) if len(t) > 2 else np.nan
        spear = float(spearmanr(t, p).correlation) if len(t) > 2 else np.nan
        corr_rows.append(
            {
                "param": name,
                "pearson": pear,
                "spearman": spear,
                "mae_abs": float(np.mean(ae)),
                "mre": float(np.mean(re)),
                "true_min": float(np.min(t)),
                "true_max": float(np.max(t)),
                "rec_min": float(np.min(p)),
                "rec_max": float(np.max(p)),
            }
        )
    with open(args.out_param_corr_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "param",
                "pearson",
                "spearman",
                "mae_abs",
                "mre",
                "true_min",
                "true_max",
                "rec_min",
                "rec_max",
            ],
        )
        w.writeheader()
        for r in corr_rows:
            w.writerow(r)

    # Print summary
    inv_mse = np.array([r["inv_fit_mse_norm"] for r in rows], dtype=float)
    cyc_mae = np.array([r["cycle_mae_phys"] for r in rows], dtype=float)
    cyc_rmse = np.array([r["cycle_rmse_phys"] for r in rows], dtype=float)
    cyc_lmae = np.array([r["cycle_log_mae"] for r in rows], dtype=float)
    cyc_lrmse = np.array([r["cycle_log_rmse"] for r in rows], dtype=float)
    pmre = np.array([r["param_mre_mean"] for r in rows], dtype=float)
    print("Saved:", args.out_csv)
    print("Saved:", args.out_plot)
    print("Saved:", args.out_param_corr_csv)
    print(
        f"[summary] n={len(rows)} "
        f"inv_fit_mse_norm mean={inv_mse.mean():.4e} p90={np.percentile(inv_mse,90):.4e} | "
        f"cycle_mae_norm mean={np.mean([r['cycle_mae_norm'] for r in rows]):.4e} | "
        f"cycle_mae_phys mean={np.nanmean(cyc_mae):.4e} | "
        f"cycle_rmse_phys mean={np.nanmean(cyc_rmse):.4e} | "
        f"cycle_log_mae mean={np.nanmean(cyc_lmae):.4e} | "
        f"cycle_log_rmse mean={np.nanmean(cyc_lrmse):.4e} | "
        f"param_mre_mean mean={pmre.mean():.4e}"
    )


if __name__ == "__main__":
    main()
