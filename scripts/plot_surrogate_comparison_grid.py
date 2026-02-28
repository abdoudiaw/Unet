# Copyright 2025-2026 Oak Ridge National Laboratory
# @authors: Abdourahmane (Abdou) Diaw - diawa@ornl.gov
#
# SPDX-License-Identifier: MIT

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from solpex.predict import load_checkpoint, predict_fields, scale_params
from solpex.utils import pick_device


def load_npz_single(npz_path, y_key):
    d = np.load(npz_path, allow_pickle=True)
    if "Y" in d.files:
        y_keys = [str(k) for k in d["y_keys"]]
        if y_key not in y_keys:
            raise KeyError(f"{y_key!r} not in y_keys={y_keys}")
        yi = y_keys.index(y_key)
        y = d["Y"][:, yi].astype(np.float32)
    elif y_key == "Te" and "Te" in d.files:
        y = d["Te"].astype(np.float32)
    else:
        raise KeyError(f"Cannot find field {y_key!r} in dataset.")

    if "mask" in d.files:
        m = d["mask"]
        if m.ndim == 2:
            m = np.repeat(m[None, :, :], y.shape[0], axis=0)
        m = (m > 0.5).astype(np.float32)
    else:
        m = np.ones_like(y, dtype=np.float32)

    if "params" in d.files:
        p = d["params"].astype(np.float32)
    elif "X" in d.files:
        p = d["X"].astype(np.float32)
    else:
        p = np.zeros((y.shape[0], 0), dtype=np.float32)
    return y, m, p


def split_indices(N, split=0.85, seed=42):
    idx = np.arange(N)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    cut = int(split * N)
    return idx[:cut], idx[cut:]


def get_channel_idx(norm, y_key):
    if hasattr(norm, "y_keys"):
        keys = [str(k) for k in norm.y_keys]
        if y_key not in keys:
            raise KeyError(f"{y_key!r} not in checkpoint y_keys={keys}")
        return keys.index(y_key)
    return 0


def norm_img(a, m, eps=1e-8):
    x = np.where(m > 0.5, a, 0.0)
    vmax = np.nanmax(np.where(m > 0.5, np.abs(a), 0.0))
    vmax = max(float(vmax), eps)
    return x / vmax


def error_map(y_true, y_pred, m, mode="percent", eps=1e-3):
    if mode == "abs":
        e = np.abs(y_pred - y_true)
    elif mode == "percent":
        denom = np.maximum(np.abs(y_true), eps)
        e = 100.0 * np.abs(y_pred - y_true) / denom
    elif mode == "smape":
        denom = np.abs(y_true) + np.abs(y_pred) + eps
        e = 200.0 * np.abs(y_pred - y_true) / denom
    else:
        raise ValueError(f"Unknown mode {mode}")
    return np.where(m > 0.5, e, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--ckpt-proposed", required=True)
    ap.add_argument("--ckpt-baseline", default=None)
    ap.add_argument("--y-key", default="Te")
    ap.add_argument("--out", default="outputs/comparison_grid.png")
    ap.add_argument("--n-cases", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", type=float, default=0.85)
    ap.add_argument("--error-mode", choices=["abs", "percent", "smape"], default="percent")
    ap.add_argument("--error-eps", type=float, default=1e-3)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    device = pick_device()
    y_all, m_all, p_all = load_npz_single(args.npz, args.y_key)
    _, val_idx = split_indices(y_all.shape[0], split=args.split, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    picks = rng.choice(val_idx, size=min(args.n_cases, len(val_idx)), replace=False)
    n = len(picks)

    model_p, norm_p, (mu_p, sd_p) = load_checkpoint(args.ckpt_proposed, device)
    ci_p = get_channel_idx(norm_p, args.y_key)

    has_baseline = args.ckpt_baseline is not None
    if has_baseline:
        model_b, norm_b, (mu_b, sd_b) = load_checkpoint(args.ckpt_baseline, device)
        ci_b = get_channel_idx(norm_b, args.y_key)

    truth_tiles = []
    pred_tiles = []
    err_p_tiles = []
    err_b_tiles = []

    for i in picks:
        yt = y_all[i]
        m = m_all[i]
        p_raw = p_all[i]

        pin_p = scale_params(p_raw, mu_p, sd_p)
        yp = predict_fields(model_p, norm_p, m, pin_p, device=device, as_numpy=True)[ci_p]

        truth_tiles.append(norm_img(yt, m))
        pred_tiles.append(norm_img(yp, m))
        err_p_tiles.append(error_map(yt, yp, m, mode=args.error_mode, eps=args.error_eps))

        if has_baseline:
            pin_b = scale_params(p_raw, mu_b, sd_b)
            yb = predict_fields(model_b, norm_b, m, pin_b, device=device, as_numpy=True)[ci_b]
            err_b_tiles.append(error_map(yt, yb, m, mode=args.error_mode, eps=args.error_eps))

    all_err = np.concatenate([np.ravel(e[np.isfinite(e)]) for e in err_p_tiles + (err_b_tiles if has_baseline else [])])
    evmax = float(np.percentile(all_err, 95)) if all_err.size else 1.0

    rows = 4 if has_baseline else 3
    fig, axes = plt.subplots(rows, n, figsize=(1.35 * n, 1.3 * rows), constrained_layout=True)
    if n == 1:
        axes = axes[:, None]

    for j in range(n):
        axes[0, j].imshow(truth_tiles[j], origin="lower", cmap="winter", vmin=0.0, vmax=1.0)
        axes[1, j].imshow(pred_tiles[j], origin="lower", cmap="winter", vmin=0.0, vmax=1.0)
        axes[0, j].set_xticks([]); axes[0, j].set_yticks([])
        axes[1, j].set_xticks([]); axes[1, j].set_yticks([])

        if has_baseline:
            axes[2, j].imshow(err_b_tiles[j], origin="lower", cmap="RdBu_r", vmin=0.0, vmax=evmax)
            axes[2, j].set_xticks([]); axes[2, j].set_yticks([])

            axes[3, j].imshow(err_p_tiles[j], origin="lower", cmap="RdBu_r", vmin=0.0, vmax=evmax)
            axes[3, j].set_xticks([]); axes[3, j].set_yticks([])

            bmean = np.nanmean(err_b_tiles[j])
            pmean = np.nanmean(err_p_tiles[j])
            if np.isfinite(bmean) and np.isfinite(pmean) and bmean < pmean:
                for r in (2, 3):
                    rect = patches.Rectangle((0, 0), err_p_tiles[j].shape[1] - 1, err_p_tiles[j].shape[0] - 1,
                                             linewidth=2.0, edgecolor="red", facecolor="none")
                    axes[r, j].add_patch(rect)
        else:
            axes[2, j].imshow(err_p_tiles[j], origin="lower", cmap="RdBu_r", vmin=0.0, vmax=evmax)
            axes[2, j].set_xticks([]); axes[2, j].set_yticks([])

    axes[0, 0].set_ylabel("Truth", fontsize=11)
    axes[1, 0].set_ylabel("Proposed", fontsize=11)
    if has_baseline:
        axes[2, 0].set_ylabel("Baseline\nError", fontsize=11)
        axes[3, 0].set_ylabel("Proposed\nError", fontsize=11)
    else:
        axes[2, 0].set_ylabel("Error", fontsize=11)

    fig.suptitle(
        f"{args.y_key} comparison grid | error={args.error_mode} | red box: baseline better",
        fontsize=13,
    )
    fig.savefig(args.out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
