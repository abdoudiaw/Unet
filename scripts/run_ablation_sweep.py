#!/usr/bin/env python3
"""
Ablation sweep for UNet hyperparameters.

Runs a grid of experiments varying architecture and loss settings,
logs results to a CSV for paper tables.

Usage:
    python scripts/run_ablation_sweep.py --npz solps.npz --epochs 200 --device cuda
    python scripts/run_ablation_sweep.py --npz solps.npz --epochs 200 --device mps  # Apple Silicon
    python scripts/run_ablation_sweep.py --npz solps.npz --subset arch   # only architecture experiments
    python scripts/run_ablation_sweep.py --npz solps.npz --subset loss   # only loss experiments
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solpex import data
from solpex.data import MaskedLogStandardizer, MaskedSymLogStandardizer
from solpex.train import train_unet


# ======================================================================
# Experiment definitions
# ======================================================================

def build_experiments(subset="all"):
    """
    Build the list of ablation experiments.

    Each experiment is a dict with:
      - tag: unique name
      - group: category (for the paper table)
      - overrides: kwargs that differ from the baseline
    """
    # Baseline config (the current "best" from the paper)
    BASELINE = dict(
        base=32, depth=3, lr=3e-4, batch=4, dropout=0.0,
        lam_grad=0.1, film_hidden=128,
    )

    experiments = []

    # ---- Architecture ablation ----
    arch_exps = [
        # Baseline
        {"tag": "baseline_d3_b32", "group": "baseline", "overrides": {}},

        # Depth
        {"tag": "depth2_b32", "group": "depth", "overrides": {"depth": 2}},
        {"tag": "depth4_b32", "group": "depth", "overrides": {"depth": 4}},

        # Base channels (capacity)
        {"tag": "d3_b16", "group": "capacity", "overrides": {"base": 16}},
        {"tag": "d3_b64", "group": "capacity", "overrides": {"base": 64}},

        # Dropout
        {"tag": "d3_b32_drop01", "group": "regularization", "overrides": {"dropout": 0.1}},
        {"tag": "d3_b32_drop02", "group": "regularization", "overrides": {"dropout": 0.2}},

        # FiLM hidden size
        {"tag": "d3_b32_film64", "group": "film", "overrides": {"film_hidden": 64}},
        {"tag": "d3_b32_film256", "group": "film", "overrides": {"film_hidden": 256}},

        # No FiLM (params concatenated as channels instead)
        # This is tested by setting P=0 — handled specially below
    ]

    # ---- Loss ablation ----
    loss_exps = [
        # No gradient loss
        {"tag": "d3_b32_nograd", "group": "loss", "overrides": {"lam_grad": 0.0}},

        # Stronger gradient loss
        {"tag": "d3_b32_grad03", "group": "loss", "overrides": {"lam_grad": 0.3}},

        # Different learning rates
        {"tag": "d3_b32_lr1e3", "group": "lr", "overrides": {"lr": 1e-3}},
        {"tag": "d3_b32_lr1e4", "group": "lr", "overrides": {"lr": 1e-4}},

        # Batch size
        {"tag": "d3_b32_bs8", "group": "batch", "overrides": {"batch": 8}},
        {"tag": "d3_b32_bs2", "group": "batch", "overrides": {"batch": 2}},
    ]

    # ---- Combined winners (best settings from first round) ----
    combo_exps = [
        {"tag": "nograd_b64", "group": "combo",
         "overrides": {"lam_grad": 0.0, "base": 64}},
        {"tag": "nograd_depth4", "group": "combo",
         "overrides": {"lam_grad": 0.0, "depth": 4}},
        {"tag": "nograd_b64_depth4", "group": "combo",
         "overrides": {"lam_grad": 0.0, "base": 64, "depth": 4}},
        {"tag": "nograd_film256", "group": "combo",
         "overrides": {"lam_grad": 0.0, "film_hidden": 256}},
    ]

    if subset in ("all", "arch"):
        experiments.extend(arch_exps)
    if subset in ("all", "loss"):
        experiments.extend(loss_exps)
    if subset in ("all", "combo"):
        experiments.extend(combo_exps)

    # Fill in baseline values for any missing keys
    for exp in experiments:
        for k, v in BASELINE.items():
            if k not in exp["overrides"]:
                exp["overrides"][k] = v

    return experiments


# ======================================================================
# Runner
# ======================================================================

NORMS_BANK = {
    "Te": MaskedLogStandardizer(eps=1e-2),
    "Ti": MaskedLogStandardizer(eps=1e-2),
    "ne": MaskedLogStandardizer(eps=1e16),
    "ni": MaskedLogStandardizer(eps=1e16),
    "ua": MaskedSymLogStandardizer(scale=5e3),
    "Sp": MaskedLogStandardizer(eps=1e10),
    "Qe": MaskedSymLogStandardizer(scale=1e2),
    "Qi": MaskedSymLogStandardizer(scale=1e2),
    "Sm": MaskedSymLogStandardizer(scale=1e-2),
}


def run_experiment(exp, npz_path, y_keys, epochs, device, out_dir,
                   early_stop_patience, param_transform):
    ov = exp["overrides"]
    tag = exp["tag"]
    print(f"\n{'='*60}")
    print(f"[ablation] {tag} | {exp['group']} | {ov}")
    print(f"{'='*60}")

    norms_by_name = {k: NORMS_BANK[k] for k in y_keys}
    train_loader, val_loader, norm, Pdim, (H, W), param_scaler, pt_name, pt_keys = data.make_loaders(
        npz_path=npz_path,
        inputs_mode="params",
        batch_size=ov["batch"],
        split=0.85,
        y_keys=y_keys,
        norms_by_name=norms_by_name,
        norm_fit_batch=4,
        num_workers=0,
        param_transform=param_transform,
    )
    b0 = next(iter(train_loader))
    in_ch = b0["x"].shape[1]
    out_ch = b0["y"].shape[1]

    ckpt_path = os.path.join(out_dir, f"ablation_{tag}.pt")
    t0 = time.time()

    model, hist = train_unet(
        train_loader=train_loader,
        val_loader=val_loader,
        norm=norm,
        in_ch=in_ch,
        out_ch=out_ch,
        device=device,
        inputs_mode="params",
        epochs=epochs,
        base=ov["base"],
        depth=ov["depth"],
        dropout=ov["dropout"],
        amp=False,
        lam_grad=ov["lam_grad"],
        lam_grad_warmup_end=60,
        lam_w=0.5,
        multiscale=0,
        grad_accum_steps=1,
        save_path=ckpt_path,
        param_scaler=param_scaler,
        lr_init=ov["lr"],
        film_hidden=ov["film_hidden"],
        early_stop_patience=early_stop_patience,
        early_stop_min_delta=1e-4,
        P=Pdim,
        z_dim=0,
        param_transform=pt_name,
        param_keys=pt_keys,
    )

    elapsed = time.time() - t0
    va_mse_arr = np.asarray(hist.get("va_mse", [np.inf]), dtype=float)
    best_val = float(np.min(va_mse_arr))
    best_epoch = int(np.argmin(va_mse_arr))
    n_params = sum(p.numel() for p in model.parameters())

    result = {
        "tag": tag,
        "group": exp["group"],
        "base": ov["base"],
        "depth": ov["depth"],
        "lr": ov["lr"],
        "batch": ov["batch"],
        "dropout": ov["dropout"],
        "lam_grad": ov["lam_grad"],
        "film_hidden": ov["film_hidden"],
        "best_val_mse": best_val,
        "best_epoch": best_epoch,
        "total_epochs": len(va_mse_arr),
        "n_params": n_params,
        "n_params_M": round(n_params / 1e6, 2),
        "time_s": round(elapsed, 1),
    }
    print(f"[ablation] {tag} done: best_val={best_val:.6g} @ epoch {best_epoch}, "
          f"params={n_params/1e6:.2f}M, time={elapsed:.0f}s")
    return result


def main():
    ap = argparse.ArgumentParser(description="UNet ablation sweep for paper")
    ap.add_argument("--npz", required=True, help="Path to solps.npz")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--device", default="cpu", help="cpu, cuda, or mps")
    ap.add_argument("--out-dir", default="outputs/ablation")
    ap.add_argument("--out-csv", default="outputs/ablation/results.csv")
    ap.add_argument("--subset", default="all", choices=["all", "arch", "loss", "combo"],
                    help="Run only a subset of experiments")
    ap.add_argument("--y-keys", default="Te,Ti,ne,ni,ua,Sp,Qe,Qi,Sm")
    ap.add_argument("--early-stop", type=int, default=30)
    ap.add_argument("--param-transform", default="throughput_ratio")
    ap.add_argument("--tags", default=None,
                    help="Comma-separated list of specific experiment tags to run")
    args = ap.parse_args()

    y_keys = [k.strip() for k in args.y_keys.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)

    experiments = build_experiments(args.subset)
    if args.tags:
        tag_set = {t.strip() for t in args.tags.split(",")}
        experiments = [e for e in experiments if e["tag"] in tag_set]

    print(f"Running {len(experiments)} ablation experiments")
    print(f"  epochs={args.epochs} device={args.device} y_keys={y_keys}")
    for e in experiments:
        print(f"  - {e['tag']} ({e['group']}): {e['overrides']}")

    results = []
    for exp in experiments:
        try:
            r = run_experiment(
                exp, args.npz, y_keys, args.epochs, args.device,
                args.out_dir, args.early_stop, args.param_transform,
            )
            results.append(r)
        except Exception as e:
            print(f"[ablation] {exp['tag']} FAILED: {e}")
            results.append({"tag": exp["tag"], "group": exp["group"],
                            "best_val_mse": float("inf"), "error": str(e)})

        # Save incrementally so we don't lose progress
        _write_csv(results, args.out_csv)

    print(f"\n{'='*60}")
    print("ABLATION SWEEP COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {args.out_csv}")
    for r in sorted(results, key=lambda x: x.get("best_val_mse", float("inf"))):
        print(f"  {r['tag']:25s}  val_mse={r.get('best_val_mse', 'FAIL'):>12s}"
              if isinstance(r.get("best_val_mse"), str) else
              f"  {r['tag']:25s}  val_mse={r['best_val_mse']:.6g}  "
              f"params={r.get('n_params_M', '?')}M  time={r.get('time_s', '?')}s")


def _write_csv(results, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not results:
        return
    fieldnames = sorted({k for r in results for k in r.keys()})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(r)


if __name__ == "__main__":
    main()
