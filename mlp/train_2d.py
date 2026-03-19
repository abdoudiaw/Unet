#!/usr/bin/env python3
"""
Train an MLP ensemble on per-cell 2D SOLPS data.

Inputs per cell:  [Gamma_D2, Ptot_W, Gamma_core, dna, hci, psi_n, |B|]
Outputs per cell: [Te, Ti, ne, ua, Sp, Qe, Qi, Sm]

Uses the same ensemble architecture as transport_learner.py but adapted
for the 2D cell-based dataset from build_2d_database.py.

Usage:
    python mlp/train_2d.py --data mlp/dataset_2d.npz --output mlp/ensemble_2d.pt
    python mlp/train_2d.py --data mlp/dataset_2d.npz --output mlp/ensemble_2d.pt \
        --n-hidden 128 --n-layers 8 --n-members 5 --epochs 2000 --device cuda
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import sklearn.metrics
import torch
import torch.utils.data

torch.set_default_dtype(torch.float32)


# ======================================================================
# Per-field normalization config
# ======================================================================

# "log" for positive fields, "symlog" for signed fields, "linear" for small-range
NORM_CONFIG = {
    "Te":  ("log",    1e-2),    # log(Te + eps), eps avoids log(0)
    "Ti":  ("log",    1e-2),
    "ne":  ("log",    1e16),
    "ua":  ("symlog", 5e3),     # sign(x)*log1p(|x|/scale)
    "Sp":  ("symlog", 1e10),
    "Qe":  ("symlog", 1e2),
    "Qi":  ("symlog", 1e2),
    "Sm":  ("symlog", 1e-2),
}


def _apply_field_transform(data, columns):
    """Apply per-field log/symlog transform. Returns transformed data (float32)."""
    out = np.empty_like(data, dtype=np.float64)
    for j, col in enumerate(columns):
        mode, scale = NORM_CONFIG.get(col, ("linear", 1.0))
        x = data[:, j].astype(np.float64)
        if mode == "log":
            out[:, j] = np.log(np.maximum(x, 0) + scale)
        elif mode == "symlog":
            out[:, j] = np.sign(x) * np.log1p(np.abs(x) / scale)
        else:
            out[:, j] = x
    return out.astype(np.float32)


def _invert_field_transform(data, columns):
    """Inverse of _apply_field_transform. Returns physical-space data."""
    out = np.empty_like(data, dtype=np.float64)
    for j, col in enumerate(columns):
        mode, scale = NORM_CONFIG.get(col, ("linear", 1.0))
        x = data[:, j].astype(np.float64)
        if mode == "log":
            out[:, j] = np.exp(x) - scale
        elif mode == "symlog":
            out[:, j] = np.sign(x) * scale * (np.exp(np.abs(x)) - 1.0)
        else:
            out[:, j] = x
    return out.astype(np.float32)


# ======================================================================
# Scaler (standard mean/std, applied AFTER field transform)
# ======================================================================

class Scaler(torch.nn.Module):
    def __init__(self, means, stds, eps=1e-300):
        super().__init__()
        self.means = torch.nn.Parameter(means, requires_grad=False)
        self.stds = torch.nn.Parameter(stds, requires_grad=False)
        self.eps = eps

    @classmethod
    def from_tensor(cls, tensor):
        return cls(tensor.mean(dim=0), tensor.std(dim=0))

    @classmethod
    def from_inversion(cls, other):
        new_stds = 1 / (other.stds + other.eps)
        new_means = -other.means / (other.stds + other.eps)
        return cls(new_means, new_stds)

    def forward(self, tensor):
        return (tensor - self.means) / (self.stds + self.eps)


# ======================================================================
# Model wrapper
# ======================================================================

class EnsembleModel:
    """Ensemble of MLP networks with mean/std prediction."""

    def __init__(self, networks, feature_columns, target_columns, err_info=None):
        self.networks = networks
        self.feature_columns = tuple(feature_columns)
        self.target_columns = tuple(target_columns)
        self.err_info = err_info

    def predict(self, features, device="cpu"):
        features = np.asarray(features, dtype=np.float32)
        batched = features.ndim > 1
        tensor = torch.as_tensor(features).to(device)
        if not batched:
            tensor = tensor.unsqueeze(0)
        with torch.no_grad():
            results = np.stack([net(tensor).cpu().numpy() for net in self.networks])
        mean = results.mean(axis=0)
        std = results.std(axis=0)
        if not batched:
            return mean[0], std[0]
        return mean, std

    def to(self, device):
        self.networks = [n.to(device) for n in self.networks]
        return self


# ======================================================================
# Training functions
# ======================================================================

def build_network(n_inputs, n_outputs, n_hidden, n_layers, activation, inscaler):
    """Build a Sequential MLP: inscaler → hidden layers → output."""
    layers = [inscaler, torch.nn.Linear(n_inputs, n_hidden), activation()]
    for _ in range(n_layers - 2):
        layers.append(torch.nn.Linear(n_hidden, n_hidden))
        layers.append(activation())
    layers.append(torch.nn.Linear(n_hidden, n_outputs))
    return torch.nn.Sequential(*layers)


def train_epoch(loader, network, cost_fn, opt, cost_scaler, device):
    network.train()
    for batch_in, batch_out in loader:
        batch_in = batch_in.to(device, non_blocking=True)
        batch_out = batch_out.to(device, non_blocking=True)
        opt.zero_grad()
        pred = network(batch_in)
        loss = cost_fn(pred, cost_scaler(batch_out))
        loss.backward()
        opt.step()


def evaluate(loader, network, cost_scaler=None, device="cpu"):
    network.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch_in, batch_out in loader:
            batch_in = batch_in.to(device, non_blocking=True)
            batch_out = batch_out.to(device, non_blocking=True)
            pred = network(batch_in)
            if cost_scaler is not None:
                batch_out = cost_scaler(batch_out)
            preds.append(pred)
            trues.append(batch_out)
    return torch.cat(preds, 0), torch.cat(trues, 0)


def score_model(predicted, true):
    """Compute per-output R² and RMSE in float64 to avoid overflow."""
    pred_np = predicted.cpu().numpy().astype(np.float64)
    true_np = true.cpu().numpy().astype(np.float64)
    # Guard against NaN/Inf from the model
    valid = np.isfinite(pred_np) & np.isfinite(true_np)
    r2s, rmses = [], []
    for j, (p, t) in enumerate(zip(pred_np.T, true_np.T)):
        m = valid[:, j]
        if m.sum() < 2 or t[m].std() < 1e-12:
            r2s.append(0.0)
            rmses.append(float("inf"))
            continue
        rmse = np.sqrt(np.mean((t[m] - p[m]) ** 2))
        ss_res = np.sum((t[m] - p[m]) ** 2)
        ss_tot = np.sum((t[m] - t[m].mean()) ** 2)
        r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
        r2s.append(r2)
        rmses.append(rmse)
    return np.array(r2s), np.array(rmses)


def _preload_batches(feat, tgt, batch_size, device, shuffle_seed=None):
    """Split data into GPU-resident batches. Returns list of (x, y) tuples."""
    n = len(feat)
    idx = torch.arange(n)
    if shuffle_seed is not None:
        g = torch.Generator()
        g.manual_seed(shuffle_seed)
        idx = idx[torch.randperm(n, generator=g)]
    batches = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        bi = idx[start:end]
        batches.append((feat[bi].to(device), tgt[bi].to(device)))
    return batches


def train_epoch_preloaded(batches, network, cost_fn, opt, cost_scaler):
    """Train one epoch from preloaded GPU batches."""
    network.train()
    for batch_in, batch_out in batches:
        opt.zero_grad()
        pred = network(batch_in)
        loss = cost_fn(pred, cost_scaler(batch_out))
        loss.backward()
        opt.step()


def evaluate_preloaded(batches, network, cost_scaler):
    """Evaluate on preloaded GPU batches."""
    network.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch_in, batch_out in batches:
            pred = network(batch_in)
            preds.append(pred)
            trues.append(cost_scaler(batch_out))
    return torch.cat(preds, 0), torch.cat(trues, 0)


def train_single(features_train, targets_train, config, device="cpu"):
    """Train one MLP member. Returns the network in eval mode (transformed space).

    Targets are already in transformed space (log/symlog applied before calling).
    The network output is in transformed space — caller must invert field
    transform for physical-space scoring.
    """
    n_total = len(features_train)
    n_valid = max(2, int(config["validation_fraction"] * n_total))
    n_train = n_total - n_valid

    preload_gpu = config.get("preload_gpu", False) and str(device) != "cpu"

    # Split into train/valid (targets already in transformed space)
    dataset = torch.utils.data.TensorDataset(
        torch.as_tensor(features_train),
        torch.as_tensor(targets_train),
    )
    train_ds, valid_ds = torch.utils.data.random_split(dataset, [n_train, n_valid])

    train_feat = train_ds[:][0]
    train_tgt = train_ds[:][1]

    inscaler = Scaler.from_tensor(train_feat).to(device)
    cost_scaler = Scaler.from_tensor(train_tgt).to(device)
    outscaler = Scaler.from_inversion(cost_scaler).to(device)

    network = build_network(
        n_inputs=features_train.shape[1],
        n_outputs=targets_train.shape[1],
        n_hidden=config["n_hidden"],
        n_layers=config["n_layers"],
        activation=config["activation"],
        inscaler=inscaler,
    ).to(device)

    cost_fn = torch.nn.MSELoss()
    opt = torch.optim.Adam(network.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=config["patience"], factor=0.5
    )

    if preload_gpu:
        # Load all data to GPU once — no DataLoader overhead
        all_train_feat = train_ds[:][0].to(device)
        all_train_tgt = train_ds[:][1].to(device)
        all_valid_feat = valid_ds[:][0].to(device)
        all_valid_tgt = valid_ds[:][1].to(device)
        # Pre-split valid into batches (no shuffle needed)
        valid_batches = _preload_batches(
            all_valid_feat, all_valid_tgt, config["eval_batch_size"], device
        )
        train_loader = None
        valid_loader = None
    else:
        use_cuda = (str(device) != "cpu")
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=config["batch_size"], shuffle=True,
            pin_memory=use_cuda, num_workers=2 if use_cuda else 0,
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_ds, batch_size=config["eval_batch_size"], shuffle=False,
            pin_memory=use_cuda, num_workers=2 if use_cuda else 0,
        )

    best_cost = float("inf")
    best_params = copy.deepcopy(network.state_dict())
    boredom = 0
    max_boredom = 2 * config["patience"] + 1

    log_every = max(1, config["n_epochs"] // 20)  # ~20 log lines per member
    for epoch in range(config["n_epochs"]):
        if preload_gpu:
            # Reshuffle train batches each epoch
            train_batches = _preload_batches(
                all_train_feat, all_train_tgt, config["batch_size"],
                device, shuffle_seed=epoch,
            )
            train_epoch_preloaded(train_batches, network, cost_fn, opt, cost_scaler)
            pred, true = evaluate_preloaded(valid_batches, network, cost_scaler)
        else:
            train_epoch(train_loader, network, cost_fn, opt, cost_scaler, device)
            pred, true = evaluate(valid_loader, network, cost_scaler, device)
        eval_cost = (pred - true).abs().mean().item()
        scheduler.step(eval_cost)

        improved = eval_cost < best_cost
        if improved:
            best_cost = eval_cost
            best_params = copy.deepcopy(network.state_dict())
            boredom = 0
        else:
            boredom += 1

        if epoch % log_every == 0 or improved or boredom > max_boredom:
            lr_now = opt.param_groups[0]["lr"]
            mark = "*" if improved else " "
            print(f"    epoch {epoch:4d}/{config['n_epochs']}  "
                  f"val_mae={eval_cost:.6f}  best={best_cost:.6f}  "
                  f"lr={lr_now:.2e}  boredom={boredom} {mark}")

        if boredom > max_boredom:
            print(f"    early stop at epoch {epoch}")
            break

    network.load_state_dict(best_params)
    # Append output scaler: network outputs transformed space (log/symlog)
    # Caller must apply _invert_field_transform for physical units
    final = torch.nn.Sequential(*network[:], outscaler).to(device)
    for p in final.parameters():
        p.requires_grad_(False)
    final.eval()
    return final


def train_ensemble(features, targets, run_ids, config, target_columns, device="cpu"):
    """Train an ensemble with group-based train/test split.

    Applies per-field log/symlog transform to targets before training.
    Scoring is done in physical space (inverse-transformed).
    """
    # Group split by run
    unique_runs = np.unique(run_ids)
    rng = np.random.RandomState(config["seed"])
    rng.shuffle(unique_runs)
    n_test = max(1, int(config["test_fraction"] * len(unique_runs)))
    test_runs = set(unique_runs[:n_test].tolist())
    train_mask = np.array([r not in test_runs for r in run_ids])
    test_mask = ~train_mask

    # Apply per-field transform (log/symlog) before training
    targets_t = _apply_field_transform(targets, target_columns)

    feat_train = features[train_mask]
    tgt_train = targets_t[train_mask]
    feat_test = features[test_mask]
    tgt_test_t = targets_t[test_mask]        # transformed (for model eval)
    tgt_test_phys = targets[test_mask]       # physical (for R² scoring)

    print(f"Train: {train_mask.sum()} cells ({len(unique_runs) - n_test} runs)")
    print(f"Test:  {test_mask.sum()} cells ({n_test} runs)")

    networks = []
    err_info = []
    attempts = 0

    while len(networks) < config["n_members"] and attempts < config["max_model_tries"]:
        attempts += 1
        t0 = time.time()
        net = train_single(feat_train, tgt_train, config, device=device)
        elapsed = time.time() - t0

        # Score on test set — model outputs transformed space, invert for physical R²
        test_ds = torch.utils.data.TensorDataset(
            torch.as_tensor(feat_test),
            torch.as_tensor(tgt_test_t),
        )
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=config["eval_batch_size"], shuffle=False,
            pin_memory=(str(device) != "cpu"),
        )
        pred_t, _ = evaluate(test_loader, net, cost_scaler=None, device=device)
        # Invert field transform for physical-space scoring
        pred_phys = _invert_field_transform(pred_t.cpu().numpy(), target_columns)
        true_phys = tgt_test_phys
        r2s, rmses = score_model(
            torch.from_numpy(pred_phys), torch.from_numpy(true_phys)
        )

        status = "ACCEPT" if np.all(r2s >= config["score_thresh"]) else "REJECT"
        print(f"  Attempt {attempts}: R²={r2s} RMSE={rmses} [{status}] ({elapsed:.0f}s)")

        if status == "REJECT":
            continue
        networks.append(net.cpu())
        err_info.append(rmses)

    if not networks:
        raise RuntimeError(
            f"Failed to train acceptable members after {attempts} attempts. "
            f"Try lowering --score-thresh or increasing --max-tries."
        )

    return networks, np.mean(err_info, axis=0) if err_info else None


# ======================================================================
# Main
# ======================================================================

def main():
    ap = argparse.ArgumentParser(description="Train MLP ensemble on 2D SOLPS cell data")
    ap.add_argument("--data", required=True, help="Path to dataset_2d.npz")
    ap.add_argument("--output", default="mlp/ensemble_2d.pt")
    ap.add_argument("--device", default="cpu", help="cpu, cuda, or mps")

    # Architecture
    ap.add_argument("--n-hidden", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--activation", default="silu", choices=["relu", "silu", "gelu"])

    # Training
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--validation-fraction", type=float, default=0.1)

    # Ensemble
    ap.add_argument("--n-members", type=int, default=5)
    ap.add_argument("--test-fraction", type=float, default=0.2)
    ap.add_argument("--score-thresh", type=float, default=0.85)
    ap.add_argument("--max-tries", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)

    # Performance
    ap.add_argument("--preload-gpu", action="store_true",
                    help="Load entire dataset to GPU memory (faster, needs ~200MB VRAM)")

    # Output
    ap.add_argument("--results-csv", default=None)
    args = ap.parse_args()

    activation_map = {
        "relu": torch.nn.ReLU,
        "silu": torch.nn.SiLU,
        "gelu": torch.nn.GELU,
    }

    config = dict(
        n_hidden=args.n_hidden,
        n_layers=args.n_layers,
        activation=activation_map[args.activation],
        n_epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        eval_batch_size=8192,
        patience=args.patience,
        validation_fraction=args.validation_fraction,
        n_members=args.n_members,
        test_fraction=args.test_fraction,
        score_thresh=args.score_thresh,
        preload_gpu=args.preload_gpu,
        max_model_tries=args.max_tries,
        seed=args.seed,
    )

    print(f"Loading {args.data} ...")
    d = np.load(args.data, allow_pickle=True)
    features = d["features"]
    targets = d["targets"]
    run_ids = d["run_ids"]
    feat_cols = list(d["feature_columns"])
    tgt_cols = list(d["target_columns"])
    print(f"  {features.shape[0]} cells, {features.shape[1]} features, {targets.shape[1]} targets")
    print(f"  Features: {feat_cols}")
    print(f"  Targets:  {tgt_cols}")
    print(f"  Config: hidden={args.n_hidden} layers={args.n_layers} act={args.activation} "
          f"members={args.n_members} lr={args.lr} batch={args.batch_size}")

    t0 = time.time()
    networks, err_info = train_ensemble(features, targets, run_ids, config,
                                        target_columns=tgt_cols, device=args.device)
    elapsed = time.time() - t0

    model = EnsembleModel(
        networks=networks,
        feature_columns=feat_cols,
        target_columns=tgt_cols,
        err_info=err_info,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save({
        "model": model,
        "config": config,
        "feature_columns": feat_cols,
        "target_columns": tgt_cols,
        "norm_config": {k: v for k, v in NORM_CONFIG.items() if k in tgt_cols},
        "err_info": err_info,
        "n_cells": features.shape[0],
        "n_runs": int(d["n_runs"]),
    }, args.output)

    print(f"\nEnsemble saved: {args.output}")
    print(f"  {len(networks)} members, {elapsed:.0f}s total")
    if err_info is not None:
        for name, rmse in zip(tgt_cols, err_info):
            print(f"  {name}: mean RMSE = {rmse:.4g}")

    # Optional CSV log
    if args.results_csv:
        os.makedirs(os.path.dirname(args.results_csv) or ".", exist_ok=True)
        with open(args.results_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["target", "mean_rmse"])
            for name, rmse in zip(tgt_cols, err_info):
                w.writerow([name, f"{rmse:.6g}"])
        print(f"  Results CSV: {args.results_csv}")


if __name__ == "__main__":
    main()
