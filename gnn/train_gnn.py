#!/usr/bin/env python3
"""
Train the conditional GNN surrogate: params → plasma fields on native SOLPS mesh.

Requires coupling_dataset.npz built from balance.nc files:
    python scripts/build_coupling_dataset.py --base-dir <SOLPS_runs> --out coupling_dataset.npz

Usage:
    python gnn/train_gnn.py --data coupling_dataset.npz --device cuda
    python gnn/train_gnn.py --data coupling_dataset.npz --device cuda --hidden 128 --n-layers 6
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
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gnn.model import ConditionalGNN
from gnn.data import load_coupling_dataset


# ======================================================================
# Normalisation
# ======================================================================

class FieldNormalizer:
    """Per-field symlog normalization fitted on training graphs."""

    def __init__(self, target_keys):
        self.target_keys = list(target_keys)
        self.mu = None
        self.std = None

    def fit(self, graphs):
        all_y = torch.cat([g.y for g in graphs], dim=0)
        y_t = torch.sign(all_y) * torch.log1p(torch.abs(all_y))
        self.mu = y_t.mean(dim=0)
        self.std = y_t.std(dim=0).clamp(min=1e-6)
        return self

    def transform(self, y):
        y_t = torch.sign(y) * torch.log1p(torch.abs(y))
        return (y_t - self.mu.to(y.device)) / self.std.to(y.device)

    def inverse(self, y_n):
        y_t = y_n * self.std.to(y_n.device) + self.mu.to(y_n.device)
        return torch.sign(y_t) * (torch.exp(torch.abs(y_t)) - 1.0)


# ======================================================================
# Training
# ======================================================================

def train_epoch(model, loader, opt, norm, device):
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        batch = batch.to(device)
        opt.zero_grad()
        pred = model(batch)
        target_n = norm.transform(batch.y)
        loss = nn.functional.mse_loss(pred, target_n)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item() * batch.num_graphs
        n += batch.num_graphs
    return total_loss / max(n, 1)


@torch.no_grad()
def eval_epoch(model, loader, norm, device):
    model.eval()
    total_loss = 0.0
    n = 0
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        target_n = norm.transform(batch.y)
        loss = nn.functional.mse_loss(pred, target_n)
        total_loss += loss.item() * batch.num_graphs
        n += batch.num_graphs
    return total_loss / max(n, 1)


@torch.no_grad()
def compute_r2(model, loader, norm, device, target_keys):
    """Compute per-field R² on physical-space predictions."""
    model.eval()
    all_pred, all_true = [], []
    for batch in loader:
        batch = batch.to(device)
        pred_n = model(batch)
        pred_phys = norm.inverse(pred_n)
        all_pred.append(pred_phys.cpu())
        all_true.append(batch.y.cpu())
    pred = torch.cat(all_pred, dim=0).numpy()
    true = torch.cat(all_true, dim=0).numpy()

    r2s = {}
    for j, key in enumerate(target_keys):
        ss_res = ((true[:, j] - pred[:, j]) ** 2).sum()
        ss_tot = ((true[:, j] - true[:, j].mean()) ** 2).sum()
        r2s[key] = float(1.0 - ss_res / max(ss_tot, 1e-12))
    return r2s


def main():
    ap = argparse.ArgumentParser(description="Train conditional GNN surrogate")
    ap.add_argument("--data", required=True,
                    help="Path to coupling_dataset.npz (from build_coupling_dataset.py)")
    ap.add_argument("--output", default="gnn/cond_gnn.pt")
    ap.add_argument("--device", default="cpu")

    # Architecture
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--film-hidden", type=int, default=128)

    # Training
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--split", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--results-csv", default=None)
    args = ap.parse_args()

    target_keys = ["Te", "Ti", "ne", "ua", "Sp", "Qe", "Qi", "Sm"]

    # --- Load data ---
    print(f"Loading {args.data} ...")
    graphs, meta = load_coupling_dataset(args.data, target_keys=target_keys)

    if not graphs:
        sys.exit("No valid graphs built from dataset")

    # --- Train/val split ---
    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(len(graphs))
    n_train = int(args.split * len(graphs))
    train_graphs = [graphs[i] for i in idx[:n_train]]
    val_graphs = [graphs[i] for i in idx[n_train:]]
    print(f"  Train: {len(train_graphs)}, Val: {len(val_graphs)}")

    # --- Normalizer ---
    norm = FieldNormalizer(target_keys).fit(train_graphs)

    # --- Model ---
    device = torch.device(args.device)
    param_dim = int(graphs[0].params.shape[1])
    model = ConditionalGNN(
        node_features=2,   # psi_n, |B|
        param_dim=param_dim,
        out_features=len(target_keys),
        hidden=args.hidden,
        n_layers=args.n_layers,
        edge_dim=3,
        dropout=args.dropout,
        film_hidden=args.film_hidden,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params/1e6:.2f}M params, hidden={args.hidden}, "
          f"layers={args.n_layers}, film_hidden={args.film_hidden}")

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=args.patience // 3, factor=0.5, min_lr=1e-5
    )

    # --- Training loop ---
    best_val = float("inf")
    best_state = None
    boredom = 0
    log_every = max(1, args.epochs // 40)
    t0 = time.time()

    for epoch in range(args.epochs):
        tr_loss = train_epoch(model, train_loader, opt, norm, device)
        va_loss = eval_epoch(model, val_loader, norm, device)
        scheduler.step(va_loss)

        improved = va_loss < best_val
        if improved:
            best_val = va_loss
            best_state = copy.deepcopy(model.state_dict())
            boredom = 0
        else:
            boredom += 1

        if epoch % log_every == 0 or improved or boredom > args.patience:
            lr_now = opt.param_groups[0]["lr"]
            mark = "*" if improved else " "
            print(f"  epoch {epoch:3d}/{args.epochs}  "
                  f"train={tr_loss:.6f}  val={va_loss:.6f}  best={best_val:.6f}  "
                  f"lr={lr_now:.2e}  bored={boredom} {mark}")

        if boredom > args.patience:
            print(f"  early stop at epoch {epoch}")
            break

    elapsed = time.time() - t0

    # --- Evaluate best model ---
    model.load_state_dict(best_state)
    r2s = compute_r2(model, val_loader, norm, device, target_keys)
    print(f"\nTraining complete: {elapsed:.0f}s, best_val={best_val:.6f}")
    print("Per-field R² (validation):")
    for k, v in r2s.items():
        print(f"  {k:6s}: {v:.4f}")

    # --- Save ---
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save({
        "model_state": best_state,
        "config": {
            "node_features": 2,
            "param_dim": param_dim,
            "out_features": len(target_keys),
            "hidden": args.hidden,
            "n_layers": args.n_layers,
            "dropout": args.dropout,
            "film_hidden": args.film_hidden,
        },
        "norm_mu": norm.mu,
        "norm_std": norm.std,
        "target_keys": target_keys,
        "n_params": n_params,
        "best_val": best_val,
        "r2": {k: float(v) for k, v in r2s.items()},
        "meta": meta,
    }, args.output)
    print(f"Saved: {args.output}")

    if args.results_csv:
        os.makedirs(os.path.dirname(args.results_csv) or ".", exist_ok=True)
        with open(args.results_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["field", "r2"])
            for k, v in r2s.items():
                w.writerow([k, f"{v:.6f}"])
        print(f"Results CSV: {args.results_csv}")


if __name__ == "__main__":
    main()
