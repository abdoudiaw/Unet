#!/usr/bin/env python3
"""
Train the EIRENE-replacement GNN: plasma state → sources + neutrals.

Requires coupling_dataset.npz built from balance.nc files.

Usage:
    python gnn/train_eirene_gnn.py --data coupling_dataset.npz --device cuda
    python gnn/train_eirene_gnn.py --data coupling_dataset.npz --device cuda --hidden 128 --epochs 500
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
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gnn.eirene_gnn import EireneGNN
from gnn.data import _build_edges_masked

# Input: all 14 plasma channels
# Output: all 9 EIRENE outputs
PLASMA_KEYS = ["Te", "Ti", "ne", "ni", "ua", "vol", "hx", "hy",
               "bb0", "bb1", "bb2", "bb3", "R", "Z"]
SOURCE_KEYS = ["Sp", "Sne", "Qe", "Qi", "Sm", "dab2", "dmb2", "tab2", "tmb2"]


def masked_mse(pred, target, mask):
    """MSE over valid nodes only."""
    diff = (pred - target) ** 2
    valid = mask.unsqueeze(-1) & torch.isfinite(pred) & torch.isfinite(target)
    diff = torch.where(valid, diff, torch.zeros_like(diff))
    return diff.sum() / valid.float().sum().clamp_min(1.0)


def build_eirene_graphs(npz_path):
    """Build PyG graphs for EIRENE training from coupling_dataset.npz."""
    d = np.load(npz_path, allow_pickle=True)
    plasma = d["plasma"]     # (N, 14, H, W)
    sources = d["sources"]   # (N, 9, H, W)
    mask_arr = d["mask"]     # (N, H, W)
    runs = list(d["runs"]) if "runs" in d.files else [str(i) for i in range(len(plasma))]

    N, Cp, H, W = plasma.shape
    Cs = sources.shape[1]

    # Reference edge structure from first mask
    ref_mask = mask_arr[0].astype(bool)
    ref_edges = _build_edges_masked(ref_mask)
    n_nodes_ref = int(ref_mask.sum())

    # Edge attrs from R, Z of first run
    R_ref = plasma[0, 12][ref_mask]  # R is channel 12
    Z_ref = plasma[0, 13][ref_mask]  # Z is channel 13
    if ref_edges.shape[1] > 0:
        dR = R_ref[ref_edges[1]] - R_ref[ref_edges[0]]
        dZ = Z_ref[ref_edges[1]] - Z_ref[ref_edges[0]]
        dist = np.sqrt(dR**2 + dZ**2)
        ref_edge_attr = np.stack([dR, dZ, dist], axis=-1).astype(np.float32)
    else:
        ref_edge_attr = np.zeros((0, 3), dtype=np.float32)

    edge_index_t = torch.from_numpy(ref_edges)
    edge_attr_t = torch.from_numpy(ref_edge_attr)

    graphs = []
    for i in range(N):
        m = mask_arr[i].astype(bool)
        if m.sum() != n_nodes_ref:
            continue

        # Node features: all 14 plasma channels
        x = np.stack([plasma[i, c][m] for c in range(Cp)], axis=-1).astype(np.float32)
        # Targets: all 9 source channels
        y = np.stack([sources[i, c][m] for c in range(Cs)], axis=-1).astype(np.float32)
        # Per-node validity
        node_mask = np.all(np.isfinite(x), axis=-1) & np.all(np.isfinite(y), axis=-1)

        data = Data(
            x=torch.from_numpy(x),
            edge_index=edge_index_t,
            edge_attr=edge_attr_t,
            y=torch.from_numpy(y),
            mask=torch.from_numpy(node_mask),
            num_nodes=n_nodes_ref,
        )
        data.run_name = runs[i] if i < len(runs) else str(i)
        graphs.append(data)

    return graphs, n_nodes_ref, ref_edges.shape[1]


def main():
    ap = argparse.ArgumentParser(description="Train EIRENE-replacement GNN")
    ap.add_argument("--data", required=True, help="coupling_dataset.npz")
    ap.add_argument("--output", default="gnn/eirene_gnn.pt")
    ap.add_argument("--device", default="cpu")

    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--patience", type=int, default=100)
    ap.add_argument("--split", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--results-csv", default=None)
    args = ap.parse_args()

    print(f"Loading {args.data} ...")
    graphs, n_nodes, n_edges = build_eirene_graphs(args.data)
    print(f"  {len(graphs)} graphs, {n_nodes} nodes, {n_edges} edges")

    # Split
    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(len(graphs))
    n_train = int(args.split * len(graphs))
    train_graphs = [graphs[i] for i in idx[:n_train]]
    val_graphs = [graphs[i] for i in idx[n_train:]]
    print(f"  Train: {len(train_graphs)}, Val: {len(val_graphs)}")

    # Preload all graphs to device
    device = torch.device(args.device)
    train_graphs = [g.to(device) for g in train_graphs]
    val_graphs = [g.to(device) for g in val_graphs]
    print(f"  Preloaded all graphs to {device}")
    model = EireneGNN(
        in_features=14,
        out_features=9,
        hidden=args.hidden,
        n_layers=args.n_layers,
        edge_dim=3,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params/1e6:.2f}M params")

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=0.5, patience=args.patience // 3, min_lr=1e-6)

    best_val = float("inf")
    best_state = None
    boredom = 0
    log_every = max(1, args.epochs // 40)
    t0 = time.time()

    for epoch in range(args.epochs):
        # Train
        model.train()
        tr_loss_sum, tr_n = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            pred = model(batch)
            loss = masked_mse(pred, batch.y, batch.mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss_sum += loss.item() * batch.num_graphs
            tr_n += batch.num_graphs
        tr_loss = tr_loss_sum / max(tr_n, 1)

        # Val
        model.eval()
        va_loss_sum, va_n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch)
                loss = masked_mse(pred, batch.y, batch.mask)
                va_loss_sum += loss.item() * batch.num_graphs
                va_n += batch.num_graphs
        va_loss = va_loss_sum / max(va_n, 1)
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
            print(f"  epoch {epoch:3d}/{args.epochs}  train={tr_loss:.6f}  "
                  f"val={va_loss:.6f}  best={best_val:.6f}  lr={lr_now:.2e}  "
                  f"bored={boredom} {mark}")

        if boredom > args.patience:
            print(f"  early stop at epoch {epoch}")
            break

    elapsed = time.time() - t0

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save({
        "model_state": best_state,
        "config": {
            "in_features": 14,
            "out_features": 9,
            "hidden": args.hidden,
            "n_layers": args.n_layers,
            "dropout": args.dropout,
        },
        "plasma_keys": PLASMA_KEYS,
        "source_keys": SOURCE_KEYS,
        "n_params": n_params,
        "best_val": best_val,
    }, args.output)

    print(f"\nDone: {elapsed:.0f}s, best_val={best_val:.6f}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
