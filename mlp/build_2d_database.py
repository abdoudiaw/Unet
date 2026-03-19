#!/usr/bin/env python3
"""
Build per-cell MLP training database from coupling_dataset.npz (native SOLPS mesh).

Each row = one cell from one simulation:
    features: [Gamma_D2, Ptot_W, Gamma_core, dna, hci,
               R, Z, crx0, cry0, crx1, cry1, crx2, cry2, crx3, cry3,
               psi_n, Bmag]   → 17 features
    targets:  [Te, Ti, ne, ua, Sp, Qe, Qi, Sm]

Uses the native SOLPS cell geometry (4 corner points + centre) rather
than rasterized coordinates.  This gives the MLP full positional
information for a fair comparison with the GNN/UNet.

Usage:
    python mlp/build_2d_database.py --data coupling_dataset.npz --out mlp/dataset_2d.npz
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


PARAM_KEYS = ["Gamma_D2", "Ptot_W", "Gamma_core", "dna", "hci"]

FEATURE_COLUMNS = (
    "Gamma_D2", "Ptot_W", "Gamma_core", "dna", "hci",
    "R", "Z",
    "crx0", "cry0", "crx1", "cry1", "crx2", "cry2", "crx3", "cry3",
    "psi_n", "Bmag",
)

TARGET_COLUMNS = ("Te", "Ti", "ne", "ua", "Sp", "Qe", "Qi", "Sm")

# Indices into the plasma array from build_coupling_dataset.py (22 channels):
# ["Te","Ti","ne","ni","ua","vol","hx","hy","bb0","bb1","bb2","bb3","R","Z",
#  "crx0","cry0","crx1","cry1","crx2","cry2","crx3","cry3"]
PI = {"Te": 0, "Ti": 1, "ne": 2, "ni": 3, "ua": 4,
      "vol": 5, "hx": 6, "hy": 7,
      "bb0": 8, "bb1": 9, "bb2": 10, "bb3": 11,
      "R": 12, "Z": 13,
      "crx0": 14, "cry0": 15, "crx1": 16, "cry1": 17,
      "crx2": 18, "cry2": 19, "crx3": 20, "cry3": 21}

# Indices into the source array (9 channels)
SI = {"Sp": 0, "Sne": 1, "Qe": 2, "Qi": 3, "Sm": 4,
      "dab2": 5, "dmb2": 6, "tab2": 7, "tmb2": 8}


def compute_psi_n(R, Z, R0=None, Z0=None):
    """Approximate normalised psi from cell-centre coordinates."""
    if R0 is None:
        R0 = float(R.mean())
    if Z0 is None:
        Z0 = float(Z.mean())
    rho = np.sqrt((R - R0)**2 + (Z - Z0)**2)
    mx = rho.max()
    return (rho / mx if mx > 0 else np.zeros_like(rho)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(
        description="Build per-cell MLP database from coupling_dataset.npz (native SOLPS mesh)")
    ap.add_argument("--data", required=True,
                    help="Path to coupling_dataset.npz (from build_coupling_dataset.py)")
    ap.add_argument("--out", default="mlp/dataset_2d.npz", help="Output path")
    args = ap.parse_args()

    print(f"Loading {args.data} ...")
    d = np.load(args.data, allow_pickle=True)

    plasma = d["plasma"]          # (N, Cp, H, W)
    sources = d["sources"]        # (N, Cs, H, W)
    mask = d["mask"]              # (N, H, W)
    params = d["params"]          # (N, 5)
    plasma_keys = list(d["plasma_keys"])
    source_keys = list(d["source_keys"])
    runs = list(d["runs"]) if "runs" in d.files else [str(i) for i in range(len(plasma))]

    N, Cp, H, W = plasma.shape
    Cs = sources.shape[1]
    print(f"  {N} runs, plasma {Cp} ch, sources {Cs} ch, native grid {H}x{W}")
    print(f"  plasma_keys: {plasma_keys}")
    print(f"  source_keys: {source_keys}")

    # Build index maps for plasma and source arrays
    pk_idx = {k: i for i, k in enumerate(plasma_keys)}
    sk_idx = {k: i for i, k in enumerate(source_keys)}

    # Check that corners are available
    has_corners = "crx0" in pk_idx
    if not has_corners:
        print("  WARNING: coupling_dataset.npz missing corner coords (crx0..cry3).")
        print("  Rebuild with updated build_coupling_dataset.py to include corners.")
        print("  Falling back to centre-only features.")

    # Map target columns to (plasma/source, channel_index)
    target_map = []
    available_targets = []
    for t in TARGET_COLUMNS:
        if t in pk_idx:
            target_map.append(("plasma", pk_idx[t]))
            available_targets.append(t)
        elif t in sk_idx:
            target_map.append(("source", sk_idx[t]))
            available_targets.append(t)
        else:
            print(f"  WARNING: target {t} not found, skipping")

    print(f"  Targets: {available_targets}")

    # Reference geometry from first run
    R_ref = plasma[0, pk_idx["R"]]   # (H, W)
    Z_ref = plasma[0, pk_idx["Z"]]
    Bmag_ref = plasma[0, pk_idx["bb3"]]  # |B|
    psi_n_ref = compute_psi_n(R_ref, Z_ref)

    # Build per-cell rows
    all_features = []
    all_targets = []
    all_run_ids = []

    for i in range(N):
        m = mask[i].astype(bool)
        n_valid = m.sum()
        if n_valid == 0:
            continue

        # Control params broadcast
        p = params[i].astype(np.float32)
        p_broadcast = np.tile(p, (n_valid, 1))  # (n_valid, 5)

        # Geometry features
        R_i = plasma[i, pk_idx["R"]][m]
        Z_i = plasma[i, pk_idx["Z"]][m]
        Bmag_i = plasma[i, pk_idx["bb3"]][m]
        psi_n_i = compute_psi_n(plasma[i, pk_idx["R"]], plasma[i, pk_idx["Z"]])[m]

        if has_corners:
            # 4 corner points: crx0,cry0, crx1,cry1, crx2,cry2, crx3,cry3
            corners = np.column_stack([
                plasma[i, pk_idx["crx0"]][m], plasma[i, pk_idx["cry0"]][m],
                plasma[i, pk_idx["crx1"]][m], plasma[i, pk_idx["cry1"]][m],
                plasma[i, pk_idx["crx2"]][m], plasma[i, pk_idx["cry2"]][m],
                plasma[i, pk_idx["crx3"]][m], plasma[i, pk_idx["cry3"]][m],
            ])  # (n_valid, 8)
            # Features: [params(5), R, Z, crx0..cry3(8), psi_n, Bmag] = 17
            feat = np.column_stack([
                p_broadcast, R_i, Z_i, corners, psi_n_i, Bmag_i,
            ])
        else:
            # Fallback: [params(5), R, Z, psi_n, Bmag] = 9
            feat = np.column_stack([
                p_broadcast, R_i, Z_i, psi_n_i, Bmag_i,
            ])

        all_features.append(feat.astype(np.float32))

        # Targets
        tgt_cols = []
        for arr_name, ch_idx in target_map:
            arr = plasma[i] if arr_name == "plasma" else sources[i]
            tgt_cols.append(arr[ch_idx][m].astype(np.float32))
        all_targets.append(np.column_stack(tgt_cols))

        all_run_ids.append(np.full(n_valid, i, dtype=np.int32))

    features = np.concatenate(all_features, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    run_ids = np.concatenate(all_run_ids, axis=0)

    # Determine actual feature columns
    if has_corners:
        feat_cols = list(FEATURE_COLUMNS)
    else:
        feat_cols = list(PARAM_KEYS) + ["R", "Z", "psi_n", "Bmag"]

    print(f"\nDatabase: {features.shape[0]} cells from {N} runs")
    print(f"  Features: {features.shape} ({feat_cols})")
    print(f"  Targets:  {targets.shape} ({available_targets})")

    for j, name in enumerate(feat_cols):
        col = features[:, j]
        print(f"  {name:12s}: min={col.min():.4g}  max={col.max():.4g}  mean={col.mean():.4g}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(
        args.out,
        features=features,
        targets=targets,
        run_ids=run_ids,
        feature_columns=np.array(feat_cols),
        target_columns=np.array(available_targets),
        param_keys=np.array(PARAM_KEYS),
        n_runs=N,
    )
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
