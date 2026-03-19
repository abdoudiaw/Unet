#!/usr/bin/env python3
"""
Build per-cell training database from solps.npz for the MLP ensemble.

Each row = one cell from one simulation:
    features: [gas_puff, p_tot, core_flux, dna, hci, psi_n, Bmag]
    targets:  [Te, Ti, ne, ua, Sp, Qe, Qi, Sm]

The database is saved as a compressed .npz file (not SQLite) for speed
and compatibility with the training script.

Usage:
    python mlp/build_2d_database.py --npz solps.npz --out mlp/dataset_2d.npz
    python mlp/build_2d_database.py --npz solps.npz --out mlp/dataset_2d.npz --bfield bfield.h5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


PARAM_KEYS = ["Gamma_D2", "Ptot_W", "n_core", "dna", "hci"]

FEATURE_COLUMNS = (
    "Gamma_D2", "Ptot_W", "n_core", "dna", "hci",
    "psi_n", "Bmag",
)

TARGET_COLUMNS = ("Te", "Ti", "ne", "ua", "Sp", "Qe", "Qi", "Sm")


def compute_psi_n(Rg, Zg, R0=None, Z0=None):
    """
    Approximate normalised poloidal flux from (R,Z) grid.

    If an equilibrium is available, use the real psi. Otherwise,
    approximate as distance from magnetic axis normalised to the
    domain extent.  This is a placeholder — the real psi_n should
    be computed from the equilibrium file when available.
    """
    if R0 is None:
        R0 = float(Rg.mean())
    if Z0 is None:
        Z0 = float(Zg.mean())
    rho = np.sqrt((Rg - R0) ** 2 + (Zg - Z0) ** 2)
    rho_max = rho.max()
    if rho_max > 0:
        return rho / rho_max
    return np.zeros_like(rho)


def compute_Bmag(Rg, Bt0=None, R0=None):
    """
    Approximate |B| assuming dominant toroidal field Bt ~ Bt0*R0/R.

    When a bfield.h5 is provided, use the actual (Br, Bt, Bz) instead.
    """
    if Bt0 is None:
        Bt0 = 2.0   # typical DIII-D on-axis Bt
    if R0 is None:
        R0 = 1.7     # typical DIII-D axis
    Bt = Bt0 * R0 / np.where(Rg > 0.5, Rg, 0.5)
    return np.abs(Bt)


def load_bfield_h5(path, Rg, Zg):
    """Load actual |B| from a bfield.h5 and interpolate onto the data grid."""
    import h5py
    from scipy.interpolate import RegularGridInterpolator

    with h5py.File(path, "r") as f:
        r = f["r"][:]
        z = f["z"][:]
        br = f["br"][:]
        bt = f["bt"][:]
        bz = f["bz"][:]

    bmag = np.sqrt(br**2 + bt**2 + bz**2)
    interp = RegularGridInterpolator((z, r), bmag, method="linear",
                                      bounds_error=False, fill_value=None)
    pts = np.column_stack([Zg.ravel(), Rg.ravel()])
    return interp(pts).reshape(Rg.shape)


def main():
    ap = argparse.ArgumentParser(description="Build per-cell MLP database from solps.npz")
    ap.add_argument("--npz", required=True, help="Path to solps.npz")
    ap.add_argument("--out", default="mlp/dataset_2d.npz", help="Output path")
    ap.add_argument("--bfield", default=None, help="Optional bfield.h5 for real |B|")
    ap.add_argument("--R0", type=float, default=None, help="Magnetic axis R (m)")
    ap.add_argument("--Z0", type=float, default=None, help="Magnetic axis Z (m)")
    ap.add_argument("--Bt0", type=float, default=None, help="On-axis Bt (T)")
    args = ap.parse_args()

    print(f"Loading {args.npz} ...")
    d = np.load(args.npz, allow_pickle=True)

    Y = d["Y"]              # (N, C, H, W)
    mask = d["mask"]         # (N, H, W)
    params = d["params"]     # (N, 5)
    y_keys = list(d["y_keys"])
    Rg = d["Rg"]             # (H, W)
    Zg = d["Zg"]             # (H, W)
    runs = d["runs"]

    N, C, H, W = Y.shape
    print(f"  {N} runs, {C} channels, grid {H}x{W}")
    print(f"  y_keys: {y_keys}")

    # Map y_keys to target columns
    target_idx = []
    for t in TARGET_COLUMNS:
        if t in y_keys:
            target_idx.append(y_keys.index(t))
        else:
            target_idx.append(None)
    available_targets = [t for t, i in zip(TARGET_COLUMNS, target_idx) if i is not None]
    print(f"  Available targets: {available_targets}")

    # Compute spatial features
    psi_n = compute_psi_n(Rg, Zg, R0=args.R0, Z0=args.Z0)

    if args.bfield is not None:
        print(f"Loading |B| from {args.bfield} ...")
        Bmag = load_bfield_h5(args.bfield, Rg, Zg)
    else:
        print("Approximating |B| from Bt~1/R ...")
        Bmag = compute_Bmag(Rg, Bt0=args.Bt0, R0=args.R0)

    # Build per-cell rows
    all_features = []
    all_targets = []
    all_run_ids = []

    for i in range(N):
        m = mask[i].astype(bool)  # (H, W)
        n_valid = m.sum()
        if n_valid == 0:
            continue

        # Control params broadcast to all cells
        p = params[i]  # (5,)
        p_broadcast = np.tile(p, (n_valid, 1))  # (n_valid, 5)

        # Spatial features for valid cells
        psi_cells = psi_n[m]      # (n_valid,)
        bmag_cells = Bmag[m]      # (n_valid,)

        # Features: [params(5), psi_n, Bmag] = 7 columns
        feat = np.column_stack([p_broadcast, psi_cells, bmag_cells])
        all_features.append(feat)

        # Targets
        tgt_cols = []
        for ti in target_idx:
            if ti is not None:
                tgt_cols.append(Y[i, ti][m])
            else:
                tgt_cols.append(np.zeros(n_valid, dtype=np.float32))
        tgt = np.column_stack(tgt_cols)
        all_targets.append(tgt)

        all_run_ids.append(np.full(n_valid, i, dtype=np.int32))

    features = np.concatenate(all_features, axis=0).astype(np.float32)
    targets = np.concatenate(all_targets, axis=0).astype(np.float32)
    run_ids = np.concatenate(all_run_ids, axis=0)

    print(f"\nDatabase: {features.shape[0]} cells from {N} runs")
    print(f"  Features: {features.shape} ({FEATURE_COLUMNS})")
    print(f"  Targets:  {targets.shape} ({available_targets})")

    # Feature statistics
    for j, name in enumerate(FEATURE_COLUMNS):
        col = features[:, j]
        print(f"  {name:12s}: min={col.min():.4g}  max={col.max():.4g}  mean={col.mean():.4g}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(
        args.out,
        features=features,
        targets=targets,
        run_ids=run_ids,
        feature_columns=np.array(FEATURE_COLUMNS),
        target_columns=np.array(available_targets),
        param_keys=np.array(PARAM_KEYS),
        n_runs=N,
        Rg=Rg,
        Zg=Zg,
        psi_n=psi_n,
        Bmag=Bmag,
    )
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
