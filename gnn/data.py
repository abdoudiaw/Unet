# Authors: Abdourahmane (Abdou) Diaw - diawa@ornl.gov
# SPDX-License-Identifier: MIT
"""
Build PyG graph dataset from coupling_dataset.npz (native SOLPS mesh).

Each simulation becomes one graph on the native (ny, nx) grid:
  - Nodes  = valid cells (mask > 0)
  - Edges  = 4-connected neighbors on the native grid
  - Node features: [psi_n, |B|]  (from bb3 = |B| in the plasma array)
  - Global params: control parameters broadcast to all nodes
  - Targets: [Te, Ti, ne, ua, Sp, Qe, Qi, Sm]

The coupling_dataset.npz is built by:
    python scripts/build_coupling_dataset.py --base-dir <SOLPS_runs> --out coupling_dataset.npz

Alternatively, you can build graphs directly from balance.nc directories.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data


TARGET_KEYS = ("Te", "Ti", "ne", "ua", "Sp", "Qe", "Qi", "Sm")

# Indices into the 14-channel plasma array from build_coupling_dataset.py:
# ["Te", "Ti", "ne", "ni", "ua", "vol", "hx", "hy", "bb0", "bb1", "bb2", "bb3", "R", "Z"]
PLASMA_IDX = {"Te": 0, "Ti": 1, "ne": 2, "ni": 3, "ua": 4,
              "vol": 5, "hx": 6, "hy": 7,
              "bb0": 8, "bb1": 9, "bb2": 10, "bb3": 11, "R": 12, "Z": 13}

# Indices into the 9-channel source array:
# ["Sp", "Sne", "Qe", "Qi", "Sm", "dab2", "dmb2", "tab2", "tmb2"]
SOURCE_IDX = {"Sp": 0, "Sne": 1, "Qe": 2, "Qi": 3, "Sm": 4,
              "dab2": 5, "dmb2": 6, "tab2": 7, "tmb2": 8}


def _build_edges_masked(mask_2d):
    """
    Build undirected edge list from a 2D boolean mask using 4-connectivity.

    Connects adjacent valid cells in the (ny, nx) native grid.
    Returns edge_index (2, E) and per-edge (dR, dZ, dist) placeholders
    (actual geometry filled in per-graph from R, Z coordinates).
    """
    H, W = mask_2d.shape
    node_id = np.full((H, W), -1, dtype=np.int64)
    node_id[mask_2d] = np.arange(mask_2d.sum())

    src, dst = [], []
    # Right neighbor (j+1) and down neighbor (i+1)
    for di, dj in [(0, 1), (1, 0)]:
        for i in range(H - di):
            for j in range(W - dj):
                ni, nj = i + di, j + dj
                if mask_2d[i, j] and mask_2d[ni, nj]:
                    a, b = node_id[i, j], node_id[ni, nj]
                    src.extend([a, b])  # undirected
                    dst.extend([b, a])

    return np.array([src, dst], dtype=np.int64)


def compute_psi_n_from_grid(R, Z, R0=None, Z0=None):
    """
    Approximate normalised psi from native grid coordinates.
    R, Z are (ny, nx) arrays — the actual SOLPS cell centres.
    """
    if R0 is None:
        R0 = float(R.mean())
    if Z0 is None:
        Z0 = float(Z.mean())
    rho = np.sqrt((R - R0)**2 + (Z - Z0)**2)
    mx = rho.max()
    return (rho / mx if mx > 0 else np.zeros_like(rho)).astype(np.float32)


def load_coupling_dataset(npz_path, target_keys=TARGET_KEYS):
    """
    Load coupling_dataset.npz and build PyG graph list.

    Parameters
    ----------
    npz_path : path to coupling_dataset.npz (from build_coupling_dataset.py)
    target_keys : which fields to predict

    Returns
    -------
    graphs : list[Data]
    meta : dict with dataset info
    """
    d = np.load(npz_path, allow_pickle=True)

    plasma = d["plasma"]          # (N, 14, H, W) — native grid
    sources = d["sources"]        # (N, 9, H, W)
    mask = d["mask"]              # (N, H, W)
    params = d["params"]          # (N, 5)
    plasma_keys = list(d["plasma_keys"])
    source_keys = list(d["source_keys"])
    runs = list(d["runs"]) if "runs" in d.files else [str(i) for i in range(len(plasma))]

    N, _, H, W = plasma.shape
    print(f"  Coupling dataset: {N} runs, native grid {H}x{W}")

    # Map target keys to (array, channel_index)
    target_map = []
    for tk in target_keys:
        if tk in PLASMA_IDX:
            target_map.append(("plasma", PLASMA_IDX[tk]))
        elif tk in SOURCE_IDX:
            target_map.append(("source", SOURCE_IDX[tk]))
        else:
            raise ValueError(f"Unknown target key: {tk}")

    # Use first valid run to build reference edge structure
    ref_mask = mask[0].astype(bool)
    ref_edges = _build_edges_masked(ref_mask)
    n_nodes_ref = int(ref_mask.sum())

    # Reference geometry from first run
    R_ref = plasma[0, PLASMA_IDX["R"]]  # (H, W)
    Z_ref = plasma[0, PLASMA_IDX["Z"]]

    # Edge attributes from reference geometry
    R_flat = R_ref[ref_mask]
    Z_flat = Z_ref[ref_mask]
    if ref_edges.shape[1] > 0:
        dR = R_flat[ref_edges[1]] - R_flat[ref_edges[0]]
        dZ = Z_flat[ref_edges[1]] - Z_flat[ref_edges[0]]
        dist = np.sqrt(dR**2 + dZ**2)
        ref_edge_attr = np.stack([dR, dZ, dist], axis=-1).astype(np.float32)
    else:
        ref_edge_attr = np.zeros((0, 3), dtype=np.float32)

    edge_index_t = torch.from_numpy(ref_edges)
    edge_attr_t = torch.from_numpy(ref_edge_attr)

    print(f"  Graph: {n_nodes_ref} nodes, {ref_edges.shape[1]} edges")

    graphs = []
    for i in range(N):
        m = mask[i].astype(bool)
        if m.sum() != n_nodes_ref:
            continue  # skip if different mask shape

        # Node geometry features: [psi_n, |B|]
        # |B| = bb3 (4th component of bb array in SOLPS)
        Bmag = plasma[i, PLASMA_IDX["bb3"]][m]
        R_i = plasma[i, PLASMA_IDX["R"]]
        Z_i = plasma[i, PLASMA_IDX["Z"]]
        psi_n = compute_psi_n_from_grid(R_i, Z_i)[m]

        node_feat = np.stack([psi_n, Bmag], axis=-1).astype(np.float32)

        # Control params broadcast to all nodes
        p = params[i].astype(np.float32)
        params_broadcast = np.tile(p, (n_nodes_ref, 1))

        # Targets
        tgt_cols = []
        for arr_name, ch_idx in target_map:
            arr = plasma[i] if arr_name == "plasma" else sources[i]
            tgt_cols.append(arr[ch_idx][m].astype(np.float32))
        targets = np.stack(tgt_cols, axis=-1)

        data = Data(
            x=torch.from_numpy(node_feat),
            edge_index=edge_index_t,
            edge_attr=edge_attr_t,
            params=torch.from_numpy(params_broadcast),
            y=torch.from_numpy(targets),
            num_nodes=n_nodes_ref,
        )
        data.run_name = runs[i] if i < len(runs) else str(i)
        data.run_idx = i
        graphs.append(data)

    print(f"  Built {len(graphs)} graphs")
    return graphs, dict(
        target_keys=list(target_keys),
        n_nodes=n_nodes_ref,
        n_edges=ref_edges.shape[1],
        H=H, W=W,
    )
