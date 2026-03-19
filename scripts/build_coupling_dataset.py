#!/usr/bin/env python3
# Authors: Abdourahmane (Abdou) Diaw - diawa@ornl.gov
# Project: APP-FPP Fire
# SPDX-License-Identifier: MIT
"""
Build plasma-to-sources training dataset directly from SOLPS-ITER balance.nc.

Reads balance.nc (NetCDF4) from each run directory and exports an NPZ with
all 9 EIRENE outputs that B2.5 needs:
  - 5 volumetric sources: Sp, Sne, Qe, Qi, Sm
  - 2 neutral densities:  dab2 (D0 atoms), dmb2 (D2 molecules)
  - 2 neutral temperatures: tab2 (D0), tmb2 (D2)

Produces:
  - plasma  : (N, 14, H, W) — Te, Ti, ne, ni, ua, vol, hx, hy, bb0-bb3, R, Z
  - sources : (N, Cs, H, W) — Sp, Sne, Qe, Qi, Sm, dab2, dmb2, tab2, tmb2
  - mask    : (N, H, W)     — validity mask
  - params  : (N, 5)        — control parameters (if params.json exists)
  - Rg, Zg  : (H, W)       — geometry

Usage:
    python scripts/build_coupling_dataset.py \
        --base-dir /home/cloud/solps-runs/diii-d/runners/d2-only \
        --out coupling_dataset.npz

    # Or find all balance.nc recursively:
    python scripts/build_coupling_dataset.py \
        --base-dir /home/cloud/solps-runs/diii-d/runners/d2-only \
        --recursive --out coupling_dataset.npz
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

J_PER_EV = 1.602176634e-19

PARAM_KEYS = ["Gamma_D2", "Ptot_W", "Gamma_core", "dna", "hci"]
PLASMA_KEYS = ["Te", "Ti", "ne", "ni", "ua", "vol", "hx", "hy",
               "bb0", "bb1", "bb2", "bb3", "R", "Z"]
SOURCE_KEYS = ["Sp", "Sne", "Qe", "Qi", "Sm", "dab2", "dmb2", "tab2", "tmb2"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sum_eirene_group(ds, var_names, species_dim=False):
    """Sum EIRENE balance variables over strata, optionally pick D+ (index 1).

    species_dim=True  → shape (nstra, ns, ny+2, nx+2), pick species index 1
    species_dim=False → shape (nstra, ny+2, nx+2)

    Returns: (ny, nx) array with ghost cells stripped, or None.
    """
    s = (slice(1, -1), slice(1, -1))
    acc = None
    for vn in var_names:
        if vn not in ds.variables:
            continue
        a = np.nan_to_num(np.array(ds.variables[vn]), nan=0.0)
        if species_dim:
            a = a[:, 1, s[0], s[1]].sum(axis=0)
        else:
            a = a[:, s[0], s[1]].sum(axis=0)
        acc = a if acc is None else acc + a
    return acc


def _neutral_field(ds, varname, species_idx=0):
    """Read a neutral field from EIRENE grid, map to B2.5 interior.

    EIRENE grid: (natm/nmol, eirny=38, eirnx=100)
    B2.5 interior: slice [1:-1, 2:-2] → (ny=36, nx=96)
    """
    if varname not in ds.variables:
        return None
    a = np.array(ds.variables[varname])
    return a[species_idx, 1:-1, 2:-2]


def _deep_get(d, path, default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load_params(params_path):
    """Load control parameters from params.json."""
    try:
        with open(params_path, "r") as f:
            data = json.load(f)

        pe = _deep_get(data, ("inputs", "power", "Pe_W"))
        pi = _deep_get(data, ("inputs", "power", "Pi_W"))
        n_core = _deep_get(data, ("inputs", "core", "density_m-3"))
        gamma_d2 = _deep_get(data, ("inputs", "gas_puffing", "targets", "D2", "value"))
        dna = _deep_get(data, ("inputs", "transport", "dna", "value"))
        hci = _deep_get(data, ("inputs", "transport", "hci", "value"))
        if hci is None:
            hci = _deep_get(data, ("inputs", "transport", "hce", "value"))

        if None not in (pe, pi, n_core, gamma_d2, dna, hci):
            return np.array([float(gamma_d2),
                             float(pe) + float(pi),
                             float(n_core),
                             float(dna),
                             float(hci)], dtype=np.float32)
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Core reader
# ---------------------------------------------------------------------------

def load_run(run_dir):
    """Load one SOLPS-ITER run from balance.nc.

    Returns dict with plasma, source, and neutral fields, or (None, reason).
    """
    from netCDF4 import Dataset

    nc_path = os.path.join(run_dir, "balance.nc")
    if not os.path.exists(nc_path):
        return None, "no_balance_nc"

    try:
        ds = Dataset(nc_path, "r")
    except Exception as e:
        return None, f"nc_open_error:{e}"

    try:
        nx = len(ds.dimensions["nx_plus2"]) - 2
        ny = len(ds.dimensions["ny_plus2"]) - 2
        s = (slice(1, -1), slice(1, -1))

        # --- Plasma profiles ---
        te = np.array(ds.variables["te"])[s]
        ti = np.array(ds.variables["ti"])[s]
        ne = np.array(ds.variables["ne"])[s]
        na = np.array(ds.variables["na"])[:, s[0], s[1]]
        ua_all = np.array(ds.variables["ua"])[:, s[0], s[1]]
        ni = na[1]       # D+ is species index 1
        ua = ua_all[1]

        # --- Geometry ---
        crx = np.array(ds.variables["crx"])
        cry = np.array(ds.variables["cry"])
        Rg = crx[-1][s]
        Zg = cry[-1][s]
        vol = np.array(ds.variables["vol"])[s]
        hx = np.array(ds.variables["hx"])[s]
        hy = np.array(ds.variables["hy"])[s]

        # Magnetic field: (4, ny+2, nx+2)
        bb = np.array(ds.variables["bb"])[:, s[0], s[1]]

        # --- EIRENE volumetric sources ---
        Sp = _sum_eirene_group(ds, [
            "eirene_mc_papl_sna_bal", "eirene_mc_pmpl_sna_bal",
            "eirene_mc_pipl_sna_bal", "eirene_mc_pppl_sna_bal",
        ], species_dim=True)

        Sne = _sum_eirene_group(ds, [
            "eirene_mc_pael_sne_bal", "eirene_mc_pmel_sne_bal",
        ], species_dim=False)

        Qe = _sum_eirene_group(ds, [
            "eirene_mc_eael_she_bal", "eirene_mc_emel_she_bal",
            "eirene_mc_eiel_she_bal", "eirene_mc_epel_she_bal",
        ], species_dim=False)

        Qi = _sum_eirene_group(ds, [
            "eirene_mc_eapl_shi_bal", "eirene_mc_empl_shi_bal",
            "eirene_mc_eipl_shi_bal", "eirene_mc_eppl_shi_bal",
        ], species_dim=False)

        Sm = _sum_eirene_group(ds, [
            "eirene_mc_mapl_smo_bal", "eirene_mc_mmpl_smo_bal",
            "eirene_mc_mipl_smo_bal", "eirene_mc_mppl_smo_bal",
        ], species_dim=True)

        # --- Neutral densities and temperatures (EIRENE grid → B2.5) ---
        dab2 = _neutral_field(ds, "dab2", species_idx=0)
        dmb2 = _neutral_field(ds, "dmb2", species_idx=0)
        tab2 = _neutral_field(ds, "tab2", species_idx=0)
        tmb2 = _neutral_field(ds, "tmb2", species_idx=0)

    except Exception as e:
        ds.close()
        return None, f"read_error:{e}"

    ds.close()

    if any(v is None for v in [Sp, Sne, Qe, Qi, Sm]):
        return None, "missing_eirene_source_vars"

    # Fill missing neutral fields with zeros
    zeros = np.zeros((ny, nx), dtype=np.float32)
    if dab2 is None:
        dab2 = zeros.copy()
    if dmb2 is None:
        dmb2 = zeros.copy()
    if tab2 is None:
        tab2 = zeros.copy()
    if tmb2 is None:
        tmb2 = zeros.copy()

    # J→eV conversion if needed (SOLPS stores temperatures in Joules)
    te_max = float(np.nanmax(te))
    if 0.0 < te_max < 1e-6:
        te = te / J_PER_EV
        ti = ti / J_PER_EV
        tab2 = tab2 / J_PER_EV
        tmb2 = tmb2 / J_PER_EV
        te_max = float(np.nanmax(te))

    if te_max < 0.1:
        return None, "unconverged_te_zero"

    return dict(
        Te=te.astype(np.float32), Ti=ti.astype(np.float32),
        ne=ne.astype(np.float32), ni=ni.astype(np.float32),
        ua=ua.astype(np.float32),
        vol=vol.astype(np.float32), hx=hx.astype(np.float32),
        hy=hy.astype(np.float32),
        bb0=bb[0].astype(np.float32), bb1=bb[1].astype(np.float32),
        bb2=bb[2].astype(np.float32), bb3=bb[3].astype(np.float32),
        R=Rg.astype(np.float32), Z=Zg.astype(np.float32),
        Sp=Sp.astype(np.float32), Sne=Sne.astype(np.float32),
        Qe=Qe.astype(np.float32), Qi=Qi.astype(np.float32),
        Sm=Sm.astype(np.float32),
        dab2=dab2.astype(np.float32), dmb2=dmb2.astype(np.float32),
        tab2=tab2.astype(np.float32), tmb2=tmb2.astype(np.float32),
    ), None


# ---------------------------------------------------------------------------
# QC
# ---------------------------------------------------------------------------

VALID_RULES = {
    "Te": "pos", "Ti": "pos", "ne": "pos", "ni": "pos",
    "ua": "finite", "vol": "pos", "hx": "pos", "hy": "pos",
    "bb0": "finite", "bb1": "finite", "bb2": "finite", "bb3": "finite",
    "R": "finite", "Z": "finite",
    "Sp": "finite", "Sne": "finite", "Qe": "finite", "Qi": "finite", "Sm": "finite",
    "dab2": "nonneg", "dmb2": "nonneg", "tab2": "nonneg", "tmb2": "nonneg",
}


def validate(fields):
    """Return (mask, details) where mask is per-cell validity."""
    shape = fields["Te"].shape
    mask = np.ones(shape, dtype=bool)
    details = {}

    for key, rule in VALID_RULES.items():
        a = fields[key]
        f = np.isfinite(a)
        if rule == "pos":
            v = f & (a > 0)
        elif rule == "nonneg":
            v = f & (a >= 0)
        else:
            v = f
        mask &= v
        details[f"valid_{key}"] = float(np.mean(v))

    return mask, details


def find_run_dirs(base_dir, run_prefix="run_", recursive=False):
    """Find SOLPS run directories."""
    base = Path(base_dir)
    if recursive:
        # Find all directories containing balance.nc
        dirs = sorted(str(nc.parent) for nc in base.rglob("balance.nc"))
    else:
        # Look for subdirectories matching prefix
        dirs = sorted(
            str(base / d) for d in os.listdir(base)
            if d.startswith(run_prefix) and os.path.isdir(base / d)
        )
    return dirs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-dir", required=True,
                    help="Root directory containing SOLPS run subdirectories")
    ap.add_argument("--out", default="coupling_dataset.npz",
                    help="Output NPZ path")
    ap.add_argument("--out-qc-csv", default=None,
                    help="QC report CSV path (default: <out>.qc.csv)")
    ap.add_argument("--run-prefix", default="run_",
                    help="Subdirectory prefix filter (ignored with --recursive)")
    ap.add_argument("--recursive", action="store_true",
                    help="Recursively search for balance.nc files")
    ap.add_argument("--min-mask-frac", type=float, default=0.01,
                    help="Minimum valid-cell fraction to accept a run")
    ap.add_argument("--max-runs", type=int, default=0, help="0 = all")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.out_qc_csv is None:
        args.out_qc_csv = args.out.replace(".npz", "") + "_qc.csv"

    run_dirs = find_run_dirs(args.base_dir, args.run_prefix, args.recursive)
    if args.max_runs > 0:
        run_dirs = run_dirs[:args.max_runs]
    if not run_dirs:
        sys.exit(f"No runs found under {args.base_dir}")

    print(f"Found {len(run_dirs)} candidate run directories")

    # ---- Collect ----
    all_plasma, all_sources, all_masks = [], [], []
    all_params, all_names = [], []
    qc_rows = []
    ref_shape = None
    ref_Rg, ref_Zg = None, None

    for rd in run_dirs:
        rn = os.path.basename(rd)
        row = {"run": rn, "status": "rejected", "reason": ""}

        # Load fields from balance.nc
        fields, err = load_run(rd)
        if fields is None:
            row["reason"] = err
            qc_rows.append(row)
            if args.verbose:
                print(f"  [skip] {rn}: {err}")
            continue

        shape = fields["Te"].shape
        if ref_shape is None:
            ref_shape = shape
            ref_Rg = fields["R"]
            ref_Zg = fields["Z"]
        elif shape != ref_shape:
            row["reason"] = f"shape_mismatch_{shape}"
            qc_rows.append(row)
            continue

        # QC
        mask, details = validate(fields)
        row.update(details)
        mfrac = float(np.mean(mask))
        row["mask_frac"] = mfrac

        if mfrac < args.min_mask_frac:
            row["reason"] = f"mask_too_small_{mfrac:.4f}"
            qc_rows.append(row)
            continue

        # Stack channels: (C, H, W)
        plasma = np.stack([fields[k] for k in PLASMA_KEYS], axis=0)
        sources = np.stack([fields[k] for k in SOURCE_KEYS], axis=0)

        all_plasma.append(plasma)
        all_sources.append(sources)
        all_masks.append(mask.astype(np.uint8))
        all_names.append(rn)

        # Params (optional — don't reject runs without params.json)
        pvec = load_params(os.path.join(rd, "params.json"))
        all_params.append(pvec if pvec is not None
                          else np.full(len(PARAM_KEYS), np.nan, dtype=np.float32))

        row["status"] = "accepted"
        row["reason"] = ""
        qc_rows.append(row)

    # ---- QC report ----
    os.makedirs(os.path.dirname(args.out_qc_csv) or ".", exist_ok=True)
    fieldnames = sorted({k for r in qc_rows for k in r.keys()})
    with open(args.out_qc_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in qc_rows:
            w.writerow(r)

    n_acc = len(all_plasma)
    n_rej = len(qc_rows) - n_acc
    reasons = {}
    for r in qc_rows:
        if r["status"] != "accepted":
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1

    if n_acc == 0:
        print(f"ERROR: 0 runs accepted out of {len(run_dirs)}")
        if reasons:
            for k, v in sorted(reasons.items(), key=lambda x: -x[1])[:10]:
                print(f"  {k}: {v}")
        sys.exit(1)

    # ---- Save ----
    plasma_arr = np.stack(all_plasma)     # (N, 14, H, W)
    source_arr = np.stack(all_sources)    # (N, 9, H, W)
    mask_arr = np.stack(all_masks)        # (N, H, W)
    param_arr = np.stack(all_params)      # (N, 5)

    save = dict(
        plasma=plasma_arr,
        sources=source_arr,
        plasma_keys=np.array(PLASMA_KEYS),
        source_keys=np.array(SOURCE_KEYS),
        mask=mask_arr,
        params=param_arr,
        param_keys=np.array(PARAM_KEYS),
        runs=np.array(all_names),
    )
    if ref_Rg is not None:
        save["Rg"] = ref_Rg
        save["Zg"] = ref_Zg

    np.savez_compressed(args.out, **save)

    print(f"\nSaved: {args.out}")
    print(f"  accepted : {n_acc} / {len(run_dirs)}")
    print(f"  plasma   : {plasma_arr.shape}  ({', '.join(PLASMA_KEYS)})")
    print(f"  sources  : {source_arr.shape}  ({', '.join(SOURCE_KEYS)})")
    print(f"  mask mean: {mask_arr.mean():.4f}")
    print(f"  params   : {param_arr.shape}")
    if reasons:
        print(f"\nRejection summary ({n_rej} total):")
        for k, v in sorted(reasons.items(), key=lambda x: -x[1])[:10]:
            print(f"  {k}: {v}")
    print(f"\nQC report: {args.out_qc_csv}")


if __name__ == "__main__":
    main()
## Build NPZ from your local D2-only runs:
#  python scripts/build_coupling_dataset.py \
#      --base-dir /home/cloud/solps-runs/diii-d/runners/d2-only \
#      --recursive \
#      --out coupling_dataset.npz \
#      --verbose
