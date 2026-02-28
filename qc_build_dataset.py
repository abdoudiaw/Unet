# Copyright 2025-2026 Oak Ridge National Laboratory
# @authors: Abdourahmane (Abdou) Diaw - diawa@ornl.gov
#
# SPDX-License-Identifier: MIT

import argparse
import csv
import json
import os

import numpy as np
from matplotlib.path import Path as MplPath
from scipy.interpolate import griddata

import quixote
from quixote import SolpsData


PARAM_KEYS = ["Gamma_D2", "Ptot_W", "n_core", "dna", "hci"]
DEFAULT_RULES = {
    "Te": ("pos",),
    "Ti": ("pos",),
    "ne": ("pos",),
    "ni": ("pos",),
    "ua": ("finite",),
    "Sp": ("finite",),
    "Qe": ("finite",),
    "Qi": ("finite",),
    "Sm": ("finite",),
}

ELECTRON_BASES = ["eael_she", "eiel_she", "emel_she", "epel_she"]
ION_BASES = ["eapl_shi", "eipl_shi", "empl_shi", "eppl_shi"]
MOMENTUM_BASES = ["mapl_smo", "mipl_smo", "mmpl_smo", "mppl_smo"]


def _deep_get(d, path, default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load_params(params_json_path):
    try:
        with open(params_json_path, "r") as f:
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
            return {
                "Gamma_D2": float(gamma_d2),
                "Ptot_W": float(pe) + float(pi),
                "n_core": float(n_core),
                "dna": float(dna),
                "hci": float(hci),
            }

        if "solps-iter-params" in data:
            p = data["solps-iter-params"][0]
        elif "solps_iter_params" in data:
            p = data["solps_iter_params"][0]
        else:
            return None

        pe, pi = p.get("Pe"), p.get("Pi")
        if pe is None or pi is None:
            return None

        out = {
            "Gamma_D2": p.get("gas_puff") or p.get("Gamma_D2"),
            "Ptot_W": float(pe) + float(pi),
            "n_core": p.get("core_density") or p.get("n_core"),
            "dna": p.get("dna"),
            "hci": p.get("hci") or p.get("hce"),
        }
        if any(v is None for v in out.values()):
            return None
        return {k: float(v) for k, v in out.items()}
    except Exception:
        return None


def load_poly_csv(path):
    m = np.loadtxt(path, delimiter=",", skiprows=1)
    m = m[np.all(np.isfinite(m), axis=1)]
    if m.shape[1] >= 3:
        r, z = m[:, 1], m[:, 2]
    else:
        r, z = m[:, 0], m[:, 1]
    if (r[0], z[0]) != (r[-1], z[-1]):
        r = np.r_[r, r[0]]
        z = np.r_[z, z[0]]
    return r, z


def valid_from_rule(a, rule):
    a = np.asarray(a)
    f = np.isfinite(a)
    if "pos" in rule:
        return f & (a > 0)
    return f


def _sum_species_strata(a):
    a = np.nan_to_num(np.asarray(a), nan=0.0)
    if a.ndim == 4:
        return a.sum(axis=(2, 3))
    if a.ndim == 3:
        return a.sum(axis=-1)
    if a.ndim == 2:
        return a
    raise ValueError(f"Unexpected rank {a.ndim} for EIRENE array")


def _interior(arr, n0, n1):
    shp = arr.shape
    if shp == (n0, n1):
        return arr
    if shp == (n0 + 2, n1 + 2):
        return arr[1:n0 + 1, 1:n1 + 1]
    if shp == (n1, n0):
        return arr.T
    if shp == (n1 + 2, n0 + 2):
        return arr.T[1:n0 + 1, 1:n1 + 1]
    raise ValueError(f"Unexpected shape {shp} vs ({n0},{n1})")


def _interior_like(arr, ref2d):
    ny, nx = ref2d.shape
    return _interior(arr, ny, nx)


def _best_field(bal, base):
    name_bal = f"eirene_mc_{base}_bal"
    name_raw = f"eirene_mc_{base}"
    if hasattr(bal, name_bal):
        return getattr(bal, name_bal)
    if hasattr(bal, name_raw):
        return getattr(bal, name_raw)
    return None


def _sum_group_like(bal, bases, ref2d):
    acc = None
    for base in bases:
        a = _best_field(bal, base)
        if a is None:
            continue
        m = _sum_species_strata(a)
        m = _interior_like(m, ref2d)
        acc = m if acc is None else (acc + m)
    if acc is None:
        raise RuntimeError(f"No matching fields for bases={bases}")
    return acc.astype(np.float32, copy=False)


def _sum_strata(a):
    return np.nan_to_num(a, nan=0.0).sum(axis=-1)


def particle_source_deuterium(bal, ref2d, comp_idx=-1):
    s = (
        _sum_strata(bal.eirene_mc_pipl_sna_bal)
        + _sum_strata(bal.eirene_mc_pmpl_sna_bal)
        + _sum_strata(bal.eirene_mc_pppl_sna_bal)
        + _sum_strata(bal.eirene_mc_papl_sna_bal)
    )
    s2 = s[..., comp_idx]
    s2 = _interior_like(s2, ref2d)
    return s2.astype(np.float32, copy=False)


def load_case_native(run_dir_abs, y_keys):
    shot = SolpsData(os.path.join(quixote.module_path(), run_dir_abs))
    r2d = np.array(shot.crx)[:, :, -1].astype(np.float32)
    z2d = np.array(shot.cry)[:, :, -1].astype(np.float32)

    te = np.array(shot.te).astype(np.float32)
    fields = {"Te": te}

    if "Ti" in y_keys:
        fields["Ti"] = np.array(shot.ti).astype(np.float32)
    if "ne" in y_keys:
        fields["ne"] = np.array(shot.ne).astype(np.float32)
    if "ni" in y_keys:
        fields["ni"] = np.array(shot.ni).astype(np.float32)
    if "ua" in y_keys:
        ua = np.array(shot.ua)
        if ua.ndim == 3:
            if ua.shape[2] > 1:
                ua = ua[:, :, 1]
            else:
                ua = ua[:, :, 0]
        fields["ua"] = np.asarray(ua, dtype=np.float32)

    need_balance = any(k in y_keys for k in ("Sp", "Qe", "Qi", "Sm"))
    if need_balance:
        bal = shot.balance
        if "Sp" in y_keys:
            fields["Sp"] = particle_source_deuterium(bal, te)
        if "Qe" in y_keys:
            fields["Qe"] = _sum_group_like(bal, ELECTRON_BASES, te)
        if "Qi" in y_keys:
            fields["Qi"] = _sum_group_like(bal, ION_BASES, te)
        if "Sm" in y_keys:
            fields["Sm"] = _sum_group_like(bal, MOMENTUM_BASES, te)

    for k in y_keys:
        if k not in fields:
            raise RuntimeError(f"Field {k} could not be loaded for run {run_dir_abs}")

    return r2d, z2d, fields


def interp_to_raster(r2d, z2d, a2d, rg, zg):
    return griddata((r2d.ravel(), z2d.ravel()), a2d.ravel(), (rg, zg), method="linear", fill_value=np.nan).astype(np.float32)


def build_geom_mask(R, Z, core_csv, edge_csv):
    if core_csv is None or edge_csv is None:
        return np.ones_like(R, dtype=bool)
    rcore, zcore = load_poly_csv(core_csv)
    redge, zedge = load_poly_csv(edge_csv)
    pts = np.c_[R.ravel(), Z.ravel()]
    inside_core = MplPath(np.c_[rcore, zcore]).contains_points(pts)
    inside_edge = MplPath(np.c_[redge, zedge]).contains_points(pts)
    return (inside_edge & ~inside_core).reshape(R.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--out-npz", required=True)
    ap.add_argument("--out-qc-csv", default="outputs/qc_report.csv")
    ap.add_argument("--y-keys", default="Te")
    ap.add_argument("--mode", choices=["native", "raster"], default="native")
    ap.add_argument("--H", type=int, default=256)
    ap.add_argument("--W", type=int, default=128)
    ap.add_argument("--core-csv", default=None)
    ap.add_argument("--edge-csv", default=None)
    ap.add_argument("--min-mask-frac", type=float, default=0.01)
    ap.add_argument("--max-runs", type=int, default=0, help="0 means all")
    ap.add_argument("--run-prefix", default="run_")
    ap.add_argument("--pad-multiple", type=int, default=8, help="Pad H/W to multiple (0 disables).")
    ap.add_argument("--verbose-errors", action="store_true")
    args = ap.parse_args()

    y_keys = [k.strip() for k in args.y_keys.split(",") if k.strip()]
    for k in y_keys:
        if k not in DEFAULT_RULES:
            raise ValueError(f"Unsupported y_key={k!r}.")

    runs = sorted(
        d for d in os.listdir(args.base_dir)
        if d.startswith(args.run_prefix) and os.path.isdir(os.path.join(args.base_dir, d))
    )
    if args.max_runs > 0:
        runs = runs[:args.max_runs]
    if not runs:
        raise RuntimeError(f"No runs found under {args.base_dir} with prefix {args.run_prefix!r}")

    ref_run = next((rn for rn in runs if os.path.exists(os.path.join(args.base_dir, rn, "params.json"))), None)
    if ref_run is None:
        raise RuntimeError("No params.json found in candidate runs.")

    r_ref, z_ref, _ = load_case_native(os.path.join(args.base_dir, ref_run), y_keys=y_keys)
    native_ref_shape = r_ref.shape
    if args.mode == "raster":
        r_lin = np.linspace(np.nanmin(r_ref), np.nanmax(r_ref), args.W)
        z_lin = np.linspace(np.nanmin(z_ref), np.nanmax(z_ref), args.H)
        Rg, Zg = np.meshgrid(r_lin, z_lin)
    else:
        Rg, Zg = r_ref, z_ref

    mask_geom = build_geom_mask(Rg, Zg, args.core_csv, args.edge_csv)

    pad_cfg = None
    if args.mode == "native" and args.pad_multiple and args.pad_multiple > 1:
        H0, W0 = Rg.shape
        H2 = int(np.ceil(H0 / args.pad_multiple) * args.pad_multiple)
        W2 = int(np.ceil(W0 / args.pad_multiple) * args.pad_multiple)
        ph = H2 - H0
        pw = W2 - W0
        pt, pb = ph // 2, ph - ph // 2
        pl, pr = pw // 2, pw - pw // 2
        if ph > 0 or pw > 0:
            pad_cfg = (pt, pb, pl, pr)
            Rg = np.pad(Rg, ((pt, pb), (pl, pr)), mode="edge")
            Zg = np.pad(Zg, ((pt, pb), (pl, pr)), mode="edge")
            mask_geom = np.pad(mask_geom, ((pt, pb), (pl, pr)), mode="constant", constant_values=False)
            print(f"[pad] native grid padded from ({H0},{W0}) to ({H2},{W2})")

    Ys = []
    Ms = []
    Ps = []
    RunNames = []
    qc_rows = []
    target_shape = None

    for rn in runs:
        row = {"run": rn, "status": "rejected", "reason": ""}
        try:
            p = load_params(os.path.join(args.base_dir, rn, "params.json"))
            if p is None:
                row["reason"] = "missing_or_bad_params"
                qc_rows.append(row)
                continue
            pvec = np.array([p[k] for k in PARAM_KEYS], dtype=np.float32)

            r2d, z2d, native = load_case_native(os.path.join(args.base_dir, rn), y_keys=y_keys)

            if args.mode == "native":
                if r2d.shape != native_ref_shape or z2d.shape != native_ref_shape:
                    row["reason"] = f"shape_mismatch_native_{r2d.shape}_vs_{native_ref_shape}"
                    qc_rows.append(row)
                    continue
                chans = [np.asarray(native[k], dtype=np.float32) for k in y_keys]
            else:
                chans = [interp_to_raster(r2d, z2d, np.asarray(native[k], dtype=np.float32), Rg, Zg) for k in y_keys]

            Y = np.stack(chans, axis=0)
            if pad_cfg is not None:
                pt, pb, pl, pr = pad_cfg
                Y = np.pad(Y, ((0, 0), (pt, pb), (pl, pr)), mode="constant", constant_values=0.0)
            m = mask_geom.copy()
            valid_fracs = {}
            for ci, k in enumerate(y_keys):
                v = valid_from_rule(Y[ci], DEFAULT_RULES[k])
                m &= v
                valid_fracs[f"valid_{k}"] = float(np.mean(v))

            mfrac = float(np.mean(m))
            row["mask_frac"] = mfrac
            row.update(valid_fracs)
            row["shape"] = str(tuple(Y.shape[1:]))

            if mfrac < args.min_mask_frac:
                row["reason"] = f"mask_too_small_{mfrac:.4f}"
                qc_rows.append(row)
                continue

            if target_shape is None:
                target_shape = Y.shape[1:]
            if Y.shape[1:] != target_shape:
                row["reason"] = f"target_shape_mismatch_{Y.shape[1:]}_vs_{target_shape}"
                qc_rows.append(row)
                continue

            row["status"] = "accepted"
            row["reason"] = ""
            qc_rows.append(row)

            Ys.append(Y.astype(np.float32))
            Ms.append(m.astype(np.uint8))
            Ps.append(pvec)
            RunNames.append(rn)
        except Exception as e:
            if args.verbose_errors:
                print(f"[reject] {rn}: {type(e).__name__}: {e}")
            row["reason"] = f"exception:{type(e).__name__}"
            qc_rows.append(row)

    os.makedirs(os.path.dirname(args.out_qc_csv) or ".", exist_ok=True)
    fieldnames = sorted({k for r in qc_rows for k in r.keys()})
    with open(args.out_qc_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in qc_rows:
            w.writerow(r)

    accepted = len(Ys)
    rej = sum(1 for r in qc_rows if r.get("status") != "accepted")
    reasons = {}
    for r in qc_rows:
        if r.get("status") != "accepted":
            key = r.get("reason", "unknown")
            reasons[key] = reasons.get(key, 0) + 1
    if reasons:
        top = sorted(reasons.items(), key=lambda kv: kv[1], reverse=True)[:8]
        print("Top rejection reasons:")
        for k, v in top:
            print(f"  {k}: {v}")

    if accepted == 0:
        print(f"Saved QC CSV: {args.out_qc_csv}")
        raise RuntimeError("No valid runs accepted. See QC CSV and rejection summary above.")

    Y_all = np.stack(Ys, axis=0)  # (N,C,H,W)
    M_all = np.stack(Ms, axis=0)  # (N,H,W)
    P_all = np.stack(Ps, axis=0)  # (N,P)
    runs_np = np.array(RunNames)

    os.makedirs(os.path.dirname(args.out_npz) or ".", exist_ok=True)
    np.savez_compressed(
        args.out_npz,
        Y=Y_all,
        y_keys=np.array(y_keys),
        mask=M_all,
        params=P_all,
        param_keys=np.array(PARAM_KEYS),
        Rg=Rg.astype(np.float32),
        Zg=Zg.astype(np.float32),
        runs=runs_np,
        mode=np.array(args.mode),
    )

    print(f"Saved dataset: {args.out_npz}")
    print(f"Saved QC CSV: {args.out_qc_csv}")
    print(f"Accepted runs: {accepted} | Rejected runs: {rej}")
    print(f"Y shape: {Y_all.shape} | mask mean: {M_all.mean():.4f} | params shape: {P_all.shape}")


if __name__ == "__main__":
    main()
