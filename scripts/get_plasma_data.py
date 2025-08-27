import os, json, numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from scipy.interpolate import griddata

from functools import reduce
import operator as op

from quixote import SolpsData

# ============================== CONFIG ===============================
BASE_DIR = "/Users/42d/ORNL Dropbox/Abdou DIaw/SOLPS DB"
CORE_CSV = "/Users/42d/ML_Projects/Tokamak_Pulse_Simulation_ML/scripts/data/core.csv"
EDGE_CSV = "/Users/42d/ML_Projects/Tokamak_Pulse_Simulation_ML/scripts/data/edge.csv"
H, W     = 256, 128
OUT_NPZ  = "/Users/42d/ML_Projects/Tokamak_Pulse_Simulation_ML/scripts/data/solps_raster_dataset.npz"

# Fixed param order -> columns in params matrix
PARAM_KEYS = ["gas_puff", "p_tot", "core_flux", "dna", "hci"]

# Targets to extract from SOLPS; (name, require_positive)
# Set require_positive=False for things like potential φ that can be negative.
TARGET_SPECS = [
    ("Te", True),
    ("ne", True),
    ("ni", True),
    ("ti", True),
    # ("phi", False),   # uncomment if available in your SolpsData
]
# ====================================================================

def load_params(params_json_path):
    try:
        with open(params_json_path, "r") as f:
            data = json.load(f)
        P = data.get("solps-iter-params", data.get("solps_iter_params"))[0]
        pe, pi = P.get("Pe"), P.get("Pi")
        if pe is None or pi is None:
            return None
        out = {
            "gas_puff":  P.get("gas_puff"),
            "core_flux": P.get("core_flux"),
            "dna":       P.get("dna"),
            "hci":       P.get("hci"),
            "p_tot":     float(pe) + float(pi),
        }
        return None if any(v is None for v in out.values()) else out
    except Exception:
        return None

def load_poly_csv(path):
    M = np.loadtxt(path, delimiter=",", skiprows=1)
    M = M[np.all(np.isfinite(M), axis=1)]
    if M.shape[1] >= 3: R, Z = M[:,1], M[:,2]
    else:               R, Z = M[:,0], M[:,1]
    if (R[0], Z[0]) != (R[-1], Z[-1]):
        R = np.r_[R, R[0]]; Z = np.r_[Z, Z[0]]
    return R, Z

def _maybe_get(shot, name):
    """Return a numpy array for the given field name, trying a couple aliases."""
    aliases = {
        "Te": ["te"],
        "ne": ["ne"],
        "ni": ["ni"],
        "ti": ["ti"],
        "phi": ["phi", "po"],   # try both, depending on your SolpsData wrapper
    }
    for a in aliases.get(name, [name]):
        if hasattr(shot, a):
            return np.array(getattr(shot, a))
    return None

def load_case(run_dir_abs):
    """Return r2d, z2d and a dict of native 2D fields keyed by TARGET_SPECS names."""
    # IMPORTANT: pass absolute path directly
    shot = SolpsData(run_dir_abs)

    r2d = np.array(shot.crx)[:, :, -1]
    z2d = np.array(shot.cry)[:, :, -1]

    fields = {}
    for name, _ in TARGET_SPECS:
        arr = _maybe_get(shot, name)
        if arr is None:
            fields[name] = None
        else:
            fields[name] = np.array(arr)  # native (Ny,Nx)
    return r2d, z2d, fields

# -------------------- discover runs and lock reference --------------------
runs = sorted(d for d in os.listdir(BASE_DIR)
              if d.startswith("run_") and os.path.isdir(os.path.join(BASE_DIR, d)))

if not runs:
    raise RuntimeError(f"No run_* folders found under {BASE_DIR}")

ref_run = next((rn for rn in runs
                if os.path.exists(os.path.join(BASE_DIR, rn, "params.json"))), None)
if ref_run is None:
    raise RuntimeError("No params.json found in any run_* folder.")

# reference extents for raster grid
r2d_ref, z2d_ref, _ = load_case(os.path.join(BASE_DIR, ref_run))
Rmin, Rmax = np.nanmin(r2d_ref), np.nanmax(r2d_ref)
Zmin, Zmax = np.nanmin(z2d_ref), np.nanmax(z2d_ref)
r_lin = np.linspace(Rmin, Rmax, W)
z_lin = np.linspace(Zmin, Zmax, H)
Rg, Zg = np.meshgrid(r_lin, z_lin)   # (H,W) — used for ALL runs

# -------------------- geometry mask (between core & edge) -----------------
Rcore, Zcore = load_poly_csv(CORE_CSV)
Redge, Zedge = load_poly_csv(EDGE_CSV)
pts = np.c_[Rg.ravel(), Zg.ravel()]
inside_core = MplPath(np.c_[Rcore, Zcore]).contains_points(pts)
inside_edge = MplPath(np.c_[Redge, Zedge]).contains_points(pts)
mask_geometry = (inside_edge & ~inside_core).reshape(H, W)

# -------------------- loop runs and build dataset ------------------------
Y_list       = []  # (C,H,W) per run
Mask_list    = []
Params_list  = []
RunNames     = []

target_keys_kept = [name for name, _ in TARGET_SPECS]  # will prune if missing entirely

for rn in runs:
    run_dir_abs = os.path.join(BASE_DIR, rn)

    # params
    p = load_params(os.path.join(run_dir_abs, "params.json"))
    if p is None:
        print(f"skip (no/invalid params): {rn}")
        continue
    param_vec = np.array([p[k] for k in PARAM_KEYS], dtype=np.float32)

    # native → raster
    try:
        r2d, z2d, native = load_case(run_dir_abs)
    except Exception as e:
        print(f"skip (bad SOLPS case): {rn} ({e})")
        continue

    # Interpolate each requested target onto (H,W)
    rasters = {}
    valids  = {}
    all_missing = True
    for name, require_pos in TARGET_SPECS:
        arr = native.get(name)
        if arr is None:
            rasters[name] = None
            valids[name]  = None
            continue
        all_missing = False
        raster = griddata(
            (r2d.ravel(), z2d.ravel()),
            arr.ravel(),
            (Rg, Zg),
            method="linear",
            fill_value=np.nan,
        )
        if require_pos:
            valid = np.isfinite(raster) & (raster > 0)
        else:
            valid = np.isfinite(raster)
        rasters[name] = raster.astype(np.float32)
        valids[name]  = valid

    if all_missing:
        print(f"skip (no requested fields present): {rn}")
        continue

    # Combined mask: geometry AND valid for all present fields
    present_valids = [v for v in valids.values() if v is not None]
    valid_all = reduce(op.and_, present_valids) if present_valids else np.zeros_like(mask_geometry, dtype=bool)
    mask = mask_geometry & valid_all

    if mask.mean() < 0.01:
        print(f"skip (mask too small): {rn} coverage={mask.mean():.3f}")
        continue

    # Stack channels in TARGET_SPECS order; drop channels that were missing for this run
    chans = []
    kept_for_run = []
    for name, _ in TARGET_SPECS:
        if rasters[name] is not None:
            chans.append(rasters[name][None, ...])  # (1,H,W)
            kept_for_run.append(name)
    Y = np.concatenate(chans, axis=0)  # (C,H,W)

    # On the first accepted run, lock the final target order based on present channels
    if len(Y_list) == 0:
        target_keys_kept = kept_for_run
    else:
        # Ensure channel consistency across runs
        if kept_for_run != target_keys_kept:
            # Reorder/align by name
            reord = []
            for k in target_keys_kept:
                reord.append(rasters[k][None, ...])
            Y = np.concatenate(reord, axis=0)

    Y_list.append(Y.astype(np.float32))
    Mask_list.append(mask.astype(np.uint8))
    Params_list.append(param_vec)
    RunNames.append(rn)

# stack
if not Y_list:
    raise RuntimeError("No valid runs collected.")
Y       = np.stack(Y_list, axis=0)             # (N, C, H, W)
Mask    = np.stack(Mask_list, axis=0)          # (N, H, W) uint8
Params  = np.stack(Params_list, axis=0)        # (N, P)
runs_np = np.array(RunNames)

# -------------------- save one file --------------------
np.savez_compressed(
    OUT_NPZ,
    Y=Y, mask=Mask,
    Rg=Rg.astype(np.float32), Zg=Zg.astype(np.float32),
    params=Params, param_keys=np.array(PARAM_KEYS),
    target_keys=np.array(target_keys_kept),
    runs=runs_np
)
print(f"Saved {OUT_NPZ}")
print(f"  Y shape      : {Y.shape} with channels {target_keys_kept}")
print(f"  mask shape   : {Mask.shape} (mean coverage={Mask.mean():.3f})")
print(f"  params shape : {Params.shape} with keys {PARAM_KEYS}")

