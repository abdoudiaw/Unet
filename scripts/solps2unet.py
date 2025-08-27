import os, json, numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from scipy.interpolate import griddata

import quixote
from quixote import SolpsData

# ============================== CONFIG ===============================
BASE_DIR = "/Users/42d/ORNL Dropbox/Abdou DIaw/SOLPS DB"   # folder with run_* dirs
CORE_CSV = "/Users/42d/ML_Projects/Tokamak_Pulse_Simulation_ML/scripts/data/core.csv"
EDGE_CSV = "/Users/42d/ML_Projects/Tokamak_Pulse_Simulation_ML/scripts/data/edge.csv"
H, W     = 256, 128    # raster size used by your U-Net
OUT_NPZ  = "/Users/42d/ML_Projects/Tokamak_Pulse_Simulation_ML/scripts/data/solps_raster_dataset.npz"

# Fixed param order -> columns in params matrix
PARAM_KEYS = ["gas_puff", "p_tot", "core_flux", "dna", "hci"]
# ====================================================================

def load_params(params_json_path):
    """
    Read your params.json and return a dict with PARAM_KEYS
    (p_tot := Pe + Pi). Return None if missing/incomplete.
    """
    try:
        with open(params_json_path, "r") as f:
            data = json.load(f)
        # handle both "solps-iter-params" and "solps_iter_params"
        if "solps-iter-params" in data:
            P = data["solps-iter-params"][0]
        else:
            P = data["solps_iter_params"][0]
        pe, pi = P.get("Pe"), P.get("Pi")
        if pe is None or pi is None:  # need both for p_tot
            return None
        out = {
            "gas_puff":  P.get("gas_puff"),
            "core_flux": P.get("core_flux"),
            "dna":       P.get("dna"),
            "hci":       P.get("hci"),
            "p_tot":     float(pe) + float(pi),
        }
        if any(v is None for v in out.values()):
            return None
        return out
    except Exception:
        return None

def load_poly_csv(path):
    """Return closed (R,Z) polygon from CSV with either (R,Z) or (i,R,Z)."""
    M = np.loadtxt(path, delimiter=",", skiprows=1)
    M = M[np.all(np.isfinite(M), axis=1)]
    if M.shape[1] >= 3: R, Z = M[:,1], M[:,2]
    else:               R, Z = M[:,0], M[:,1]
    if (R[0], Z[0]) != (R[-1], Z[-1]):   # close the loop
        R = np.r_[R, R[0]]; Z = np.r_[Z, Z[0]]
    return R, Z

def load_case(run_dir_abs):
    """Read one SOLPS run and return r2d,z2d,te2d (native grid)."""
    shot = SolpsData(os.path.join(quixote.module_path(), run_dir_abs))
    r2d  = np.array(shot.crx)[:, :, -1]
    z2d  = np.array(shot.cry)[:, :, -1]
    te2d = np.array(shot.te)           # eV
    return r2d, z2d, te2d

# -------------------- discover runs and lock reference --------------------
runs = sorted(d for d in os.listdir(BASE_DIR)
              if d.startswith("run_") and os.path.isdir(os.path.join(BASE_DIR, d)))

if not runs:
    raise RuntimeError(f"No run_* folders found under {BASE_DIR}")

# find first run that has params.json -> lock grid extents to it
ref_run = None
for rn in runs:
    if os.path.exists(os.path.join(BASE_DIR, rn, "params.json")):
        ref_run = rn; break
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
mask_between_geom = (inside_edge & ~inside_core).reshape(H, W)       # geometry only

# -------------------- loop runs and build dataset ------------------------
Te_list      = []
Mask_list    = []             # per-run mask = geometry & valid raster sampling
Params_list  = []
RunNames     = []

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
        r2d, z2d, te2d = load_case(run_dir_abs)
    except Exception as e:
        print(f"skip (bad SOLPS case): {rn} ({e})"); continue

    te_raster = griddata((r2d.ravel(), z2d.ravel()),
                         te2d.ravel(), (Rg, Zg), method="linear")
    valid = np.isfinite(te_raster) & (te_raster > 0)

    mask = mask_between_geom & valid

    if mask.mean() < 0.01:   # tiny usable region — likely bad run
        print(f"skip (mask too small): {rn} coverage={mask.mean():.3f}")
        continue

    Te_list.append(te_raster.astype(np.float32))
    Mask_list.append(mask.astype(np.uint8))
    Params_list.append(param_vec)
    RunNames.append(rn)

# stack
if not Te_list:
    raise RuntimeError("No valid runs collected.")
Te      = np.stack(Te_list, axis=0)            # (N, H, W)
Mask    = np.stack(Mask_list, axis=0)          # (N, H, W) uint8
Params  = np.stack(Params_list, axis=0)        # (N, P)
runs_np = np.array(RunNames)

# -------------------- save one file --------------------
np.savez_compressed(
    OUT_NPZ,
    Te=Te, mask=Mask,
    Rg=Rg.astype(np.float32), Zg=Zg.astype(np.float32),
    params=Params, param_keys=np.array(PARAM_KEYS),
    runs=runs_np
)
print(f"Saved {OUT_NPZ}")
print(f"  Te shape     : {Te.shape}")
print(f"  mask shape   : {Mask.shape} (mean coverage={Mask.mean():.3f})")
print(f"  params shape : {Params.shape} with keys {PARAM_KEYS}")

