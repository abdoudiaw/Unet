# build_energy_npz_native.py
import os, numpy as np
from collections import defaultdict
import quixote
from quixote import SolpsData

BASE_DIR = "/Users/42d/ORNL Dropbox/Abdou DIaw/SOLPS DB"
OUT_DIR  = "/Users/42d/ML_Projects/Tokamak_Pulse_Simulation_ML/scripts/data"

# ---------------- utils ----------------
def _interior2d(M, ny, nx):
    """Return interior (ny,nx) regardless of input orientation/margins."""
    shp = M.shape
    if shp == (ny, nx):            return M
    if shp == (nx, ny):            return M.T
    if shp == (ny+2, nx+2):        return M[1:ny+1, 1:nx+1]
    if shp == (nx+2, ny+2):        return M[1:nx+1, 1:ny+1].T
    raise ValueError(f"Unexpected 2D shape {shp} for target (ny,nx)=({ny},{nx})")

def _sum_species_strata(A):
    """
    Sum over species and strata. Accepts:
      - (nx+2, ny+2, nsp, nstrata)
      - (ny+2, nx+2, nsp, nstrata)
      - already-summed 2D arrays
    """
    A = np.nan_to_num(np.asarray(A), nan=0.0)
    if A.ndim == 4:
        return A.sum(axis=(2,3))   # -> (nx+2, ny+2) or (ny+2, nx+2)
    if A.ndim == 2:
        return A
    if A.ndim == 3:  # rare: if strata already merged, but species not
        return A.sum(axis=-1)
    raise ValueError(f"Unexpected rank {A.ndim} for EIRENE array")

def _best_field(bal, base):
    """Prefer *_bal if present, else raw. Returns ndarray or None."""
    name_bal = f"eirene_mc_{base}_bal"
    name_raw = f"eirene_mc_{base}"
    if hasattr(bal, name_bal): return getattr(bal, name_bal)
    if hasattr(bal, name_raw): return getattr(bal, name_raw)
    return None

def _sum_group(bal, bases, ny, nx):
    acc = None
    for base in bases:
        A = _best_field(bal, base)
        if A is None:
            continue
        M = _sum_species_strata(A)     # 2D with edges
        M = _interior2d(M, ny, nx)     # interior (ny, nx)
        acc = M if acc is None else (acc + M)
    if acc is None:
        raise RuntimeError(f"No matching fields found for {bases}")
    return acc.astype(np.float32, copy=False)

# ---------------- field groups ----------------
# Electron energy equation sources from species X: eael/eiel/emel/epel (…_she)
ELECTRON_BASES = ["eael_she", "eiel_she", "emel_she", "epel_she"]
# Ion energy equation sources from species X: eapl/eipl/empl/eppl (…_shi)
ION_BASES      = ["eapl_shi", "eipl_shi", "empl_shi", "eppl_shi"]

def load_Qe_Qi(run_dir_abs):
    shot = SolpsData(run_dir_abs)  # path to the run directory
    ny, nx = int(shot.balance.ny), int(shot.balance.nx)
    Qe = _sum_group(shot.balance, ELECTRON_BASES, ny, nx)  # [W m^-3], signed
    Qi = _sum_group(shot.balance, ION_BASES,      ny, nx)  # [W m^-3], signed
    assert Qe.shape == (ny, nx) and Qi.shape == (ny, nx)
    return Qe, Qi

# ---------------- collect & write ----------------
by_shape_e = defaultdict(list)   # (ny, nx) -> list of arrays
by_shape_i = defaultdict(list)

runs = sorted(
    d for d in os.listdir(BASE_DIR)
    if d.startswith("run_") and os.path.isdir(os.path.join(BASE_DIR, d))
)

for rn in runs:
    run_path = os.path.join(BASE_DIR, rn)
    try:
        Qe, Qi = load_Qe_Qi(run_path)
    except Exception as e:
        print("skip:", rn, "(", e, ")")
        continue
    by_shape_e[Qe.shape].append(Qe)
    by_shape_i[Qi.shape].append(Qi)

if not by_shape_e:
    raise RuntimeError("No valid runs found / no energy fields present.")

os.makedirs(OUT_DIR, exist_ok=True)
for (ny, nx), lst in by_shape_e.items():
    Qe_all = np.stack(lst, axis=0)                 # (N, ny, nx)
    out_e = os.path.join(OUT_DIR, f"solps_electron_power_native_{ny}x{nx}.npz")
    np.savez_compressed(out_e, Qe=Qe_all)
    print(f"Saved {out_e}   Qe.shape={Qe_all.shape}")

for (ny, nx), lst in by_shape_i.items():
    Qi_all = np.stack(lst, axis=0)                 # (N, ny, nx)
    out_i = os.path.join(OUT_DIR, f"solps_ion_power_native_{ny}x{nx}.npz")
    np.savez_compressed(out_i, Qi=Qi_all)
    print(f"Saved {out_i}   Qi.shape={Qi_all.shape}")

