# build_particle_source_npz_native.py
import os, numpy as np
from collections import defaultdict
from scipy.interpolate import griddata  # not used here but often handy
import quixote
from quixote import SolpsData

BASE_DIR = "/Users/42d/ORNL Dropbox/Abdou DIaw/SOLPS DB"
OUT_DIR  = "/Users/42d/ML_Projects/Tokamak_Pulse_Simulation_ML/scripts/data"

def _interior(arr, ny, nx):
    shp = arr.shape
    if shp == (ny, nx):            return arr
    if shp == (ny+2, nx+2):        return arr[1:ny+1, 1:nx+1]
    if shp == (nx, ny):            return arr.T
    if shp == (nx+2, ny+2):        return arr.T[1:ny+1, 1:nx+1]
    raise ValueError(f"Unexpected shape {shp} vs (ny,nx)=({ny},{nx})")

def _sum_strata(A):
    # (Ny, Nx, ncomp, nstrata) -> (Ny, Nx, ncomp)
    return np.nan_to_num(A, nan=0.0).sum(axis=-1)

def particle_source_deuterium(bal, comp_idx=-1):
    S = (_sum_strata(bal.eirene_mc_pipl_sna_bal) +
         _sum_strata(bal.eirene_mc_pmpl_sna_bal) +
         _sum_strata(bal.eirene_mc_pppl_sna_bal) +
         _sum_strata(bal.eirene_mc_papl_sna_bal))
    ny, nx = int(bal.ny), int(bal.nx)
    S = _interior(S[..., comp_idx], ny, nx)
    return S.astype(np.float32, copy=False)

def load_native_S(run_dir_abs):
    shot = SolpsData(os.path.join(quixote.module_path(), run_dir_abs))
    S = particle_source_deuterium(shot.balance)      # (ny, nx) interior
    ny, nx = int(shot.balance.ny), int(shot.balance.nx)
    assert S.shape == (ny, nx)
    return S

# --- collect by shape ---
by_shape = defaultdict(list)   # (ny, nx) -> list of S
runs = sorted(d for d in os.listdir(BASE_DIR)
              if d.startswith("run_") and os.path.isdir(os.path.join(BASE_DIR, d)))
for rn in runs:
    try:
        S = load_native_S(os.path.join(BASE_DIR, rn))
    except Exception as e:
        print("skip:", rn, "(", e, ")"); continue
    by_shape[S.shape].append(S)

if not by_shape:
    raise RuntimeError("No valid runs.")

# --- write one NPZ per shape ---
os.makedirs(OUT_DIR, exist_ok=True)
for (ny, nx), lst in by_shape.items():
    S_all = np.stack(lst, axis=0)  # (N, ny, nx)
    out = os.path.join(OUT_DIR, f"solps_particle_source_native_{ny}x{nx}.npz")
    np.savez_compressed(out, S=S_all)
    print(f"Saved {out}   S.shape={S_all.shape}")

