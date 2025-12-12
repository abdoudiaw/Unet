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
OUT_NPZ  = "/Users/42d/Unet/scripts/solps_raster_dataset_source_particle.npz"

# Fixed param order -> columns in params matrix
PARAM_KEYS = ["gas_puff", "p_tot", "core_flux", "dna", "hci"]
# ====================================================================

import numpy as np

import numpy as np

def particle_source_deuterium(bal, comp_idx: int = -1):
    """
    Total particle source to plasma (ions + molecules + photons + atoms)
    for Deuterium (component index -1 by default).

    Returns
    -------
    S_D : (ny, nx) float64
        2-D interior map on the SOLPS grid.
    Tot_Sp_D : float
        Toroidally integrated total [s^-1] (sqrt(g) already included).
    """
    def _sum_strata(A):  # (Ny, Nx, ncomp, nstrata) -> (Ny, Nx, ncomp)
        return np.nan_to_num(A, nan=0.0).sum(axis=-1)

    # sum over strata for each channel
    S = (_sum_strata(bal.eirene_mc_pipl_sna_bal)
         + _sum_strata(bal.eirene_mc_pmpl_sna_bal)
         + _sum_strata(bal.eirene_mc_pppl_sna_bal)
         + _sum_strata(bal.eirene_mc_papl_sna_bal))      # (Ny, Nx, ncomp)

    # pick Deuterium component (last)
    S = S[..., comp_idx]                                   # (Ny, Nx)

    # normalize to (ny, nx)
    ny, nx = int(bal.ny), int(bal.nx)
    ny2, nx2 = getattr(bal, "ny_plus2", ny+2), getattr(bal, "nx_plus2", nx+2)

    if S.shape == (ny, nx):
        pass
    elif S.shape == (ny2, nx2):           # ghosts present
        S = S[1:ny+1, 1:nx+1]
    elif S.shape == (nx, ny):             # transposed interior
        S = S.T
    elif S.shape == (nx2, ny2):           # transposed ghosts
        S = S.T[1:ny+1, 1:nx+1]
    else:
        raise ValueError(f"Unexpected shape {S.shape}; expected {(ny,nx)} or ghosts { (ny2,nx2) }.")

    Tot_Sp_D = float(np.nansum(S))
    return S.astype(np.float64, copy=False), Tot_Sp_D



import numpy as np
from scipy.interpolate import griddata

def particle_source_deuterium(bal, comp_idx: int = -1,
                              r_target=None, z_target=None,
                              target_grid: str = "eirene"):
    """
    Compute the total particle source (ions + molecules + photons + atoms)
    for Deuterium.

    Parameters
    ----------
    bal : Balance object from SolpsData.balance
        Contains EIRENE Monte Carlo source tallies and geometry.
    comp_idx : int, optional
        Deuterium component index (-1 = last).
    r_target, z_target : 2D arrays, optional
        Target plasma grid coordinates for interpolation (if target_grid='plasma').
    target_grid : {'eirene','plasma'}
        'eirene' -> return S on EIRENE grid (no interpolation)
        'plasma' -> remap/interpolate S to the provided plasma grid.

    Returns
    -------
    S_D : 2D float array
        Source term [m^-3 s^-1].
    Tot_Sp_D : float
        Toroidally integrated total [s^-1].
    R_src, Z_src : 2D float arrays
        Coordinates of the returned S_D field (either EIRENE or plasma grid).
    mask_src : 2D bool array
        Mask of finite cells (True = valid data).
    """

    def _sum_strata(A):  # (Ny, Nx, ncomp, nstrata) -> (Ny, Nx, ncomp)
        return np.nan_to_num(A, nan=0.0).sum(axis=-1)

    # --- 1. Combine channels ---
    S = (_sum_strata(bal.eirene_mc_pipl_sna_bal)
         + _sum_strata(bal.eirene_mc_pmpl_sna_bal)
         + _sum_strata(bal.eirene_mc_pppl_sna_bal)
         + _sum_strata(bal.eirene_mc_papl_sna_bal))  # (Ny, Nx, ncomp)

    S = S[..., comp_idx]  # pick deuterium component

    # --- 2. Trim ghosts if present ---
    ny, nx = int(bal.ny), int(bal.nx)
    ny2, nx2 = getattr(bal, "ny_plus2", ny + 2), getattr(bal, "nx_plus2", nx + 2)

    if S.shape == (ny2, nx2):
        S = S[1:ny+1, 1:nx+1]
    elif S.shape == (nx2, ny2):
        S = S.T[1:ny+1, 1:nx+1]
    elif S.shape == (nx, ny):
        S = S.T
    elif S.shape != (ny, nx):
        raise ValueError(f"Unexpected shape {S.shape}, expected {(ny, nx)}")

    # --- 3. Coordinates for EIRENE grid ---
    if hasattr(bal, "crx") and hasattr(bal, "cry"):
        R_src = np.array(bal.crx)[1:ny+1, 1:nx+1, -1]
        Z_src = np.array(bal.cry)[1:ny+1, 1:nx+1, -1]
    else:
        # fallback: approximate using uniform spacing
        r_lin = np.linspace(1.0, 2.0, nx)
        z_lin = np.linspace(-1.5, 1.5, ny)
        R_src, Z_src = np.meshgrid(r_lin, z_lin)

    mask_src = np.isfinite(S) & (np.abs(S) > 0)

    # --- 4. Optional remap to plasma grid ---
    if target_grid.lower() == "plasma":
        if r_target is None or z_target is None:
            raise ValueError("Must provide r_target and z_target for plasma remap.")
        S_interp = griddata(
            (R_src[mask_src].ravel(), Z_src[mask_src].ravel()),
            S[mask_src].ravel(),
            (r_target, z_target),
            method="linear"
        )
        S = np.nan_to_num(S_interp, nan=0.0)
        R_src, Z_src = r_target, z_target
        mask_src = np.isfinite(S) & (np.abs(S) > 0)

    Tot_Sp_D = float(np.nansum(S))
    return S.astype(np.float64, copy=False), Tot_Sp_D, R_src, Z_src, mask_src


def particle_source_deuterium(bal, comp_idx: int = -1):
    """
    Compute total D particle source (ions + neutrals + molecules + photons).

    Returns
    -------
    S_D : ndarray (ny, nx)
        2-D interior map on SOLPS grid, same as bal.eirene_mc_* arrays.
    Tot_Sp_D : float
        Toroidally integrated total [s^-1].
    """
    def _sum_strata(A):
        # Collapse strata dimension and handle NaNs
        return np.nan_to_num(A, nan=0.0).sum(axis=-1)

    # Combine EIRENE source channels
    S = (
        _sum_strata(bal.eirene_mc_pipl_sna_bal)
        + _sum_strata(bal.eirene_mc_pmpl_sna_bal)
        + _sum_strata(bal.eirene_mc_pppl_sna_bal)
        + _sum_strata(bal.eirene_mc_papl_sna_bal)
    )

    # Select the Deuterium component
    S = S[..., comp_idx]  # (ny, nx) ideally

    # If ghosts exist, strip them cleanly
    ny, nx = int(bal.ny), int(bal.nx)
    if S.shape == (ny + 2, nx + 2):
        S = S[1:ny + 1, 1:nx + 1]

    # Ensure float64
    S = S.astype(np.float64, copy=False)
    Tot_Sp_D = np.nansum(S)
    return S, Tot_Sp_D


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

    # Source term for Deuterium
    S_D, Tot_Sp_D = particle_source_deuterium(shot.balance)
    print(S_D.shape, shot.balance.ny, shot.balance.nx)

    S = np.nan_to_num(S_D, nan=0.0)
#
#    fig, ax = plt.subplots(figsize=(7, 9))
#    pcm = ax.pcolormesh(r2d, z2d, S, shading='auto', cmap='inferno')  # note: transpose here
#    ax.set_xlabel("R [m]")
#    ax.set_ylabel("Z [m]")
#    ax.set_title("Deuterium Particle Source $S_D$")
#    fig.colorbar(pcm, ax=ax, label="source [a.u.]")
#    plt.tight_layout()
#    plt.show()




    return r2d, z2d, S

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
        r2d, z2d, te = load_case(run_dir_abs)
    except Exception as e:
        print(f"skip (bad SOLPS case): {rn} ({e})"); continue

    te_raster = griddata((r2d.ravel(), z2d.ravel()),
                                 te.ravel(), (Rg, Zg), method="linear")
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
te      = np.stack(Te_list, axis=0)            # (N, H, W)
Mask    = np.stack(Mask_list, axis=0)          # (N, H, W) uint8
Params  = np.stack(Params_list, axis=0)        # (N, P)
runs_np = np.array(RunNames)

# -------------------- save one file --------------------
np.savez_compressed(
    OUT_NPZ,
    Te=te, mask=Mask,
    Rg=Rg.astype(np.float32), Zg=Zg.astype(np.float32),
    params=Params, param_keys=np.array(PARAM_KEYS),
    runs=runs_np
)
print(f"Saved {OUT_NPZ}")
print(f"  te shape     : {te.shape}")
print(f"  mask shape   : {Mask.shape} (mean coverage={Mask.mean():.3f})")
print(f"  params shape : {Params.shape} with keys {PARAM_KEYS}")

