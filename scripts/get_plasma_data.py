# Copyright 2025-2026 Oak Ridge National Laboratory
# @authors: Abdourahmane (Abdou) Diaw - diawa@ornl.gov
#
# SPDX-License-Identifier: MIT


import os, json, numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from scipy.interpolate import griddata
import quixote
from functools import reduce
import operator as op
import json
from quixote import SolpsData
import json
import numpy as np
import os
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from quixote import GridDataPlot

# ============================== CONFIG ===============================
BASE_DIR = "/Users/42d/ORNL Dropbox/Abdou DIaw/SOLPS_DB/ens__DIII-D__APP-FPP__D_C_Ne__ss__lhs__20250124_144649"   # folder with run_* dirs
CORE_CSV = "/Users/42d/ML_Projects/Tokamak_Pulse_Simulation_ML/scripts/data/core.csv"
EDGE_CSV = "/Users/42d/ML_Projects/Tokamak_Pulse_Simulation_ML/scripts/data/edge.csv"
H, W     = 256, 128    # raster size used by your U-Net
OUT_NPZ  = "solps_raster_dataset_small.npz"

run='/Users/42d/ORNL Dropbox/Abdou Diaw/SOLPS_DB/ens__DIII-D__APP-FPP__D_C_Ne__ss__lhs__20250124_144649/run_7cbd3c21__D_C_Ne'
shot = SolpsData(os.path.join(quixote.module_path(), run))

#'psi', 'psisep',
print(dir(shot))
print(shot.psi.shape, shot.psisep)
te2d = np.array(shot.te)
ti2d = np.array(shot.ti)
ne   = np.array(shot.ne)
ni   = np.array(shot.ni)
ua   = np.array(shot.ua[:, :, 1])

print(ua.shape)


# --- load ---
psi = np.array(shot.psi)
psisep = float(shot.psisep)
te2d = np.array(shot.te)

R = np.array(shot.crx[:,:,-1])  # or shot.crx_triang depending on grid type
Z = np.array(shot.cry[:,:,-1])


psi = np.array(shot.psi)
psisep = float(shot.psisep)
R = np.array(shot.crx[:, :, -1])
Z = np.array(shot.cry[:, :, -1])

te2d = np.array(shot.te)

fig, ax = plt.subplots(1, 1, figsize=(6, 8))

pc = ax.pcolormesh(R, Z, te2d, shading="auto", cmap="inferno")
fig.colorbar(pc, ax=ax, label="Te [eV]")

ax.contour(R, Z, psi, levels=[psisep], colors="cyan", linewidths=2.0)

ax.set_aspect("equal", adjustable="box")
ax.set_xlim(np.nanmin(R), np.nanmax(R))
ax.set_ylim(np.nanmin(Z), np.nanmax(Z))

ax.set_title("Te with separatrix (psi = psisep)")
ax.set_xlabel("R [m]")
ax.set_ylabel("Z [m]")
plt.show()



# Targets to extract from SOLPS; (name, require_positive)
# Set require_positive=False for things like potential φ that can be negative.
TARGET_SPECS = [
    ("Te", True),
    ("Ti", True),
    ("ne", True),
    ("ua", False),
    ("Sp", False),
    ("Qe", False),
    ("Qi", False),
    ("Sm", False),
]

#TARGET_SPECS = [
#    "Te","Ti","ne","ni","ua","Sp","Qe","Qi","Sm"] #,"Prad"]

# ====================================================================


# Fixed param order -> columns in params matrix
PARAM_KEYS = ["Gamma_D2", "Ptot_W", "n_core", "dna", "hci"]

def _deep_get(d, path, default=None):
    """Safely get nested keys: path like ('inputs','power','Pe_W')"""
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# ---------------- field groups ----------------
# Electron energy equation sources from species X: eael/eiel/emel/epel (…_she)
ELECTRON_BASES = ["eael_she", "eiel_she", "emel_she", "epel_she"]
# Ion energy equation sources from species X: eapl/eipl/empl/eppl (…_shi)
ION_BASES      = ["eapl_shi", "eipl_shi", "empl_shi", "eppl_shi"]
# Particle momentum equation sources from species X: ma/mi/mm/mp (…_smo)
MOMENTUM_BASES = ["mapl_smo", "mipl_smo", "mmpl_smo", "mppl_smo"]

# Fixed param order
PARAM_KEYS = ["Gamma_D2", "Ptot_W", "n_core", "dna", "hci"]

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

        # ---- NEW solpsmeta schema ----
        Pe = _deep_get(data, ("inputs", "power", "Pe_W"))
        Pi = _deep_get(data, ("inputs", "power", "Pi_W"))
        n_core = _deep_get(data, ("inputs", "core", "density_m-3"))
        Gamma_D2 = _deep_get(data, ("inputs", "gas_puffing", "targets", "D2", "value"))
        dna = _deep_get(data, ("inputs", "transport", "dna", "value"))
        hci = _deep_get(data, ("inputs", "transport", "hci", "value"))
        if hci is None:
            hci = _deep_get(data, ("inputs", "transport", "hce", "value"))

        if None not in (Pe, Pi, n_core, Gamma_D2, dna, hci):
            return {
                "Gamma_D2": float(Gamma_D2),
                "Ptot_W": float(Pe) + float(Pi),
                "n_core": float(n_core),
                "dna": float(dna),
                "hci": float(hci),
            }

        # ---- OPTIONAL fallback to OLD schema ----
        if "solps-iter-params" in data:
            P = data["solps-iter-params"][0]
        elif "solps_iter_params" in data:
            P = data["solps_iter_params"][0]
        else:
            return None

        pe, pi = P.get("Pe"), P.get("Pi")
        if pe is None or pi is None:
            return None

        out = {
            "Gamma_D2": P.get("gas_puff") or P.get("Gamma_D2"),
            "Ptot_W": float(pe) + float(pi),
            "n_core": P.get("core_density") or P.get("n_core"),
            "dna": P.get("dna"),
            "hci": P.get("hci") or P.get("hce"),
        }
        if any(v is None for v in out.values()):
            return None
        return {k: float(v) for k, v in out.items()}

    except Exception:
        return None



def _interior_like(arr, ref2d):
    ny, nx = ref2d.shape  # "ny,nx" here means "ref.shape"
    return _interior(arr, ny, nx)

def _interior(arr, n0, n1):
    shp = arr.shape
    if shp == (n0, n1):            return arr
    if shp == (n0+2, n1+2):        return arr[1:n0+1, 1:n1+1]
    if shp == (n1, n0):            return arr.T
    if shp == (n1+2, n0+2):        return arr.T[1:n0+1, 1:n1+1]
    raise ValueError(f"Unexpected shape {shp} vs (n0,n1)=({n0},{n1})")

        # ---------------- utils ----------------

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


def _sum_group_like(bal, bases, ref2d):
    acc = None
    for base in bases:
        A = _best_field(bal, base)
        if A is None:
            continue
        M = _sum_species_strata(A)       # 2D (maybe with ghost cells)
        M = _interior_like(M, ref2d)     # <-- force orientation to match te2d
        acc = M if acc is None else (acc + M)
    if acc is None:
        raise RuntimeError(f"No matching fields found for {bases}")
    return acc.astype(np.float32, copy=False)


def load_poly_csv(path):
    """Return closed (R,Z) polygon from CSV with either (R,Z) or (i,R,Z)."""
    M = np.loadtxt(path, delimiter=",", skiprows=1)
    M = M[np.all(np.isfinite(M), axis=1)]
    if M.shape[1] >= 3: R, Z = M[:,1], M[:,2]
    else:               R, Z = M[:,0], M[:,1]
    if (R[0], Z[0]) != (R[-1], Z[-1]):   # close the loop
        R = np.r_[R, R[0]]; Z = np.r_[Z, Z[0]]
    return R, Z

def _sum_strata(A):
    # (Ny, Nx, ncomp, nstrata) -> (Ny, Nx, ncomp)
    return np.nan_to_num(A, nan=0.0).sum(axis=-1)

def particle_source_deuterium(bal, ref2d, comp_idx=-1):
    S = (_sum_strata(bal.eirene_mc_pipl_sna_bal) +
         _sum_strata(bal.eirene_mc_pmpl_sna_bal) +
         _sum_strata(bal.eirene_mc_pppl_sna_bal) +
         _sum_strata(bal.eirene_mc_papl_sna_bal))
    S2 = S[..., comp_idx]              # 2D with edges or interior
    S2 = _interior_like(S2, ref2d)     # match Te orientation
    return S2.astype(np.float32, copy=False)

def load_case(run_dir_abs):
    shot = SolpsData(os.path.join(quixote.module_path(), run_dir_abs))

    te2d = np.array(shot.te)
    ti2d = np.array(shot.ti)
    ne   = np.array(shot.ne)
    ni   = np.array(shot.ni)
    ua   = np.array(shot.ua[:, :, 1])

    r2d  = np.array(shot.crx)[:, :, -1]
    z2d  = np.array(shot.cry)[:, :, -1]

    Sp = particle_source_deuterium(shot.balance, te2d)
    Qe = _sum_group_like(shot.balance, ELECTRON_BASES, te2d)
    Qi = _sum_group_like(shot.balance, ION_BASES,      te2d)
    Sm = _sum_group_like(shot.balance, MOMENTUM_BASES, te2d)

    native = {
        "Te": te2d,
        "Ti": ti2d,
        "ne": ne,
        "ni": ni,
        "ua": ua,
        "Sp": Sp,
        "Qe": Qe,
        "Qi": Qi,
        "Sm": Sm,
    }
    return r2d, z2d, native

    

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from scipy.interpolate import griddata
from functools import reduce
import operator as op
import quixote
from quixote import SolpsData

# ============================== CONFIG ===============================
BASE_DIR = "/Users/42d/ORNL Dropbox/Abdou DIaw/SOLPS_DB/ens__DIII-D__APP-FPP__D_C_Ne__ss__lhs__20250124_144649"
CORE_CSV = "/Users/42d/ML_Projects/Tokamak_Pulse_Simulation_ML/scripts/data/core.csv"
EDGE_CSV = "/Users/42d/ML_Projects/Tokamak_Pulse_Simulation_ML/scripts/data/edge.csv"
H, W     =  96,40 #256, 128
OUT_NPZ  = "solps_raster_dataset_new_.npz"

PARAM_KEYS = ["Gamma_D2", "Ptot_W", "n_core", "dna", "hci"]

# Field validity rules:
# - Te, Ti, ne, ni : finite & > 0
# - ua            : finite (can be +/-)
# - Qe, Qi, Sm    : finite (can be +/-)
# - Sp            : finite & > 0   (your requirement)
TARGET_RULES = {
    "Te": ("pos",),
    "Ti": ("pos",),
    "ne": ("pos",),
    "ni": ("pos",),
    "ua": ("finite",),
    "Qe": ("finite",),
    "Qi": ("finite",),
    "Sm": ("finite",),
    "Sp": ("pos",),
}
Y_KEYS = ["Te", "Ti", "ne", "ni", "ua", "Sp", "Qe", "Qi", "Sm"]  # no Prad


# ============================== HELPERS ===============================
def load_poly_csv(path):
    M = np.loadtxt(path, delimiter=",", skiprows=1)
    M = M[np.all(np.isfinite(M), axis=1)]
    if M.shape[1] >= 3:
        R, Z = M[:, 1], M[:, 2]
    else:
        R, Z = M[:, 0], M[:, 1]
    if (R[0], Z[0]) != (R[-1], Z[-1]):
        R = np.r_[R, R[0]]
        Z = np.r_[Z, Z[0]]
    return R, Z

def build_geometry_mask(Rg, Zg, core_csv, edge_csv):
    Rcore, Zcore = load_poly_csv(core_csv)
    Redge, Zedge = load_poly_csv(edge_csv)
    pts = np.c_[Rg.ravel(), Zg.ravel()]
    inside_core = MplPath(np.c_[Rcore, Zcore]).contains_points(pts)
    inside_edge = MplPath(np.c_[Redge, Zedge]).contains_points(pts)
    return (inside_edge & ~inside_core).reshape(Rg.shape)

def valid_from_rule(A, rule):
    A = np.asarray(A)
    finite = np.isfinite(A)
    if "pos" in rule:
        return finite & (A > 0)
    return finite

def interp_to_raster(r2d, z2d, arr2d, Rg, Zg):
    arr2d = np.asarray(arr2d)
    if arr2d.shape != r2d.shape:
        raise ValueError(f"Field shape {arr2d.shape} != r2d shape {r2d.shape}")
    return griddata(
        (r2d.ravel(), z2d.ravel()),
        arr2d.ravel(),
        (Rg, Zg),
        method="linear",
        fill_value=np.nan,
    ).astype(np.float32)


# ============================== DISCOVER RUNS ===============================
runs = sorted(
    d for d in os.listdir(BASE_DIR)
    if d.startswith("run_6aabb6c2__D_C_Ne") and os.path.isdir(os.path.join(BASE_DIR, d))
)
if not runs:
    raise RuntimeError(f"No run_* folders found under {BASE_DIR}")

ref_run = next((rn for rn in runs if os.path.exists(os.path.join(BASE_DIR, rn, "params.json"))), None)
if ref_run is None:
    raise RuntimeError("No params.json found in any run_* folder.")

# ============================== BUILD COMMON RASTER GRID ===============================
r2d_ref, z2d_ref, native_ref = load_case(os.path.join(BASE_DIR, ref_run))
Rmin, Rmax = np.nanmin(r2d_ref), np.nanmax(r2d_ref)
Zmin, Zmax = np.nanmin(z2d_ref), np.nanmax(z2d_ref)
r_lin = np.linspace(Rmin, Rmax, W)
z_lin = np.linspace(Zmin, Zmax, H)
Rg, Zg = np.meshgrid(r_lin, z_lin)  # (H,W)

mask_geometry = build_geometry_mask(Rg, Zg, CORE_CSV, EDGE_CSV)


import h5py
import numpy as np

geom_ref = "/Users/42d/ML_Projects/Tokamak_Pulse_Simulation_ML/scripts/data/geom_ref.h5"

with h5py.File(geom_ref, "w") as f:
    f.create_dataset("R2D", data=Rg.astype(np.float32))
    f.create_dataset("Z2D", data=Zg.astype(np.float32))

print("Saved geometry:")
print("R2D shape:", Rg.shape)
print("Z2D shape:", Zg.shape)


# ============================== OPTIONAL: ONE-RUN SMOKE TEST ===============================
SMOKE_TEST = False
if SMOKE_TEST:
    test_rn  = runs[0]
    r2d, z2d, native = load_case(os.path.join(BASE_DIR, test_rn))
    te_r = interp_to_raster(r2d, z2d, native["Te"], Rg, Zg)
    m = mask_geometry & valid_from_rule(te_r, TARGET_RULES["Te"])

    extent = [Rmin, Rmax, Zmin, Zmax]
    plt.figure(figsize=(10,4))
    ax = plt.subplot(1,2,1)
    im = ax.imshow(te_r, origin="lower", extent=extent, aspect="auto")
    plt.colorbar(im, ax=ax, label="Te")
    ax.contour(Rg, Zg, m.astype(float), levels=[0.5], linewidths=2)
    ax.set_title(f"Te raster + mask ({test_rn})"); ax.set_xlabel("R"); ax.set_ylabel("Z")

    ax2 = plt.subplot(1,2,2)
    ax2.imshow(mask_geometry, origin="lower", extent=extent, aspect="auto")
    ax2.set_title("mask_geometry"); ax2.set_xlabel("R"); ax2.set_ylabel("Z")
    plt.tight_layout(); plt.show()

