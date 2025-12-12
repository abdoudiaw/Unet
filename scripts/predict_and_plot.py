import numpy as np
import torch
from solps_ai.predict import load_checkpoint, predict_te, scale_params
from solps_ai.plotting import plot_te_curvilinear, plot_te_rectilinear

import h5py
from matplotlib import pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def plot_truth_pred_percent_error(Te_true, Te_pred, mask, fname="Te_truth_pred_pcterr.png",
                                  use_smape=False, eps=1e-3, vmax_pct=100):
    """
    Te_true, Te_pred: (H,W) in eV
    mask: (H,W) float/bool; >0.5 means valid region
    use_smape: False -> % error relative to truth; True -> symmetric %
    eps: floor to avoid division by ~0
    vmax_pct: clip color scale at this percentile value (e.g. 100%)
    """
    m = mask > 0.5
    # percent error map
    if use_smape:
        denom = (np.abs(Te_true) + np.abs(Te_pred) + eps)
        pct = 200.0 * np.abs(Te_pred - Te_true) / denom
    else:
        denom = np.maximum(Te_true, eps)
        pct = 100.0 * np.abs(Te_pred - Te_true) / denom

    # mask outside ROI
    Te_true_m = np.where(m, Te_true, np.nan)
    Te_pred_m = np.where(m, Te_pred, np.nan)
    pct_m     = np.where(m, pct, np.nan)

    # common Te color scale based on truth
    vmin = np.nanpercentile(Te_true_m, 1)
    vmax = np.nanpercentile(Te_true_m, 99)

    # robust cap for the percent error heatmap
    vmax_err = np.nanpercentile(pct_m, 95)  # robust auto cap
    vmax_err = min(vmax_err, vmax_pct)      # don't let it explode visually

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    im0 = axes[0].imshow(Te_true_m, origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)
    axes[0].set_title("Truth Te [eV]"); plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(Te_pred_m, origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)
    axes[1].set_title("Predicted Te [eV]"); plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(pct_m, origin='lower', cmap='magma', vmin=0.0, vmax=vmax_err)
    axes[2].set_title(("sMAPE [%]" if use_smape else "Percent error [%]"))
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xlabel("X (pixels)"); ax.set_ylabel("Y (pixels)")

    # masked summary stats
    valid = np.isfinite(pct_m)
    mean_pct = float(np.nanmean(pct_m[valid])) if np.any(valid) else float("nan")
    p90_pct  = float(np.nanpercentile(pct_m, 90)) if np.any(valid) else float("nan")
    print(f"Mean % error: {mean_pct:.2f} | 90th %%: {p90_pct:.2f} | vmax shown: {vmax_err:.1f}")

    plt.savefig(fname, dpi=300)
    plt.show()


def load_geometry_h5(path):
    with h5py.File(path, "r") as f:
        return f["R2D"][:], f["Z2D"][:]

def _load_truth_and_params(npz_path, idx=0):
    """Return (Te_true, mask, params_raw, target_name) for sample idx."""
    d = np.load(npz_path, allow_pickle=True)
    # mask per-sample (H,W)
    mask = (d["mask"][idx] > 0.5).astype(np.float32)

    # params if present
    if "params" in d.files and d["params"].size:
        params_raw = d["params"][idx].astype(np.float32).tolist()
    else:
        params_raw = []  # model will still run if you trained autoencoder-style

    # target: legacy (Te) or unified (Y)
    if "Te" in d.files:
        Te_true = d["Te"][idx].astype(np.float32)                    # (H,W)
        target_name = "Te"
    elif "Y" in d.files:
        Y = d["Y"][idx].astype(np.float32)                           # (C,H,W)
        # pick channel index for Te
        if "target_keys" in d.files:
            keys = [str(k) for k in d["target_keys"]]
            ch = keys.index("Te") if "Te" in keys else 0
        else:
            ch = 0
        Te_true = Y[ch]                                              # (H,W)
        target_name = "Te"
    else:
        raise KeyError("Dataset must contain 'Te' or 'Y'.")

    return Te_true, mask, params_raw, target_name


def image_to_solps(img, r2d, z2d, Rmin, Rmax, Zmin, Zmax):
    """
    Map (H,W) image to SOLPS grid (ny, nx) by inverse affine lookup.
    """
    H, W = img.shape
    # compute fractional pixel coords (u,v) corresponding to each (R,Z) cell center
    u = (r2d - Rmin) * (W - 1) / (Rmax - Rmin)   # (ny, nx)
    v = (z2d - Zmin) * (H - 1) / (Zmax - Zmin)   # (ny, nx)
    # order=1 -> bilinear; mode='nearest' to avoid NaNs at borders
    return map_coordinates(img, [v, u], order=1, mode="nearest")
    

#npz_path = "/Users/42d/Unet/scripts/solps_raster_dataset_source_particle.npz"
#npz_path = "/Users/42d/ML_Projects/Tokamak_Pulse_Simulation_ML/scripts/data/solps_raster_dataset_te.npz"
npz_path = "/Users/42d/ML_Projects/Tokamak_Pulse_Simulation_ML/scripts/data/solps_raster_dataset_te.npz"

def main(idx=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load checkpoint + normalizer (+ optional param scaler)
    model, norm, (param_mu, param_std) = load_checkpoint("/Users/42d/Downloads/unet_best.pt", device)

    # 2) Load truth + mask + that sample’s params
    Te_true, mask_ref, params_raw, _ = _load_truth_and_params(npz_path, idx=idx)

    # 3) Predict Te for the SAME sample (scale params if scaler exists)
    params = scale_params(params_raw, param_mu, param_std) if len(params_raw) else None
    Te_pred = predict_te(model, norm, mask_ref, params, device=device, as_numpy=True)  # (H,W) in eV:contentReference[oaicite:3]{index=3}

    # after you already have:
    # Te_true, mask_ref, params_raw = ...
    # Te_pred = ...
    plot_truth_pred_percent_error(Te_true, Te_pred, mask_ref, use_smape=False, eps=1e-3, vmax_pct=150)


if __name__ == "__main__":
    main(idx=2)  # change idx to visualize other samples

