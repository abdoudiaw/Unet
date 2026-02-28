# Copyright 2025-2026 Oak Ridge National Laboratory
# @authors: Abdourahmane (Abdou) Diaw - diawa@ornl.gov
#
# SPDX-License-Identifier: MIT

import os
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Set random seed
np.random.seed(4321)

# Imports from solpex
from solpex import data
from solpex.data import MaskedLogStandardizer, MaskedSymLogStandardizer
from solpex.models import UNet, bottleneck_to_z, ParamToZ, z_to_bottleneck
from solpex.train import train_unet
from solpex.latent import extract_z_dataset, train_param2z
from solpex.utils import pick_device, sample_from_loader, eval_param2z_one, nearest_neighbor_in_Z
from solpex.plotting import plot_ae_recon_one

# ============================================================================
# Helper Functions
# ============================================================================

def _as_torch(x, device):
    """Convert input to torch tensor on specified device."""
    if torch.is_tensor(x):
        return x.to(device)
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def denorm_z(z_n: torch.Tensor, z_mu: np.ndarray, z_std: np.ndarray, device):
    """Denormalize latent vector z."""
    mu = torch.as_tensor(z_mu, device=device, dtype=z_n.dtype)
    std = torch.as_tensor(z_std, device=device, dtype=z_n.dtype)
    return z_n * std + mu


def fit_z_scaler(Z):
    """Compute mean and std for z normalization."""
    mu = Z.mean(axis=0)
    std = Z.std(axis=0)
    std[std < 1e-8] = 1.0
    return mu.astype(np.float32), std.astype(np.float32)


def apply_z_scaler(Z, mu, std):
    """Apply z normalization."""
    return (Z - mu) / std


def fit_p_scaler(P):
    """Compute mean and std for parameter normalization."""
    x = P.astype(np.float64, copy=False)
    mu = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(~np.isfinite(std) | (std < 1e-8), 1.0, std)
    mu = np.where(~np.isfinite(mu), 0.0, mu)
    return mu.astype(np.float32), std.astype(np.float32)


def apply_p_scaler(P, mu, std):
    """Apply parameter normalization."""
    return (P - mu) / std


def masked_mse(pred, target, mask):
    """Masked MSE on tensors shaped (B,C,H,W), (B,C,H,W), (B,1,H,W)."""
    if mask.shape[1] == 1 and pred.shape[1] != 1:
        mask = mask.expand_as(pred)
    diff2 = (pred - target) ** 2
    return float((diff2 * mask).sum().item() / max(mask.sum().item(), 1e-8))


def assert_finite(name, x):
    """Raise if tensor/array/scalar contains NaN/Inf."""
    arr = x
    if torch.is_tensor(x):
        arr = x.detach().cpu().numpy()
    arr = np.asarray(arr)
    if not np.all(np.isfinite(arr)):
        raise RuntimeError(f"{name} has non-finite values.")
    
# ============================================================================
# Surrogate Testing
# ============================================================================

@torch.no_grad()
def test_full_surrogate(ae, mlp, sample, z_mu, z_std, device):
    """
    Test full surrogate pipeline: params -> z -> reconstruction.
    
    Args:
        ae: Autoencoder model
        mlp: Parameter-to-latent MLP
        sample: Data sample dict with 'x', 'params', 'y', 'mask'
        z_mu: Latent mean for denormalization
        z_std: Latent std for denormalization
        device: Torch device
        
    Returns:
        dict: Contains y_true, y_pred, z_true, z_pred, etc.
    """
    ae.eval()
    mlp.eval()

    # Convert inputs to tensors
    x      = _as_torch(sample["x"], device).unsqueeze(0)          # (1,C,H,W)
    p_mu = sample.get("p_mu", None)
    p_std = sample.get("p_std", None)
    params_np = np.asarray(sample["params"], dtype=np.float32)
    if p_mu is not None and p_std is not None:
        params_np = apply_p_scaler(params_np, p_mu, p_std)
    params = _as_torch(params_np, device).unsqueeze(0)            # (1,P)
    y_true = _as_torch(sample["y"], device).unsqueeze(0)          # (1,C,H,W)
    m      = _as_torch(sample["mask"], device).unsqueeze(0)       # (1,1,H,W)

    # Get true bottleneck and latent
    b_true, _ = ae.encode(x)
    z_true = bottleneck_to_z(b_true)                              # (1, z_dim)

    # Predict latent from parameters (denormalize if trained on normalized Z)
    z_pred_n = mlp(params)                                        # (1, z_dim)
    z_mu_t  = torch.as_tensor(z_mu,  device=device).view(1, -1)
    z_std_t = torch.as_tensor(z_std, device=device).view(1, -1)
    z_pred  = z_pred_n * z_std_t + z_mu_t

    # Expand z to bottleneck feature map
    B, Cb, Hb, Wb = b_true.shape
    b_pred = z_pred.view(B, Cb, 1, 1).expand(B, Cb, Hb, Wb)

    # Decode to get reconstruction
    y_pred = ae.decode_from_bottleneck(x, b_pred)

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "mask": m,
        "z_true": z_true,
        "z_pred": z_pred,
        "z_pred_n": z_pred_n,
        "params": params,
    }

# ============================================================================
# Main Training Pipeline
# ============================================================================

def run(smoke_test: bool = False):
    """Main training and evaluation pipeline."""
    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    npz_path = "/Users/42d/Downloads/solps_raster_dataset_new.npz"
    device = pick_device()
    print("Device:", device)

    # Training parameters (laptop-safe)
    nepochs = 1 if smoke_test else 10
    nepochs_mlp = 1 if smoke_test else (nepochs * 2)
    num_workers = 0
    batch_size = 1
    split = 0.85  # train fraction used by make_loaders
    amp = False  # Keep off on MPS/CPU
    if smoke_test:
        print("[SMOKE] Enabled: running minimal 1-epoch workflow checks.")

    # Output fields
    y_keys = ["Te"] #, "Ti", "ne", "ua", "Sp", "Qe", "Qi", "Sm"]

    # Field normalizers
    norms_by_name = {
        "Te": MaskedLogStandardizer(eps=1e-2),
        "Ti": MaskedLogStandardizer(eps=1e-2),
        "ne": MaskedLogStandardizer(eps=1e16),
        "ua": MaskedSymLogStandardizer(scale=5e3),
        "Sp": MaskedLogStandardizer(eps=1e10),
        "Qe": MaskedSymLogStandardizer(scale=1e2),
        "Qi": MaskedSymLogStandardizer(scale=1e2),
        "Sm": MaskedSymLogStandardizer(scale=1e-2),
    }

    # -------------------------------------------------------------------------
    # Data Loading
    # -------------------------------------------------------------------------
    train_loader, val_loader, norm, Pdim, (H, W), _ = data.make_loaders(
        npz_path,
        inputs_mode="autoencoder",
        batch_size=batch_size,
        num_workers=num_workers,
        split=split,
        y_keys=y_keys,
        norms_by_name=norms_by_name,
        norm_fit_batch=2,
    )

    # Inspect dataset
    d = np.load(npz_path, allow_pickle=True)
    if "X" in d.files:
        print("X shape:", d["X"].shape)
    elif "params" in d.files:
        print("params shape:", d["params"].shape)
    else:
        print("No parameter matrix found in dataset keys:", d.files)

    if "x_keys" in d.files:
        print("x_keys:", d["x_keys"])
    elif "param_keys" in d.files:
        print("param_keys:", d["param_keys"])

    b0 = next(iter(train_loader))
    C = b0["x"].shape[1]
    print("C:", C, "H,W:", H, W)

    # -------------------------------------------------------------------------
    # Autoencoder Training
    # -------------------------------------------------------------------------
    os.makedirs("outputs", exist_ok=True)
    ae_ckpt = "outputs/ae_smoke.pt" if smoke_test else "outputs/ae.pt"
    mlp_ckpt = "outputs/param2z_smoke.pt" if smoke_test else "outputs/param2z.pt"

    model = UNet(in_ch=C, out_ch=C, base=16).to(device)

    if (not smoke_test) and os.path.exists(ae_ckpt):
        ckpt = torch.load(ae_ckpt, map_location=device)
        
        # Check if architecture matches
        try:
            model.load_state_dict(ckpt["model"], strict=True)
            print(f"[AE] Resumed from {ae_ckpt}")
        except RuntimeError as e:
            print(f"[AE] Checkpoint architecture mismatch: {e}")
            print("[AE] Training from scratch with new architecture...")
            model, _ = train_unet(
                train_loader, val_loader, norm,
                in_ch=C, out_ch=C, device=device,
                inputs_mode="autoencoder",
                epochs=nepochs,
                base=16,
                amp=amp,
                lam_grad=0.0,
                lam_w=0.0,
                multiscale=0,
                grad_accum_steps=1,
                save_path=ae_ckpt,
            )
    else:
        print("[AE] Training from scratch...")
        model, _ = train_unet(
            train_loader, val_loader, norm,
            in_ch=C, out_ch=C, device=device,
            inputs_mode="autoencoder",
            epochs=nepochs,
            base=16,
            amp=amp,
            lam_grad=0.0,
            lam_w=0.0,
            multiscale=0,
            grad_accum_steps=1,
            save_path=ae_ckpt,
        )
    # -------------------------------------------------------------------------
    # Extract Latent Dataset
    # -------------------------------------------------------------------------
    bb = next(iter(torch.utils.data.DataLoader(train_loader.dataset, batch_size=2)))
    print("Batch keys:", bb.keys())
    print("x:", bb["x"].shape, "params:", bb["params"].shape, "idx:", bb["idx"])

    train_ds = train_loader.dataset
    Z, P, idxs = extract_z_dataset(
        ae_model=model,
        dataset=train_ds,
        device=device,
        batch_size=4,
        num_workers=0,
        max_batches=None,
    )

    # -------------------------------------------------------------------------
    # Train Parameter-to-Latent MLP
    # -------------------------------------------------------------------------
    # Train/val split
    N = Z.shape[0]
    cut = max(1, int(0.8 * N))
    P_tr, Z_tr = P[:cut], Z[:cut]
    P_va, Z_va = P[cut:], Z[cut:]

    # Normalize parameters for MLP stability
    p_mu, p_std = fit_p_scaler(P_tr)
    P_tr_n = apply_p_scaler(P_tr, p_mu, p_std)
    P_va_n = apply_p_scaler(P_va, p_mu, p_std)

    # Normalize latents
    z_mu, z_std = fit_z_scaler(Z_tr)
    Z_tr_n = apply_z_scaler(Z_tr, z_mu, z_std)
    Z_va_n = apply_z_scaler(Z_va, z_mu, z_std)
    Z_tr_n = np.clip(Z_tr_n, -5.0, 5.0)
    Z_va_n = np.clip(Z_va_n, -5.0, 5.0)

    # Train MLP
    mlp = ParamToZ(P=P_tr.shape[1], z_dim=Z.shape[1], hidden=(64, 64)).to(device)
    mlp, _ = train_param2z(
        P_train=P_tr_n, Z_train=Z_tr_n,
        P_val=P_va_n, Z_val=Z_va_n,
        model=mlp,
        device=device,
        lr=1e-3,
        epochs=nepochs_mlp,
        batch_size=16,
        save_path=mlp_ckpt,
    )

    print("Saved:", ae_ckpt, mlp_ckpt)

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------
    # Sample from train and val sets
    train_gidx, train_s = sample_from_loader(train_loader, k=0)
    val_gidx, val_s = sample_from_loader(val_loader, k=0)
    print("Train global idx:", train_gidx, "Val global idx:", val_gidx)

    # Test full surrogate
    train_s["p_mu"], train_s["p_std"] = p_mu, p_std
    val_s["p_mu"], val_s["p_std"] = p_mu, p_std
    out_tr = test_full_surrogate(model, mlp, train_s, z_mu, z_std, device)
    out_va = test_full_surrogate(model, mlp, val_s, z_mu, z_std, device)

    print("TRAIN z MSE:", torch.mean((out_tr["z_pred"] - out_tr["z_true"])**2).item())
    print("VAL   z MSE:", torch.mean((out_va["z_pred"] - out_va["z_true"])**2).item())

#    # 1) Forward test: AE reconstruction
#    plot_ae_recon_one(ae=model, norm=norm, sample=train_s, device=device, title="TRAIN")
#    plot_ae_recon_one(ae=model, norm=norm, sample=val_s, device=device, title="VAL")

    # 2) Backward test: params -> z
    out_train = eval_param2z_one(
        ae=model, mlp=mlp, sample=train_s, device=device, label="TRAIN", z_mu=z_mu, z_std=z_std
    )
    print("TRAIN z MSE (eval):", torch.mean((out_train["z_pred"] - out_train["z_true"])**2).item())

    out_val = eval_param2z_one(
        ae=model, mlp=mlp, sample=val_s, device=device, label="VAL", z_mu=z_mu, z_std=z_std
    )
    print("VAL z prediction:", out_val["z_pred"])
    print("VAL z MSE (eval):", torch.mean((out_val["z_pred"] - out_val["z_true"])**2).item())

    # 3) Workflow checks: forward, inverse, cycle
    with torch.no_grad():
        x_tr = _as_torch(train_s["x"], device).unsqueeze(0)
        y_tr = _as_torch(train_s["y"], device).unsqueeze(0)
        m_tr = _as_torch(train_s["mask"], device).unsqueeze(0)
        yhat_tr = model(x_tr)
        fwd_mse_tr = masked_mse(yhat_tr, y_tr, m_tr)

        x_va = _as_torch(val_s["x"], device).unsqueeze(0)
        y_va = _as_torch(val_s["y"], device).unsqueeze(0)
        m_va = _as_torch(val_s["mask"], device).unsqueeze(0)
        yhat_va = model(x_va)
        fwd_mse_va = masked_mse(yhat_va, y_va, m_va)

    inv_mse_tr = float(torch.mean((out_train["z_pred"] - out_train["z_true"])**2).item())
    inv_mse_va = float(torch.mean((out_val["z_pred"] - out_val["z_true"])**2).item())

    cyc_mse_tr = masked_mse(out_tr["y_pred"], out_tr["y_true"], out_tr["mask"])
    cyc_mse_va = masked_mse(out_va["y_pred"], out_va["y_true"], out_va["mask"])

    z_cons_tr = float(torch.mean((out_tr["z_pred"].detach().cpu() - out_train["z_pred"])**2).item())
    z_cons_va = float(torch.mean((out_va["z_pred"].detach().cpu() - out_val["z_pred"])**2).item())

    print(
        f"[CHECK] forward_mse(train/val)=({fwd_mse_tr:.4e}, {fwd_mse_va:.4e}) "
        f"inverse_z_mse(train/val)=({inv_mse_tr:.4e}, {inv_mse_va:.4e}) "
        f"cycle_mse(train/val)=({cyc_mse_tr:.4e}, {cyc_mse_va:.4e}) "
        f"z_consistency(train/val)=({z_cons_tr:.4e}, {z_cons_va:.4e})"
    )

    for name, val in [
        ("fwd_mse_tr", fwd_mse_tr), ("fwd_mse_va", fwd_mse_va),
        ("inv_mse_tr", inv_mse_tr), ("inv_mse_va", inv_mse_va),
        ("cyc_mse_tr", cyc_mse_tr), ("cyc_mse_va", cyc_mse_va),
        ("z_cons_tr", z_cons_tr), ("z_cons_va", z_cons_va),
    ]:
        assert_finite(name, val)

    if smoke_test:
        limits = {
            "fwd_mse_max": 1e2,
            "inv_mse_max": 1e2,
            "cyc_mse_max": 1e2,
            "z_cons_max": 1e-4,
        }
        if fwd_mse_tr > limits["fwd_mse_max"] or fwd_mse_va > limits["fwd_mse_max"]:
            raise RuntimeError(f"[SMOKE] Forward check failed: {fwd_mse_tr=:.3e}, {fwd_mse_va=:.3e}")
        if inv_mse_tr > limits["inv_mse_max"] or inv_mse_va > limits["inv_mse_max"]:
            raise RuntimeError(f"[SMOKE] Inverse check failed: {inv_mse_tr=:.3e}, {inv_mse_va=:.3e}")
        if cyc_mse_tr > limits["cyc_mse_max"] or cyc_mse_va > limits["cyc_mse_max"]:
            raise RuntimeError(f"[SMOKE] Cycle check failed: {cyc_mse_tr=:.3e}, {cyc_mse_va=:.3e}")
        if z_cons_tr > limits["z_cons_max"] or z_cons_va > limits["z_cons_max"]:
            raise RuntimeError(f"[SMOKE] z-consistency failed: {z_cons_tr=:.3e}, {z_cons_va=:.3e}")
        print("[SMOKE] PASS: forward/inverse/cycle checks are finite and within limits.")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    smoke = os.environ.get("SMOKE_TEST", "0") == "1"
    run(smoke_test=smoke)

# Usage: srun -n 1 -N 1 -p pbatch -A lbpm --time=3:00:00 --pty /bin/sh
