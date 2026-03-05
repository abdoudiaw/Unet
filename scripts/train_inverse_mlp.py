# Copyright 2025-2026 Oak Ridge National Laboratory
# @authors: Abdourahmane (Abdou) Diaw - diawa@ornl.gov
#
# SPDX-License-Identifier: MIT

"""
Train an amortized inverse MLP (z -> params) from a trained forward UNet.

Steps:
  1. Load the trained forward UNet checkpoint.
  2. Pass all training samples through the encoder to extract bottleneck z.
  3. Train a ZToParam MLP mapping z -> scaled_params.
  4. Save the inverse checkpoint.

Usage:
  python train_inverse_mlp.py \
      --npz data/solps_native_all_qc.npz \
      --ckpt outputs/cond_unet.pt \
      --out  outputs/inverse_mlp.pt \
      --epochs 400 --hidden 128,128
"""

import argparse
import os

import numpy as np
import torch

from solpex.predict import load_checkpoint
from solpex.data import apply_param_transform
from solpex.models import bottleneck_to_z
from solpex.latent import ZToParam, ParamToZ, train_z2param, train_cycle_consistent
from solpex.utils import pick_device


def split_indices(N, split=0.85, seed=42):
    idx = np.arange(N)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    cut = int(split * N)
    return idx[:cut], idx[cut:]


@torch.no_grad()
def extract_z_from_forward(model, Y, M, P_raw, p_mu, p_std, device, batch_size=16):
    """Run the forward UNet encoder on all samples, return Z and P_scaled."""
    model.eval()
    N = Y.shape[0]
    H, W = Y.shape[2], Y.shape[3]
    Zs = []

    for i in range(0, N, batch_size):
        j = min(i + batch_size, N)

        # Build input x: mask only (params passed via FiLM)
        mask_b = torch.from_numpy(M[i:j]).float().unsqueeze(1).to(device)  # (B,1,H,W)
        p_scaled = (P_raw[i:j] - p_mu) / p_std
        p_t = torch.from_numpy(p_scaled).float().to(device)  # (B,P)

        b, _ = model.encode(mask_b, params=p_t)  # bottleneck
        z = bottleneck_to_z(b)  # (B, base*8)
        if hasattr(model, 'z_proj') and model.z_proj is not None:
            z = model.z_proj(z)
        Zs.append(z.cpu().numpy())

    Z = np.concatenate(Zs, axis=0).astype(np.float32)
    P_scaled = ((P_raw - p_mu) / p_std).astype(np.float32)
    return Z, P_scaled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--ckpt", required=True, help="Forward UNet checkpoint")
    ap.add_argument("--out", default="outputs/inverse_mlp.pt")
    ap.add_argument("--split", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=str, default="128,128",
                    help="Comma-separated hidden layer sizes")
    ap.add_argument("--cycle", action="store_true",
                    help="Enable cycle consistency training (joint p2z + z2p)")
    ap.add_argument("--lam-cycle", type=float, default=0.1,
                    help="Weight for cycle consistency losses")
    ap.add_argument("--use-layernorm", action="store_true",
                    help="Add LayerNorm to MLP hidden layers")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    device = pick_device()
    print("Device:", device)

    # Load forward model
    model, norm, (p_mu, p_std), ckpt_param_transform, ckpt_param_keys = load_checkpoint(args.ckpt, device)
    if p_mu is None or p_std is None:
        raise RuntimeError("Checkpoint missing param_mu / param_std.")
    p_mu = np.asarray(p_mu, dtype=np.float32)
    p_std = np.asarray(p_std, dtype=np.float32)
    print(f"param_transform={ckpt_param_transform}, param_keys={ckpt_param_keys}")
    print(f"Forward model loaded: in_ch={model.enc1.net[0].in_channels}")

    # Load dataset
    d = np.load(args.npz, allow_pickle=True)
    if "Y" in d.files:
        Y = d["Y"].astype(np.float32)
        y_keys = [str(k) for k in d["y_keys"]]
    elif "Te" in d.files:
        Y = d["Te"][:, None].astype(np.float32)
        y_keys = ["Te"]
    else:
        raise KeyError("Dataset must contain Y or Te.")

    if "mask" in d.files:
        M = d["mask"]
        if M.ndim == 2:
            M = np.repeat(M[None], Y.shape[0], axis=0)
        M = (M > 0.5).astype(np.float32)
    else:
        M = np.ones((Y.shape[0], Y.shape[2], Y.shape[3]), dtype=np.float32)

    P_raw = d["params"].astype(np.float32) if "params" in d.files else d["X"].astype(np.float32)
    p_keys = ([str(k) for k in d["param_keys"]] if "param_keys" in d.files
              else [str(k) for k in d["x_keys"]] if "x_keys" in d.files
              else [f"p{i}" for i in range(P_raw.shape[1])])

    # Apply same param transform as forward model
    if ckpt_param_transform and ckpt_param_transform != "none":
        P_raw, p_keys = apply_param_transform(P_raw, p_keys, ckpt_param_transform)

    N = Y.shape[0]
    print(f"Dataset: N={N}, Y={Y.shape}, P={P_raw.shape}, fields={y_keys}, p_keys={p_keys}")

    # Extract latent Z from forward model encoder
    print("Extracting bottleneck latents...")
    Z, P_scaled = extract_z_from_forward(model, Y, M, P_raw, p_mu, p_std, device)
    z_dim = Z.shape[1]
    P_dim = P_scaled.shape[1]
    print(f"Z shape: {Z.shape} (z_dim={z_dim}), P_scaled shape: {P_scaled.shape}")

    # Train/val split
    tr_idx, va_idx = split_indices(N, split=args.split, seed=args.seed)
    Z_tr, Z_va = Z[tr_idx], Z[va_idx]
    P_tr, P_va = P_scaled[tr_idx], P_scaled[va_idx]

    # Normalize Z for training stability
    z_mu = Z_tr.mean(axis=0).astype(np.float32)
    z_std = Z_tr.std(axis=0).astype(np.float32)
    z_std[z_std < 1e-8] = 1.0
    Z_tr_n = np.clip((Z_tr - z_mu) / z_std, -5.0, 5.0)
    Z_va_n = np.clip((Z_va - z_mu) / z_std, -5.0, 5.0)

    # Build and train inverse MLP(s)
    hidden = tuple(int(x) for x in args.hidden.split(","))
    use_ln = args.use_layernorm

    if args.cycle:
        print(f"Training cycle-consistent: z_dim={z_dim} <-> P={P_dim}, "
              f"hidden={hidden}, lam_cycle={args.lam_cycle}, layernorm={use_ln}")
        z2p = ZToParam(z_dim=z_dim, P=P_dim, hidden=hidden, use_layernorm=use_ln).to(device)
        p2z = ParamToZ(P=P_dim, latent_dim=z_dim, hidden=hidden, use_layernorm=use_ln).to(device)

        p2z, z2p, hist = train_cycle_consistent(
            Z_train=Z_tr_n, P_train=P_tr,
            Z_val=Z_va_n, P_val=P_va,
            p2z_model=p2z, z2p_model=z2p,
            device=device,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lam_cycle=args.lam_cycle,
            save_path=args.out,
        )

        # Save extra metadata into checkpoint
        ckpt = torch.load(args.out, map_location="cpu", weights_only=False)
        ckpt.update({
            "z_mu": z_mu, "z_std": z_std,
            "p_mu": p_mu, "p_std": p_std,
            "p_keys": p_keys,
            "hidden": list(hidden),
            "forward_ckpt": args.ckpt,
            "use_layernorm": use_ln,
            "cycle": True,
            "lam_cycle": args.lam_cycle,
            "param_transform": ckpt_param_transform,
        })
        torch.save(ckpt, args.out)
        print(f"Saved cycle-consistent inverse checkpoint to {args.out}")
        inv_mlp = z2p  # for eval below
    else:
        print(f"Training ZToParam: z_dim={z_dim} -> hidden={hidden} -> P={P_dim}, layernorm={use_ln}")
        inv_mlp = ZToParam(z_dim=z_dim, P=P_dim, hidden=hidden, use_layernorm=use_ln).to(device)

        inv_mlp, hist = train_z2param(
            Z_train=Z_tr_n, P_train=P_tr,
            Z_val=Z_va_n, P_val=P_va,
            model=inv_mlp,
            device=device,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            save_path=args.out,
        )

        # Save extra metadata into checkpoint
        ckpt = torch.load(args.out, map_location="cpu", weights_only=False)
        ckpt.update({
            "z_mu": z_mu, "z_std": z_std,
            "p_mu": p_mu, "p_std": p_std,
            "p_keys": p_keys,
            "hidden": list(hidden),
            "forward_ckpt": args.ckpt,
            "use_layernorm": use_ln,
            "param_transform": ckpt_param_transform,
        })
        torch.save(ckpt, args.out)
        print(f"Saved inverse MLP to {args.out}")

    # Quick eval
    inv_mlp.eval()
    with torch.no_grad():
        z_va_t = torch.from_numpy(Z_va_n).float().to(device)
        p_pred = inv_mlp(z_va_t).cpu().numpy()
    mae = np.mean(np.abs(p_pred - P_va), axis=0)
    for i, k in enumerate(p_keys):
        print(f"  {k}: MAE(scaled) = {mae[i]:.4f}")
    print(f"  mean MAE(scaled) = {mae.mean():.4f}")


if __name__ == "__main__":
    main()
