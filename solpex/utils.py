# Copyright 2025-2026 Oak Ridge National Laboratory
# @authors: Abdourahmane (Abdou) Diaw - diawa@ornl.gov
#
# SPDX-License-Identifier: MIT

import h5py, time, numpy as np
import torch
from .models import UNet, bottleneck_to_z, ParamToZ


def sample_from_loader(loader, k=0):
    base_ds = loader.dataset.dataset      # underlying SOLPSDataset
    gidx = loader.dataset.indices[k]      # global index into base_ds
    sample = base_ds[gidx]                # dict with x,y,mask,params,idx
    return gidx, sample


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def scale_params_for_inference(params, param_mu, param_std):
    params = np.asarray(params, dtype=np.float32)
    if param_mu is not None and param_std is not None:
        params = (params - np.asarray(param_mu, dtype=np.float32)) / np.asarray(param_std, dtype=np.float32)
    return params

def save_geometry_h5(path, r2d, z2d, case_name=None, units="m", level=4):
    r = np.ascontiguousarray(np.asarray(r2d, dtype=np.float32))
    z = np.ascontiguousarray(np.asarray(z2d, dtype=np.float32))
    H, W = r.shape
    chunk = (min(512, H), min(512, W))
    with h5py.File(path, "w") as f:
        for name, arr in [("R2D", r), ("Z2D", z)]:
            d = f.create_dataset(name, data=arr, compression="gzip",
                                 compression_opts=int(level), shuffle=True, chunks=chunk)
            d.attrs["units"] = units
            d.attrs["grid"] = "cell centers"
        f.attrs["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
        if case_name is not None: f.attrs["case_name"] = str(case_name)

def nearest_neighbor_in_Z(z_pred, Z_ref):
    # z_pred: (z_dim,), Z_ref: (N,z_dim)
    d2 = np.sum((Z_ref - z_pred[None,:])**2, axis=1)
    return int(np.argmin(d2)), float(d2.min())

@torch.no_grad()
def eval_param2z_one(
    *, ae, mlp, sample, device, label="", z_mu=None, z_std=None, p_mu=None, p_std=None
):
    """
    Returns dict with z_true, z_pred (both torch tensors on CPU), plus cos similarity + norms.
    If z_mu/z_std provided, assumes mlp outputs normalized z and will unnormalize here.
    """
    ae.eval()
    mlp.eval()

    x = sample["x"].unsqueeze(0).to(device)
    p_raw = sample["params"].detach().cpu().numpy().astype(np.float32, copy=False)
    if (p_mu is None or p_std is None) and ("p_mu" in sample and "p_std" in sample):
        p_mu = sample["p_mu"]
        p_std = sample["p_std"]
    if (p_mu is not None) and (p_std is not None):
        p_raw = (p_raw - np.asarray(p_mu, dtype=np.float32)) / np.asarray(p_std, dtype=np.float32)
    p = torch.from_numpy(p_raw).unsqueeze(0).to(device)

    # true latent
    _, b = ae(x, return_bottleneck=True)
    z_true = bottleneck_to_z(b)  # (1, zdim)

    # predicted latent (maybe normalized)
    z_pred = mlp(p)
    if (z_mu is not None) and (z_std is not None):
        mu = torch.as_tensor(z_mu, device=device).view(1, -1)
        sd = torch.as_tensor(z_std, device=device).view(1, -1)
        z_pred = z_pred * sd + mu

    # metrics
    eps = 1e-12
    a = z_true / (z_true.norm(dim=1, keepdim=True) + eps)
    b = z_pred / (z_pred.norm(dim=1, keepdim=True) + eps)
    cos = (a * b).sum(dim=1).item()

    out = {
        "z_true": z_true.detach().cpu(),
        "z_pred": z_pred.detach().cpu(),
        "cos": float(cos),
        "z_true_norm": float(z_true.norm().item()),
        "z_pred_norm": float(z_pred.norm().item()),
    }
    if label:
        print(f"[{label}] cos={out['cos']:.4f} |z_true|={out['z_true_norm']:.4g} |z_pred|={out['z_pred_norm']:.4g}")
    return out
