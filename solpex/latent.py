# Copyright 2025-2026 Oak Ridge National Laboratory
# @authors: Abdourahmane (Abdou) Diaw - diawa@ornl.gov
#
# SPDX-License-Identifier: MIT

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .models import bottleneck_to_z

from torch import nn

class ParamToZ(nn.Module):
    def __init__(self, P, latent_dim, hidden=(256,256), dropout=0.0, use_layernorm=False):
        super().__init__()
        layers = []
        d = P
        for h in hidden:
            layers += [nn.Linear(d, h), nn.SiLU()]
            if use_layernorm:
                layers += [nn.LayerNorm(h)]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, latent_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, p):
        return self.net(p)
class ZToParam(nn.Module):
    """Amortized inverse: maps latent z -> control parameters."""
    def __init__(self, z_dim, P, hidden=(256, 256), dropout=0.0, use_layernorm=False):
        super().__init__()
        layers = []
        d = z_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.SiLU()]
            if use_layernorm:
                layers += [nn.LayerNorm(h)]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, P)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


def train_param2z(
    *,
    P_train, Z_train,
    P_val, Z_val,
    model,
    device,
    lr=1e-3,
    epochs=200,
    batch_size=256,
    wd=1e-4,
    num_workers=0,
    save_path=None,
):
    device = torch.device(device)
    model = model.to(device)

    tr_ds = TensorDataset(torch.from_numpy(P_train), torch.from_numpy(Z_train))
    va_ds = TensorDataset(torch.from_numpy(P_val), torch.from_numpy(Z_val))

    tr = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    va = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = torch.nn.SmoothL1Loss(beta=0.05)  # robust

    best = float("inf")
    hist = {"tr": [], "va": []}

    for ep in range(epochs):
        model.train()
        tr_sum = 0.0
        n = 0
        for p, z in tr:
            p = p.to(device)
            z = z.to(device)
            pred = model(p)
            loss = loss_fn(pred, z)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tr_sum += float(loss.item()) * p.shape[0]
            n += p.shape[0]
        tr_loss = tr_sum / max(n, 1)

        model.eval()
        va_sum = 0.0
        n = 0
        with torch.no_grad():
            for p, z in va:
                p = p.to(device)
                z = z.to(device)
                pred = model(p)
                loss = loss_fn(pred, z)
                va_sum += float(loss.item()) * p.shape[0]
                n += p.shape[0]
        va_loss = va_sum / max(n, 1)

        hist["tr"].append(tr_loss)
        hist["va"].append(va_loss)

        if (ep % 10) == 0 or ep == epochs - 1:
            print(f"[param2z] ep {ep:04d} tr {tr_loss:.5f} va {va_loss:.5f}")

        if va_loss < best:
            best = va_loss
            if save_path:
                torch.save({"model": model.state_dict(), "best": best}, save_path)

    return model, hist


def train_z2param(
    *,
    Z_train, P_train,
    Z_val, P_val,
    model,
    device,
    lr=1e-3,
    epochs=200,
    batch_size=256,
    wd=1e-4,
    num_workers=0,
    save_path=None,
):
    """Train ZToParam inverse MLP: latent z -> scaled params."""
    device = torch.device(device)
    model = model.to(device)

    tr_ds = TensorDataset(torch.from_numpy(Z_train), torch.from_numpy(P_train))
    va_ds = TensorDataset(torch.from_numpy(Z_val), torch.from_numpy(P_val))

    tr = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    va = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = torch.nn.SmoothL1Loss(beta=0.05)

    best = float("inf")
    hist = {"tr": [], "va": []}

    for ep in range(epochs):
        model.train()
        tr_sum = 0.0
        n = 0
        for z, p in tr:
            z = z.to(device)
            p = p.to(device)
            pred = model(z)
            loss = loss_fn(pred, p)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_sum += float(loss.item()) * z.shape[0]
            n += z.shape[0]
        tr_loss = tr_sum / max(n, 1)

        model.eval()
        va_sum = 0.0
        n = 0
        with torch.no_grad():
            for z, p in va:
                z = z.to(device)
                p = p.to(device)
                pred = model(z)
                loss = loss_fn(pred, p)
                va_sum += float(loss.item()) * z.shape[0]
                n += z.shape[0]
        va_loss = va_sum / max(n, 1)

        hist["tr"].append(tr_loss)
        hist["va"].append(va_loss)

        if (ep % 10) == 0 or ep == epochs - 1:
            print(f"[z2param] ep {ep:04d} tr {tr_loss:.5f} va {va_loss:.5f}")

        if va_loss < best:
            best = va_loss
            if save_path:
                torch.save({
                    "model": model.state_dict(),
                    "best": best,
                    "z_dim": Z_train.shape[1],
                    "P": P_train.shape[1],
                }, save_path)

    return model, hist


def train_cycle_consistent(
    *,
    Z_train, P_train,
    Z_val, P_val,
    p2z_model, z2p_model,
    device,
    lr=1e-3,
    epochs=400,
    batch_size=64,
    wd=1e-4,
    lam_cycle=0.1,
    num_workers=0,
    save_path=None,
):
    """Joint training of ParamToZ and ZToParam with cycle consistency losses."""
    device = torch.device(device)
    p2z_model = p2z_model.to(device)
    z2p_model = z2p_model.to(device)

    tr_ds = TensorDataset(torch.from_numpy(Z_train), torch.from_numpy(P_train))
    va_ds = TensorDataset(torch.from_numpy(Z_val), torch.from_numpy(P_val))

    tr = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    va = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    all_params = list(p2z_model.parameters()) + list(z2p_model.parameters())
    opt = torch.optim.AdamW(all_params, lr=lr, weight_decay=wd)
    loss_fn = torch.nn.SmoothL1Loss(beta=0.05)

    best = float("inf")
    hist = {"tr": [], "va": [], "tr_fwd": [], "tr_inv": [], "tr_cyc_pzp": [], "tr_cyc_zpz": []}

    for ep in range(epochs):
        p2z_model.train()
        z2p_model.train()
        sums = {"total": 0.0, "fwd": 0.0, "inv": 0.0, "cyc_pzp": 0.0, "cyc_zpz": 0.0}
        n = 0
        for z, p in tr:
            z = z.to(device)
            p = p.to(device)

            # Forward losses
            L_fwd = loss_fn(z2p_model(z), p)
            L_inv = loss_fn(p2z_model(p), z)

            # Cycle consistency losses
            L_cyc_pzp = loss_fn(z2p_model(p2z_model(p)), p)
            L_cyc_zpz = loss_fn(p2z_model(z2p_model(z)), z)

            loss = L_fwd + L_inv + lam_cycle * (L_cyc_pzp + L_cyc_zpz)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            opt.step()

            bs = z.shape[0]
            sums["total"] += float(loss.item()) * bs
            sums["fwd"] += float(L_fwd.item()) * bs
            sums["inv"] += float(L_inv.item()) * bs
            sums["cyc_pzp"] += float(L_cyc_pzp.item()) * bs
            sums["cyc_zpz"] += float(L_cyc_zpz.item()) * bs
            n += bs

        tr_loss = sums["total"] / max(n, 1)
        hist["tr"].append(tr_loss)
        hist["tr_fwd"].append(sums["fwd"] / max(n, 1))
        hist["tr_inv"].append(sums["inv"] / max(n, 1))
        hist["tr_cyc_pzp"].append(sums["cyc_pzp"] / max(n, 1))
        hist["tr_cyc_zpz"].append(sums["cyc_zpz"] / max(n, 1))

        # Validate
        p2z_model.eval()
        z2p_model.eval()
        va_sum = 0.0
        nv = 0
        with torch.no_grad():
            for z, p in va:
                z = z.to(device)
                p = p.to(device)
                L_fwd = loss_fn(z2p_model(z), p)
                L_inv = loss_fn(p2z_model(p), z)
                L_cyc_pzp = loss_fn(z2p_model(p2z_model(p)), p)
                L_cyc_zpz = loss_fn(p2z_model(z2p_model(z)), z)
                loss = L_fwd + L_inv + lam_cycle * (L_cyc_pzp + L_cyc_zpz)
                va_sum += float(loss.item()) * z.shape[0]
                nv += z.shape[0]
        va_loss = va_sum / max(nv, 1)
        hist["va"].append(va_loss)

        if (ep % 10) == 0 or ep == epochs - 1:
            print(
                f"[cycle] ep {ep:04d} tr {tr_loss:.5f} va {va_loss:.5f} "
                f"fwd {hist['tr_fwd'][-1]:.5f} inv {hist['tr_inv'][-1]:.5f} "
                f"cyc_pzp {hist['tr_cyc_pzp'][-1]:.5f} cyc_zpz {hist['tr_cyc_zpz'][-1]:.5f}"
            )

        if va_loss < best:
            best = va_loss
            if save_path:
                torch.save({
                    "z2p_model": z2p_model.state_dict(),
                    "p2z_model": p2z_model.state_dict(),
                    "best": best,
                    "z_dim": Z_train.shape[1],
                    "P": P_train.shape[1],
                }, save_path)

    return p2z_model, z2p_model, hist


@torch.no_grad()
def extract_z_dataset(
    *,
    ae_model,
    dataset,
    device,
    batch_size=64,
    num_workers=0,
    max_batches=None,
):
    """
    Returns:
      Z: (N, z_dim) np.float32
      P: (N, P)     np.float32
      idxs: (N,)    np.int64
    """
    device = torch.device(device)
    ae_model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    Zs, Ps, Is = [], [], []
    seen = 0

    for bi, batch in enumerate(loader):
        x = batch["x"].to(device)
        params = batch["params"].to(device) if "params" in batch else None

        # model must support return_bottleneck=True
        yhat, b = ae_model(x, params=params, return_bottleneck=True)  # b: (B,C,h,w)

        z_t = bottleneck_to_z(b)  # (B, base*8) torch on device
        if hasattr(ae_model, 'z_proj') and ae_model.z_proj is not None:
            z_t = ae_model.z_proj(z_t)
        z = z_t.detach().float().cpu().numpy()  # ALWAYS numpy on CPU
        Zs.append(z)

        # params: prefer batch params if provided
        if "params" in batch:
            p = batch["params"].detach().float().cpu().numpy()
        elif "idx" in batch and hasattr(dataset, "params"):
            ii = batch["idx"]
            if torch.is_tensor(ii):
                ii = ii.detach().cpu().numpy()
            p = dataset.params[np.asarray(ii, dtype=int)].astype(np.float32, copy=False)
        else:
            raise KeyError("Need params for param→z: provide batch['params'] or batch['idx'] with dataset.params.")
        Ps.append(p)

        bs = z.shape[0]
        Is.append(np.arange(seen, seen + bs, dtype=np.int64))
        seen += bs

        if max_batches is not None and (bi + 1) >= max_batches:
            break

    Z = np.concatenate(Zs, axis=0).astype(np.float32, copy=False)
    P = np.concatenate(Ps, axis=0).astype(np.float32, copy=False)
    idxs = np.concatenate(Is, axis=0).astype(np.int64, copy=False)
    return Z, P, idxs
