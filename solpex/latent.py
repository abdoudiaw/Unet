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
    def __init__(self, P, latent_dim, hidden=(256,256), dropout=0.0):
        super().__init__()
        layers = []
        d = P
        for h in hidden:
            layers += [nn.Linear(d, h), nn.SiLU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, latent_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, p):
        return self.net(p)
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

import numpy as np
import torch
from torch.utils.data import DataLoader
from .models import bottleneck_to_z

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

        # model must support return_bottleneck=True
        yhat, b = ae_model(x, return_bottleneck=True)  # b: (B,C,h,w)

        z_t = bottleneck_to_z(b)  # (B, z_dim) torch on device
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
