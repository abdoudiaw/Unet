# Copyright 2025-2026 Oak Ridge National Laboratory
# @authors: Abdourahmane (Abdou) Diaw - diawa@ornl.gov
#
# SPDX-License-Identifier: MIT

# losses.py
import torch
import torch.nn.functional as F
import numpy as np 

# ------------------------
# Sobel cache (per device/dtype/C)
# ------------------------
_SOBEL_CACHE = {}  # key -> (kx, ky)

def _sobel_kernels_cached(dtype, device, channels: int):
    """
    Returns kx, ky shaped (C,1,3,3) for grouped conv (groups=C).
    Cached to avoid re-allocating every batch.
    """
    dev = device if isinstance(device, torch.device) else torch.device(device)
    key = (dev.type, dev.index, str(dtype), int(channels))
    hit = _SOBEL_CACHE.get(key, None)
    if hit is not None:
        return hit

    kx1 = torch.tensor(
        [[[-1., 0., 1.],
          [-2., 0., 2.],
          [-1., 0., 1.]]],
        dtype=dtype, device=dev
    ).unsqueeze(0)  # (1,1,3,3)

    ky1 = torch.tensor(
        [[[-1., -2., -1.],
          [ 0.,  0.,  0.],
          [ 1.,  2.,  1.]]],
        dtype=dtype, device=dev
    ).unsqueeze(0)  # (1,1,3,3)

    kx = kx1.repeat(channels, 1, 1, 1)  # (C,1,3,3)
    ky = ky1.repeat(channels, 1, 1, 1)  # (C,1,3,3)

    _SOBEL_CACHE[key] = (kx, ky)
    return kx, ky


def _grad_xy_grouped(x, kx, ky):
    """
    x: (B,C,H,W)
    kx,ky: (C,1,3,3)
    """
    C = x.shape[1]
    # grouped conv: weight must be (C,1,kh,kw) and groups=C
    gx = F.conv2d(x, kx, padding=1, groups=C)
    gy = F.conv2d(x, ky, padding=1, groups=C)
    return gx, gy


def _masked_mean(px, mask, w=None, channel_weights=None):
    # px: (B,C,H,W) or (B,1,H,W)
    # mask: (B,1,H,W)
    if mask.shape[1] == 1 and px.shape[1] != 1:
        mask = mask.expand(px.shape[0], px.shape[1], px.shape[2], px.shape[3])
    wf = torch.ones_like(px)
    if w is not None:
        if w.ndim == 2:
            w = w[None, None, :, :]
        if w.shape[1] == 1 and px.shape[1] != 1:
            w = w.expand(px.shape[0], px.shape[1], px.shape[2], px.shape[3])
        wf = wf * w
    if channel_weights is not None:
        cw = channel_weights.view(1, -1, 1, 1).to(px.device, px.dtype)
        if cw.shape[1] == 1 and px.shape[1] != 1:
            cw = cw.expand(px.shape[0], px.shape[1], px.shape[2], px.shape[3])
        wf = wf * cw
    num = (px * mask * wf).sum()
    den = (mask * wf).sum().clamp_min(1e-8)
    return num / den


def masked_weighted_loss(
    pred, target, mask, *,
    w=None,
    lam_grad=0.2,
    lam_w=1.0,
    base="huber",          # "huber" | "mse" | "l1"
    huber_beta=0.05,
    grad_mode="vector",    # "vector" | "magnitude"
    grad_base="l1",        # "l1" | "huber" | "mse"
    multiscale=0,
    ms_weight=0.5,

    # NEW: compute gradient loss on downsampled fields for speed
    grad_ds: int = 1,      # 1=full res, 2=half, 4=quarter, ...
    channel_weights=None,  # optional tensor-like of shape (C,)
):
    """
    pred/target: (B,C,H,W) normalized
    mask:        (B,1,H,W) 0/1
    w:           optional weights (1,1,H,W) or broadcastable
    """

    # ----- base pixel loss -----
    if base == "huber":
        base_px = F.smooth_l1_loss(pred, target, beta=huber_beta, reduction="none")
    elif base == "l1":
        base_px = (pred - target).abs()
    elif base == "mse":
        base_px = (pred - target) ** 2
    else:
        raise ValueError(f"Unknown base={base}")

    L_base = _masked_mean(base_px, mask, w=None, channel_weights=channel_weights)
    L_edge = _masked_mean(base_px, mask, w=w, channel_weights=channel_weights) if (w is not None and lam_w != 0.0) \
             else torch.zeros((), device=pred.device, dtype=pred.dtype)

    # ----- gradient loss (optional) -----
    if lam_grad != 0.0:
        # downsample for grad loss if requested
        if grad_ds is None or grad_ds < 1:
            grad_ds = 1

        if grad_ds > 1:
            # avg pooling keeps things stable; also downsample mask
            p_g = F.avg_pool2d(pred, kernel_size=grad_ds, stride=grad_ds)
            t_g = F.avg_pool2d(target, kernel_size=grad_ds, stride=grad_ds)
            m_g = F.avg_pool2d(mask, kernel_size=grad_ds, stride=grad_ds)
            m_g = (m_g > 0.5).to(pred.dtype)
        else:
            p_g, t_g, m_g = pred, target, mask

        C = p_g.shape[1]
        kx, ky = _sobel_kernels_cached(p_g.dtype, p_g.device, channels=C)
        pgx, pgy = _grad_xy_grouped(p_g, kx, ky)
        tgx, tgy = _grad_xy_grouped(t_g, kx, ky)

        if grad_mode == "magnitude":
            p = torch.sqrt(pgx*pgx + pgy*pgy + 1e-12)
            t = torch.sqrt(tgx*tgx + tgy*tgy + 1e-12)

            if grad_base == "l1":
                grad_px = (p - t).abs()
            elif grad_base == "mse":
                grad_px = (p - t) ** 2
            elif grad_base == "huber":
                grad_px = F.smooth_l1_loss(p, t, beta=huber_beta, reduction="none")
            else:
                raise ValueError(f"Unknown grad_base={grad_base}")

        else:  # "vector"
            dx = pgx - tgx
            dy = pgy - tgy

            if grad_base == "l1":
                grad_px = dx.abs() + dy.abs()
            elif grad_base == "mse":
                grad_px = dx*dx + dy*dy
            elif grad_base == "huber":
                grad_px = (
                    F.smooth_l1_loss(pgx, tgx, beta=huber_beta, reduction="none")
                    + F.smooth_l1_loss(pgy, tgy, beta=huber_beta, reduction="none")
                )
            else:
                raise ValueError(f"Unknown grad_base={grad_base}")

        L_grad = _masked_mean(grad_px, m_g, w=None, channel_weights=channel_weights)

    else:
        L_grad = torch.zeros((), device=pred.device, dtype=pred.dtype)

    loss = L_base + lam_w * L_edge + lam_grad * L_grad

    # ----- multiscale (optional) -----
    if multiscale and multiscale > 0:
        def ds(t): return F.avg_pool2d(t, kernel_size=2, stride=2)
        p2, t2, m2 = ds(pred), ds(target), ds(mask)
        w2 = ds(w) if w is not None else None

        loss2 = masked_weighted_loss(
            p2, t2, m2,
            w=w2,
            lam_grad=lam_grad,
            lam_w=lam_w,
            base=base,
            huber_beta=huber_beta,
            grad_mode=grad_mode,
            grad_base=grad_base,
            multiscale=0,
            grad_ds=max(1, int(grad_ds // 2)),  # keep grad downsampling consistent
            channel_weights=channel_weights,
        )
        loss = loss + ms_weight * loss2

        if multiscale >= 2:
            p4, t4, m4 = ds(p2), ds(t2), ds(m2)
            w4 = ds(w2) if w2 is not None else None
            loss4 = masked_weighted_loss(
                p4, t4, m4,
                w=w4,
                lam_grad=lam_grad,
                lam_w=lam_w,
                base=base,
                huber_beta=huber_beta,
                grad_mode=grad_mode,
                grad_base=grad_base,
                multiscale=0,
                grad_ds=max(1, int(grad_ds // 4)),
                channel_weights=channel_weights,
            )
            loss = loss + ms_weight * 0.5 * loss4

    return loss

def edge_weights(mask_np: np.ndarray, sigma_px: float = 3.0) -> np.ndarray:
    from scipy.ndimage import binary_erosion, distance_transform_edt
    m = mask_np.astype(bool)
    if not m.any():
        return np.zeros_like(mask_np, dtype=np.float32)
    inner    = binary_erosion(m, iterations=1, border_value=0)
    boundary = m & (~inner)
    d        = distance_transform_edt(~boundary)
    w        = np.exp(-(d**2) / (2.0 * sigma_px**2))
    return (w * m).astype(np.float32)

def mae_norm(pred, target, mask):
    return (torch.abs(pred - target) * mask).sum() / mask.sum().clamp_min(1e-8)

def batch_error_sums_ev(pred, target, mask, norm):
    """
    pred/target: (B,C,H,W) normalized
    mask:        (B,1,H,W)
    Returns: abs_sum, sq_sum, px  (all scalars)
    """
    with torch.no_grad():
        p_ev = norm.inverse(pred, mask)    # (B,C,H,W) physical
        y_ev = norm.inverse(target, mask)

        # kill any NaN/Inf that sneaks in (symlog inverse can blow up if z huge)
        p_ev = torch.nan_to_num(p_ev, nan=0.0, posinf=0.0, neginf=0.0)
        y_ev = torch.nan_to_num(y_ev, nan=0.0, posinf=0.0, neginf=0.0)

        # expand mask to channels
        if mask.shape[1] == 1 and p_ev.shape[1] != 1:
            m = mask.expand_as(p_ev)
        else:
            m = mask

        diff = (p_ev - y_ev) * m
        abs_sum = diff.abs().sum()
        sq_sum  = (diff * diff).sum()
        px      = m.sum()  # counts pixels * channels
        return abs_sum, sq_sum, px
