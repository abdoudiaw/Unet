"""
SOLPEx autoresearch train template with in-file loss stack copied from SOLPEx.

This keeps autoresearch single-file editable while preserving your current loss
definitions and adding staged cycle/parameter terms.
"""

import time
import torch
import torch.nn.functional as F

from prepare import MetricBundle, compute_val_score, print_summary

# ---------------------------------------------------------------------------
# Tunable weights (agent edits these)
# ---------------------------------------------------------------------------

LAMBDA_W = 1.0
LAMBDA_G = 0.2

ALPHA_TARGET = 0.05  # cycle weight
BETA_TARGET = 0.05   # parameter loss weight

FORWARD_ONLY_EPOCHS = 20
CYCLE_RAMP_EPOCHS = 40
PARAM_RAMP_EPOCHS = 40
TOTAL_EPOCHS = 120

ALPHA_EVAL = 0.1
BETA_EVAL = 0.1

# Forward loss controls (copied defaults from SOLPEx)
BASE_LOSS = "huber"
HUBER_BETA = 0.05
GRAD_MODE = "vector"
GRAD_BASE = "l1"
MULTISCALE = 1
MS_WEIGHT = 0.5
GRAD_DS = 2


def ramp(epoch: int, start: int, length: int, target: float) -> float:
    if epoch < start:
        return 0.0
    if length <= 0:
        return target
    frac = min((epoch - start) / length, 1.0)
    return target * frac


# ---------------------------------------------------------------------------
# Loss stack copied from /Users/42d/SOLPEx/solpex/losses.py
# ---------------------------------------------------------------------------

_SOBEL_CACHE = {}


def _sobel_kernels_cached(dtype, device, channels: int):
    dev = device if isinstance(device, torch.device) else torch.device(device)
    key = (dev.type, dev.index, str(dtype), int(channels))
    hit = _SOBEL_CACHE.get(key, None)
    if hit is not None:
        return hit

    kx1 = torch.tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
        dtype=dtype,
        device=dev,
    ).unsqueeze(0)
    ky1 = torch.tensor(
        [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
        dtype=dtype,
        device=dev,
    ).unsqueeze(0)

    kx = kx1.repeat(channels, 1, 1, 1)
    ky = ky1.repeat(channels, 1, 1, 1)
    _SOBEL_CACHE[key] = (kx, ky)
    return kx, ky


def _grad_xy_grouped(x, kx, ky):
    c = x.shape[1]
    gx = F.conv2d(x, kx, padding=1, groups=c)
    gy = F.conv2d(x, ky, padding=1, groups=c)
    return gx, gy


def _masked_mean(px, mask, w=None, channel_weights=None):
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
    pred,
    target,
    mask,
    *,
    w=None,
    lam_grad=0.2,
    lam_w=1.0,
    base="huber",
    huber_beta=0.05,
    grad_mode="vector",
    grad_base="l1",
    multiscale=0,
    ms_weight=0.5,
    grad_ds=1,
    channel_weights=None,
):
    if base == "huber":
        base_px = F.smooth_l1_loss(pred, target, beta=huber_beta, reduction="none")
    elif base == "l1":
        base_px = (pred - target).abs()
    elif base == "mse":
        base_px = (pred - target) ** 2
    else:
        raise ValueError(f"Unknown base={base}")

    l_base = _masked_mean(base_px, mask, w=None, channel_weights=channel_weights)
    l_edge = (
        _masked_mean(base_px, mask, w=w, channel_weights=channel_weights)
        if (w is not None and lam_w != 0.0)
        else torch.zeros((), device=pred.device, dtype=pred.dtype)
    )

    if lam_grad != 0.0:
        if grad_ds is None or grad_ds < 1:
            grad_ds = 1
        if grad_ds > 1:
            p_g = F.avg_pool2d(pred, kernel_size=grad_ds, stride=grad_ds)
            t_g = F.avg_pool2d(target, kernel_size=grad_ds, stride=grad_ds)
            m_g = F.avg_pool2d(mask, kernel_size=grad_ds, stride=grad_ds)
            m_g = (m_g > 0.5).to(pred.dtype)
        else:
            p_g, t_g, m_g = pred, target, mask

        c = p_g.shape[1]
        kx, ky = _sobel_kernels_cached(p_g.dtype, p_g.device, channels=c)
        pgx, pgy = _grad_xy_grouped(p_g, kx, ky)
        tgx, tgy = _grad_xy_grouped(t_g, kx, ky)

        if grad_mode == "magnitude":
            p = torch.sqrt(pgx * pgx + pgy * pgy + 1e-12)
            t = torch.sqrt(tgx * tgx + tgy * tgy + 1e-12)
            if grad_base == "l1":
                grad_px = (p - t).abs()
            elif grad_base == "mse":
                grad_px = (p - t) ** 2
            elif grad_base == "huber":
                grad_px = F.smooth_l1_loss(p, t, beta=huber_beta, reduction="none")
            else:
                raise ValueError(f"Unknown grad_base={grad_base}")
        else:
            dx = pgx - tgx
            dy = pgy - tgy
            if grad_base == "l1":
                grad_px = dx.abs() + dy.abs()
            elif grad_base == "mse":
                grad_px = dx * dx + dy * dy
            elif grad_base == "huber":
                grad_px = (
                    F.smooth_l1_loss(pgx, tgx, beta=huber_beta, reduction="none")
                    + F.smooth_l1_loss(pgy, tgy, beta=huber_beta, reduction="none")
                )
            else:
                raise ValueError(f"Unknown grad_base={grad_base}")

        l_grad = _masked_mean(grad_px, m_g, w=None, channel_weights=channel_weights)
    else:
        l_grad = torch.zeros((), device=pred.device, dtype=pred.dtype)

    loss = l_base + lam_w * l_edge + lam_grad * l_grad

    if multiscale and multiscale > 0:
        def ds(t):
            return F.avg_pool2d(t, kernel_size=2, stride=2)

        p2, t2, m2 = ds(pred), ds(target), ds(mask)
        w2 = ds(w) if w is not None else None
        loss2 = masked_weighted_loss(
            p2,
            t2,
            m2,
            w=w2,
            lam_grad=lam_grad,
            lam_w=lam_w,
            base=base,
            huber_beta=huber_beta,
            grad_mode=grad_mode,
            grad_base=grad_base,
            multiscale=0,
            grad_ds=max(1, int(grad_ds // 2)),
            channel_weights=channel_weights,
        )
        loss = loss + ms_weight * loss2
        if multiscale >= 2:
            p4, t4, m4 = ds(p2), ds(t2), ds(m2)
            w4 = ds(w2) if w2 is not None else None
            loss4 = masked_weighted_loss(
                p4,
                t4,
                m4,
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


def edge_weights(mask: torch.Tensor, sigma_px: float = 3.0) -> torch.Tensor:
    """
    mask: (H,W) float/bool tensor on any device
    Returns edge emphasis map in [0,1], same shape/device as input.
    Pure-torch approximation to avoid scipy/numpy dependency.
    """
    _ = sigma_px  # kept for signature compatibility; currently unused in fallback
    m = (mask > 0.5).to(dtype=torch.float32)
    if m.sum() == 0:
        return torch.zeros_like(m)
    # simple edge proxy: pixels with at least one 4-neighbor outside mask
    up = F.pad(m[1:, :], (0, 0, 0, 1))
    dn = F.pad(m[:-1, :], (0, 0, 1, 0))
    lf = F.pad(m[:, 1:], (0, 1, 0, 0))
    rt = F.pad(m[:, :-1], (1, 0, 0, 0))
    neigh_min = torch.minimum(torch.minimum(up, dn), torch.minimum(lf, rt))
    boundary = m * (1.0 - neigh_min)
    # blend boundary boost with interior mask
    w = 0.5 * m + 0.5 * boundary
    return w


def mse_masked(pred, target, mask):
    diff2 = (pred - target) ** 2
    return _masked_mean(diff2, mask)


def compute_losses(
    pred_fwd,
    target_field,
    mask,
    pred_cycle,
    pred_params,
    true_params,
    edge_w=None,
    alpha_train=0.0,
    beta_train=0.0,
):
    l_fwd = masked_weighted_loss(
        pred_fwd,
        target_field,
        mask,
        w=edge_w,
        lam_grad=LAMBDA_G,
        lam_w=LAMBDA_W,
        base=BASE_LOSS,
        huber_beta=HUBER_BETA,
        grad_mode=GRAD_MODE,
        grad_base=GRAD_BASE,
        multiscale=MULTISCALE,
        ms_weight=MS_WEIGHT,
        grad_ds=GRAD_DS,
    )
    l_cycle = mse_masked(pred_cycle, target_field, mask)
    l_param = F.smooth_l1_loss(pred_params, true_params)
    l_total = l_fwd + alpha_train * l_cycle + beta_train * l_param
    return l_total, l_fwd, l_cycle, l_param


def train_one_run() -> MetricBundle:
    """
    Drop in your real model/data loop here.
    Current body is a runnable smoke test with synthetic tensors.
    """
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    b, c, h, w = 4, 8, 104, 40
    p = 5

    base_mask = torch.ones((1, 1, h, w), dtype=torch.float32, device=device)
    edge_w = edge_weights(base_mask[0, 0], sigma_px=3.0).view(1, 1, h, w)

    num_steps = 0
    forward_score = 0.0
    cycle_score = 0.0
    param_score = 0.0

    for epoch in range(TOTAL_EPOCHS):
        alpha_train = ramp(epoch, FORWARD_ONLY_EPOCHS, CYCLE_RAMP_EPOCHS, ALPHA_TARGET)
        beta_train = ramp(
            epoch,
            FORWARD_ONLY_EPOCHS + CYCLE_RAMP_EPOCHS,
            PARAM_RAMP_EPOCHS,
            BETA_TARGET,
        )

        # synthetic batch placeholders (replace with real SOLPEx tensors)
        target_field = torch.randn((b, c, h, w), device=device)
        pred_fwd = target_field + 0.05 * torch.randn_like(target_field)
        pred_cycle = target_field + 0.08 * torch.randn_like(target_field)
        true_params = torch.randn((b, p), device=device)
        pred_params = true_params + 0.10 * torch.randn_like(true_params)
        mask = base_mask.expand(b, 1, h, w)

        l_total, l_fwd, l_cycle, l_param = compute_losses(
            pred_fwd=pred_fwd,
            target_field=target_field,
            mask=mask,
            pred_cycle=pred_cycle,
            pred_params=pred_params,
            true_params=true_params,
            edge_w=edge_w,
            alpha_train=alpha_train,
            beta_train=beta_train,
        )

        _ = l_total
        forward_score += float(l_fwd.detach().item())
        cycle_score += float(l_cycle.detach().item())
        param_score += float(l_param.detach().item())
        num_steps += 1

    n = float(max(num_steps, 1))
    forward_score /= n
    cycle_score /= n
    param_score /= n

    training_seconds = time.time() - t0
    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
        if torch.cuda.is_available()
        else 0.0
    )

    val_score = compute_val_score(
        forward_score=forward_score,
        cycle_score=cycle_score,
        param_score=param_score,
        alpha_eval=ALPHA_EVAL,
        beta_eval=BETA_EVAL,
    )

    return MetricBundle(
        val_score=val_score,
        forward_score=forward_score,
        cycle_score=cycle_score,
        param_score=param_score,
        training_seconds=training_seconds,
        peak_vram_mb=peak_vram_mb,
        num_steps=num_steps,
    )


if __name__ == "__main__":
    metrics = train_one_run()
    print_summary(metrics)
