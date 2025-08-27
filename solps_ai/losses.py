# losses.py
import numpy as np
import torch
import torch.nn.functional as F

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

def _total_variation(x):
    # x: (B,C,H,W)
    dx = torch.abs(x[..., 1:, :] - x[..., :-1, :]).mean()
    dy = torch.abs(x[..., :, 1:] - x[..., :, :-1]).mean()
    return dx + dy

def masked_weighted_loss(pred, target, mask, w=None, lam_grad=0.2, lam_w=1.0):
    """
    pred/target: (B,C,H,W)
    mask      : (B,1,H,W) or (B,H,W)
    w         : (1,1,H,W) optional edge-weight map (multiplies mask)
    """
    if mask.dim() == 3:  # (B,H,W) -> (B,1,H,W)
        mask = mask.unsqueeze(1)
    if mask.size(1) == 1 and pred.size(1) > 1:
        mask = mask.expand(-1, pred.size(1), -1, -1)  # broadcast over channels
    m = mask

    if w is not None:
        if w.dim() == 3:  # (1,H,W) -> (1,1,H,W)
            w = w.unsqueeze(1)
        if w.size(1) == 1 and pred.size(1) > 1:
            w = w.expand(-1, pred.size(1), -1, -1)
        m = m * w

    # masked L1 (per-pixel per-channel), average over masked pixels
    l1 = (torch.abs(pred - target) * m).sum() / m.sum().clamp_min(1e-8)

    # optional TV smoothness on predictions inside ROI
    tv = _total_variation(pred * (m > 0).float())

    return lam_w * l1 + lam_grad * tv

def mae_norm(pred, target, mask):
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    if mask.size(1) == 1 and pred.size(1) > 1:
        mask = mask.expand(-1, pred.size(1), -1, -1)
    return (torch.abs(pred - target) * mask).sum() / mask.sum().clamp_min(1e-8)

def batch_error_sums_ev(pred, target, mask, norm):
    """
    Compute sums across all channels in physical units.
    Returns (abs_sum, sq_sum, px) aggregated over B and C.
    """
    with torch.no_grad():
        p_ev = norm.inverse(pred, mask)   # (B,C,H,W)
        y_ev = norm.inverse(target, mask) # (B,C,H,W)
        if mask.dim() == 3: mask = mask.unsqueeze(1)
        if mask.size(1) == 1 and p_ev.size(1) > 1:
            mask = mask.expand(-1, p_ev.size(1), -1, -1)
        diff = (p_ev - y_ev) * mask
        abs_sum = diff.abs().sum()
        sq_sum  = (diff**2).sum()
        px      = mask.sum()
        return abs_sum, sq_sum, px

