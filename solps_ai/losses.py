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

def masked_weighted_loss(pred, target, mask, w=None, lam_grad=0.2, lam_w=1.0):
    mse = (((pred-target)**2) * mask).sum() / mask.sum().clamp_min(1e-8)
    # gradient consistency (Sobel-like)
    kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=pred.dtype, device=pred.device).view(1,1,3,3)/8
    ky = kx.transpose(-1,-2)
    def g(x):
        gx = F.conv2d(x, kx, padding=1); gy = F.conv2d(x, ky, padding=1)
        return torch.sqrt(gx*gx + gy*gy + 1e-12)
    gl = (((g(pred)-g(target))**2) * mask).sum() / mask.sum().clamp_min(1e-8)
    if w is None:
        return mse + lam_grad*gl
    if isinstance(w, np.ndarray):
        w = torch.as_tensor(w, device=pred.device, dtype=pred.dtype)
    if w.dim() == 2:  # (H,W)
        w = w.unsqueeze(0).unsqueeze(0)
    if w.size(0) == 1:  # broadcast to batch
        w = w.expand(pred.size(0), 1, pred.size(2), pred.size(3))
    wmse = (((pred-target)**2) * mask * w).sum() / (mask*w).sum().clamp_min(1e-8)
    return mse + lam_grad*gl + lam_w*wmse

def mae_norm(pred, target, mask):
    return (torch.abs(pred - target) * mask).sum() / mask.sum().clamp_min(1e-8)

def batch_error_sums_ev(pred, target, mask, norm):
    with torch.no_grad():
        p_ev = norm.inverse(pred, mask)
        y_ev = norm.inverse(target, mask)
        diff = (p_ev - y_ev) * mask
        abs_sum = diff.abs().sum()
        sq_sum  = (diff**2).sum()
        px      = mask.sum()
        return abs_sum, sq_sum, px

