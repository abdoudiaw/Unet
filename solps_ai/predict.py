# predict.py
import numpy as np
import torch
from .data import MaskedLogStandardizer

def _normalize_device(model, device):
    if device is None:
        return next(model.parameters()).device
    if isinstance(device, (list, tuple)):      # <- handle tuples
        device = device[0]
    if not isinstance(device, torch.device):
        device = torch.device(device)
    return device

def scale_params(params, mu, std):
    if mu is None or std is None:
        return np.asarray(params, dtype=np.float32)
    return (np.asarray(params, dtype=np.float32) - np.asarray(mu, dtype=np.float32)) / np.asarray(std, dtype=np.float32)


def _normalize_device(model, device):
    """Return a torch.device. Accepts None/'cuda'/('cuda',) etc."""
    if device is None:
        return next(model.parameters()).device
    if isinstance(device, (list, tuple)):
        device = device[0]
    if not isinstance(device, torch.device):
        device = torch.device(device)
    return device

@torch.no_grad()
def predict_multi(model, norm, mask, params=None, device=None, as_numpy=True):
    """
    Predict ALL channels for a single case in physical units.
    Inputs
      - mask: (H,W) or (1,H,W) numpy/torch, any dtype
      - params: None or 1D array-like of length P (scaled params expected)
    Returns
      - (C,H,W) in torch.Tensor (if as_numpy=False) or np.float32
    """
    device = _normalize_device(model, device)
    model.eval()

    # --- mask -> (1,1,H,W) float on device
    mask_t = torch.as_tensor(mask, dtype=torch.float32)             # no device= here
    if mask_t.dim() == 2:
        mask_t = mask_t.unsqueeze(0)                                # (1,H,W)
    elif mask_t.dim() != 3 or mask_t.size(0) != 1:
        # force (1,H,W)
        mask_t = mask_t.squeeze()
        if mask_t.dim() != 2:
            raise ValueError(f"mask must be (H,W) or (1,H,W); got shape {tuple(mask_t.shape)}")
        mask_t = mask_t.unsqueeze(0)
    mask_t = (mask_t > 0.5).float().to(device)                      # binarize & move
    H, W = mask_t.shape[-2:]

    # --- build model input: x = [mask, (params per-pixel)...] -> (1, 1+P, H, W)
    xlist = [mask_t.unsqueeze(0)]                                   # (1,1,H,W)

    if params is not None:
        p = torch.as_tensor(params, dtype=torch.float32)            # CPU first
        if p.dim() == 0:
            p = p.view(1)                                          # scalar -> (1,)
        if p.dim() != 1:
            p = p.view(-1)                                         # flatten to (P,)
        P = p.numel()
        p = p.view(1, P, 1, 1).expand(1, P, H, W).to(device)        # (1,P,H,W)
        xlist.append(p)

    x = torch.cat(xlist, dim=1)                                     # (1,1+P,H,W)

    # --- forward
    yN = model(x)                                                   # (1,C,H,W) normalized
    # inverse to physical units; keep tensors on device
    y  = norm.inverse(yN, mask_t.unsqueeze(0))                      # (1,C,H,W)
    # zero outside mask
    y  = y * mask_t.unsqueeze(0)

    y = y.squeeze(0)                                                # (C,H,W)
    if as_numpy:
        return y.detach().cpu().numpy().astype(np.float32)
    return y

def predict_te(model, norm, mask, params, device=None, as_numpy=True):
    y = predict_multi(model, norm, mask, params, device=device, as_numpy=False)  # (C,H,W)
    te = y[0]  # assume channel 0 = Te
    return te.detach().cpu().numpy().astype(np.float32) if as_numpy else te

def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    in_ch   = ckpt.get("in_ch", 1)
    out_ch  = ckpt.get("out_ch", 1)
    norm_mu, norm_sigma, norm_eps, pos_flags = ckpt.get("norm")
    pos_flags = torch.as_tensor(pos_flags, dtype=torch.uint8)

    from .models import UNet
    model = UNet(in_ch=in_ch, out_ch=out_ch).to(device)
    model.load_state_dict(ckpt["model"])

    norm = MaskedLogStandardizer(eps=norm_eps)
    norm.mu    = torch.as_tensor(norm_mu)
    norm.sigma = torch.as_tensor(norm_sigma)
    norm.pos   = pos_flags.bool()

    param_mu  = ckpt.get("param_mu", None)
    param_std = ckpt.get("param_std", None)
    return model, norm, (param_mu, param_std)

