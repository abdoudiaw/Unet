# predict.py
import numpy as np
import torch
from .data import MaskedLogStandardizer

def scale_params(params, mu, std):
    if mu is None or std is None:
        return np.asarray(params, dtype=np.float32)
    return (np.asarray(params, dtype=np.float32) - np.asarray(mu, dtype=np.float32)) / np.asarray(std, dtype=np.float32)

def predict_multi(model, norm, mask, params, device=None, as_numpy=True):
    """
    Returns physical-units prediction for ALL channels: (C,H,W).
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    mask_t = torch.as_tensor(mask, dtype=torch.float32, device=device)
    if mask_t.dim() == 2: mask_t = mask_t.unsqueeze(0)  # (1,H,W)
    H, W = mask_t.shape[-2:]
    xlist = [mask_t.unsqueeze(0)]  # (1,1,H,W)

    if params is not None:
        p = torch.as_tensor(params, dtype=torch.float32, device=device).view(-1,1,1).expand(-1,H,W)
        xlist.append(p.unsqueeze(0))  # (1,P,H,W)

    x = torch.cat(xlist, dim=1)
    with torch.no_grad():
        yN = model(x)  # (1,C,H,W)
        y  = norm.inverse(yN, mask_t.unsqueeze(0))  # (1,C,H,W)
        y  = (y * mask_t.unsqueeze(0))  # zero outside
    return y.squeeze(0).detach().cpu().numpy().astype(np.float32) if as_numpy else y.squeeze(0)

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

