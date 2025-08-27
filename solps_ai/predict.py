import numpy as np
import torch
from .data import MaskedLogStandardizer

def scale_params(params, mu, std):
    if mu is None or std is None: return np.asarray(params, dtype=np.float32)
    return (np.asarray(params, dtype=np.float32) - np.asarray(mu, dtype=np.float32)) / np.asarray(std, dtype=np.float32)

def predict_te(model, norm, mask, params, device=None, as_numpy=True):
    """
    Predict Te (eV) for one mask + parameter vector.
    - mask: (H,W) or (1,H,W) binary
    - params: (P,) scaled to match training (O(1))
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    # mask tensor
    if isinstance(mask, np.ndarray):
        m = torch.from_numpy(mask).float()
    else:
        m = mask.float()
    if m.dim() == 2:   m = m.unsqueeze(0).unsqueeze(0)
    elif m.dim() == 3: m = m.unsqueeze(0)
    m = (m > 0.5).float().to(device)  # (1,1,H,W)
    H, W = m.shape[-2:]

    # params tensor
    params = np.asarray(params, dtype=np.float32).ravel()
    P = int(params.shape[0])
    p = torch.from_numpy(params).view(1, P, 1, 1).expand(1, P, H, W).to(device)

    # sanity: model first conv expects 1+P
    expected_in = getattr(getattr(model, "enc1", [None])[0], "in_channels", None)
    if expected_in is not None and expected_in != 1 + P:
        raise RuntimeError(f"Model expects in_ch={expected_in}, but params gives {1+P}.")

    x = torch.cat([m, p], dim=1)

    with torch.no_grad():
        z = model(x)               # normalized
        te = norm.inverse(z, m)    # eV
    return te.squeeze().detach().cpu().numpy().astype(np.float32) if as_numpy else te

def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)  # your own file => trusted
    model_in_ch = ckpt.get("in_ch", 1)
    from .models import UNet
    model = UNet(in_ch=model_in_ch, out_ch=1).to(device)
    model.load_state_dict(ckpt["model"])
    mu, sigma, eps = ckpt["norm"]
    norm = MaskedLogStandardizer(eps=eps); norm.mu, norm.sigma = mu, sigma
    param_mu  = ckpt.get("param_mu", None)
    param_std = ckpt.get("param_std", None)
    return model, norm, (param_mu, param_std)

