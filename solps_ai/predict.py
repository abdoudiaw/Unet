import numpy as np
import torch
from .data import (
    MaskedLinearStandardizer,
    MaskedLogStandardizer,
    MaskedSymLogStandardizer,
    MultiChannelNormalizer,
)

def scale_params(params, mu, std):
    if mu is None or std is None: return np.asarray(params, dtype=np.float32)
    return (np.asarray(params, dtype=np.float32) - np.asarray(mu, dtype=np.float32)) / np.asarray(std, dtype=np.float32)

def _norm_from_ckpt(norm_ckpt):
    # Backward-compatible loader:
    # - legacy: (mu, sigma, eps)
    # - current: {"kind": "...", ...}
    if isinstance(norm_ckpt, (tuple, list)) and len(norm_ckpt) == 3:
        mu, sigma, eps = norm_ckpt
        norm = MaskedLogStandardizer(eps=float(eps))
        norm.mu, norm.sigma = mu, sigma
        return norm

    if not isinstance(norm_ckpt, dict):
        raise ValueError("Unsupported norm checkpoint format.")

    kind = norm_ckpt.get("kind", "")
    if kind == "MaskedLogStandardizer":
        norm = MaskedLogStandardizer(eps=float(norm_ckpt.get("eps", 1.0)))
    elif kind == "MaskedLinearStandardizer":
        norm = MaskedLinearStandardizer(eps=float(norm_ckpt.get("eps", 1e-12)))
    elif kind == "MaskedSymLogStandardizer":
        norm = MaskedSymLogStandardizer(
            scale=float(norm_ckpt.get("scale", 1.0)),
            eps=float(norm_ckpt.get("eps", 1e-12)),
        )
    elif kind == "MultiChannelNormalizer":
        y_keys = norm_ckpt.get("y_keys", [])
        norms_pack = norm_ckpt.get("norms", {})
        norms_by_name = {k: _norm_from_ckpt(v) for k, v in norms_pack.items()}
        return MultiChannelNormalizer(y_keys=y_keys, norms_by_name=norms_by_name)
    else:
        raise ValueError(f"Unsupported normalizer kind: {kind!r}")

    norm.mu = norm_ckpt.get("mu")
    norm.sigma = norm_ckpt.get("sigma")
    return norm

def _get_model_in_channels(model):
    # Prefer explicit module metadata if available.
    if hasattr(model, "enc1") and hasattr(model.enc1, "net") and len(model.enc1.net) > 0:
        conv0 = model.enc1.net[0]
        if hasattr(conv0, "in_channels"):
            return int(conv0.in_channels)
    return None

def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)

    from .models import UNet
    in_ch = ckpt.get("in_ch", 1)
    out_ch = ckpt.get("out_ch", 1)
    base  = ckpt.get("base", 2)  # fallback if missing

    model = UNet(in_ch=in_ch, out_ch=out_ch, base=base).to(device)
    model.load_state_dict(ckpt["model"], strict=True)

    norm = _norm_from_ckpt(ckpt["norm"])

    if "param_mu" in ckpt and "param_std" in ckpt:
        param_scaler = (ckpt.get("param_mu"), ckpt.get("param_std"))
    else:
        param_scaler = ckpt.get("param_scaler", (None, None))

    return model, norm, param_scaler

def predict_fields(model, norm, mask, params, Rn=None, Zn=None, device=None, as_numpy=True):
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    # mask tensor -> (1,1,H,W)
    m = torch.from_numpy(mask).float() if isinstance(mask, np.ndarray) else mask.float()
    if m.dim() == 2:   m = m.unsqueeze(0).unsqueeze(0)
    elif m.dim() == 3: m = m.unsqueeze(0)
    m = (m > 0.5).float().to(device)
    H, W = m.shape[-2:]

    # expected in channels (works for your UNet style)
    expected_in = _get_model_in_channels(model)

    channels = [m]  # (1,1,H,W)

    # geometry channels -> (1,1,H,W)
    if Rn is not None and Zn is not None:
        Rch = torch.from_numpy(np.asarray(Rn, np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        Zch = torch.from_numpy(np.asarray(Zn, np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        if Rch.shape[-2:] != (H, W) or Zch.shape[-2:] != (H, W):
            raise ValueError(f"Rn/Zn shape mismatch: got {Rch.shape[-2:]},{Zch.shape[-2:]}, expected {(H,W)}")
        channels += [Rch, Zch]

    # params channels -> (1,P,H,W)
    if params is not None:
        params = np.asarray(params, dtype=np.float32).ravel()
        P = int(params.shape[0])
        p = torch.from_numpy(params).view(1, P, 1, 1).expand(1, P, H, W).to(device)
        channels.append(p)

    x = torch.cat(channels, dim=1)  # (1,C,H,W)

    if expected_in is not None and x.shape[1] != expected_in:
        raise RuntimeError(f"Model expects in_ch={expected_in}, but you built C={x.shape[1]}.")

    with torch.no_grad():
        z = model(x)
        y = norm.inverse(z, m)

    if as_numpy:
        out = y.squeeze(0).detach().cpu().numpy().astype(np.float32)  # (C,H,W)
        if out.ndim == 2:
            out = out[None, ...]
        return out
    return y


def predict_te(model, norm, mask, params, Rn=None, Zn=None, device=None, as_numpy=True):
    """
    Backward-compatible single-field helper.
    For multi-output models, returns channel 0 by default (expected Te if y_keys start with Te).
    """
    out = predict_fields(
        model=model, norm=norm, mask=mask, params=params,
        Rn=Rn, Zn=Zn, device=device, as_numpy=as_numpy
    )
    if as_numpy:
        return out[0]
    return out[:, 0:1]
