import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

# ---------- Normalizer ----------
class MaskedLogStandardizer:
    def __init__(self, eps=1.0, sigma_floor=1e-6):
        self.mu = None; self.sigma = None; self.eps = float(eps)
        self.sigma_floor = float(sigma_floor)

    def fit(self, loader):
        sums = 0.0; sums2 = 0.0; count = 0.0
        with torch.no_grad():
            for b in loader:
                y = b["y"].float()
                m = b["mask"].float()
                y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
                z = torch.log1p(torch.clamp(y, min=0) + self.eps)
                z = torch.where(torch.isfinite(z), z, torch.zeros_like(z))
                z = z*m
                sums  += z.sum().item()
                sums2 += (z*z).sum().item()
                count += m.sum().item()
        if count <= 0:
            raise RuntimeError("Normalizer.fit(): mask has zero valid pixels.")
        self.mu = sums / count
        var = max(sums2 / count - self.mu**2, 0.0)
        self.sigma = float(max(var**0.5, self.sigma_floor))
        print(f"[normalizer] mu={self.mu:.6g} sigma={self.sigma:.6g} count={int(count)}")

    def transform(self, y, mask):
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        z = torch.log1p(torch.clamp(y, min=0) + self.eps)
        z = (z - self.mu) / (self.sigma + 1e-8)
        z = torch.where(torch.isfinite(z), z, torch.zeros_like(z))
        return z * mask

    def inverse(self, y_norm, mask=None):
        z = y_norm * (self.sigma + 1e-8) + self.mu
        y = torch.expm1(z) - self.eps
        y = torch.where(torch.isfinite(y), y, torch.zeros_like(y))
        return y if mask is None else y * mask


# ---------- Dataset ----------
class SOLPSDataset(Dataset):
    """
    Expects an NPZ with Te: (N,H,W), mask: (N,H,W), optional params: (N,P)
    inputs="autoencoder": x=y (normalized); inputs!="autoencoder": x=[mask, params...]
    """
    def __init__(self, path, normalizer=None, inputs="autoencoder"):
        self.normalizer = normalizer
        self.inputs = inputs
        d = np.load(path, allow_pickle=True)
        self.Te     = d["Te"].astype(np.float32)     # (N,H,W)
        self.mask   = d["mask"].astype(np.float32)   # (N,H,W)
        self.params = d.get("params", None)
        if self.params is None:
            self.params = np.zeros((self.Te.shape[0], 0), dtype=np.float32)

    def __len__(self):
        return self.Te.shape[0]

    def __getitem__(self, idx):
        Te_raw = torch.from_numpy(self.Te[idx]).unsqueeze(0).float()   # (1,H,W)
        mask   = torch.from_numpy(self.mask[idx]).unsqueeze(0).float() # (1,H,W)
        mask   = (mask > 0.5).float()

        Te_raw = torch.nan_to_num(Te_raw, nan=0.0, posinf=0.0, neginf=0.0)
        Te_roi = Te_raw * mask
        y = self.normalizer.transform(Te_raw, mask) if self.normalizer else Te_roi

        if self.inputs == "autoencoder":
            x = y.clone()
        else:
            params = torch.from_numpy(self.params[idx]).float()
            P = params.numel()
            H, W = mask.shape[-2:]
            x = torch.cat([mask, params.view(P,1,1).expand(P,H,W)], dim=0) if P else mask

        return {"x": x, "y": y, "mask": mask}


def fit_param_scaler(raw_params_train):
    mu = raw_params_train.mean(axis=0).astype(np.float32)
    std = raw_params_train.std(axis=0).astype(np.float32)
    std[std < 1e-12] = 1.0
    return mu, std


def apply_param_scaler(dataset, mu, std):
    if dataset.params.size and mu is not None:
        dataset.params = (dataset.params - mu) / std


def make_loaders(npz_path, inputs_mode="params", batch_size=16, split=0.85, seed=42,
                 device="cuda" if torch.cuda.is_available() else "cpu"):
    # raw (no normalization) to compute split & fit normalizer
    raw_all = SOLPSDataset(npz_path, normalizer=None, inputs=inputs_mode)
    N = len(raw_all)
    idx = np.arange(N); rng = np.random.default_rng(seed); rng.shuffle(idx)
    cut = int(split * N)
    idx_tr, idx_va = idx[:cut], idx[cut:]

    # normalizer on TRAIN only
    norm = MaskedLogStandardizer(eps=1.0)
    raw_train = Subset(raw_all, idx_tr)
    norm.fit(DataLoader(raw_train, batch_size=16, shuffle=False, num_workers=0))

    # datasets WITH normalization
    ds_tr = SOLPSDataset(npz_path, normalizer=norm, inputs=inputs_mode)
    ds_va = SOLPSDataset(npz_path, normalizer=norm, inputs=inputs_mode)

    # param scaling if in params mode
    param_mu = param_std = None
    if inputs_mode != "autoencoder":
        param_mu, param_std = fit_param_scaler(raw_all.params[idx_tr])
        apply_param_scaler(ds_tr, param_mu, param_std)
        apply_param_scaler(ds_va, param_mu, param_std)

    train_set = Subset(ds_tr, idx_tr)
    val_set   = Subset(ds_va, idx_va)

    use_cuda = (str(device) == "cuda")
    num_w    = 0  # start safe in notebooks; bump to 2 later
    pin_mem  = use_cuda
    pers_w   = False

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_w, pin_memory=pin_mem, persistent_workers=pers_w)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              num_workers=num_w, pin_memory=pin_mem, persistent_workers=pers_w)

    P = train_set.dataset.params.shape[1] if inputs_mode != "autoencoder" else 0
    H, W = train_set[0]["mask"].shape[-2:]

    return train_loader, val_loader, norm, P, (H, W), (param_mu, param_std)

