import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

# ---------- Normalizer ----------
class MaskedLogStandardizer:
    """
    Per-channel masked normalizer.
    Set self.pos (bool per channel) to force log1p on strictly-positive targets.
    """
    def __init__(self, eps=1.0, sigma_floor=1e-6):
        self.mu = None          # (C,)
        self.sigma = None       # (C,)
        self.eps = float(eps)
        self.sigma_floor = float(sigma_floor)
        self.pos = None         # (C,) torch.bool; if None we'll detect from data

    def _transform_raw(self, y, mask):
        # y: (C,H,W) or (B,C,H,W). mask: (1,H,W) or (B,1,H,W)
        if y.dim() == 3: y = y.unsqueeze(0)
        if mask is not None and mask.dim() == 3: mask = mask.unsqueeze(1)
        B, C, H, W = y.shape

        # if not set, detect positive channels on masked pixels
        if self.pos is None:
            self.pos = torch.zeros(C, dtype=torch.bool, device=y.device)
            mC = (mask if mask is not None else torch.ones(B,1,H,W, device=y.device)).expand(-1, C, -1, -1)
            for c in range(C):
                yc = y[:, c]
                v  = yc[mC[:, c] > 0]
                if v.numel() > 0:
                    self.pos[c] = bool(torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).min() > 0)

        out = []
        for c in range(C):
            yc = y[:, c]
            if bool(self.pos[c]):
                out.append(torch.log1p(torch.clamp(yc, min=1e-12) + self.eps))
            else:
                out.append(yc)
        return torch.stack(out, dim=1)  # (B,C,H,W)

    def fit(self, loader):
        """Robust masked fit in float64 to avoid overflow."""
        import torch
        sums = None; sums2 = None; count = None
        with torch.no_grad():
            for b in loader:
                y = b["y"].to(torch.float64)        # RAW targets (B,C,H,W)
                m = b["mask"].to(torch.float64)     # (B,1,H,W)
                yt = self._transform_raw(y, m).to(torch.float64)  # (B,C,H,W)
                if m.dim() == 3: m = m.unsqueeze(1)
                mC = m.expand(-1, yt.size(1), -1, -1)

                yt = torch.nan_to_num(yt)
                mC = torch.nan_to_num(mC)

                s  = (yt * mC).sum(dim=(0,2,3))
                s2 = (yt * yt * mC).sum(dim=(0,2,3))
                c  = mC.sum(dim=(0,2,3))
                if sums is None:
                    sums, sums2, count = s, s2, c
                else:
                    sums += s; sums2 += s2; count += c

        count = torch.clamp(count, min=1.0)
        mu  = sums / count
        var = torch.clamp(sums2 / count - mu*mu, min=self.sigma_floor**2)

        self.mu    = mu.to(torch.float32).cpu()
        self.sigma = var.sqrt().to(torch.float32).cpu()

    def transform(self, y, mask):
        yt = self._transform_raw(y, mask)           # (B,C,H,W)
        mu = torch.nan_to_num(self.mu.to(yt.device)).view(1, -1, 1, 1)
        sg = torch.nan_to_num(self.sigma.to(yt.device), nan=1.0).view(1, -1, 1, 1)
        return (yt - mu) / sg                        # (B,C,H,W)

    def inverse(self, yN, mask=None):
        if yN.dim() == 3: yN = yN.unsqueeze(0)
        mu = torch.nan_to_num(self.mu.to(yN.device)).view(1, -1, 1, 1)
        sg = torch.nan_to_num(self.sigma.to(yN.device), nan=1.0).view(1, -1, 1, 1)
        yt = yN * sg + mu
        out = []
        for c in range(yt.size(1)):
            yc = yt[:, c]
            if bool(self.pos[c]):
                out.append(torch.expm1(yc) - self.eps)
            else:
                out.append(yc)
        return torch.stack(out, dim=1)               # (B,C,H,W)

# ---------- Dataset ----------
# data.py

class SOLPSDataset(Dataset):
    """
    Expects NPZ with either:
      - legacy: Te: (N,H,W)
      - new:    Y:  (N,C,H,W), target_keys
    Always returns:
      x: (in_ch,H,W), y: (C,H,W), mask: (1,H,W)
    """
    def __init__(self, path, normalizer=None, inputs="autoencoder", include_coords=False):
        self.normalizer = normalizer
        self.inputs = inputs
        self.include_coords = include_coords

        d = np.load(path, allow_pickle=True)
        if "Y" in d.files:
            self.Y = d["Y"].astype(np.float32)              # (N,C,H,W)
        elif "Te" in d.files:
            self.Y = d["Te"].astype(np.float32)[:, None]    # (N,1,H,W) legacy -> add channel dim
        else:
            raise KeyError("Dataset must contain 'Y' or 'Te'.")

        self.mask   = d["mask"].astype(np.float32)          # (N,H,W)
        self.params = d.get("params", None)
        if self.params is None:
            self.params = np.zeros((self.Y.shape[0], 0), dtype=np.float32)

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, idx):
        Y_raw = torch.from_numpy(self.Y[idx]).float()                 # (C,H,W)
        mask  = torch.from_numpy(self.mask[idx]).unsqueeze(0).float() # (1,H,W)
        mask  = (mask > 0.5).float()

        # clean NaNs/infs
        Y_raw = torch.nan_to_num(Y_raw, nan=0.0, posinf=0.0, neginf=0.0)

        # y (normalized if normalizer provided)
        if self.normalizer:
            y = self.normalizer.transform(Y_raw, mask)  # may return (1,C,H,W)
            if y.dim() == 4 and y.size(0) == 1:
                y = y.squeeze(0)                        # -> (C,H,W)
        else:
            y = Y_raw

        if self.inputs == "autoencoder":
            x = y.clone()
        else:
            H, W = mask.shape[-2:]
            xs = [mask]  # geometry channel
            if self.include_coords:
                grid_r = torch.linspace(0, 1, W).view(1, 1, 1, W).expand(1, 1, H, W)
                grid_z = torch.linspace(0, 1, H).view(1, 1, H, 1).expand(1, 1, H, W)
                xs += [grid_r, grid_z]
            if self.params is not None and self.params.shape[1] > 0:
                p = torch.from_numpy(self.params[idx]).view(-1,1,1).float().expand(-1,H,W)
                xs.append(p)
            x = torch.cat(xs, dim=0)                                  # (1 [+coords] + P, H, W)

        return {"x": x, "y": y, "mask": mask}

def fit_param_scaler(raw_params_train):
    x = raw_params_train.astype(np.float64)
    mu  = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(~np.isfinite(std) | (std < 1e-12), 1.0, std)
    mu  = np.where(~np.isfinite(mu), 0.0, mu)
    return mu.astype(np.float32), std.astype(np.float32)  # keep storage small

def apply_param_scaler(dataset, mu, std):
    if dataset.params.size and mu is not None:
        dataset.params = (dataset.params.astype(np.float32) - mu) / std


def make_loaders(npz_path, inputs_mode="params", batch_size=16, split=0.85, seed=42,
                 device="cuda" if torch.cuda.is_available() else "cpu"):
    # ---- read once to get channel names ----
    with np.load(npz_path, allow_pickle=True) as d:
        target_keys = [str(k) for k in d.get("target_keys", [])]

    # ---- raw dataset (no normalization) for split & fitting ----
    raw_all = SOLPSDataset(npz_path, normalizer=None, inputs=inputs_mode)
    N = len(raw_all)
    idx = np.arange(N); rng = np.random.default_rng(seed); rng.shuffle(idx)
    cut = int(split * N)
    idx_tr, idx_va = idx[:cut], idx[cut:]

    # ---- build & configure normalizer ----
    norm = MaskedLogStandardizer(eps=1.0, sigma_floor=1e-6)

    # Force log1p for strictly-positive targets (prevents overflow on ne/ni)
    if target_keys:
        pos_flags = [k in ("Te", "ne", "ni", "ti") for k in target_keys]
        norm.pos = torch.tensor(pos_flags, dtype=torch.bool)

    # Fit on TRAIN ONLY (raw targets)
    raw_train = Subset(raw_all, idx_tr)
    fit_loader = DataLoader(raw_train, batch_size=32, shuffle=False, num_workers=0)
    norm.fit(fit_loader)

    # ---- datasets WITH normalization for training/inference ----
    ds_tr = SOLPSDataset(npz_path, normalizer=norm, inputs=inputs_mode)
    ds_va = SOLPSDataset(npz_path, normalizer=norm, inputs=inputs_mode)

    # ---- parameter scaling (for params-mode inputs) ----
    param_mu = param_std = None
    if inputs_mode != "autoencoder":
        param_mu, param_std = fit_param_scaler(raw_all.params[idx_tr])
        apply_param_scaler(ds_tr, param_mu, param_std)
        apply_param_scaler(ds_va, param_mu, param_std)

    # ---- splits ----
    train_set = Subset(ds_tr, idx_tr)
    val_set   = Subset(ds_va, idx_va)

    # ---- loaders ----
    use_cuda = (str(device) == "cuda")
    num_w    = 0
    pin_mem  = use_cuda
    pers_w   = False

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_w, pin_memory=pin_mem, persistent_workers=pers_w, drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              num_workers=num_w, pin_memory=pin_mem, persistent_workers=pers_w)

    # ---- shapes/meta ----
    sample = train_set[0]
    C = int(sample["y"].shape[0])                              # number of output channels
    P = int(train_set.dataset.params.shape[1] if inputs_mode != "autoencoder" else 0)
    H, W = int(sample["mask"].shape[-2]), int(sample["mask"].shape[-1])

    return train_loader, val_loader, norm, P, (H, W), (param_mu, param_std), C
