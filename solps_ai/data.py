import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import h5py

def normalize_coords(R2d, Z2d, mask=None):
    if mask is not None:
        m = mask > 0.5
        Rmin, Rmax = R2d[m].min(), R2d[m].max()
        Zmin, Zmax = Z2d[m].min(), Z2d[m].max()
    else:
        Rmin, Rmax = R2d.min(), R2d.max()
        Zmin, Zmax = Z2d.min(), Z2d.max()

    Rn = 2.0 * (R2d - Rmin) / (Rmax - Rmin) - 1.0
    Zn = 2.0 * (Z2d - Zmin) / (Zmax - Zmin) - 1.0
    return Rn.astype(np.float32), Zn.astype(np.float32)
    
def load_geometry_h5(fname):
    """Load (R,Z) grids from geom_ref.h5."""
    with h5py.File(fname, "r") as f:
        print(f.keys())
        R2d = np.array(f["R2D"])
        Z2d = np.array(f["Z2D"])
    return R2d, Z2d
    
# ---------- Normalizer ----------
class MaskedLogStandardizer:
    def __init__(self, eps=1.0, sigma_floor=1e-6):
        self.mu = None; self.sigma = None; self.eps = float(eps)
        self.sigma_floor = float(sigma_floor)

    def fit(self, loader, channel_index: int = 0, key: str = "y", mask_key: str = "mask"):
        """
        Fit mu/sigma on z = log(y + eps), masked.
        Supports y shaped (B,1,H,W) or (B,C,H,W) via channel_index.
        """
        import torch

        s1 = 0.0
        s2 = 0.0
        n  = 0.0

        for b in loader:
            y = b[key]
            m = b[mask_key]

            # ensure tensors
            if not torch.is_tensor(y):
                y = torch.as_tensor(y)
            if not torch.is_tensor(m):
                m = torch.as_tensor(m)

            # select channel
            if y.dim() == 4:  # (B,C,H,W) or (B,1,H,W)
                yc = y[:, channel_index:channel_index+1]
            elif y.dim() == 3:  # (B,H,W) (rare)
                yc = y.unsqueeze(1)
            else:
                raise ValueError(f"Unexpected y shape {tuple(y.shape)}")

            if m.dim() == 3:   # (B,H,W)
                mc = m.unsqueeze(1)
            elif m.dim() == 4: # (B,1,H,W)
                mc = m
            else:
                raise ValueError(f"Unexpected mask shape {tuple(m.shape)}")

            mc = (mc > 0.5).float()
            yc = torch.nan_to_num(yc, nan=0.0, posinf=0.0, neginf=0.0)

#            z = torch.log(yc + float(self.eps))
            # ALWAYS use this
            z = torch.log(torch.clamp(yc, min=0) + self.eps)
            z = torch.where(torch.isfinite(z), z, torch.zeros_like(z))
            z = z * mc

            vals = z[mc.bool()]
            if vals.numel() == 0:
                continue

            s1 += vals.sum().item()
            s2 += (vals * vals).sum().item()
            n  += float(vals.numel())

        mu = s1 / max(n, 1.0)
        var = s2 / max(n, 1.0) - mu * mu
        sigma = max((max(var, 0.0) ** 0.5), self.sigma_floor)

        # store as torch tensors like before
        self.mu = torch.tensor(mu, dtype=torch.float32)
        self.sigma = torch.tensor(sigma, dtype=torch.float32)

    def transform(self, y, mask):
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        # ALWAYS use this
        z = torch.log(torch.clamp(y, min=0) + self.eps)

        mu = self.mu.to(z.device)
        sigma = self.sigma.to(z.device)
        z = (z - mu) / (sigma + 1e-8)
        z = torch.where(torch.isfinite(z), z, torch.zeros_like(z))
        return z * mask

    def inverse(self, y_norm, mask=None):
        z = y_norm * (self.sigma + 1e-8) + self.mu
        y = torch.exp(z) - self.eps
        y = torch.where(torch.isfinite(y), y, torch.zeros_like(y))
        return y if mask is None else y * mask

class MaskedLinearStandardizer:
    def __init__(self, eps=1e-12):
        self.eps = eps
        self.mu = None
        self.sigma = None

    def fit(self, loader, key="y", mask_key="mask", channel_index=0):
        s1 = 0.0
        s2 = 0.0
        n  = 0.0
        for b in loader:
            y = b[key]          # (B,C,H,W)
            m = b[mask_key]     # (B,1,H,W)
            yc = y[:, channel_index:channel_index+1]
            mc = (m > 0.5).float()
            vals = yc[mc.bool()]
            if vals.numel() == 0:
                continue
            s1 += vals.sum().item()
            s2 += (vals * vals).sum().item()
            n  += float(vals.numel())
        mu = s1 / max(n, 1.0)
        var = s2 / max(n, 1.0) - mu * mu
        sigma = (max(var, 0.0) ** 0.5) + self.eps
        self.mu = torch.tensor(mu, dtype=torch.float32)
        self.sigma = torch.tensor(sigma, dtype=torch.float32)

    def transform(self, y, m):
        return (y - self.mu.to(y.device)) / self.sigma.to(y.device)

    def inverse(self, z, m):
        return z * self.sigma.to(z.device) + self.mu.to(z.device)


class MaskedSymLogStandardizer:
    """
    z = sign(y) * log1p(|y|/s)
    then standardize z with masked mean/std
    """
    def __init__(self, scale, eps=1e-12):
        self.scale = float(scale)
        self.eps = eps
        self.mu = None
        self.sigma = None

    def _symlog(self, y):
        s = self.scale
        return torch.sign(y) * torch.log1p(torch.abs(y) / s)

    def _symexp(self, z):
        s = self.scale
        return torch.sign(z) * s * torch.expm1(torch.abs(z))

    def fit(self, loader, key="y", mask_key="mask", channel_index=0):
        s1 = 0.0
        s2 = 0.0
        n  = 0.0
        for b in loader:
            y = b[key]
            m = b[mask_key]
            yc = y[:, channel_index:channel_index+1]
            mc = (m > 0.5).float()
            z = self._symlog(torch.nan_to_num(yc))
            vals = z[mc.bool()]
            if vals.numel() == 0:
                continue
            s1 += vals.sum().item()
            s2 += (vals * vals).sum().item()
            n  += float(vals.numel())
        mu = s1 / max(n, 1.0)
        var = s2 / max(n, 1.0) - mu * mu
        sigma = (max(var, 0.0) ** 0.5) + self.eps
        self.mu = torch.tensor(mu, dtype=torch.float32)
        self.sigma = torch.tensor(sigma, dtype=torch.float32)

    def transform(self, y, m):
        z = self._symlog(torch.nan_to_num(y))
        mu = self.mu.to(z.device); sigma = self.sigma.to(z.device)
        zhat = (z - mu) / sigma
        zhat = torch.where(torch.isfinite(zhat), zhat, torch.zeros_like(zhat))
        return zhat * m

    def inverse(self, zhat, m=None):
        mu = self.mu.to(zhat.device); sigma = self.sigma.to(zhat.device)
        z = zhat * sigma + mu
        y = self._symexp(z)
        y = torch.where(torch.isfinite(y), y, torch.zeros_like(y))
        return y if m is None else y * m


class MultiChannelNormalizer:
    def __init__(self, y_keys, norms_by_name):
        self.y_keys = list(map(str, y_keys))
        self.norms = {k: norms_by_name[k] for k in self.y_keys}

    def fit(self, loader):
        # assumes loader yields y as (B,C,H,W) already in RAW physical space
        for ci, name in enumerate(self.y_keys):
            self.norms[name].fit(loader, channel_index=ci)

    def transform(self, y, m):
        # y: (B,C,H,W), m: (B,1,H,W)
        outs = []
        for ci, name in enumerate(self.y_keys):
            outs.append(self.norms[name].transform(y[:, ci:ci+1], m))
        return torch.cat(outs, dim=1)

    def inverse(self, z, m):
        outs = []
        for ci, name in enumerate(self.y_keys):
            outs.append(self.norms[name].inverse(z[:, ci:ci+1], m))
        return torch.cat(outs, dim=1)


class SOLPSDataset(Dataset):
    """
    Supports:
      - legacy: Te: (N,H,W), mask: (N,H,W) or (H,W)
      - multi:  Y: (N,C,H,W), y_keys: (C,), mask: (N,H,W) or (H,W)

    Args:
      path: npz
      geom_h5: optional geom_ref.h5 to add Rn/Zn to x in params mode
      normalizer: object with .transform(y, mask) and optionally supports multi-channel y
      inputs: "autoencoder" or "params"
      y_key: single output name (backwards compat)
      y_keys: list of output names (multi-output). If provided, overrides y_key.
    Returns dict:
      - x: (Cin,H,W)
      - y: (Cout,H,W)
      - mask: (1,H,W)
      - params: (P,) if inputs != "autoencoder"
    """
    def __init__(self, path, geom_h5=None, normalizer=None, inputs="autoencoder",
                 y_key="Te", y_keys=None):
        self.normalizer = normalizer
        self.inputs = str(inputs)
        self.y_key = str(y_key)
        self.y_keys_req = None if y_keys is None else [str(k) for k in y_keys]

        d = np.load(path, allow_pickle=True)

        # ---- outputs first (so we know N,H,W) ----
        if "Y" in d.files:
            self.Y = d["Y"].astype(np.float32)  # (N,C,H,W)
            self.y_keys_all = np.array(d["y_keys"]).astype(str)  # (C,)
            self.N, self.C_all, self.H, self.W = self.Y.shape
            self._has_Y = True
        else:
            self.Te = d["Te"].astype(np.float32)  # (N,H,W)
            self.y_keys_all = np.array(["Te"], dtype=str)
            self.N, self.H, self.W = self.Te.shape
            self.C_all = 1
            self._has_Y = False

        # ---- choose outputs ----
        if self.y_keys_req is None:
            # single-output mode
            if self._has_Y:
                hits = np.where(self.y_keys_all == self.y_key)[0]
                if hits.size == 0:
                    raise KeyError(f"y_key={self.y_key!r} not found in y_keys={list(self.y_keys_all)}")
                self.ks = [int(hits[0])]
                self.y_keys = np.array([self.y_key], dtype=str)
            else:
                if self.y_key != "Te":
                    raise KeyError("Legacy dataset has only 'Te'.")
                self.ks = [0]
                self.y_keys = np.array(["Te"], dtype=str)
        else:
            # multi-output mode
            if not self._has_Y:
                raise KeyError("Requested y_keys but dataset has no 'Y' array (only legacy Te).")
            ks = []
            for name in self.y_keys_req:
                hits = np.where(self.y_keys_all == name)[0]
                if hits.size == 0:
                    raise KeyError(f"y_key={name!r} not found in y_keys={list(self.y_keys_all)}")
                ks.append(int(hits[0]))
            self.ks = ks
            self.y_keys = np.array(self.y_keys_req, dtype=str)

        self.C_out = len(self.ks)

        # ---- mask (robust) ----
        mask = d.get("mask", None)
        bad_mask = (mask is None) or (isinstance(mask, np.ndarray) and mask.shape == ())
        if bad_mask:
            mask = np.ones((self.N, self.H, self.W), dtype=np.float32)
        else:
            mask = np.array(mask)
            if mask.ndim == 2 and mask.shape == (self.H, self.W):
                mask = np.repeat(mask[None, :, :], self.N, axis=0)
            if mask.shape != (self.N, self.H, self.W):
                raise ValueError(f"mask has wrong shape {mask.shape}, expected {(self.N, self.H, self.W)}")
            mask = mask.astype(np.float32)
        self.mask = mask  # (N,H,W)

        # ---- geometry channels (H,W), used only for params-mode x ----
        if geom_h5 is not None:
            R2d, Z2d = load_geometry_h5(geom_h5)
            if R2d.shape != (self.H, self.W) or Z2d.shape != (self.H, self.W):
                raise ValueError(f"Geometry shape mismatch: R2d{R2d.shape}, Z2d{Z2d.shape}, expected {(self.H, self.W)}")
            mask2d = (self.mask[0] > 0.5)
            Rn, Zn = normalize_coords(R2d, Z2d, mask=mask2d)
            self.Rn = Rn.astype(np.float32)
            self.Zn = Zn.astype(np.float32)
        else:
            self.Rn = None
            self.Zn = None

        # ---- params ----
        params = d.get("params", None)
        if params is None:
            params = d.get("X", None)
        if params is None:
            params = np.zeros((self.N, 0), dtype=np.float32)
        else:
            params = np.array(params).astype(np.float32)
            if params.ndim != 2 or params.shape[0] != self.N:
                raise ValueError(f"params has shape {params.shape}, expected (N,P) with N={self.N}")
        self.params = params  # (N,P)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # ---- y_raw: (C_out,H,W) ----
        if self._has_Y:
            y_raw = self.Y[idx, self.ks, :, :]      # (C_out,H,W)
        else:
            y_raw = self.Te[idx, :, :][None, :, :]  # (1,H,W)

        y_raw = torch.from_numpy(y_raw).float()
        y_raw = torch.nan_to_num(y_raw, nan=0.0, posinf=0.0, neginf=0.0)

        # ---- mask: (1,H,W) ----
        mask = torch.from_numpy(self.mask[idx]).unsqueeze(0).float()
        mask = (mask > 0.5).float()

        # ---- normalize target ----
        if self.normalizer is None:
            y = y_raw * mask
        else:
            y1 = y_raw.unsqueeze(0)     # (1,C,H,W)
            m1 = mask.unsqueeze(0)      # (1,1,H,W)
            y1n = self.normalizer.transform(y1, m1)  # (1,C,H,W)
            y = y1n.squeeze(0)

        # ---- params ALWAYS available (from X in npz) ----
        params = torch.from_numpy(self.params[idx]).float()  # (P,)

        # ---- build x ----
        if self.inputs == "autoencoder":
            # AE uses normalized y as input
            x = y.clone()
            return {"x": x, "y": y, "mask": mask, "params": params, "idx": int(idx)}

        # params-mode input (mask + optional geom + broadcast params)
        H, W = mask.shape[-2:]
        channels = [mask]  # (1,H,W)

        if self.Rn is not None:
            channels.append(torch.from_numpy(self.Rn).unsqueeze(0).float())
            channels.append(torch.from_numpy(self.Zn).unsqueeze(0).float())

        P = int(params.numel())
        if P > 0:
            channels.append(params.view(P, 1, 1).expand(P, H, W))

        x = torch.cat(channels, dim=0)  # (Cin,H,W)
#        return {"x": x, "y": y, "mask": mask, "params": params, "idx": int(idx)}

        item = {
            "x": x, "y": y, "mask": mask,
            "params": torch.from_numpy(self.params[idx]).float(),
            "idx": int(idx),
        }
        return item
    

#        return {"x": x, "y": y, "mask": mask, "params": params}

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


class SelectYKey(Dataset):
    """
    Wraps a SOLPSDataset that returns dicts containing "y" and optionally exposes .y_keys.
    Slices output channels so training sees a single channel: (1,H,W).
    """
    def __init__(self, base_ds, y_key="Te"):
        self.base = base_ds
        self.y_key = y_key

        # Find channel index
        y_keys = getattr(base_ds, "y_keys", None)
        if y_keys is None:
            # fallback: assume Te is channel 0
            self.k = 0
            self._y_keys = np.array([y_key])
        else:
            y_keys_arr = np.array(y_keys).astype(str)
            matches = np.where(y_keys_arr == str(y_key))[0]
            if matches.size == 0:
                raise KeyError(f"y_key={y_key!r} not found in y_keys={list(y_keys_arr)}")
            self.k = int(matches[0])
            self._y_keys = np.array([str(y_key)])

        # keep convenient passthroughs if you use them elsewhere
        self.params = getattr(base_ds, "params", None)
        self.y_keys = self._y_keys

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        d = self.base[i]  # expects dict
        y = d["y"]        # shape (C,H,W) or (1,H,W)

        # Slice to (1,H,W)
        if y.ndim == 3:
            y = y[self.k:self.k+1, :, :]
        elif y.ndim == 4:
            # just in case some dataset returns (C,1,H,W) etc
            y = y[:, self.k:self.k+1, :, :]
        else:
            raise ValueError(f"Unexpected y shape: {tuple(y.shape)}")

        d = dict(d)
        d["y"] = y
        return d

def make_loaders(
    npz_path,
    inputs_mode="params",
    batch_size=16,
    split=0.85,
    seed=42,
    device="cuda" if torch.cuda.is_available() else "cpu",
    y_key="Te",
    y_keys=None,                 # NEW
    norm_fit_batch=16,
    geom_h5=None,
    norms_by_name=None,          # NEW (dict)
    num_workers=0,
):
    # raw dataset (no normalization) to compute split & fit normalizer
    raw_all = SOLPSDataset(
        npz_path,
        geom_h5=geom_h5 if inputs_mode != "autoencoder" else None,
        normalizer=None,
        inputs=inputs_mode,
        y_key=y_key,
        y_keys=y_keys,
    )
    N = len(raw_all)

    idx = np.arange(N)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    cut = int(split * N)
    idx_tr, idx_va = idx[:cut], idx[cut:]

    raw_train = Subset(raw_all, idx_tr)
    fit_loader = DataLoader(raw_train, batch_size=norm_fit_batch, shuffle=False, num_workers=num_workers)

    # -------- choose normalizer --------
    if y_keys is None:
        # single-channel legacy
        norm = MaskedLogStandardizer(eps=1.0)
        norm.fit(fit_loader)
    else:
        if norms_by_name is None:
            raise ValueError("y_keys provided but norms_by_name is None.")
        # MultiChannelNormalizer expects norms in the same order as y_keys
        norm = MultiChannelNormalizer(y_keys=y_keys, norms_by_name=norms_by_name)
        norm.fit(fit_loader)

    # datasets WITH normalization
    ds_tr = SOLPSDataset(
        npz_path,
        geom_h5=geom_h5 if inputs_mode != "autoencoder" else None,
        inputs=inputs_mode,
        y_key=y_key,
        y_keys=y_keys,
        normalizer=norm,
    )
    ds_va = SOLPSDataset(
        npz_path,
        geom_h5=geom_h5 if inputs_mode != "autoencoder" else None,
        inputs=inputs_mode,
        y_key=y_key,
        y_keys=y_keys,
        normalizer=norm,
    )

    # param scaling if in params mode
    param_mu = param_std = None
    if inputs_mode != "autoencoder":
        param_mu, param_std = fit_param_scaler(raw_all.params[idx_tr])
        apply_param_scaler(ds_tr, param_mu, param_std)
        apply_param_scaler(ds_va, param_mu, param_std)

    train_set = Subset(ds_tr, idx_tr)
    val_set   = Subset(ds_va, idx_va)

    use_cuda = (torch.device(device).type == "cuda")

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(4 if num_workers > 0 else None),
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(4 if num_workers > 0 else None),
    )


#    P = train_set.dataset.params.shape[1] if inputs_mode != "autoencoder" else 0
    P = train_set.dataset.params.shape[1] if hasattr(train_set.dataset, "params") else 0

    H, W = train_set[0]["mask"].shape[-2:]
    return train_loader, val_loader, norm, P, (H, W), (param_mu, param_std)


def _load_truth_and_params(npz_path, idx=0, y_keys_wanted=None):
    D = np.load(npz_path, allow_pickle=True)
    Y = D["Y"]          # (N,K,H,W)
    mask = D["mask"]    # (N,H,W)
    X = D["X"]          # (N,P)
    y_keys_all = list(D["y_keys"])

    if y_keys_wanted is None:
        y_keys_wanted = ["Te"]

    ks = [y_keys_all.index(k) for k in y_keys_wanted]
    Y_true = Y[idx, ks]           # (C,H,W)
    mask_ref = mask[idx]          # (H,W)
    params_raw = X[idx]           # (P,)
    return Y_true, mask_ref, params_raw, y_keys_wanted
