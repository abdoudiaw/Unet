import json
import os
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from solps_ai.data import (
    MaskedLogStandardizer,
    MaskedSymLogStandardizer,
    MultiChannelNormalizer,
    fit_param_scaler,
)
from solps_ai.train import train_unet
from solps_ai.utils import pick_device, sample_from_loader


def _env_int(name, default):
    v = os.environ.get(name, "")
    if v is None:
        return int(default)
    v = str(v).strip()
    return int(v) if v else int(default)


def _env_float(name, default):
    v = os.environ.get(name, "")
    if v is None:
        return float(default)
    v = str(v).strip()
    return float(v) if v else float(default)


def split_indices(N, split=0.85, seed=42):
    idx = np.arange(N)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    cut = int(split * N)
    return idx[:cut], idx[cut:]


def _norm_factory():
    # Keep this consistent with your all-field training setup.
    return {
        "Te": lambda: MaskedLogStandardizer(eps=1e-2),
        "Ti": lambda: MaskedLogStandardizer(eps=1e-2),
        "ne": lambda: MaskedLogStandardizer(eps=1e16),
        "ni": lambda: MaskedLogStandardizer(eps=1e16),
        "ua": lambda: MaskedSymLogStandardizer(scale=5e3),
        "Sp": lambda: MaskedLogStandardizer(eps=1e10),
        "Qp": lambda: MaskedSymLogStandardizer(scale=1e2),
        "Qe": lambda: MaskedSymLogStandardizer(scale=1e2),
        "Qi": lambda: MaskedSymLogStandardizer(scale=1e2),
        "Sm": lambda: MaskedSymLogStandardizer(scale=1e-2),
    }


class _FitDataset(Dataset):
    def __init__(self, Y, M):
        self.Y = torch.from_numpy(Y).float()
        self.M = torch.from_numpy(M).float()

    def __len__(self):
        return int(self.Y.shape[0])

    def __getitem__(self, idx):
        return {
            "y": self.Y[idx],                              # (C,H,W)
            "mask": self.M[idx].unsqueeze(0),             # (1,H,W)
        }


class PlasmaToSourceDataset(Dataset):
    def __init__(
        self,
        Y_in,
        Y_out,
        M,
        P,
        x_norm,
        y_norm,
        include_params=True,
        p_mu=None,
        p_std=None,
    ):
        self.Y_in = Y_in.astype(np.float32)
        self.Y_out = Y_out.astype(np.float32)
        self.M = M.astype(np.float32)
        self.P = P.astype(np.float32)
        self.x_norm = x_norm
        self.y_norm = y_norm
        self.include_params = bool(include_params)
        self.p_mu = p_mu
        self.p_std = p_std

    def __len__(self):
        return int(self.Y_in.shape[0])

    def __getitem__(self, idx):
        yin = torch.from_numpy(self.Y_in[idx]).float()                # (Cin,H,W)
        yout = torch.from_numpy(self.Y_out[idx]).float()              # (Cout,H,W)
        m = torch.from_numpy(self.M[idx]).unsqueeze(0).float()        # (1,H,W)
        m = (m > 0.5).float()

        yin_b = yin.unsqueeze(0)
        yout_b = yout.unsqueeze(0)
        m_b = m.unsqueeze(0)

        xin_n = self.x_norm.transform(yin_b, m_b).squeeze(0) if self.x_norm is not None else yin * m
        yout_n = self.y_norm.transform(yout_b, m_b).squeeze(0) if self.y_norm is not None else yout * m

        channels = [m, xin_n]
        p = torch.from_numpy(self.P[idx]).float()
        if self.include_params and p.numel() > 0:
            H, W = m.shape[-2:]
            channels.append(p.view(-1, 1, 1).expand(-1, H, W))
        x = torch.cat(channels, dim=0)
        return {"x": x, "y": yout_n, "mask": m, "params": p, "idx": int(idx)}


def load_npz(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    if "Y" not in d.files or "y_keys" not in d.files:
        raise KeyError("Dataset must contain Y and y_keys for plasma->sources training.")
    Y = d["Y"].astype(np.float32)
    y_keys = [str(k) for k in d["y_keys"]]
    if "mask" in d.files:
        M = d["mask"]
        if M.ndim == 2:
            M = np.repeat(M[None, :, :], Y.shape[0], axis=0)
        M = (M > 0.5).astype(np.float32)
    else:
        M = np.ones((Y.shape[0], Y.shape[2], Y.shape[3]), dtype=np.float32)
    if "params" in d.files:
        P = d["params"].astype(np.float32)
    elif "X" in d.files:
        P = d["X"].astype(np.float32)
    else:
        P = np.zeros((Y.shape[0], 0), dtype=np.float32)
    if "param_keys" in d.files:
        p_keys = [str(k) for k in d["param_keys"]]
    elif "x_keys" in d.files:
        p_keys = [str(k) for k in d["x_keys"]]
    else:
        p_keys = [f"p{i}" for i in range(P.shape[1])]
    return Y, y_keys, M, P, p_keys


def select_channels(Y, y_keys_all, keys_wanted):
    misses = [k for k in keys_wanted if k not in y_keys_all]
    if misses:
        raise KeyError(f"Requested keys not found in dataset: {misses}; available={y_keys_all}")
    idx = [y_keys_all.index(k) for k in keys_wanted]
    return Y[:, idx], idx


def _pack_multi_norm(norm):
    pack = {"kind": norm.__class__.__name__, "y_keys": list(norm.y_keys), "norms": {}}
    for name, n in norm.norms.items():
        rec = {"kind": n.__class__.__name__}
        if hasattr(n, "eps"):
            rec["eps"] = float(n.eps)
        if hasattr(n, "scale"):
            rec["scale"] = float(n.scale)
        if getattr(n, "mu", None) is not None:
            rec["mu"] = float(n.mu.detach().cpu().item())
        if getattr(n, "sigma", None) is not None:
            rec["sigma"] = float(n.sigma.detach().cpu().item())
        pack["norms"][name] = rec
    return pack


def masked_mae(pred, target, mask):
    if mask.shape[1] == 1 and pred.shape[1] != 1:
        mask = mask.expand_as(pred)
    err = (pred - target).abs()
    return float((err * mask).sum().item() / max(mask.sum().item(), 1e-8))


@torch.no_grad()
def eval_one(model, y_norm, sample, device):
    x = sample["x"].unsqueeze(0).to(device)
    y = sample["y"].unsqueeze(0).to(device)
    m = sample["mask"].unsqueeze(0).to(device)
    p = model(x)
    y_phys = y_norm.inverse(y, m)
    p_phys = y_norm.inverse(p, m)
    mae_n = masked_mae(p, y, m)
    mae_phys = masked_mae(p_phys, y_phys, m)
    return mae_n, mae_phys


def run():
    npz_path = os.environ.get("NPZ_PATH", "data/solps_native_all_qc.npz")
    in_keys_req = [k.strip() for k in os.environ.get("IN_KEYS", "Te,Ti,ne,ni,ua").split(",") if k.strip()]
    out_keys_req = [k.strip() for k in os.environ.get("OUT_KEYS", "Qp,Sp,Qe,Qi,Sm").split(",") if k.strip()]
    include_params = os.environ.get("INCLUDE_PARAMS", "1") == "1"

    epochs = _env_int("EPOCHS", 120)
    batch_size = _env_int("BATCH_SIZE", 4)
    base = _env_int("BASE", 32)
    lr = _env_float("LR", 3e-4)
    split = _env_float("SPLIT", 0.85)
    seed = _env_int("SEED", 42)
    early_stop_patience = _env_int("EARLY_STOP_PATIENCE", 25)
    early_stop_min_delta = _env_float("EARLY_STOP_MIN_DELTA", 1e-4)

    device = pick_device()
    print("Device:", device)
    with np.load(npz_path, allow_pickle=True) as dchk:
        y_keys_avail = [str(k) for k in dchk["y_keys"]] if "y_keys" in dchk.files else (["Te"] if "Te" in dchk.files else [])
    in_keys = [k for k in in_keys_req if k in y_keys_avail]
    out_keys = [k for k in out_keys_req if k in y_keys_avail]
    drop_in = [k for k in in_keys_req if k not in y_keys_avail]
    drop_out = [k for k in out_keys_req if k not in y_keys_avail]
    if drop_in:
        print(f"[warn] dropping unavailable IN_KEYS: {drop_in}")
    if drop_out:
        print(f"[warn] dropping unavailable OUT_KEYS: {drop_out}")
    if not in_keys:
        raise RuntimeError(f"No valid IN_KEYS left. requested={in_keys_req}, available={y_keys_avail}")
    if not out_keys:
        raise RuntimeError(f"No valid OUT_KEYS left. requested={out_keys_req}, available={y_keys_avail}")

    print(
        f"cfg: in_keys={in_keys} out_keys={out_keys} include_params={include_params} "
        f"epochs={epochs} batch={batch_size} base={base} lr={lr:.2e}"
    )

    Y, y_keys_all, M, P_raw, p_keys = load_npz(npz_path)
    Y_in, _ = select_channels(Y, y_keys_all, in_keys)
    Y_out, _ = select_channels(Y, y_keys_all, out_keys)

    idx_tr, idx_va = split_indices(Y.shape[0], split=split, seed=seed)
    if include_params and P_raw.shape[1] > 0:
        p_mu, p_std = fit_param_scaler(P_raw[idx_tr])
        P = (P_raw - p_mu) / p_std
    else:
        p_mu, p_std = None, None
        P = np.zeros_like(P_raw) if not include_params else P_raw

    nf = _norm_factory()
    miss_in = [k for k in in_keys if k not in nf]
    miss_out = [k for k in out_keys if k not in nf]
    if miss_in or miss_out:
        raise KeyError(f"Missing normalizer config in_keys={miss_in} out_keys={miss_out}")

    x_norm = MultiChannelNormalizer(y_keys=in_keys, norms_by_name={k: nf[k]() for k in in_keys})
    y_norm = MultiChannelNormalizer(y_keys=out_keys, norms_by_name={k: nf[k]() for k in out_keys})
    fit_loader_x = DataLoader(_FitDataset(Y_in[idx_tr], M[idx_tr]), batch_size=4, shuffle=True, num_workers=0)
    fit_loader_y = DataLoader(_FitDataset(Y_out[idx_tr], M[idx_tr]), batch_size=4, shuffle=True, num_workers=0)
    x_norm.fit(fit_loader_x)
    y_norm.fit(fit_loader_y)

    ds = PlasmaToSourceDataset(
        Y_in=Y_in,
        Y_out=Y_out,
        M=M,
        P=P,
        x_norm=x_norm,
        y_norm=y_norm,
        include_params=include_params,
        p_mu=p_mu,
        p_std=p_std,
    )
    tr_ds = Subset(ds, idx_tr.tolist())
    va_ds = Subset(ds, idx_va.tolist())
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    b0 = next(iter(tr_loader))
    in_ch = int(b0["x"].shape[1])
    out_ch = int(b0["y"].shape[1])
    print(f"shapes: in_ch={in_ch} out_ch={out_ch} H={b0['x'].shape[-2]} W={b0['x'].shape[-1]}")

    os.makedirs("outputs", exist_ok=True)
    trial_ckpt = "outputs/source_from_plasma_tmp.pt"
    final_ckpt = "outputs/source_from_plasma.pt"
    model, hist = train_unet(
        train_loader=tr_loader,
        val_loader=va_loader,
        norm=y_norm,
        in_ch=in_ch,
        out_ch=out_ch,
        device=device,
        inputs_mode="plasma_plus_params_to_sources",
        epochs=epochs,
        base=base,
        amp=False,
        lam_grad=0.1,
        lam_w=0.5,
        multiscale=0,
        grad_accum_steps=1,
        save_path=trial_ckpt,
        param_scaler=(p_mu, p_std),
        lr_init=lr,
        early_stop_patience=early_stop_patience,
        early_stop_min_delta=early_stop_min_delta,
    )
    shutil.copyfile(trial_ckpt, final_ckpt)

    # PyTorch>=2.6 defaults to weights_only=True; we need full checkpoint dict here.
    ck = torch.load(final_ckpt, map_location="cpu", weights_only=False)
    ck["x_norm"] = _pack_multi_norm(x_norm)
    ck["input_keys"] = list(in_keys)
    ck["output_keys"] = list(out_keys)
    ck["include_params"] = bool(include_params)
    ck["param_keys"] = list(p_keys)
    torch.save(ck, final_ckpt)

    tr_idx, tr_s = sample_from_loader(tr_loader, k=0)
    va_idx, va_s = sample_from_loader(va_loader, k=0)
    tr_mae_n, tr_mae_phys = eval_one(model, y_norm, tr_s, device)
    va_mae_n, va_mae_phys = eval_one(model, y_norm, va_s, device)
    best_val = float(np.min(np.asarray(hist.get("va_mse", [np.inf]), dtype=float)))
    print(
        f"[CHECK] best_val={best_val:.4e} "
        f"mae_norm(train/val)=({tr_mae_n:.4e}, {va_mae_n:.4e}) "
        f"mae_phys(train/val)=({tr_mae_phys:.4e}, {va_mae_phys:.4e})"
    )
    print(f"Train global idx: {tr_idx} Val global idx: {va_idx}")
    print(f"Saved: {final_ckpt}")

    meta = {
        "npz_path": npz_path,
        "input_keys": in_keys,
        "output_keys": out_keys,
        "include_params": include_params,
        "split": split,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "base": base,
        "lr": lr,
        "best_val_mse_norm": best_val,
    }
    with open("outputs/source_from_plasma_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved: outputs/source_from_plasma_meta.json")


if __name__ == "__main__":
    run()
