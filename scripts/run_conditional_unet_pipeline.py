import os
import shutil
import numpy as np
import torch

from solps_ai import data
from solps_ai.data import MaskedLogStandardizer, MaskedSymLogStandardizer
from solps_ai.train import train_unet
from solps_ai.utils import pick_device, sample_from_loader
from solps_ai.predict import load_checkpoint, predict_fields, scale_params


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


def _parse_custom_sweep(s):
    """
    Parse env string like:
      "base=32,lr=3e-4,batch=4; base=32,lr=1e-4,batch=4; base=16,lr=3e-4,batch=8"
    """
    out = []
    chunks = [c.strip() for c in s.split(";") if c.strip()]
    for i, ch in enumerate(chunks):
        kv = {}
        for tok in [t.strip() for t in ch.split(",") if t.strip()]:
            if "=" not in tok:
                raise ValueError(f"Bad SWEEP_TRIALS token {tok!r}, expected key=value.")
            k, v = tok.split("=", 1)
            kv[k.strip().lower()] = v.strip()
        try:
            b = int(kv["base"])
            lr = float(kv["lr"])
            bs = int(kv["batch"])
        except KeyError as e:
            raise ValueError(f"Missing key in SWEEP_TRIALS chunk {ch!r}: {e}") from e
        out.append({"tag": f"cust{i+1}_b{b}_lr{lr:g}_bs{bs}", "base": b, "lr": lr, "batch": bs})
    return out


def _build_sweep_trials(default_base, default_lr, default_batch, smoke_test):
    if smoke_test:
        return [{"tag": "smoke", "base": default_base, "lr": default_lr, "batch": default_batch}]

    custom = os.environ.get("SWEEP_TRIALS", "").strip()
    if custom:
        trials = _parse_custom_sweep(custom)
        if not trials:
            raise ValueError("SWEEP_TRIALS was provided but no valid trials were parsed.")
        return trials

    preset = os.environ.get("SWEEP_PRESET", "strong3").strip().lower()
    if preset == "strong3":
        return [
            {"tag": "b32_lr3e4_bs4", "base": 32, "lr": 3e-4, "batch": 4},
            {"tag": "b32_lr1e4_bs4", "base": 32, "lr": 1e-4, "batch": 4},
            {"tag": "b16_lr3e4_bs8", "base": 16, "lr": 3e-4, "batch": 8},
        ]
    if preset == "legacy3":
        return [
            {"tag": "b16_lr1e3_bs8", "base": 16, "lr": 1e-3, "batch": 8},
            {"tag": "b24_lr7e4_bs8", "base": 24, "lr": 7e-4, "batch": 8},
            {"tag": "b16_lr7e4_bs16", "base": 16, "lr": 7e-4, "batch": 16},
        ]
    if preset == "single_as_trial":
        return [{"tag": "single_cfg", "base": default_base, "lr": default_lr, "batch": default_batch}]
    raise ValueError(
        f"Unknown SWEEP_PRESET={preset!r}. Use one of: strong3, legacy3, single_as_trial; "
        "or set SWEEP_TRIALS."
    )


def masked_mse(pred, target, mask):
    if mask.shape[1] == 1 and pred.shape[1] != 1:
        mask = mask.expand_as(pred)
    err2 = (pred - target) ** 2
    return float((err2 * mask).sum().item() / max(mask.sum().item(), 1e-8))


def masked_mae(pred, target, mask):
    if mask.shape[1] == 1 and pred.shape[1] != 1:
        mask = mask.expand_as(pred)
    err = (pred - target).abs()
    return float((err * mask).sum().item() / max(mask.sum().item(), 1e-8))


def _assert_finite(name, value):
    arr = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    if not np.all(np.isfinite(arr)):
        raise RuntimeError(f"{name} has non-finite values.")


@torch.no_grad()
def eval_one(model, norm, sample, device):
    model.eval()
    x = sample["x"].unsqueeze(0).to(device)
    y = sample["y"].unsqueeze(0).to(device)
    m = sample["mask"].unsqueeze(0).to(device)
    p = model(x)
    mse_n = masked_mse(p, y, m)
    mae_n = masked_mae(p, y, m)

    y_ev = norm.inverse(y, m)
    p_ev = norm.inverse(p, m)
    mae_ev = masked_mae(p_ev, y_ev, m)
    return mse_n, mae_n, mae_ev


def run(smoke_test=False):
    npz_path = os.environ.get("NPZ_PATH", "/Users/42d/Downloads/solps_raster_dataset_new.npz")
    device = pick_device()
    print("Device:", device)
    if smoke_test:
        print("[SMOKE] conditional U-Net smoke test enabled.")

    y_keys_env = os.environ.get("Y_KEYS", "Te,Ti,ne,ni,ua,Sp,Qp,Qe,Qi,Sm")
    y_keys_req = [k.strip() for k in y_keys_env.split(",") if k.strip()]
    with np.load(npz_path, allow_pickle=True) as d:
        if "y_keys" in d.files:
            y_keys_avail = [str(k) for k in d["y_keys"]]
        else:
            y_keys_avail = ["Te"] if "Te" in d.files else []
    y_keys = [k for k in y_keys_req if k in y_keys_avail]
    dropped = [k for k in y_keys_req if k not in y_keys_avail]
    if dropped:
        print(f"[warn] dropping unavailable y_keys from dataset: {dropped}")
    if not y_keys:
        raise RuntimeError(
            f"No requested y_keys are available. requested={y_keys_req}, available={y_keys_avail}"
        )
    norms_bank = {
        "Te": MaskedLogStandardizer(eps=1e-2),
        "Ti": MaskedLogStandardizer(eps=1e-2),
        "ne": MaskedLogStandardizer(eps=1e16),
        "ni": MaskedLogStandardizer(eps=1e16),
        "ua": MaskedSymLogStandardizer(scale=5e3),
        "Sp": MaskedLogStandardizer(eps=1e10),
        "Qp": MaskedSymLogStandardizer(scale=1e2),
        "Qe": MaskedSymLogStandardizer(scale=1e2),
        "Qi": MaskedSymLogStandardizer(scale=1e2),
        "Sm": MaskedSymLogStandardizer(scale=1e-2),
    }
    missing = [k for k in y_keys if k not in norms_bank]
    if missing:
        raise KeyError(f"No normalizer config for y_keys: {missing}")
    norms_by_name = {k: norms_bank[k] for k in y_keys}

    epochs = _env_int("EPOCHS", 1 if smoke_test else 40)
    batch_size = _env_int("BATCH_SIZE", 2 if smoke_test else 8)
    base = _env_int("BASE", 16)
    lr = _env_float("LR", 1e-3)
    sweep_enabled = (os.environ.get("SWEEP", "0" if smoke_test else "1") == "1")
    early_stop_patience = _env_int("EARLY_STOP_PATIENCE", 3 if smoke_test else 8)
    early_stop_min_delta = _env_float("EARLY_STOP_MIN_DELTA", 1e-4)
    split = 0.85
    print(
        f"cfg: epochs={epochs} batch={batch_size} base={base} lr={lr:.2e} "
        f"sweep={sweep_enabled} early_stop_patience={early_stop_patience} y_keys={y_keys} "
        f"sweep_preset={os.environ.get('SWEEP_PRESET', 'strong3')}"
    )

    def make_data(bs):
        return data.make_loaders(
            npz_path=npz_path,
            inputs_mode="params",
            batch_size=bs,
            split=split,
            y_keys=y_keys,
            norms_by_name=norms_by_name,
            norm_fit_batch=4,
            num_workers=0,
            geom_h5=None,
        )

    def train_trial(tag, *, t_base, t_lr, t_batch):
        train_loader, val_loader, norm, Pdim, (H, W), param_scaler = make_data(t_batch)
        b0 = next(iter(train_loader))
        in_ch = b0["x"].shape[1]
        out_ch = b0["y"].shape[1]
        print(f"[trial {tag}] in_ch={in_ch} out_ch={out_ch} Pdim={Pdim} H={H} W={W}")
        trial_ckpt = f"outputs/cond_unet_{tag}.pt"
        model, hist = train_unet(
            train_loader=train_loader,
            val_loader=val_loader,
            norm=norm,
            in_ch=in_ch,
            out_ch=out_ch,
            device=device,
            inputs_mode="params",
            epochs=epochs,
            base=t_base,
            amp=False,
            lam_grad=0.1,
            lam_w=0.5,
            multiscale=0,
            grad_accum_steps=1,
            save_path=trial_ckpt,
            param_scaler=param_scaler,
            lr_init=t_lr,
            early_stop_patience=early_stop_patience,
            early_stop_min_delta=early_stop_min_delta,
        )
        best_val = float(np.min(np.asarray(hist.get("va_mse", [np.inf]), dtype=float)))
        print(f"[trial {tag}] best_val_mse={best_val:.6g} ckpt={trial_ckpt}")
        return {
            "tag": tag,
            "base": t_base,
            "lr": t_lr,
            "batch": t_batch,
            "ckpt": trial_ckpt,
            "best_val": best_val,
            "train_loader": train_loader,
            "val_loader": val_loader,
            "norm": norm,
            "model": model,
        }

    os.makedirs("outputs", exist_ok=True)
    final_ckpt = "outputs/cond_unet_smoke.pt" if smoke_test else "outputs/cond_unet.pt"

    if sweep_enabled:
        sweep_trials = _build_sweep_trials(base, lr, batch_size, smoke_test)
        print("[sweep] trials:")
        for t in sweep_trials:
            print(f"  - {t['tag']}: base={t['base']} lr={t['lr']:.2e} batch={t['batch']}")
        results = []
        for t in sweep_trials:
            results.append(train_trial(t["tag"], t_base=t["base"], t_lr=t["lr"], t_batch=t["batch"]))
        print("[sweep] results:")
        for r in sorted(results, key=lambda x: x["best_val"]):
            print(
                f"  - {r['tag']}: best_val={r['best_val']:.6g} "
                f"(base={r['base']} lr={r['lr']:.2e} batch={r['batch']})"
            )
        best = min(results, key=lambda r: r["best_val"])
        shutil.copyfile(best["ckpt"], final_ckpt)
        print(
            f"[sweep] selected {best['tag']} with best_val_mse={best['best_val']:.6g} "
            f"-> {final_ckpt}"
        )
        train_loader = best["train_loader"]
        val_loader = best["val_loader"]
        norm = best["norm"]
        model = best["model"]
    else:
        best = train_trial("single", t_base=base, t_lr=lr, t_batch=batch_size)
        shutil.copyfile(best["ckpt"], final_ckpt)
        print(f"Saved best single model to: {final_ckpt}")
        train_loader = best["train_loader"]
        val_loader = best["val_loader"]
        norm = best["norm"]
        model = best["model"]

    tr_idx, tr_s = sample_from_loader(train_loader, k=0)
    va_idx, va_s = sample_from_loader(val_loader, k=0)
    print("Train global idx:", tr_idx, "Val global idx:", va_idx)

    tr_mse_n, tr_mae_n, tr_mae_ev = eval_one(model, norm, tr_s, device)
    va_mse_n, va_mae_n, va_mae_ev = eval_one(model, norm, va_s, device)
    print(
        f"[CHECK] norm_mse(train/val)=({tr_mse_n:.4e}, {va_mse_n:.4e}) "
        f"norm_mae(train/val)=({tr_mae_n:.4e}, {va_mae_n:.4e}) "
        f"mae_eV(train/val)=({tr_mae_ev:.4e}, {va_mae_ev:.4e})"
    )

    for name, value in [
        ("tr_mse_n", tr_mse_n), ("va_mse_n", va_mse_n),
        ("tr_mae_n", tr_mae_n), ("va_mae_n", va_mae_n),
        ("tr_mae_ev", tr_mae_ev), ("va_mae_ev", va_mae_ev),
    ]:
        _assert_finite(name, value)

    # Deployment-style check: load checkpoint and predict from RAW params + mask.
    model2, norm2, (p_mu, p_std) = load_checkpoint(final_ckpt, device)
    mask_ref = va_s["mask"].squeeze(0).cpu().numpy().astype(np.float32)
    p_scaled = va_s["params"].cpu().numpy().astype(np.float32)
    if p_mu is not None and p_std is not None:
        p_raw = p_scaled * np.asarray(p_std, dtype=np.float32) + np.asarray(p_mu, dtype=np.float32)
    else:
        p_raw = p_scaled
    p_in = scale_params(p_raw, p_mu, p_std)
    y_pred = predict_fields(model2, norm2, mask_ref, p_in, device=device, as_numpy=True)
    _assert_finite("deploy_y_pred", y_pred)
    print(f"[CHECK] deploy inference from raw params: PASS (shape={tuple(y_pred.shape)})")

    if smoke_test:
        if va_mse_n > 5.0:
            raise RuntimeError(f"[SMOKE] validation mse too high: {va_mse_n:.3e}")
        print("[SMOKE] PASS")


if __name__ == "__main__":
    smoke = os.environ.get("SMOKE_TEST", "0") == "1"
    run(smoke_test=smoke)
