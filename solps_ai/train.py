import numpy as np
import torch
import os 
from .losses import masked_weighted_loss, mae_norm, batch_error_sums_ev, edge_weights


def train_unet(
    train_loader, val_loader, norm, in_ch, device, *,
    out_ch: int | None = None,
    inputs_mode="params",
    lam_grad=0.2, lam_w=1.0, lam_ev=0.0,
    epochs=10, base=32, param_scaler=None, model_cls=None,
    save_path="unet_best.pt",
    return_history: bool = True,
    history_path=None,
    multiscale=1,
    grad_base="l1",
    ms_weight=0.5,
    resume_path: str | None = None,
    strict: bool = True,

    # ---- LR scheduler knobs ----
    lr_init: float = 1e-3, lr_factor: float = 0.5, lr_patience: int = 5, lr_min: float = 1e-5,
    lr_threshold: float = 1e-4,

    # ---- NEW: GPU scaling knobs ----
    amp: bool = True,                         # enable AMP on CUDA
    grad_accum_steps: int = 1,                # >1 to emulate larger batch without OOM
    non_blocking: bool = True,                # async H2D copies when pin_memory=True in DataLoader
    clip_grad: float = 1.0,                   # grad clipping (None/0 to disable)
    channels_last: bool = True,               # better conv perf on many NVIDIA GPUs
    cudnn_benchmark: bool = True,             # speed up if input sizes are constant
    log_every: int = 1,                       # print every N epochs
    early_stop_patience: int | None = None,  # stop if no val improvement for N epochs
    early_stop_min_delta: float = 0.0,       # minimum val improvement to reset patience
    lam_grad_warmup_start: int = 20,         # epoch to start ramping grad loss
    lam_grad_warmup_end: int = 60,           # epoch to reach full lam_grad
    channel_weights=None,                    # optional list/array shape (C,)
):
    """
    param_scaler: (mu, std) or None, to store for inference
    model_cls: a constructor like lambda in_ch,out_ch: UNet(in_ch, out_ch, base)
    """
    # --- device flags ---
    device = torch.device(device)
    use_cuda = (device.type == "cuda")

    if use_cuda and cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    if out_ch is None:
        out_ch = 1


    if model_cls is None:
        from .models import UNet
        model_cls = lambda in_ch, out_ch: UNet(in_ch=in_ch, out_ch=out_ch, base=base)

    model = model_cls(in_ch, out_ch).to(device)

    cw_t = None
    if channel_weights is not None:
        cw_arr = np.asarray(channel_weights, dtype=np.float32).reshape(-1)
        if cw_arr.size != int(out_ch):
            raise ValueError(f"channel_weights length={cw_arr.size} but out_ch={out_ch}")
        cw_t = torch.from_numpy(cw_arr).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr_init)
    # ---- ReduceLROnPlateau ----
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=lr_factor,
        patience=lr_patience,
        threshold=lr_threshold,
        threshold_mode="rel",
        cooldown=0,
        min_lr=lr_min,
        eps=1e-8,
    )
    
    use_amp = bool(amp and use_cuda)

    if use_amp and getattr(torch.cuda, "is_bf16_supported", lambda: False)():
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 0
    best_val = float("inf")
    
    if resume_path is not None and os.path.exists(resume_path):
            ckpt = torch.load(resume_path, map_location=device)

            model.load_state_dict(ckpt["model"], strict=strict)

            if "opt" in ckpt:
                opt.load_state_dict(ckpt["opt"])
            if "sched" in ckpt:
                scheduler.load_state_dict(ckpt["sched"])
            if "scaler" in ckpt and use_amp:
                scaler.load_state_dict(ckpt["scaler"])

            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_val = float(ckpt.get("best_val", best_val))

            print(f"[resume] loaded {resume_path} @ epoch={start_epoch} best_val={best_val:.6g}")
            
        

    # optional memory format for faster conv
    if use_cuda and channels_last:
        model = model.to(memory_format=torch.channels_last)

    # ---- weight map from first sample ----
    m0 = train_loader.dataset[0]["mask"]  # (1,H,W) CPU
    w_map = edge_weights(m0.numpy()[0], sigma_px=3.0)
    w_map = torch.from_numpy(w_map).to(device).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    # history containers
    history = {
        "tr_mse": [], "tr_maeN": [],
        "va_mse": [], "va_maeN": [],
        "va_mae_eV": [], "va_rmse_eV": [],
        "lr": [],
    }

    # ---------- TRAIN LOOP ----------
    no_improve = 0
    for epoch in range(start_epoch, start_epoch + epochs):

        # ---------------- train ----------------
        model.train()
        tr_loss_sum = tr_maeN_sum = tr_px = 0.0

        opt.zero_grad(set_to_none=True)
        accum = 0

        # Smooth ramp for gradient loss to avoid abrupt regime shift.
        if lam_grad == 0.0:
            lam_grad_epoch = 0.0
        elif epoch < int(lam_grad_warmup_start):
            lam_grad_epoch = 0.0
        elif epoch >= int(lam_grad_warmup_end):
            lam_grad_epoch = lam_grad
        else:
            span = max(int(lam_grad_warmup_end) - int(lam_grad_warmup_start), 1)
            alpha = float(epoch - int(lam_grad_warmup_start)) / float(span)
            alpha = min(max(alpha, 0.0), 1.0)
            lam_grad_epoch = float(lam_grad) * alpha


        for b in train_loader:
            # move to device (non_blocking works only if DataLoader pin_memory=True)
            x = b["x"].to(device, non_blocking=non_blocking)
            y = b["y"].to(device, non_blocking=non_blocking)
            m = b["mask"].to(device, non_blocking=non_blocking)

            # for channels_last performance
            if use_cuda and channels_last:
                x = x.contiguous(memory_format=torch.channels_last)

            w_map = w_map.to(device)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp, dtype=amp_dtype):

                p = model(x)
                loss_base = masked_weighted_loss(
                    p, y, m,
                    w=w_map.to(p.dtype),
                    lam_grad=lam_grad_epoch,
                    lam_w=lam_w,
                    base="huber",
                    huber_beta=0.05,
                    grad_mode="vector",
                    grad_base=grad_base,
                    multiscale=multiscale,
                    ms_weight=ms_weight,
                    grad_ds=2,   # <<<<<<<<<<<< big speed-up
                    channel_weights=cw_t,
                )


                if lam_ev > 0:
                    # do inverse in float32 for stability
                    p_ev = norm.inverse(p.float(), m.float())
                    y_ev = norm.inverse(y.float(), m.float())
                    loss_ev = ((p_ev - y_ev).abs() * m.float()).sum() / m.sum().clamp_min(1e-8)
                    loss = loss_base + lam_ev * loss_ev
                else:
                    loss = loss_base

                # gradient accumulation (normalize loss)
                if grad_accum_steps > 1:
                    loss = loss / grad_accum_steps

            scaler.scale(loss).backward()
            accum += 1

            # bookkeeping (use un-divided loss for reporting)
            with torch.no_grad():
                px = float(m.sum().item())
                # multiply back if we divided above
                loss_item = float(loss.item()) * (grad_accum_steps if grad_accum_steps > 1 else 1.0)
                tr_loss_sum += loss_item * px
                tr_maeN_sum += float(mae_norm(p.float(), y.float(), m.float()).item()) * px
                tr_px       += px

            # step optimizer when we've accumulated enough
            if accum >= grad_accum_steps:
                if clip_grad and clip_grad > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_grad))
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                accum = 0

        # if epoch ended mid-accum, flush remaining grads
        if accum > 0:
            if clip_grad and clip_grad > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_grad))
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

        tr_loss = tr_loss_sum / max(tr_px, 1.0)
        tr_maeN = tr_maeN_sum / max(tr_px, 1.0)

        # ---------------- validate ----------------
        model.eval()
        va_loss_sum = va_maeN_sum = va_px = 0.0
        va_abs_ev_sum = va_sq_ev_sum = 0.0

        with torch.no_grad():
            for b in val_loader:
                x = b["x"].to(device, non_blocking=non_blocking)
                y = b["y"].to(device, non_blocking=non_blocking)
                m = b["mask"].to(device, non_blocking=non_blocking)

                if use_cuda and channels_last:
                    x = x.contiguous(memory_format=torch.channels_last)

                with torch.amp.autocast(device_type="cuda", enabled=use_amp, dtype=amp_dtype):
                    p = model(x)
                    loss_b = masked_weighted_loss(
                        p, y, m,
                        w=w_map.to(p.dtype),
                        lam_grad=lam_grad_epoch,
                        lam_w=lam_w,
                        base="huber",
                        huber_beta=0.05,
                        grad_mode="vector",
                        grad_base=grad_base,
                        multiscale=multiscale,     # try 0 first, then 1
                        ms_weight=ms_weight,
                        channel_weights=cw_t,
                    )

                px = float(m.sum().item())
                va_loss_sum += float(loss_b.item()) * px
                va_maeN_sum += float(mae_norm(p.float(), y.float(), m.float()).item()) * px
                va_px       += px

                # abs_sum, sq_sum, _ = batch_error_sums_ev(p.float(), y.float(), m.float(), norm)
                # va_abs_ev_sum += float(abs_sum.item())
                # va_sq_ev_sum  += float(sq_sum.item())



                is_multi = hasattr(norm, "y_keys") and hasattr(norm, "norms")

                # inside validation loop:
                if not is_multi and lam_ev >= 0:   # only meaningful for single Te-like channel
                    abs_sum, sq_sum, _ = batch_error_sums_ev(p.float(), y.float(), m.float(), norm)
                    va_abs_ev_sum += float(abs_sum.item())
                    va_sq_ev_sum  += float(sq_sum.item())


        va_loss    = va_loss_sum / max(va_px, 1.0)
        va_maeN    = va_maeN_sum / max(va_px, 1.0)
        va_MAE_eV  = va_abs_ev_sum / max(va_px, 1.0)
        va_RMSE_eV = (va_sq_ev_sum / max(va_px, 1.0)) ** 0.5

        # ---- record history ----
        history["tr_mse"].append(tr_loss)
        history["tr_maeN"].append(tr_maeN)
        history["va_mse"].append(va_loss)
        history["va_maeN"].append(va_maeN)
        history["va_mae_eV"].append(va_MAE_eV)
        history["va_rmse_eV"].append(va_RMSE_eV)
        history["lr"].append(opt.param_groups[0]["lr"])

        # ---- log & schedule ----
        lr_before = opt.param_groups[0]["lr"]
        if (epoch % max(int(log_every), 1)) == 0:

            print(
                f"Epoch {epoch:03d} | train {tr_loss:.4f} | val {va_loss:.4f} "
                f"| lr {lr_before:.2e} | lam_grad_eff {lam_grad_epoch:.3g}"
            )


        scheduler.step(va_loss)
        lr_after = opt.param_groups[0]["lr"]
        if lr_after < lr_before and (epoch % max(int(log_every), 1)) == 0:
            print(f"  ↳ ReduceLROnPlateau: lr {lr_before:.2e} → {lr_after:.2e}")

        # ---- checkpoint best ----
        improved = (best_val - va_loss) > float(early_stop_min_delta)
        if improved:
            best_val = va_loss
            no_improve = 0

            mu, std = (None, None)
            if param_scaler is not None:
                mu, std = param_scaler

            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "sched": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_val": float(best_val),
                "norm": _norm_to_ckpt(norm),
                "in_ch": in_ch, "out_ch": out_ch, "base": base,
                "inputs_mode": inputs_mode,
                "param_mu": mu,
                "param_std": std,
                "param_scaler": (mu, std),
                "channel_weights": None if cw_t is None else cw_t.detach().cpu().numpy().tolist(),
                "history": history,
            }
            torch.save(ckpt, save_path)
        else:
            no_improve += 1

        if early_stop_patience is not None and no_improve >= int(early_stop_patience):
            print(f"[early-stop] no val improvement for {no_improve} epochs; stopping at epoch {epoch:03d}")
            break

    # optional persist of history
    if history_path is not None:
        np.savez(history_path, **history)

    return (model, history) if return_history else model

def _norm_to_ckpt(norm):
    # Single-channel normalizers
    if all(hasattr(norm, a) for a in ("mu", "sigma", "eps")):
        pack = {"kind": norm.__class__.__name__,
                "mu": norm.mu, "sigma": norm.sigma, "eps": float(norm.eps)}
        if hasattr(norm, "scale"):
            pack["scale"] = float(norm.scale)
        return pack
    # Multi-channel normalizer: assume it has y_keys + norms dict
    if hasattr(norm, "y_keys") and hasattr(norm, "norms"):
        pack = {"kind": norm.__class__.__name__, "y_keys": list(norm.y_keys), "norms": {}}
        for k, n in norm.norms.items():
            pack["norms"][k] = _norm_to_ckpt(n)
        return pack
    # Fallback
    return {"kind": norm.__class__.__name__}
