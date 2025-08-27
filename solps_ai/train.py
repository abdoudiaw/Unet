import numpy as np
import torch
from .losses import masked_weighted_loss, mae_norm, batch_error_sums_ev, edge_weights

def train_unet(
    train_loader, val_loader, norm, in_ch, device, *,
    inputs_mode="params", lam_grad=0.2, lam_w=1.0, lam_ev=0.0,
    epochs=10, base=32, param_scaler=None, model_cls=None,
    save_path="unet_best.pt", return_history: bool = True, history_path: str | None = None,
    # ---- LR scheduler knobs (tweak as needed) ----
    lr_init: float = 1e-3, lr_factor: float = 0.5, lr_patience: int = 5, lr_min: float = 1e-5,
    lr_threshold: float = 1e-4
):
    """
    param_scaler: (mu, std) or None, to store for inference
    model_cls: a constructor like lambda in_ch,out_ch: UNet(in_ch, out_ch, base)
    """
    if model_cls is None:
        from .models import UNet
        model_cls = lambda in_ch, out_ch: UNet(in_ch=in_ch, out_ch=1, base=base)

    model = model_cls(in_ch, 1).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr_init)

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


    # AMP
    use_cuda  = (str(device) == "cuda")
    use_amp   = use_cuda
    amp_dtype = torch.bfloat16 if (use_cuda and getattr(torch.cuda, "is_bf16_supported", lambda: False)()) else torch.float16
    scaler    = torch.amp.GradScaler('cuda', enabled=use_amp)

    # weight map from first sample
    m0 = train_loader.dataset[0]["mask"]  # (1,H,W) CPU
    w_map = edge_weights(m0.numpy()[0], sigma_px=3.0)
    w_map = torch.from_numpy(w_map).to(device).unsqueeze(0).unsqueeze(0)

    # history containers
    history = {
        "tr_mse": [], "tr_maeN": [],
        "va_mse": [], "va_maeN": [],
        "va_mae_eV": [], "va_rmse_eV": []
    }

    best_val = float("inf")
    for epoch in range(epochs):
        # ---------------- train ----------------
        model.train()
        tr_loss_sum = tr_maeN_sum = tr_px = 0.0
        for b in train_loader:
            x, y, m = b["x"].to(device), b["y"].to(device), b["mask"].to(device)
            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                p = model(x)
                # ensure dtype match for weighted loss under autocast
                loss_base = masked_weighted_loss(p, y, m, w=w_map.to(p.dtype), lam_grad=lam_grad, lam_w=lam_w)

                if lam_ev > 0:
                    p_ev = norm.inverse(p.float(), m.float())
                    y_ev = norm.inverse(y.float(), m.float())
                    loss_ev = ((p_ev - y_ev).abs() * m.float()).sum() / m.sum().clamp_min(1e-8)
                    loss = loss_base + lam_ev * loss_ev
                else:
                    loss = loss_base

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            with torch.no_grad():
                px = float(m.sum().item())
                tr_loss_sum += float(loss.item()) * px
                tr_maeN_sum += float(mae_norm(p.float(), y.float(), m.float()).item()) * px
                tr_px       += px

        tr_loss = tr_loss_sum / max(tr_px, 1.0)
        tr_maeN = tr_maeN_sum / max(tr_px, 1.0)

        # ---------------- validate ----------------
        model.eval()
        va_loss_sum = va_maeN_sum = va_px = 0.0
        va_abs_ev_sum = va_sq_ev_sum = 0.0
        with torch.no_grad():
            for b in val_loader:
                x, y, m = b["x"].to(device), b["y"].to(device), b["mask"].to(device)
                with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                    p = model(x)
                    loss_b = masked_weighted_loss(p, y, m, w=w_map.to(p.dtype), lam_grad=lam_grad, lam_w=lam_w)

                px = float(m.sum().item())
                va_loss_sum += float(loss_b.item()) * px
                va_maeN_sum += float(mae_norm(p.float(), y.float(), m.float()).item()) * px
                va_px       += px

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

        # ---- log & schedule ----
        lr_before = opt.param_groups[0]["lr"]
        print(
            f"epoch {epoch:02d} | "
            f"train MSE {tr_loss:.4f}  MAE_norm {tr_maeN:.4f} | "
            f"val MSE {va_loss:.4f}  MAE_norm {va_maeN:.4f} | "
            f"val MAE {va_MAE_eV:.2f} eV  RMSE {va_RMSE_eV:.2f} eV | "
            f"lr {lr_before:.2e}"
        )

        # step the LR scheduler on validation loss
        scheduler.step(va_loss)
        lr_after = opt.param_groups[0]["lr"]
        if lr_after < lr_before:
            print(f"  ↳ ReduceLROnPlateau: lr {lr_before:.2e} → {lr_after:.2e}")

        # ---- checkpoint best ----
        if va_loss < best_val:
            best_val = va_loss
            ckpt = {
                "model": model.state_dict(),
                "norm": (float(norm.mu), float(norm.sigma), float(norm.eps)),
                "inputs_mode": inputs_mode,
                "in_ch": in_ch,
            }
            if param_scaler is not None:
                mu, std = param_scaler
                if mu is not None:
                    ckpt["param_mu"]  = mu.tolist() if hasattr(mu, "tolist") else mu
                    ckpt["param_std"] = std.tolist() if hasattr(std, "tolist") else std
            torch.save(ckpt, save_path)

    # optional persist of history
    if history_path is not None:
        np.savez(history_path, **history)

    return (model, history) if return_history else model

