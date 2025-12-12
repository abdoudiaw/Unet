#from typing import Optional
#import numpy as np
#import torch
#from .losses import masked_weighted_loss, mae_norm, batch_error_sums_ev, edge_weights
#
## def train_unet(train_loader, val_loader, norm, in_ch, device,
##                epochs=20, base=32, inputs_mode="params",
##                lam_grad=0.2, lam_w=1.0, lam_ev=0.0,
##                param_scaler=None, save_path=None,
##                return_history=False, history_path=None,
##                amp=True, grad_accum=1, clip_grad=None):
#
##     model = UNet(in_ch=in_ch, base=base, ...).to(device)
##     optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
##     scaler = torch.cuda.amp.GradScaler(enabled=amp)   # <- create ONCE
#
##     criterion = make_loss(lam_w=lam_w, lam_grad=lam_grad, lam_ev=lam_ev, norm=norm)
#
##     best_val = float("inf")
##     for epoch in range(epochs):
##         # -------- train --------
##         model.train()
##         optimizer.zero_grad(set_to_none=True)
##         for i, (x, y) in enumerate(train_loader):
##             x = x.to(device, non_blocking=True)
##             y = y.to(device, non_blocking=True)
#
##             with torch.cuda.amp.autocast(enabled=amp):   # <- forward/loss in autocast
##                 y_pred = model(x)
##                 loss = criterion(y_pred, y) / grad_accum
#
##             scaler.scale(loss).backward()
#
##             if (i + 1) % grad_accum == 0:
##                 if clip_grad is not None:
##                     scaler.unscale_(optimizer)  # required before clipping
##                     torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
##                 scaler.step(optimizer)
##                 scaler.update()
##                 optimizer.zero_grad(set_to_none=True)
#
##         # -------- validate --------
##         model.eval()
##         val_loss = 0.0
##         with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp):
##             for x, y in val_loader:
##                 x = x.to(device, non_blocking=True)
##                 y = y.to(device, non_blocking=True)
##                 y_pred = model(x)
##                 val_loss += criterion(y_pred, y).item()
#
##         val_loss /= max(1, len(val_loader))
##         if val_loss < best_val and save_path:
##             best_val = val_loss
##             torch.save(model.state_dict(), save_path)
#
##         # (optional) log history here...
#
##     return (model, None) if not return_history else (model, {"best_val": best_val})
#
#
## def train_unet(
##     train_loader, val_loader, norm, in_ch, device, *,
##     inputs_mode="params", lam_grad=0.2, lam_w=1.0, lam_ev=0.0,
##     epochs=10, base=32, param_scaler=None, model_cls=None,
## #    save_path="unet_best.pt", return_history: bool = True, history_path: str | None = None,
##     # in train_unet signature
##     save_path="unet_best.pt",
##     return_history: bool = True,
##     history_path: Optional[str] = None,
#
##     # ---- LR scheduler knobs (tweak as needed) ----
##     lr_init: float = 1e-3, lr_factor: float = 0.5, lr_patience: int = 5, lr_min: float = 1e-5,
##     lr_threshold: float = 1e-4
## ):
##     """
##     param_scaler: (mu, std) or None, to store for inference
##     model_cls: a constructor like lambda in_ch,out_ch: UNet(in_ch, out_ch, base)
##     """
##     if model_cls is None:
##         from .models import UNet
##         model_cls = lambda in_ch, out_ch: UNet(in_ch=in_ch, out_ch=1, base=base)
#
##     model = model_cls(in_ch, 1).to(device)
##     opt   = torch.optim.Adam(model.parameters(), lr=lr_init)
#
##     # ---- ReduceLROnPlateau ----
##     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
##         opt,
##         mode="min",
##         factor=lr_factor,
##         patience=lr_patience,
##         threshold=lr_threshold,
##         threshold_mode="rel",
##         cooldown=0,
##         min_lr=lr_min,
##         eps=1e-8,
##     )
#
#
##     # AMP
##     use_cuda  = (str(device) == "cuda")
##     use_amp   = use_cuda
##     amp_dtype = torch.bfloat16 if (use_cuda and getattr(torch.cuda, "is_bf16_supported", lambda: False)()) else torch.float16
##     scaler    = torch.amp.GradScaler('cuda', enabled=use_amp)
#
##     # weight map from first sample
##     m0 = train_loader.dataset[0]["mask"]  # (1,H,W) CPU
##     w_map = edge_weights(m0.numpy()[0], sigma_px=3.0)
##     w_map = torch.from_numpy(w_map).to(device).unsqueeze(0).unsqueeze(0)
#
##     # history containers
##     history = {
##         "tr_mse": [], "tr_maeN": [],
##         "va_mse": [], "va_maeN": [],
##         "va_mae_eV": [], "va_rmse_eV": []
##     }
#
##     best_val = float("inf")
##     for epoch in range(epochs):
##         # ---------------- train ----------------
##         model.train()
##         tr_loss_sum = tr_maeN_sum = tr_px = 0.0
##         for b in train_loader:
##             x, y, m = b["x"].to(device), b["y"].to(device), b["mask"].to(device)
##             opt.zero_grad(set_to_none=True)
#
##             with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
##                 p = model(x)
##                 # ensure dtype match for weighted loss under autocast
##                 loss_base = masked_weighted_loss(p, y, m, w=w_map.to(p.dtype), lam_grad=lam_grad, lam_w=lam_w)
#
##                 if lam_ev > 0:
##                     p_ev = norm.inverse(p.float(), m.float())
##                     y_ev = norm.inverse(y.float(), m.float())
##                     loss_ev = ((p_ev - y_ev).abs() * m.float()).sum() / m.sum().clamp_min(1e-8)
##                     loss = loss_base + lam_ev * loss_ev
##                 else:
##                     loss = loss_base
#
##             scaler.scale(loss).backward()
##             scaler.unscale_(opt)
##             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
##             scaler.step(opt)
##             scaler.update()
#
##             with torch.no_grad():
##                 px = float(m.sum().item())
##                 tr_loss_sum += float(loss.item()) * px
##                 tr_maeN_sum += float(mae_norm(p.float(), y.float(), m.float()).item()) * px
##                 tr_px       += px
#
##         tr_loss = tr_loss_sum / max(tr_px, 1.0)
##         tr_maeN = tr_maeN_sum / max(tr_px, 1.0)
#
##         # ---------------- validate ----------------
##         model.eval()
##         va_loss_sum = va_maeN_sum = va_px = 0.0
##         va_abs_ev_sum = va_sq_ev_sum = 0.0
##         with torch.no_grad():
##             for b in val_loader:
##                 x, y, m = b["x"].to(device), b["y"].to(device), b["mask"].to(device)
##                 with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
##                     p = model(x)
##                     loss_b = masked_weighted_loss(p, y, m, w=w_map.to(p.dtype), lam_grad=lam_grad, lam_w=lam_w)
#
##                 px = float(m.sum().item())
##                 va_loss_sum += float(loss_b.item()) * px
##                 va_maeN_sum += float(mae_norm(p.float(), y.float(), m.float()).item()) * px
##                 va_px       += px
#
##                 abs_sum, sq_sum, _ = batch_error_sums_ev(p.float(), y.float(), m.float(), norm)
##                 va_abs_ev_sum += float(abs_sum.item())
##                 va_sq_ev_sum  += float(sq_sum.item())
#
##         va_loss    = va_loss_sum / max(va_px, 1.0)
##         va_maeN    = va_maeN_sum / max(va_px, 1.0)
##         va_MAE_eV  = va_abs_ev_sum / max(va_px, 1.0)
##         va_RMSE_eV = (va_sq_ev_sum / max(va_px, 1.0)) ** 0.5
#
##         # ---- record history ----
##         history["tr_mse"].append(tr_loss)
##         history["tr_maeN"].append(tr_maeN)
##         history["va_mse"].append(va_loss)
##         history["va_maeN"].append(va_maeN)
##         history["va_mae_eV"].append(va_MAE_eV)
##         history["va_rmse_eV"].append(va_RMSE_eV)
#
##         # ---- log & schedule ----
##         lr_before = opt.param_groups[0]["lr"]
##         print(
##             f"epoch {epoch:02d} | "
##             f"train MSE {tr_loss:.4f}  MAE_norm {tr_maeN:.4f} | "
##             f"val MSE {va_loss:.4f}  MAE_norm {va_maeN:.4f} | "
##             f"val MAE {va_MAE_eV:.2f} eV  RMSE {va_RMSE_eV:.2f} eV | "
##             f"lr {lr_before:.2e}"
##         )
#
##         # step the LR scheduler on validation loss
##         scheduler.step(va_loss)
##         lr_after = opt.param_groups[0]["lr"]
##         if lr_after < lr_before:
##             print(f"  ↳ ReduceLROnPlateau: lr {lr_before:.2e} → {lr_after:.2e}")
#
##         # ---- checkpoint best ----
##         if va_loss < best_val:
##             best_val = va_loss
##             ckpt = {
##                 "model": model.state_dict(),
##                 "norm": (float(norm.mu), float(norm.sigma), float(norm.eps)),
##                 "inputs_mode": inputs_mode,
##                 "in_ch": in_ch,
##             }
##             if param_scaler is not None:
##                 mu, std = param_scaler
##                 if mu is not None:
##                     ckpt["param_mu"]  = mu.tolist() if hasattr(mu, "tolist") else mu
##                     ckpt["param_std"] = std.tolist() if hasattr(std, "tolist") else std
##             torch.save(ckpt, save_path)
#
##     # optional persist of history
##     if history_path is not None:
##         np.savez(history_path, **history)
#
##     return (model, history) if return_history else model
#
#
## train.py
#import torch
#from torch import nn
#
## wherever UNet lives in your project:
#from solps_ai.models import UNet   # adjust this import to your actual path
#
#def train_unet(train_loader, val_loader, norm, in_ch, device,
#               epochs=20, base=32, inputs_mode="params",
#               lam_grad=0.2, lam_w=1.0, lam_ev=0.0,
#               param_scaler=None, save_path=None,
#               return_history=False, history_path=None,
#               amp=True, grad_accum=1, clip_grad=None,
#               unet_kwargs=None):
#
#    if unet_kwargs is None:
#        unet_kwargs = {}
#
#    # ✅ no ellipsis here
#    model = UNet(in_ch=in_ch, base=base, **unet_kwargs).to(device)
#
#    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
#    scaler = torch.cuda.amp.GradScaler(enabled=amp)
#
#    criterion = make_loss(lam_w=lam_w, lam_grad=lam_grad,
#                          lam_ev=lam_ev, norm=norm)
#
#    best_val = float("inf")
#    history = {"train_loss": [], "val_loss": []}
#
#    for epoch in range(epochs):
#        # -------- train --------
#        model.train()
#        optimizer.zero_grad(set_to_none=True)
#        running = 0.0
#
#        for i, (x, y) in enumerate(train_loader):
#            x = x.to(device, non_blocking=True)
#            y = y.to(device, non_blocking=True)
#
#            with torch.cuda.amp.autocast(enabled=amp):
#                y_pred = model(x)
#                loss = criterion(y_pred, y) / grad_accum
#
#            scaler.scale(loss).backward()
#
#            if (i + 1) % grad_accum == 0:
#                if clip_grad is not None:
#                    scaler.unscale_(optimizer)
#                    nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
#                scaler.step(optimizer)
#                scaler.update()
#                optimizer.zero_grad(set_to_none=True)
#
#            running += loss.item() * grad_accum
#
#        train_loss = running / max(1, len(train_loader))
#        history["train_loss"].append(train_loss)
#
#        # -------- validate --------
#        model.eval()
#        val_loss = 0.0
#        with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp):
#            for x, y in val_loader:
#                x = x.to(device, non_blocking=True)
#                y = y.to(device, non_blocking=True)
#                y_pred = model(x)
#                val_loss += criterion(y_pred, y).item()
#
#        val_loss /= max(1, len(val_loader))
#        history["val_loss"].append(val_loss)
#
#        if save_path and val_loss < best_val:
#            best_val = val_loss
#            torch.save(model.state_dict(), save_path)
#
#        # (optional) write `history` to disk each epoch if history_path is set
#
#    return (model, history) if return_history else (model, None)
#
#

from typing import Optional
import numpy as np
import torch
from .losses import masked_weighted_loss, mae_norm, batch_error_sums_ev, edge_weights

# from solps_ai.losses import masked_weighted_loss, mae_norm, batch_error_sums_ev, edge_weights

def make_loss(lam_w=1.0, lam_grad=0.2, lam_ev=0.0, norm=None):
    """Factory returning a callable composite loss using existing primitives."""
    def loss_fn(pred, target):
        # basic mask handling — assume pred,target are (B,1,H,W)
        mask = torch.ones_like(pred)
        return masked_weighted_loss(pred, target, mask, w=None,
                                    lam_grad=lam_grad, lam_w=lam_w)
    return loss_fn

# train.py
import torch
from torch import nn

# wherever UNet lives in your project:
from solps_ai.models import UNet   # adjust this import to your actual path

def train_unet(train_loader, val_loader, norm, in_ch, device,
               epochs=20, base=32, inputs_mode="params",
               lam_grad=0.2, lam_w=1.0, lam_ev=0.0,
               param_scaler=None, save_path=None,
               return_history=False, history_path=None,
               amp=True, grad_accum=1, clip_grad=None,
               unet_kwargs=None):

    if unet_kwargs is None:
        unet_kwargs = {}

    # ✅ no ellipsis here
    model = UNet(in_ch=in_ch, base=base, **unet_kwargs).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    criterion = make_loss(lam_w=lam_w, lam_grad=lam_grad,
                          lam_ev=lam_ev, norm=norm)

    best_val = float("inf")
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        # -------- train --------
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0

        for i, batch in enumerate(train_loader):
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            mask = batch.get("mask", torch.ones_like(y)).to(device, non_blocking=True)


            with torch.cuda.amp.autocast(enabled=amp):
                y_pred = model(x)
                # loss = criterion(y_pred, y) / grad_accum
                # loss = masked_weighted_loss(y_pred, y, mask, lam_grad=lam_grad, lam_w=lam_w) / grad_accum
                loss = masked_weighted_loss(y_pred, y, mask,
                                                      lam_grad=lam_grad, lam_w=lam_w) / grad_accum
                                                      
            scaler.scale(loss).backward()

            if (i + 1) % grad_accum == 0:
                if clip_grad is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running += loss.item() * grad_accum

        train_loss = running / max(1, len(train_loader))
        history["train_loss"].append(train_loss)

        # -------- validate --------
        model.eval()
        val_loss = 0.0
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp):
            for batch in val_loader:
                x = batch["x"].to(device, non_blocking=True)
                y = batch["y"].to(device, non_blocking=True)
                mask = batch.get("mask", torch.ones_like(y)).to(device, non_blocking=True)
                y_pred = model(x)
                # val_loss += criterion(y_pred, y).item()
                val_loss += masked_weighted_loss(y_pred, y, mask,
                                     lam_grad=lam_grad, lam_w=lam_w).item()


        val_loss /= max(1, len(val_loader))
        history["val_loss"].append(val_loss)

        if save_path and val_loss < best_val:
            best_val = val_loss

            # pack everything needed at inference
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

        # (optional) write `history` to disk each epoch if history_path is set

    return (model, history) if return_history else (model, None)

