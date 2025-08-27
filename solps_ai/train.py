# train.py
import numpy as np
import torch
from .losses import masked_weighted_loss, mae_norm, batch_error_sums_ev, edge_weights
from .models import UNet

def train_unet(
    train_loader, val_loader, norm, in_ch, device, *,
    inputs_mode="params", lam_grad=0.2, lam_w=1.0, lam_ev=0.0,
    epochs=10, base=32, param_scaler=None, model_cls=None,
    save_path="unet_best.pt", return_history: bool = True, history_path: str | None = None,
    lr_init: float = 1e-3, lr_factor: float = 0.5, lr_patience: int = 5, lr_min: float = 1e-5,
    lr_threshold: float = 1e-4
):
    model_cls = model_cls or (lambda ic, oc: UNet(in_ch=ic, out_ch=oc, base=base))

    # peek shapes
    b0 = next(iter(train_loader))
    B, C, H, W = b0["y"].shape  # C = number of target channels

    model = model_cls(in_ch, C).to(device)
    from inspect import signature
    opt = torch.optim.AdamW(model.parameters(), lr=lr_init)

    Sched = torch.optim.lr_scheduler.ReduceLROnPlateau

    # What we *want* to pass
    want = dict(
        mode="min",
        factor=lr_factor,
        patience=lr_patience,
        min_lr=lr_min,
        threshold=lr_threshold,
        cooldown=0,
        eps=1e-8,
        verbose=False,          # some torch versions don't have this
        threshold_mode="rel",   # safe default if present
    )

    # Filter to only the params your installed torch accepts
    allowed = set(signature(Sched.__init__).parameters.keys()) - {"self", "optimizer"}
    kwargs = {k: v for k, v in want.items() if k in allowed}

    sch = Sched(opt, **kwargs)

    # edge weights (1,1,H,W) -> broadcast in loss
    m0 = train_loader.dataset[0]["mask"]  # (1,H,W) CPU
    w_map = edge_weights(m0.numpy()[0], sigma_px=3.0)
    w_map = torch.from_numpy(w_map).to(device).unsqueeze(0).unsqueeze(0)

    history = {"tr_mse": [], "tr_maeN": [], "va_mse": [], "va_maeN": [], "va_mae_eV": [], "va_rmse_eV": []}
    best_val = float("inf")

    for epoch in range(epochs):
        # ---- train ----
        model.train()
        tr_losses = []; tr_maeN = []
        for b in train_loader:
            x = b["x"].to(device).float()           # (B,in_ch,H,W)
            y = b["y"].to(device).float()           # (B,C,H,W)
            m = b["mask"].to(device).float()        # (B,1,H,W)

            opt.zero_grad()
            yhat = model(x)
            loss = masked_weighted_loss(yhat, y, m, w=w_map, lam_grad=lam_grad, lam_w=lam_w)
            loss.backward()
            opt.step()

            tr_losses.append(loss.item())
            tr_maeN.append(mae_norm(yhat, y, m).item())

        history["tr_mse"].append(float(np.mean(tr_losses)))
        history["tr_maeN"].append(float(np.mean(tr_maeN)))

        # ---- validate ----
        model.eval()
        va_losses = []; va_maeN = []
        abs_sum = 0.0; sq_sum = 0.0; px_sum = 0.0
        with torch.no_grad():
            for b in val_loader:
                x = b["x"].to(device).float()
                y = b["y"].to(device).float()
                m = b["mask"].to(device).float()

                yhat = model(x)
                va_losses.append(masked_weighted_loss(yhat, y, m, w=w_map, lam_grad=lam_grad, lam_w=lam_w).item())
                va_maeN.append(mae_norm(yhat, y, m).item())

                a, s, p = batch_error_sums_ev(yhat, y, m, norm)  # now sums over all channels
                abs_sum += a.item(); sq_sum += s.item(); px_sum += p.item()

        va_mse = float(np.mean(va_losses))
        va_maeN = float(np.mean(va_maeN))
        va_mae_eV  = abs_sum / max(px_sum, 1.0)
        va_rmse_eV = (sq_sum / max(px_sum, 1.0)) ** 0.5

        history["va_mse"].append(va_mse)
        history["va_maeN"].append(va_maeN)
        history["va_mae_eV"].append(va_mae_eV)
        history["va_rmse_eV"].append(va_rmse_eV)

        sch.step(va_mse)

        # ---- checkpoint best ----
        if va_mse < best_val:
            best_val = va_mse
            ckpt = {
                "model": model.state_dict(),
                "in_ch": in_ch,
                "out_ch": C,
                "norm": (norm.mu.numpy(), norm.sigma.numpy(), norm.eps, norm.pos.numpy().astype(np.uint8)),
            }
            if param_scaler is not None:
                mu, std = param_scaler
                if mu is not None:
                    ckpt["param_mu"]  = mu.tolist() if hasattr(mu, "tolist") else mu
                    ckpt["param_std"] = std.tolist() if hasattr(std, "tolist") else std
            torch.save(ckpt, save_path)

    if history_path is not None:
        np.savez(history_path, **history)
    return (model, history) if return_history else model

