# Copyright 2025-2026 Oak Ridge National Laboratory
# @authors: Abdourahmane (Abdou) Diaw - diawa@ornl.gov
#
# SPDX-License-Identifier: MIT

import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_ae_recon_one(*, ae, norm, sample, device, title=""):
    ae.eval()

    x = sample["x"].unsqueeze(0).to(device)
    y = sample["y"].unsqueeze(0).to(device)
    m = sample["mask"].unsqueeze(0).to(device)

    with torch.no_grad():
        p = ae(x)

    # back to physical space
    y_phys = norm.inverse(y, m).cpu().numpy()[0]   # (C,H,W)
    p_phys = norm.inverse(p, m).cpu().numpy()[0]

    mask2d = (m.cpu().numpy()[0, 0] > 0.5)

    C = y_phys.shape[0]

    fig, axes = plt.subplots(
        2, C,
        figsize=(4*C, 6),
        squeeze=False,
        constrained_layout=True
    )

    for ci in range(C):
        yt = y_phys[ci].copy()
        yp = p_phys[ci].copy()

        yt[~mask2d] = np.nan
        yp[~mask2d] = np.nan

        vmin = np.nanmin(yt)
        vmax = np.nanmax(yt)

        im0 = axes[0, ci].imshow(
            yt, origin="lower", aspect="auto",
            vmin=vmin, vmax=vmax
        )
        axes[0, ci].set_title(f"{title} true (ch {ci})")
        fig.colorbar(im0, ax=axes[0, ci], fraction=0.046, pad=0.02)

        im1 = axes[1, ci].imshow(
            yp, origin="lower", aspect="auto",
            vmin=vmin, vmax=vmax
        )
        axes[1, ci].set_title(f"{title} pred (ch {ci})")
        fig.colorbar(im1, ax=axes[1, ci], fraction=0.046, pad=0.02)

        axes[0, ci].set_xticks([])
        axes[0, ci].set_yticks([])
        axes[1, ci].set_xticks([])
        axes[1, ci].set_yticks([])

    plt.show()


def plot_training_curves(history, savepath=None, title="Training curves",
                         logy=True, smooth=None, show=True):
    """
    Plot training/validation losses from a history dict (or .npz file path).

    history keys expected:
      'tr_mse','tr_maeN','va_mse','va_maeN','va_mae_eV','va_rmse_eV'
    smooth: None or float in (0,1) -> EMA smoothing factor (e.g., 0.9)
    """
    # accept either dict or .npz file path
    if isinstance(history, (str, bytes)):
        with np.load(history) as npz:
            h = {k: npz[k] for k in npz.files}
    else:
        h = history

    def arr(key):
        return np.asarray(h.get(key, []), dtype=float)

    def ema(x, beta):
        if x.size == 0 or not (0 < beta < 1):
            return x
        y = np.empty_like(x)
        y[0] = x[0]
        for i in range(1, len(x)):
            y[i] = beta * y[i-1] + (1 - beta) * x[i]
        return y

    tr_mse    = arr("tr_mse")
    tr_maeN   = arr("tr_maeN")
    va_mse    = arr("va_mse")
    va_maeN   = arr("va_maeN")
    va_mae_eV = arr("va_mae_eV")
    va_rmse_eV= arr("va_rmse_eV")

    # common x-axis
    epochs = np.arange(1, len(tr_mse) + 1)

    # optional smoothing
    if smooth is not None:
        tr_mse  = ema(tr_mse,  smooth)
        va_mse  = ema(va_mse,  smooth)
        tr_maeN = ema(tr_maeN, smooth)
        va_maeN = ema(va_maeN, smooth)
        va_mae_eV  = ema(va_mae_eV,  smooth)
        va_rmse_eV = ema(va_rmse_eV, smooth)

    # best epoch by val MSE (unsmoothed if available)
    best_idx = int(np.argmin(arr("va_mse"))) if arr("va_mse").size else None

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

    # Panel 1: MSE (normalized)
    ax = axes[0]
    (ax.semilogy if logy else ax.plot)(epochs, tr_mse, label="train MSE")
    (ax.semilogy if logy else ax.plot)(epochs, va_mse, label="val MSE")
    if best_idx is not None:
        ax.axvline(best_idx + 1, ls="--", alpha=0.4)
        ax.text(best_idx + 1, ax.get_ylim()[0], f"  best @ {best_idx+1}",
                va="bottom", ha="left", fontsize=9)
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE (norm)")
    ax.set_title("Masked weighted MSE")
    ax.grid(alpha=0.25); ax.legend()

    # Panel 2: MAE (normalized)
    ax = axes[1]
    (ax.semilogy if logy else ax.plot)(epochs, tr_maeN, label="train MAE")
    (ax.semilogy if logy else ax.plot)(epochs, va_maeN, label="val MAE")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MAE (norm)")
    ax.set_title("MAE (normalized space)")
    ax.grid(alpha=0.25); ax.legend()

    # Panel 3: Physical metrics (eV)
    ax = axes[2]
    (ax.semilogy if logy else ax.plot)(epochs, va_mae_eV,  label="val MAE (eV)")
    (ax.semilogy if logy else ax.plot)(epochs, va_rmse_eV, label="val RMSE (eV)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("eV")
    ax.set_title("Validation (physical units)")
    ax.grid(alpha=0.25); ax.legend()

    fig.suptitle(title, y=1.02, fontsize=12)

    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axes
