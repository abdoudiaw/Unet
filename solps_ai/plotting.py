import numpy as np
import matplotlib.pyplot as plt
import torch
from .predict import predict_te  # uses norm.inverse to return eV :contentReference[oaicite:2]{index=2}
from .data import _load_truth_and_params


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




def display_random_samples(train_loader, norm, n=4):
    ds = train_loader.dataset                 # this is a Subset
    idxs = np.random.choice(len(ds), size=n, replace=False)

    # collect samples
    ys, ms, te_evs = [], [], []
    for i in idxs:
        b = ds[i]
        y = b["y"]           # (1,H,W)
        m = b["mask"]        # (1,H,W)
        with torch.no_grad():
            te_ev = norm.inverse(y.unsqueeze(0), m.unsqueeze(0)).squeeze(0)  # (1,H,W)
        ys.append(y.squeeze(0).numpy())
        ms.append(m.squeeze(0).numpy())
        te_evs.append(te_ev.squeeze(0).numpy())

    ys = np.stack(ys)        # (n,H,W)
    ms = np.stack(ms)
    te_evs = np.stack(te_evs)

    # plot
    fig, axes = plt.subplots(3, n, figsize=(4*n, 10))
    for j in range(n):
        axes[0,j].imshow(ms[j], origin='lower'); axes[0,j].set_title(f"Mask idx={idxs[j]}"); axes[0,j].axis('off')
        im1 = axes[1,j].imshow(np.where(ms[j]>0.5, ys[j], np.nan), origin='lower')
        axes[1,j].set_title("Target (normalized)"); axes[1,j].axis('off'); fig.colorbar(im1, ax=axes[1,j], fraction=0.046, pad=0.04)
        im2 = axes[2,j].imshow(np.where(ms[j]>0.5, te_evs[j], np.nan), origin='lower')
        axes[2,j].set_title("Target Te (eV)"); axes[2,j].axis('off'); fig.colorbar(im2, ax=axes[2,j], fraction=0.046, pad=0.04)
    plt.suptitle("Random training samples"); plt.tight_layout(); plt.show()

def get_subset_indices(loader):
    ds = loader.dataset
    if hasattr(ds, "indices"):  # Subset
        return np.array(ds.indices, dtype=int)
    raise TypeError("val_loader.dataset is not a Subset; can't recover original indices.")

def plot_random_val_examples_ae(
    *, npz_path, val_loader, model, norm, R2d, Z2d, device,
    n=4, seed=0
):
    val_idx = get_subset_indices(val_loader)
    rng = np.random.default_rng(seed)
    picks = rng.choice(val_idx, size=min(n, len(val_idx)), replace=False)

    for idx in picks:
        Te_true, mask_ref, _params_raw, _ = _load_truth_and_params(npz_path, idx=int(idx))

        Te_pred = predict_te_ae(model, norm, Te_true, mask_ref, device=device)

        plot_truth_pred_percent_error_RZ(
            Te_true, Te_pred, mask_ref, R2d, Z2d,
            use_smape=False, vmax_pct=150
        )



@torch.no_grad()
def predict_te_ae(model, norm, te_true, mask, device=None):
    """
    te_true: (H,W) or (1,H,W) numpy/torch
    mask:    (H,W) or (1,H,W) numpy/torch
    returns: Te_pred (H,W) numpy
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    # to torch (1,1,H,W)
    te = torch.from_numpy(te_true).float() if isinstance(te_true, np.ndarray) else te_true.float()
    m  = torch.from_numpy(mask).float()    if isinstance(mask, np.ndarray) else mask.float()

    if te.dim() == 2: te = te.unsqueeze(0).unsqueeze(0)
    if te.dim() == 3: te = te.unsqueeze(0)
    if m.dim()  == 2: m  = m.unsqueeze(0).unsqueeze(0)
    if m.dim()  == 3: m  = m.unsqueeze(0)

    m = (m > 0.5).float().to(device)
    te = torch.nan_to_num(te, nan=0.0, posinf=0.0, neginf=0.0).to(device)

    # normalize using your masked normalizer
    z_in = norm.transform(te, m)           # (1,1,H,W)

    z_out = model(z_in)                   # (1,1,H,W)
    te_rec = norm.inverse(z_out, m)        # back to eV

    return te_rec.squeeze().detach().cpu().numpy().astype(np.float32)

def _plot_training_curves(history, title="Training"):
    """
    history: dict with lists: 'tr_mse','tr_maeN','va_mse','va_maeN','va_mae_eV','va_rmse_eV'
    """
    plt.figure(figsize=(8,4))
    for k in ["tr_mse","va_mse"]:
        if k in history: plt.plot(history[k], label=k)
    plt.legend(); plt.title(title); plt.xlabel("epoch"); plt.ylabel("MSE")
    plt.grid(True, alpha=0.3); plt.show()

def plot_te_rectilinear(te_ev, R_1d, Z_1d, mask=None, title='Te (eV)', fname=None):
    te = np.asarray(te_ev, float)
    if mask is not None:
        m = np.asarray(mask, float); m = m.squeeze() if m.ndim==3 else m
        te = np.where(m > 0.5, te, np.nan)
    extent = [float(np.min(R_1d)), float(np.max(R_1d)),
              float(np.min(Z_1d)), float(np.max(Z_1d))]
    vmin, vmax = np.nanpercentile(te, [1, 99])
    plt.figure(figsize=(6,6))
    im = plt.imshow(te, origin='lower', aspect='equal', extent=extent, vmin=vmin, vmax=vmax)
    cb = plt.colorbar(im); cb.set_label('Te [eV]')
    plt.xlabel('R [m]'); plt.ylabel('Z [m]'); plt.title(title)
    if fname: plt.savefig(fname, dpi=200, bbox_inches='tight');
    plt.show()

def centers_to_corners(Rc, Zc):
    Rc = np.asarray(Rc, float); Zc = np.asarray(Zc, float)
    H, W = Rc.shape
    Rvf = np.empty((H+1, W)); Zvf = np.empty((H+1, W))
    Rvf[1:H] = 0.5*(Rc[0:H-1] + Rc[1:H]);  Zvf[1:H] = 0.5*(Zc[0:H-1] + Zc[1:H])
    Rvf[0]   = Rc[0] - 0.5*(Rc[1]-Rc[0]);  Zvf[0]   = Zc[0] - 0.5*(Zc[1]-Zc[0])
    Rvf[H]   = Rc[H-1] + 0.5*(Rc[H-1]-Rc[H-2]);  Zvf[H] = Zc[H-1] + 0.5*(Zc[H-1]-Zc[H-2])
    Rhf = np.empty((H, W+1));  Zhf = np.empty((H, W+1))
    Rhf[:,1:W] = 0.5*(Rc[:,0:W-1] + Rc[:,1:W]);  Zhf[:,1:W] = 0.5*(Zc[:,0:W-1] + Zc[:,1:W])
    Rhf[:,0]   = Rc[:,0] - 0.5*(Rc[:,1]-Rc[:,0]);  Zhf[:,0]   = Zc[:,0] - 0.5*(Zc[:,1]-Zc[:,0])
    Rhf[:,W]   = Rc[:,W-1] + 0.5*(Rc[:,W-1]-Rc[:,W-2]);  Zhf[:,W] = Zc[:,W-1] + 0.5*(Zc[:,W-1]-Zc[:,W-2])
    Rcorn = 0.25*(Rvf[:, :-1] + Rvf[:, 1:] + Rhf[:-1, :] + Rhf[1:, :])
    Zcorn = 0.25*(Zvf[:, :-1] + Zvf[:, 1:] + Zhf[:-1, :] + Zhf[1:, :])
    return Rcorn, Zcorn

def plot_te_curvilinear(te_ev, Rcenters, Zcenters, mask=None, title='Te (eV)', fname=None):
    te = np.asarray(te_ev, float)
    if mask is not None:
        m = np.asarray(mask, float); m = m.squeeze() if m.ndim==3 else m
        te = np.where(m > 0.5, te, np.nan)
    Rcorn, Zcorn = centers_to_corners(Rcenters, Zcenters)
    vmin, vmax = np.nanpercentile(te, [1, 99])
    plt.figure(figsize=(6,6))
    pm = plt.pcolormesh(Rcorn, Zcorn, te, shading='auto', vmin=vmin, vmax=vmax)
    cb = plt.colorbar(pm); cb.set_label('Te [eV]')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('R [m]'); plt.ylabel('Z [m]'); plt.title(title)
    if fname: plt.savefig(fname, dpi=200, bbox_inches='tight')
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

def plot_te_log10(te_ev, mask=None, title='Predicted log10 Te', fname=None):
     te = np.array(te_ev, dtype=float)
     te = np.where(te > 0, np.log10(te), np.nan)
     if mask is not None:
         m = np.array(mask, dtype=float)
         if m.ndim == 3:
             m = m.squeeze()
         te = np.where(m > 0.5, te, np.nan)

     vmin, vmax = np.nanpercentile(te, [1, 99])

     plt.figure(figsize=(6, 8))
     im = plt.imshow(te, origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
     cb = plt.colorbar(im)
     cb.set_label('log10 Te [eV]')
     plt.title(title)
     plt.xlabel('X (pixels)')
     plt.ylabel('Y (pixels)')
     if fname:
         plt.savefig(fname, dpi=200, bbox_inches='tight')
     plt.show()

#import numpy as np
#import torch
#import matplotlib.pyplot as plt

def show_case_prediction(
    model, loader, norm, device="cuda", *,
    idx=None, R2d=None, Z2d=None, savepath=None,
    param_scaler=None, cmap="viridis"
):
    """
    Pick a real sample from `loader` (train/val), run the model, and plot:
      [Target Te (eV)] | [Prediction Te (eV)] | [|Error| (eV)]

    If R2d/Z2d are provided (cell-center coords), axes are labeled in meters.
    Returns:
        (global_idx, params_scaled, params_raw, fig)
    """
    # --- resolve underlying dataset + global indices ---
    ds = loader.dataset
    if hasattr(ds, "dataset") and hasattr(ds, "indices"):  # torch.utils.data.Subset
        base_ds, ids = ds.dataset, ds.indices
    else:  # already the base dataset
        base_ds, ids = ds, np.arange(len(ds))

    # pick an example within this split (supports negative idx)
    j = np.random.randint(len(ids)) if idx is None else int(idx if idx >= 0 else len(ids) + idx)
    i = int(ids[j])  # global index into base dataset

    # --- fetch normalized target + mask ---
    b = base_ds[i]             # dict with "y","mask",...
    y_norm = b["y"]            # (1,H,W), normalized
    m_t    = b["mask"]         # (1,H,W)
    mask   = m_t.squeeze(0).detach().cpu().numpy().astype(np.float32)  # (H,W)

    # --- target Te in eV ---
    with torch.no_grad():
        te_target = norm.inverse(y_norm, m_t).squeeze(0).detach().cpu().numpy().astype(np.float32)

    # --- params (scaled, as used by the model) ---
    params_scaled = None
    if hasattr(base_ds, "params") and base_ds.params is not None:
        params_scaled = np.asarray(base_ds.params[i], dtype=np.float32)  # shape (P,)

    # optional: unscale to raw physical params
    params_raw = None
    if params_scaled is not None and param_scaler is not None and param_scaler[0] is not None:
        mu, std = param_scaler
        params_raw = params_scaled * np.asarray(std, np.float32) + np.asarray(mu, np.float32)

    # --- model prediction in eV ---
    te_pred = predict_te(model, norm, mask, params_scaled, device=device, as_numpy=True)

    # --- error map (inside mask only) ---
    err = np.abs(te_pred - te_target) * (mask > 0.5)

    # --- axes / extent (optional rectilinear coordinates) ---
    if R2d is not None and Z2d is not None:
        H, W = te_pred.shape
        R_axis = np.linspace(float(np.min(R2d)), float(np.max(R2d)), W)
        Z_axis = np.linspace(float(np.min(Z2d)), float(np.max(Z2d)), H)
        extent = [R_axis[0], R_axis[-1], Z_axis[0], Z_axis[-1]]
        xlab, ylab = "R [m]", "Z [m]"
    else:
        extent = None
        xlab, ylab = "x", "y"

    # --- shared color limits for target/pred ---
    both = np.where(mask > 0.5, np.stack([te_target, te_pred], 0), np.nan)
    vmin, vmax = np.nanpercentile(both, [1, 99])
    emax = float(np.nanpercentile(err, 99))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)

    def _imshow(ax, data, title, lim=None):
        d = np.where(mask > 0.5, data, np.nan)
        im = ax.imshow(
            d, origin="lower", extent=extent, cmap=cmap,
            vmin=None if lim is None else lim[0],
            vmax=None if lim is None else lim[1], aspect="equal"
        )
        ax.set_title(title); ax.set_xlabel(xlab); ax.set_ylabel(ylab)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    _imshow(axes[0], te_target, "Target Te (eV)", (vmin, vmax))
    _imshow(axes[1], te_pred,   "Prediction Te (eV)", (vmin, vmax))
    _imshow(axes[2], err,       "Abs error (eV)", (0.0, emax))

    P = 0 if params_scaled is None else int(params_scaled.shape[0])
    fig.suptitle(f"Dataset case idx={i} (split idx={j})  |  P={P} params", y=1.03, fontsize=12)

    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.show()

    return i, params_scaled, params_raw, fig

import numpy as np
import torch

def to_np(a):
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    return np.asarray(a)

def inverse_from_norm(y_norm_hw, norm, mask_hw=None):
    """
    y_norm_hw: (H,W) normalized model output (not physical)
    Returns physical Te (H,W).
    """
    y = to_np(y_norm_hw).astype(np.float32)
    m = None if mask_hw is None else (to_np(mask_hw) > 0.5).astype(np.float32)

    # norm.inverse expects (1,1,H,W) typically
    y_t = torch.from_numpy(y)[None, None, :, :]
    if m is not None:
        m_t = torch.from_numpy(m)[None, None, :, :]
    else:
        m_t = None

    with torch.no_grad():
        try:
            phys = norm.inverse(y_t, mask=m_t)
        except TypeError:
            phys = norm.inverse(y_t)

    phys = phys.squeeze().cpu().numpy()
    return phys

import matplotlib.pyplot as plt
import numpy as np

def plot_truth_pred_percent_error_RZ(
    Te_true, Te_pred, mask, R2d, Z2d,
    use_smape=False, eps=1e-3, vmax_pct=150
):
    m = mask > 0.5

    Te_true_m = np.where(m, Te_true, np.nan)
    Te_pred_m = np.where(m, Te_pred, np.nan)

    if use_smape:
        denom = (np.abs(Te_true_m) + np.abs(Te_pred_m) + eps)
        pct_m = 200.0 * np.abs(Te_pred_m - Te_true_m) / denom
        err_title = "sMAPE [%]"
    else:
        denom = np.maximum(Te_true_m, eps)
        pct_m = 100.0 * np.abs(Te_pred_m - Te_true_m) / denom
        err_title = "Percent error [%]"

    vmin = np.nanpercentile(Te_true_m, 1)
    vmax = np.nanpercentile(Te_true_m, 99)
    vmax_err = min(np.nanpercentile(pct_m, 95), vmax_pct)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)

    # ---- Truth ----
    im0 = axes[0].pcolormesh(R2d, Z2d, Te_true_m, shading="auto", cmap="inferno", vmin=vmin, vmax=vmax)
    axes[0].set_title("Truth Te [eV]")
    plt.colorbar(im0, ax=axes[0])

    # ---- Prediction ----
    im1 = axes[1].pcolormesh(R2d, Z2d, Te_pred_m, shading="auto", cmap="inferno", vmin=vmin, vmax=vmax)
    axes[1].set_title("Predicted Te [eV]")
    plt.colorbar(im1, ax=axes[1])

    # ---- Percent error ----
    im2 = axes[2].pcolormesh(R2d, Z2d, pct_m, shading="auto", cmap="magma", vmin=0, vmax=vmax_err)
    axes[2].set_title(err_title)
    plt.colorbar(im2, ax=axes[2])

    for ax in axes:
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
        ax.set_aspect("equal")

    valid = np.isfinite(pct_m)
    print(f"Mean % error: {np.nanmean(pct_m[valid]):.2f} | 90th %: {np.nanpercentile(pct_m[valid],90):.2f}")

    plt.show()

import numpy as np
import matplotlib.pyplot as plt

def plot_truth_pred_diff(Te_true, Te_pred, mask, fname="Te_truth_pred_diff.png",
                         vmax_diff=None, cmap="RdBu_r"):
    """
    Plot true, predicted, and (pred - true) difference maps.

    Parameters
    ----------
    Te_true : ndarray
        True field, shape (H, W)
    Te_pred : ndarray
        Predicted field, shape (H, W)
    mask : ndarray
        Mask array, >0.5 means valid region
    fname : str
        Output filename
    vmax_diff : float or None
        Optional fixed color limit for difference map; if None, auto-scaled at 99th percentile
    cmap : str
        Colormap for difference plot
    """
    m = mask > 0.5
    Te_true_m = np.where(m, Te_true, np.nan)
    Te_pred_m = np.where(m, Te_pred, np.nan)
    diff_m    = np.where(m, Te_pred - Te_true, np.nan)

    # Color limits for true/pred plots
    vmin = np.nanpercentile(Te_true_m, 1)
    vmax = np.nanpercentile(Te_true_m, 99)

    # Symmetric difference scaling
    if vmax_diff is None:
        vmax_diff = np.nanpercentile(np.abs(diff_m), 99)
    vmin_diff = -vmax_diff

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    # True field
    # im0 = axes[0].imshow(Te_true_m, origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)

    # replace each pcolormesh call with:
    im0 = axes[0].imshow(Te_true_m, origin="lower", extent=[R2d.min(), R2d.max(), Z2d.min(), Z2d.max()],
                        cmap="inferno", vmin=vmin, vmax=vmax, aspect="equal")


    axes[0].set_title("Truth $T_e$ [eV]")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Predicted field
    # im1 = axes[1].imshow(Te_pred_m, origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)

    # replace each pcolormesh call with:
    im1 = axes[1].imshow(Te_pred_m, origin="lower", extent=[R2d.min(), R2d.max(), Z2d.min(), Z2d.max()],
                        cmap="inferno", vmin=vmin, vmax=vmax, aspect="equal")


    axes[1].set_title("Predicted $T_e$ [eV]")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Difference
    im2 = axes[2].imshow(diff_m, origin='lower', extent=[R2d.min(), R2d.max(), Z2d.min(), Z2d.max()],
                         cmap=cmap, vmin=vmin_diff, vmax=vmax_diff)
    axes[2].set_title("Error (pred − target) [eV]")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")

    # Stats
    valid = np.isfinite(diff_m)
    mae = np.nanmean(np.abs(diff_m[valid])) if np.any(valid) else np.nan
    rmse = np.sqrt(np.nanmean(diff_m[valid]**2)) if np.any(valid) else np.nan
    print(f"MAE={mae:.2e} eV, RMSE={rmse:.2e} eV, vmax_diff={vmax_diff:.2e} eV")

    # Add text on diff panel
    axes[2].text(0.02, 0.98, f"MAE={mae:.1e} eV\nRMSE={rmse:.1e} eV",
                 transform=axes[2].transAxes, va='top', ha='left', color='white',
                 fontsize=10, bbox=dict(facecolor='black', alpha=0.4, lw=0))

    plt.savefig(fname, dpi=300)
    plt.show()





#         print("No valid pixels to compute percent error (mask/finiteness issue).")

#     plt.show()
#     return fig, axes
