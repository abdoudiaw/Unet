import numpy as np
import matplotlib.pyplot as plt
import torch
from .predict import predict_te  # uses norm.inverse to return eV :contentReference[oaicite:2]{index=2}


def display_random_samples_multi(train_loader, norm, labels=None, channels=None, n=4):
    """
    Show n random samples. For each sample, plot mask + (normalized, physical) for each selected channel.
    labels: list[str] like ['Te','ne','ni',...]; optional
    channels: list[int] indices to display; default = all channels
    """
    ds = train_loader.dataset
    idxs = np.random.choice(len(ds), size=n, replace=False)

    # collect
    YsN, Ms, YsEV = [], [], []
    for i in idxs:
        b = ds[i]
        yN = b["y"]           # (C,H,W)
        m  = b["mask"]        # (1,H,W)
        with torch.no_grad():
            yEV = norm.inverse(yN.unsqueeze(0), m.unsqueeze(0)).squeeze(0)  # (C,H,W)
        YsN.append(yN.numpy()); Ms.append(m.numpy()); YsEV.append(yEV.numpy())

    YsN = np.stack(YsN)       # (n,C,H,W)
    Ms  = np.stack(Ms)        # (n,1,H,W)
    YsEV= np.stack(YsEV)      # (n,C,H,W)

    C = YsN.shape[1]
    sel = channels if channels is not None else list(range(C))
    rows = 1 + 2*len(sel)     # mask row + (norm,phys)*per-channel

    fig, axes = plt.subplots(rows, n, figsize=(4*n, 3.4*rows), squeeze=False)
    for j,i in enumerate(idxs):
        m = Ms[j,0]
        axes[0,j].imshow(m, origin='lower'); axes[0,j].set_title(f"Mask idx={i}"); axes[0,j].axis('off')
        r = 1
        for k,c in enumerate(sel):
            lab = labels[c] if (labels and c < len(labels)) else f"ch{c}"
            im1 = axes[r,  j].imshow(np.where(m>0.5, YsN[j,c],  np.nan), origin='lower'); axes[r,  j].set_title(f"{lab} (norm)"); axes[r,  j].axis('off'); fig.colorbar(im1, ax=axes[r,  j], fraction=0.046, pad=0.04)
            im2 = axes[r+1,j].imshow(np.where(m>0.5, YsEV[j,c], np.nan), origin='lower'); axes[r+1,j].set_title(f"{lab} (phys)"); axes[r+1,j].axis('off'); fig.colorbar(im2, ax=axes[r+1,j], fraction=0.046, pad=0.04)
            r += 2
    plt.suptitle("Random training samples (multi-channel)"); plt.tight_layout(); plt.show()


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

import numpy as np
import matplotlib.pyplot as plt
import torch
from .predict import predict_multi  # (C,H,W)

def show_case_prediction_grid(
    model, loader, norm, *,
    order=("Te","Ti","ni","ne"),
    target_keys=None,               # channel names in dataset order
    device="cuda", idx=None,
    R2d=None, Z2d=None, savepath=None,
    param_scaler=None, cmap="viridis"
):
    """
    Plot one validation case as a 4x3 grid:
      rows:   Te, Ti, ni, ne
      cols:   Target | Prediction | |Error|
    Returns:
      idx, params_scaled, params_raw, fig
    """
    # ----- pick a sample -----
    ds = loader.dataset
    base_ds, ids = (ds.dataset, ds.indices) if hasattr(ds, "dataset") else (ds, np.arange(len(ds)))
    j = np.random.randint(len(ids)) if idx is None else int(idx if idx >= 0 else len(ids) + idx)
    i = int(ids[j])
    b = base_ds[i]
    yN = b["y"]; m = b["mask"]

    # physical target (already log10 if norm.inverse returns that)
    with torch.no_grad():
        y_phys = norm.inverse(yN.unsqueeze(0), m.unsqueeze(0)).squeeze(0).cpu().numpy()  # (C,H,W)

    mask = m.squeeze(0).cpu().numpy()
    params_scaled = None
    if hasattr(base_ds, "params") and base_ds.params is not None:
        params_scaled = np.asarray(base_ds.params[i], dtype=np.float32)

    # raw params from scaler
    params_raw = None
    if param_scaler is not None and params_scaled is not None:
        mu, std = param_scaler
        params_raw = params_scaled * std + mu

    # model prediction (C,H,W)
    Ypred = predict_multi(model, norm, mask, params_scaled, device=device, as_numpy=True)

    # names
    if target_keys is None:
        target_keys = [f"ch{c}" for c in range(y_phys.shape[0])]
    name_to_idx = {name: k for k, name in enumerate(target_keys)}
    names = [n for n in order if n in name_to_idx]

    # extent
    extent=None; xlab="x"; ylab="y"
    if R2d is not None and Z2d is not None:
        H,W = Ypred.shape[-2], Ypred.shape[-1]
        R_axis = np.linspace(float(np.min(R2d)), float(np.max(R2d)), W)
        Z_axis = np.linspace(float(np.min(Z2d)), float(np.max(Z2d)), H)
        extent = [R_axis[0], R_axis[-1], Z_axis[0], Z_axis[-1]]
        xlab, ylab = "R [m]", "Z [m]"

    # figure
    R = len(names); C = 3
    fig, axes = plt.subplots(R, C, figsize=(4.3*C, 3.8*R), squeeze=False, constrained_layout=True)

    def _masked(a): return np.where(mask>0.5, a, np.nan)

    for r, name in enumerate(names):
        k = name_to_idx[name]
        tgt  = _masked(y_phys[k])
        pred = _masked(Ypred[k])
        err  = np.abs(pred - tgt)

        both = np.stack([tgt, pred], 0)
        vmin, vmax = np.nanpercentile(both, [1, 99])
        emax = float(np.nanpercentile(err, 99))

        def im(ax, data, title, lim):
            imh = ax.imshow(data, origin="lower", extent=extent, cmap=cmap,
                            vmin=lim[0] if lim else None, vmax=lim[1] if lim else None,
                            aspect="equal")
            ax.set_title(title )
            ax.set_xlabel(xlab); ax.set_ylabel(ylab)
            fig.colorbar(imh, ax=ax, fraction=0.046, pad=0.04)

        im(axes[r,0], tgt,  f"{name} • Target",     (vmin, vmax))
        im(axes[r,1], pred, f"{name} • Prediction", (vmin, vmax))
        im(axes[r,2], err,  f"{name} • |Error|",    (0.0, emax))

    plt.suptitle("Validation case")
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.show()

    return i, params_scaled, params_raw, fig


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import torch
from .predict import predict_multi  # returns (C,H,W) in *physical units* per your pipeline

# ---------- gather flattened true/pred pairs across a few validation samples ----------
def gather_flat_true_pred(
    model, loader, norm, target_keys, device="cuda",
    max_samples=6, data_is_log10=False
):
    """
    Returns dict[name] = (true_flat, pred_flat) in *linear* physical units.
    If your y_phys is in log10, set data_is_log10=True to un-log back to linear.
    """
    got = {k: ([], []) for k in target_keys}

    ds = loader.dataset
    base_ds, ids = (ds.dataset, ds.indices) if hasattr(ds, "dataset") else (ds, np.arange(len(ds)))
    picks = np.random.choice(len(ids), size=min(max_samples, len(ids)), replace=False)

    for j in picks:
        i = int(ids[j])
        b = base_ds[i]
        yN, m = b["y"], b["mask"]

        with torch.no_grad():
            # targets in phys units (your pipeline)
            y_phys = norm.inverse(yN.unsqueeze(0), m.unsqueeze(0)).squeeze(0).cpu().numpy()  # (C,H,W)
        mask = m.squeeze(0).cpu().numpy() > 0.5

        # model prediction in phys units for ALL channels
        params_scaled = getattr(base_ds, "params", None)
        params_scaled = None if params_scaled is None else np.asarray(params_scaled[i], dtype=np.float32)
        Ypred = predict_multi(model, norm, mask.astype(np.float32), params_scaled, device=device, as_numpy=True)

        # flatten masked pixels per channel
        for k, name in enumerate(target_keys):
            t = y_phys[k][mask]
            p = Ypred[k][mask]

            # un-log if needed
            if data_is_log10:
                t = np.power(10.0, t)
                p = np.power(10.0, p)

            # drop any NaNs/Infs
            good = np.isfinite(t) & np.isfinite(p)
            if good.any():
                got[name][0].append(t[good])
                got[name][1].append(p[good])

    # concat
    for name in target_keys:
        tt = np.concatenate(got[name][0]) if got[name][0] else np.array([])
        pp = np.concatenate(got[name][1]) if got[name][1] else np.array([])
        got[name] = (tt, pp)
    return got

# ---------- plot one channel ----------
def plot_pred_vs_true_hex(t, p, name="Te", unit="eV", gridsize=80, savepath=None):
    """
    Hexbin pred vs true with log10 count colorbar + MAE, NMAE, R^2 in title.
    Inputs t, p are 1D (linear units).
    """
    if t.size == 0:
      raise ValueError(f"No data to plot for {name}")

    # metrics
    mae = np.mean(np.abs(p - t))
    nmae = mae / (np.mean(np.abs(t)) + 1e-12) * 100.0
    ss_res = np.sum((p - t)**2)
    ss_tot = np.sum((t - np.mean(t))**2) + 1e-12
    r2 = 1.0 - ss_res/ss_tot

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    hb = ax.hexbin(t, p, gridsize=gridsize, norm=LogNorm(), mincnt=1)
    lim = np.nanpercentile(np.concatenate([t, p]), [0.5, 99.5])
    lo, hi = float(lim[0]), float(lim[1])
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1, alpha=0.7)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(f"SOLPS-ITER: {name} ({unit})")
    ax.set_ylabel(f"UNet: {name} ({unit})")
    cbar = fig.colorbar(hb, ax=ax)
    cbar.set_label("log10(count)")

    ax.set_title(f"MAE={mae:.2g} {unit} | NMAE={nmae:.1f}% | R²={r2:.3f}")
    ax.grid(True, alpha=0.25)
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.show()
    return dict(MAE=mae, NMAE=nmae, R2=r2)



    # plotting.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import torch
from .predict import predict_multi

# def _maybe_unlog(arr, do_unlog, clip=(-50.0, 50.0)):
#     """Safely convert log10 -> linear if requested."""
#     if not do_unlog:
#         return arr
#     a = np.clip(arr, clip[0], clip[1])      # avoid overflow in 10**x
#     return np.power(10.0, a)

# def gather_flat_true_pred(
#     model, loader, norm, target_keys, device="cuda",
#     max_samples=6, data_is_log10_map=None  # dict like {"Te":False,"Ti":False,"ni":True,"ne":True}
# ):
#     if data_is_log10_map is None:
#         data_is_log10_map = {k: False for k in target_keys}

#     got = {k: ([], []) for k in target_keys}
#     ds = loader.dataset
#     base_ds, ids = (ds.dataset, ds.indices) if hasattr(ds, "dataset") else (ds, np.arange(len(ds)))
#     picks = np.random.choice(len(ids), size=min(max_samples, len(ids)), replace=False)

#     for j in picks:
#         i = int(ids[j]); b = base_ds[i]
#         yN, m = b["y"], b["mask"]
#         with torch.no_grad():
#             y_phys = norm.inverse(yN.unsqueeze(0), m.unsqueeze(0)).squeeze(0).cpu().numpy()  # (C,H,W)
#         mask = (m.squeeze(0).cpu().numpy() > 0.5)

#         params_scaled = getattr(base_ds, "params", None)
#         params_scaled = None if params_scaled is None else np.asarray(params_scaled[i], dtype=np.float32)
#         Ypred = predict_multi(model, norm, mask.astype(np.float32), params_scaled, device=device, as_numpy=True)

#         for k, name in enumerate(target_keys):
#             t = y_phys[k][mask]
#             p = Ypred[k][mask]

#             # un-log only if this channel is stored in log10
#             do_unlog = bool(data_is_log10_map.get(name, False))
#             t = _maybe_unlog(t, do_unlog)
#             p = _maybe_unlog(p, do_unlog)

#             good = np.isfinite(t) & np.isfinite(p)
#             if good.any():
#                 got[name][0].append(t[good])
#                 got[name][1].append(p[good])

#     for name in target_keys:
#         tt = np.concatenate(got[name][0]) if got[name][0] else np.array([])
#         pp = np.concatenate(got[name][1]) if got[name][1] else np.array([])
#         got[name] = (tt, pp)
#     return got

# def plot_pred_vs_true_hex(t, p, name="Te", unit="eV", gridsize=80, savepath=None):
#     if t.size == 0:
#         raise ValueError(f"No data to plot for {name} (after masking/finite filter)")
#     mae = float(np.mean(np.abs(p - t)))
#     nmae = float(mae / (np.mean(np.abs(t)) + 1e-12) * 100.0)
#     ss_res = float(np.sum((p - t) ** 2))
#     ss_tot = float(np.sum((t - np.mean(t)) ** 2) + 1e-12)
#     r2 = 1.0 - ss_res / ss_tot

#     fig, ax = plt.subplots(figsize=(6.2, 4.2))
#     hb = ax.hexbin(t, p, gridsize=gridsize, norm=LogNorm(), mincnt=1)
#     lim = np.nanpercentile(np.concatenate([t, p]), [0.5, 99.5])
#     lo, hi = float(lim[0]), float(lim[1])
#     ax.plot([lo, hi], [lo, hi], 'k--', lw=1, alpha=0.7)
#     ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
#     ax.set_xlabel(f"SOLPS-ITER: {name} ({unit})")
#     ax.set_ylabel(f"UNet: {name} ({unit})")
#     cbar = fig.colorbar(hb, ax=ax); cbar.set_label("log10(count)")
#     ax.set_title(f"MAE={mae:.2g} {unit} | NMAE={nmae:.1f}% | R²={r2:.3f}")
#     ax.grid(True, alpha=0.25)
#     if savepath:
#         fig.savefig(savepath, dpi=300, bbox_inches="tight")
#     plt.show()
#     return {"MAE": mae, "NMAE": nmae, "R2": r2}

# def plot_all_channels_scatter(
#     model, loader, norm, *, device="cuda",
#     target_keys=("Te","Ti","ni","ne"),
#     units={"Te":"eV","Ti":"eV","ni":"m⁻³","ne":"m⁻³"},
#     data_is_log10_map=None,  # per-channel
#     max_samples=6, gridsize=80, save_prefix=None
# ):
#     gathered = gather_flat_true_pred(
#         model, loader, norm, list(target_keys),
#         device=device, max_samples=max_samples,
#         data_is_log10_map=data_is_log10_map
#     )
#     metrics = {}
#     for name in target_keys:
#         t, p = gathered[name]
#         sp = f"{save_prefix}_{name}.png" if save_prefix else None
#         metrics[name] = plot_pred_vs_true_hex(t, p, name=name, unit=units.get(name, ""), gridsize=gridsize, savepath=sp)
#     return metrics
# plotting.py  (patched parts)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import torch
from .predict import predict_multi

def _maybe_unlog(arr, do_unlog, clip=(-50.0, 50.0)):
    if not do_unlog:
        return arr
    a = np.clip(arr, *clip)
    return np.power(10.0, a)

def gather_flat_true_pred(
    model, loader, norm, target_keys, *,
    dataset_keys=("Te","ne","ni","Ti"),   # <-- DATASET ORDER (y_phys channel order)
    device="cuda", max_samples=6,
    data_is_log10_map=None
):
    """
    Returns dict[name] = (true_flat, pred_flat) in *linear* physical units,
    selecting channels by NAME using dataset_keys -> indices.
    """
    if data_is_log10_map is None:
        data_is_log10_map = {k: False for k in target_keys}

    # name -> dataset index
    name_to_idx = {name: i for i, name in enumerate(dataset_keys)}

    got = {k: ([], []) for k in target_keys}

    ds = loader.dataset
    base_ds, ids = (ds.dataset, ds.indices) if hasattr(ds, "dataset") else (ds, np.arange(len(ds)))
    picks = np.random.choice(len(ids), size=min(max_samples, len(ids)), replace=False)

    for j in picks:
        i = int(ids[j]); b = base_ds[i]
        yN, m = b["y"], b["mask"]

        with torch.no_grad():
            y_phys = norm.inverse(yN.unsqueeze(0), m.unsqueeze(0)).squeeze(0).cpu().numpy()  # (C,H,W)

        mask = (m.squeeze(0).cpu().numpy() > 0.5)

        params_scaled = getattr(base_ds, "params", None)
        params_scaled = None if params_scaled is None else np.asarray(params_scaled[i], dtype=np.float32)

        Ypred = predict_multi(model, norm, mask.astype(np.float32), params_scaled, device=device, as_numpy=True)  # (C,H,W)

        for name in target_keys:
            k = name_to_idx[name]  # pick channel by NAME
            t = y_phys[k][mask]
            p = Ypred[k][mask]

            # (If any channel is stored in log10, convert back to linear)
            do_unlog = bool(data_is_log10_map.get(name, False))
            t = _maybe_unlog(t, do_unlog)
            p = _maybe_unlog(p, do_unlog)

            good = np.isfinite(t) & np.isfinite(p)
            if good.any():
                got[name][0].append(t[good].astype(np.float64))
                got[name][1].append(p[good].astype(np.float64))

    for name in target_keys:
        tt = np.concatenate(got[name][0]) if got[name][0] else np.array([], dtype=np.float64)
        pp = np.concatenate(got[name][1]) if got[name][1] else np.array([], dtype=np.float64)
        got[name] = (tt, pp)
    return got

def plot_pred_vs_true_hex(t, p, name="Te", unit="eV", gridsize=80, savepath=None):
    if t.size == 0:
        raise ValueError(f"No data to plot for {name} (after masking/finite filter)")

    # metrics (float64 for stability)
    mae = float(np.mean(np.abs(p - t)))
    denom = float(np.mean(np.abs(t)) + 1e-12)
    nmae = float(mae / denom * 100.0)
    t_mean = float(np.mean(t))
    ss_res = float(np.sum((p - t) ** 2))
    ss_tot = float(np.sum((t - t_mean) ** 2) + 1e-12)
    r2 = float(1.0 - ss_res / ss_tot)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    hb = ax.hexbin(t, p, gridsize=gridsize, norm=LogNorm(), mincnt=1)
    lim = np.nanpercentile(np.concatenate([t, p]), [0.5, 99.5])
    lo, hi = float(lim[0]), float(lim[1])
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1, alpha=0.7)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(f"SOLPS-ITER: {name} ({unit})")
    ax.set_ylabel(f"UNet: {name} ({unit})")
    cbar = fig.colorbar(hb, ax=ax); cbar.set_label("log10(count)")
    ax.set_title(f"MAE={mae:.2g} {unit} | NMAE={nmae:.1f}% | R²={r2:.3f}")
    ax.grid(True, alpha=0.25)
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.show()
    return {"MAE": mae, "NMAE": nmae, "R2": r2}

def plot_all_channels_scatter(
    model, loader, norm, *, device="cuda",
    target_keys=("Te","Ti","ni","ne"),      # PLOT order (anything you like)
    dataset_keys=("Te","ne","ni","Ti"),     # DATASET order (must match y_phys)
    units={"Te":"eV","Ti":"eV","ni":"m⁻³","ne":"m⁻³"},
    data_is_log10_map=None,                 # per-channel (all False for your data)
    max_samples=6, gridsize=80, save_prefix=None
):
    gathered = gather_flat_true_pred(
        model, loader, norm, list(target_keys),
        dataset_keys=dataset_keys,
        device=device,
        max_samples=max_samples,
        data_is_log10_map=data_is_log10_map,
    )
    metrics = {}
    for name in target_keys:
        t, p = gathered[name]
        sp = f"{save_prefix}_{name}.png" if save_prefix else None
        metrics[name] = plot_pred_vs_true_hex(t, p, name=name, unit=units.get(name, ""), gridsize=gridsize, savepath=sp)
    return metrics

