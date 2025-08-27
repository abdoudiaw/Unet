import numpy as np
import matplotlib.pyplot as plt
import torch

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
#from solps_ai.predict import predict_te  # uses norm.inverse to return eV :contentReference[oaicite:2]{index=2}

def show_case_prediction(model, loader, norm, device="cuda",
                         idx=None, R2d=None, Z2d=None, savepath=None):
    """
    Pick a case from `loader` (train/val), run model, and plot:
      Target Te (eV) | Prediction Te (eV) | |Error| (eV)
    If R2d/Z2d (cell centers) are given, axes are in meters.
    Returns: (global_idx, params_scaled, fig)
    """
    # unwrap Subset -> underlying SOLPSDataset
    subset  = loader.dataset                      # torch.utils.data.Subset
    base_ds = subset.dataset                      # SOLPSDataset with .params, .mask, .Te, .normalizer
    ids     = subset.indices

    # choose a sample within this split
    j = np.random.randint(len(ids)) if idx is None else int(idx if idx >= 0 else len(ids) + idx)
    i = int(ids[j])  # global index into base_ds

    # fetch normalized target & mask from the dataset (returns {"x","y","mask"}) :contentReference[oaicite:3]{index=3}
    b = base_ds[i]
    y_norm = b["y"]            # (1,H,W), normalized
    m_t    = b["mask"]         # (1,H,W)
    mask   = m_t.squeeze(0).cpu().numpy()  # (H,W)

    # target Te in eV
    with torch.no_grad():
        te_target = norm.inverse(y_norm, m_t).squeeze(0).cpu().numpy().astype(np.float32)

    # the SAME scaled params used for training inputs (dataset stores them scaled) :contentReference[oaicite:4]{index=4}
    params_scaled = base_ds.params[i].astype(np.float32)  # shape (P,)

    # model prediction in eV using your helper (builds [mask, params] internally) :contentReference[oaicite:5]{index=5}
    te_pred = predict_te(model, norm, mask, params_scaled, device=device, as_numpy=True)

    # error map inside mask
    err = np.abs(te_pred - te_target) * (mask > 0.5)

    # axes (optional)
    if R2d is not None and Z2d is not None:
        H, W = te_pred.shape
        R_axis = np.linspace(float(R2d.min()), float(R2d.max()), W)
        Z_axis = np.linspace(float(Z2d.min()), float(Z2d.max()), H)
        extent = [R_axis[0], R_axis[-1], Z_axis[0], Z_axis[-1]]
        xlab, ylab = "R [m]", "Z [m]"
    else:
        extent = None
        xlab, ylab = "x", "y"

    # shared color limits for target/pred
    both = np.where(mask > 0.5, np.stack([te_target, te_pred], 0), np.nan)
    vmin, vmax = np.nanpercentile(both, [1, 99])

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)

    def _imshow(ax, data, title, vlim=None):
        d = np.where(mask > 0.5, data, np.nan)
        im = ax.imshow(d, origin="lower", extent=extent, vmin=None if vlim is None else vlim[0],
                       vmax=None if vlim is None else vlim[1], aspect="equal")
        ax.set_title(title); ax.set_xlabel(xlab); ax.set_ylabel(ylab)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    _imshow(axes[0], te_target, "Target Te (eV)", vlim=(vmin, vmax))
    _imshow(axes[1], te_pred,   "Prediction Te (eV)", vlim=(vmin, vmax))
    # error gets its own scale
    e99 = float(np.nanpercentile(err, 99))
    _imshow(axes[2], err, "Abs error (eV)", vlim=(0.0, e99))

    P = params_scaled.shape[0]
    fig.suptitle(f"Dataset case idx={i} (split idx={j})  |  P={P} params", y=1.03, fontsize=12)
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.show()

    return i, params_scaled, fig

# import numpy as np
# from scipy.interpolate import RegularGridInterpolator
# import matplotlib.pyplot as plt

# # 1) define raster axes (same as Option 1)
# H, W = Te_map_eV.shape
# Rmin, Rmax = float(R2d.min()), float(R2d.max())
# Zmin, Zmax = float(Z2d.min()), float(Z2d.max())
# R_axis = np.linspace(Rmin, Rmax, W)
# Z_axis = np.linspace(Zmin, Zmax, H)

# # 2) build interpolator from raster grid -> Te
# f = RegularGridInterpolator((Z_axis, R_axis), Te_map_eV, bounds_error=False, fill_value=np.nan)

# # 3) sample at SOLPS mesh centers
# pts = np.stack([Z2d.ravel(), R2d.ravel()], axis=-1)
# Te_mesh = f(pts).reshape(Z2d.shape)   # (98, 38)

# # 4) make cell corners from centers (robust edge extrapolation)
# def centers_to_corners(Rc, Zc):
#     Rc = np.asarray(Rc, float); Zc = np.asarray(Zc, float)
#     H, W = Rc.shape
#     # pad by linear extrapolation (one row/col on each side)
#     Rp = np.empty((H+2, W+2)); Zp = np.empty((H+2, W+2))
#     Rp[1:-1,1:-1] = Rc; Zp[1:-1,1:-1] = Zc
#     # rows
#     Rp[0,1:-1]  = 2*Rc[0,:]   - Rc[1,:]
#     Rp[-1,1:-1] = 2*Rc[-1,:]  - Rc[-2,:]
#     Zp[0,1:-1]  = 2*Zc[0,:]   - Zc[1,:]
#     Zp[-1,1:-1] = 2*Zc[-1,:]  - Zc[-2,:]
#     # cols
#     Rp[1:-1,0]  = 2*Rc[:,0]   - Rc[:,1]
#     Rp[1:-1,-1] = 2*Rc[:,-1]  - Rc[:,-2]
#     Zp[1:-1,0]  = 2*Zc[:,0]   - Zc[:,1]
#     Zp[1:-1,-1] = 2*Zc[:,-1]  - Zc[:,-2]
#     # corners
#     Rp[0,0]     = 2*Rp[0,1]   - Rp[0,2]
#     Rp[0,-1]    = 2*Rp[0,-2]  - Rp[0,-3]
#     Rp[-1,0]    = 2*Rp[-1,1]  - Rp[-1,2]
#     Rp[-1,-1]   = 2*Rp[-1,-2] - Rp[-1,-3]
#     Zp[0,0]     = 2*Zp[0,1]   - Zp[0,2]
#     Zp[0,-1]    = 2*Zp[0,-2]  - Zp[0,-3]
#     Zp[-1,0]    = 2*Zp[-1,1]  - Zp[-1,2]
#     Zp[-1,-1]   = 2*Zp[-1,-2] - Zp[-1,-3]
#     # average 4 neighbors to get (H+1, W+1) corners
#     Rcorn = 0.25*(Rp[0:-1,0:-1] + Rp[0:-1,1:] + Rp[1:,0:-1] + Rp[1:,1:])
#     Zcorn = 0.25*(Zp[0:-1,0:-1] + Zp[0:-1,1:] + Zp[1:,0:-1] + Zp[1:,1:])
#     return Rcorn, Zcorn

# Rcorn, Zcorn = centers_to_corners(R2d, Z2d)

# # 5) plot on the SOLPS mesh
# vmin, vmax = np.nanpercentile(Te_mesh, [1, 99])
# plt.figure(figsize=(6,6))
# pm = plt.pcolormesh(Rcorn, Zcorn, Te_mesh, shading='auto', vmin=vmin, vmax=vmax)
# cb = plt.colorbar(pm); cb.set_label('Te [eV]')
# plt.gca().set_aspect('equal', adjustable='box')
# plt.xlabel('R [m]'); plt.ylabel('Z [m]'); plt.title('Te (eV) on SOLPS mesh')
# plt.show()
