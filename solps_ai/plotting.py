import numpy as np
import matplotlib.pyplot as plt

def plot_training_curves(history, title="Training"):
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

