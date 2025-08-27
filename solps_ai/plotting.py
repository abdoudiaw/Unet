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

