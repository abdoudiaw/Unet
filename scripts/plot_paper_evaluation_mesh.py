import argparse
import csv
import json
import os

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import numpy as np
from scipy.stats import spearmanr

import quixote
from quixote import SolpsData

from solps_ai.predict import load_checkpoint, predict_fields, scale_params
from solps_ai.utils import pick_device

SIGNED_FIELDS = {"ua", "Qp", "Qe", "Qi", "Sm"}
SPARSE_POSITIVE_FIELDS = {"Sp"}
LOG_DISPLAY_FIELDS = {"ne", "ni", "Sp"}
UNITS_BY_KEY = {
    "Te": "eV",
    "Ti": "eV",
    "ne": "m^-3",
    "ni": "m^-3",
    "ua": "m/s",
    "Sp": "m^-3 s^-1",
    "Qp": "W/m^3",
    "Qe": "W/m^3",
    "Qi": "W/m^3",
    "Sm": "N/m^3",
}


def split_indices(N, split=0.85, seed=42):
    idx = np.arange(N)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    cut = int(split * N)
    return idx[:cut], idx[cut:]


def load_npz_all(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    if "Y" in d.files:
        Y = d["Y"].astype(np.float32)
        y_keys = [str(k) for k in d["y_keys"]]
    elif "Te" in d.files:
        Y = d["Te"][:, None, :, :].astype(np.float32)
        y_keys = ["Te"]
    else:
        raise KeyError("Dataset must contain Y (+y_keys) or Te.")

    if "mask" in d.files:
        m = d["mask"]
        if m.ndim == 2:
            m = np.repeat(m[None, :, :], Y.shape[0], axis=0)
        m = (m > 0.5).astype(np.float32)
    else:
        m = np.ones((Y.shape[0], Y.shape[2], Y.shape[3]), dtype=np.float32)

    if "params" in d.files:
        p = d["params"].astype(np.float32)
    elif "X" in d.files:
        p = d["X"].astype(np.float32)
    else:
        p = np.zeros((Y.shape[0], 0), dtype=np.float32)

    if "param_keys" in d.files:
        p_keys = [str(k) for k in d["param_keys"]]
    elif "x_keys" in d.files:
        p_keys = [str(k) for k in d["x_keys"]]
    else:
        p_keys = [f"p{i}" for i in range(p.shape[1])]

    return Y, y_keys, m, p, p_keys


def pick_reference_run(base_dir):
    runs = sorted(
        d for d in os.listdir(base_dir)
        if d.startswith("run_") and os.path.isdir(os.path.join(base_dir, d))
    )
    if not runs:
        raise RuntimeError(f"No run_* folders found under {base_dir}")
    for rn in runs:
        if os.path.exists(os.path.join(base_dir, rn, "params.json")):
            return rn
    return runs[0]


def load_mesh_polygons(base_dir, run_name=None):
    rn = run_name if run_name is not None else pick_reference_run(base_dir)
    shot = SolpsData(os.path.join(quixote.module_path(), os.path.join(base_dir, rn)))
    grid = np.asarray(shot.grid, dtype=np.float32)  # (H,W,4,2)
    H, W = grid.shape[:2]
    polys = [Polygon(grid[i, j], closed=True) for i in range(H) for j in range(W)]
    return rn, grid, polys


def center_crop_2d(a, shape_hw):
    H0, W0 = shape_hw
    H, W = a.shape
    if (H, W) == (H0, W0):
        return a
    if H < H0 or W < W0:
        raise ValueError(f"Cannot crop from {(H,W)} to larger {(H0,W0)}")
    top = (H - H0) // 2
    left = (W - W0) // 2
    return a[top:top + H0, left:left + W0]


def pick_field_error_mode(requested_mode, y_key):
    if requested_mode != "auto":
        return requested_mode
    # Percent error is unstable for near-zero/sparse channels.
    if y_key in SIGNED_FIELDS or y_key in SPARSE_POSITIVE_FIELDS:
        return "scaled_abs"
    return "percent_robust"


def pick_field_log_display(requested_mode, y_key):
    if requested_mode == "on":
        return True
    if requested_mode == "off":
        return False
    # auto
    return y_key in LOG_DISPLAY_FIELDS


def compute_error_map(y_true, y_pred, mask, mode="percent", eps=1e-3, scale=None, rel_floor=0.02):
    if mode == "abs":
        e = np.abs(y_pred - y_true)
    elif mode == "percent":
        denom = np.maximum(np.abs(y_true), eps)
        e = 100.0 * np.abs(y_pred - y_true) / denom
    elif mode == "percent_robust":
        scl = eps if scale is None else max(float(scale), eps)
        denom = np.maximum(np.abs(y_true), rel_floor * scl)
        denom = np.maximum(denom, eps)
        e = 100.0 * np.abs(y_pred - y_true) / denom
    elif mode == "smape":
        denom = np.abs(y_true) + np.abs(y_pred) + eps
        e = 200.0 * np.abs(y_pred - y_true) / denom
    elif mode == "scaled_abs":
        scl = eps if scale is None else max(float(scale), eps)
        e = 100.0 * np.abs(y_pred - y_true) / scl
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return np.where(mask > 0.5, e, np.nan)


def apply_error_sign(err, y_true, y_pred, sign_mode):
    if sign_mode == "absolute":
        return err
    if sign_mode == "signed":
        return np.sign(y_pred - y_true) * err
    raise ValueError(f"Unknown error sign mode: {sign_mode}")


def compute_metrics(y_true, y_pred, m):
    v = m > 0.5
    t = y_true[v].astype(np.float64, copy=False)
    p = y_pred[v].astype(np.float64, copy=False)
    if t.size == 0:
        return {"mae": np.nan, "rmse": np.nan, "pearson": np.nan, "spearman": np.nan}
    e = p - t
    mae = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(np.square(e))))
    if t.size > 2:
        pearson = float(np.corrcoef(t, p)[0, 1])
        spearman = float(spearmanr(t, p).correlation)
    else:
        pearson = np.nan
        spearman = np.nan
    return {"mae": mae, "rmse": rmse, "pearson": pearson, "spearman": spearman}


def compute_log_metrics(y_true, y_pred, m, eps=1e-12):
    v = m > 0.5
    t = np.maximum(y_true[v].astype(np.float64, copy=False), eps)
    p = np.maximum(y_pred[v].astype(np.float64, copy=False), eps)
    if t.size == 0:
        return {"log_mae": np.nan, "log_rmse": np.nan, "log_pearson": np.nan, "log_spearman": np.nan}
    lt = np.log10(t)
    lp = np.log10(p)
    e = lp - lt
    log_mae = float(np.mean(np.abs(e)))
    log_rmse = float(np.sqrt(np.mean(np.square(e))))
    if lt.size > 2:
        log_pearson = float(np.corrcoef(lt, lp)[0, 1])
        log_spearman = float(spearmanr(lt, lp).correlation)
    else:
        log_pearson = np.nan
        log_spearman = np.nan
    return {
        "log_mae": log_mae,
        "log_rmse": log_rmse,
        "log_pearson": log_pearson,
        "log_spearman": log_spearman,
    }


def add_mesh_panel(
    fig,
    ax,
    polys,
    values2d,
    title,
    cmap,
    vmin=None,
    vmax=None,
    show_xlabel=True,
    show_ylabel=True,
    add_colorbar=True,
):
    vals = values2d.reshape(-1)
    mvals = np.ma.masked_invalid(vals)
    pc = PatchCollection(polys, cmap=cmap, edgecolor="none", linewidths=0.0)
    pc.set_array(mvals)
    if vmin is not None and vmax is not None:
        pc.set_clim(vmin, vmax)
    ax.add_collection(pc)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_xlabel("R [m]" if show_xlabel else "")
    ax.set_ylabel("Z [m]" if show_ylabel else "")
    ax.set_title(title)
    if add_colorbar:
        fig.colorbar(pc, ax=ax, fraction=0.032, pad=0.01)


def plot_mesh_triplet(
    path,
    polys,
    y_true,
    y_pred,
    mask,
    y_key,
    run_name,
    sample_idx,
    error_mode,
    error_eps,
    error_rel_floor,
    error_scale,
    error_sign,
    log_display=False,
    log_display_eps=1e-12,
):
    yt = np.where(mask > 0.5, y_true, np.nan)
    yp = np.where(mask > 0.5, y_pred, np.nan)
    err = compute_error_map(
        y_true, y_pred, mask, mode=error_mode, eps=error_eps, scale=error_scale, rel_floor=error_rel_floor
    )
    err = apply_error_sign(err, y_true, y_pred, error_sign)

    if log_display:
        yt_plot = np.log10(np.maximum(yt, log_display_eps))
        yp_plot = np.log10(np.maximum(yp, log_display_eps))
        tp0 = "Truth [log10]"
        tp1 = "Prediction [log10]"
    else:
        yt_plot = yt
        yp_plot = yp
        tp0 = "Truth"
        tp1 = "Prediction"

    vmin = float(np.nanpercentile(yt_plot, 1))
    vmax = float(np.nanpercentile(yt_plot, 99))
    if error_sign == "signed":
        emax = float(np.nanpercentile(np.abs(err), 95))
        emin = -emax
        err_cmap = "coolwarm"
    else:
        emax = float(np.nanpercentile(err, 95))
        emin = 0.0
        err_cmap = "magma"

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    add_mesh_panel(fig, axes[0], polys, yt_plot, tp0, cmap="inferno", vmin=vmin, vmax=vmax)
    add_mesh_panel(fig, axes[1], polys, yp_plot, tp1, cmap="inferno", vmin=vmin, vmax=vmax)
    err_title = {
        "abs": "Abs Error",
        "percent": "Percent Error [%]",
        "percent_robust": "Percent Error Robust [%]",
        "smape": "sMAPE [%]",
        "scaled_abs": "Scaled Abs Error [%]",
    }[error_mode]
    if error_sign == "signed":
        err_title = f"Signed {err_title}"
    add_mesh_panel(fig, axes[2], polys, err, err_title, cmap=err_cmap, vmin=emin, vmax=emax)
    fig.suptitle(f"{y_key} mesh-eval | run={run_name} | val_idx={sample_idx}")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _field_label_with_unit(y_key, log_display):
    unit = UNITS_BY_KEY.get(y_key, "")
    if unit:
        return f"{y_key} [{unit}]"
    return y_key


def plot_all_fields_grid(
    *,
    path,
    fields,
    data_map,
    pred_map,
    Y,
    M,
    P,
    model,
    norm,
    p_mu,
    p_std,
    device,
    gidx,
    mesh_hw,
    polys,
    run_name,
    error_mode,
    error_eps,
    error_rel_floor,
    error_sign,
    log_display_mode,
    log_display_eps,
    paper_grid_rows=3,
):
    H0, W0 = mesh_hw
    n_fields = len(fields)
    nrows = n_fields
    ncols = 3
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(8.8, max(2.35 * nrows, 6.0)),
        constrained_layout=False,
        gridspec_kw={"wspace": 0.28, "hspace": 0.22},
    )
    if nrows == 1:
        axes = np.array([axes])

    m_full = M[gidx]
    p_raw = P[gidx]
    p_in = scale_params(p_raw, p_mu, p_std)

    for i, k in enumerate(fields):
        rr = i
        y_idx_data = data_map[k]
        y_idx_pred = pred_map[k]
        y_true = center_crop_2d(Y[gidx, y_idx_data], (H0, W0))
        m = center_crop_2d(m_full, (H0, W0))
        y_pred = center_crop_2d(
            predict_fields(model, norm, m_full, p_in, device=device, as_numpy=True)[y_idx_pred],
            (H0, W0),
        )

        log_display = pick_field_log_display(log_display_mode, k)
        mode_eff = pick_field_error_mode(error_mode, k)
        v = m > 0.5
        vscale = float(np.nanpercentile(np.abs(y_true[v]), 95)) if np.any(v) else error_eps

        yt = np.where(m > 0.5, y_true, np.nan)
        yp = np.where(m > 0.5, y_pred, np.nan)
        err = compute_error_map(y_true, y_pred, m, mode=mode_eff, eps=error_eps, scale=vscale, rel_floor=error_rel_floor)
        err = apply_error_sign(err, y_true, y_pred, error_sign)

        if log_display:
            yt_plot = np.log10(np.maximum(yt, log_display_eps))
            yp_plot = np.log10(np.maximum(yp, log_display_eps))
            t0 = f"Truth [log10] | {_field_label_with_unit(k, log_display=True)}"
            t1 = f"Prediction [log10] | {_field_label_with_unit(k, log_display=True)}"
        else:
            yt_plot = yt
            yp_plot = yp
            t0 = f"Truth | {_field_label_with_unit(k, log_display=False)}"
            t1 = f"Prediction | {_field_label_with_unit(k, log_display=False)}"

        vmin = float(np.nanpercentile(yt_plot, 1))
        vmax = float(np.nanpercentile(yt_plot, 99))
        if error_sign == "signed":
            emax = float(np.nanpercentile(np.abs(err), 95))
            emin = -emax
            ecmap = "coolwarm"
        else:
            emin = 0.0
            if mode_eff == "abs" and k in {"ne", "ni"}:
                err = np.log10(np.maximum(err, max(error_eps, 1e-30)))
                emax = float(np.nanpercentile(err, 99))
                emin = float(np.nanpercentile(err, 5))
            else:
                emax = float(np.nanpercentile(err, 95))
            ecmap = "magma"

        err_title = {
            "abs": "Abs Error",
            "percent": "Percent Error [%]",
            "percent_robust": "Percent Error Robust [%]",
            "smape": "sMAPE [%]",
            "scaled_abs": "Scaled Abs Error [%]",
        }[mode_eff]
        if mode_eff == "abs" and k in {"ne", "ni"} and error_sign != "signed":
            err_title = "log10 Abs Error"
        if error_sign == "signed":
            err_title = f"Signed {err_title}"

        # Compact panel titles: only top row gets Truth/Prediction/Error labels.
        ttl0 = "Truth" if rr == 0 else ""
        ttl1 = "Prediction" if rr == 0 else ""
        ttl2 = err_title if rr == 0 else ""
        show_x = rr == (nrows - 1)
        add_mesh_panel(fig, axes[rr, 0], polys, yt_plot, ttl0, cmap="inferno", vmin=vmin, vmax=vmax, show_xlabel=show_x, show_ylabel=True)
        add_mesh_panel(fig, axes[rr, 1], polys, yp_plot, ttl1, cmap="inferno", vmin=vmin, vmax=vmax, show_xlabel=show_x, show_ylabel=False)
        add_mesh_panel(fig, axes[rr, 2], polys, err, ttl2, cmap=ecmap, vmin=emin, vmax=emax, show_xlabel=show_x, show_ylabel=False)
        # Per-field label once, small and clean.
        axes[rr, 0].text(
            0.02, 0.98, _field_label_with_unit(k, log_display=False),
            transform=axes[rr, 0].transAxes, ha="left", va="top", fontsize=8,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.0),
        )

    fig.suptitle(f"val_idx={int(gidx)}", fontsize=13)
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def evaluate_field(
    *,
    outdir,
    y_key,
    y_idx_data,
    y_idx_pred,
    Y,
    M,
    P,
    p_keys,
    val_idx,
    model,
    norm,
    p_mu,
    p_std,
    device,
    mesh_hw,
    polys,
    run_name,
    n_examples,
    scatter_points,
    error_mode,
    error_eps,
    error_rel_floor,
    error_sign,
    log_display_mode,
    log_display_eps,
    log_metrics,
    log_metrics_eps,
    sweep_param,
    seed,
):
    os.makedirs(outdir, exist_ok=True)
    H0, W0 = mesh_hw
    y_true_cat, y_pred_cat = [], []
    abs_err_cat, pct_err_cat = [], []
    log_true_cat, log_pred_cat = [], []
    sample_metrics = []

    error_mode_eff = pick_field_error_mode(error_mode, y_key)
    log_display_eff = pick_field_log_display(log_display_mode, y_key)

    for j, gidx in enumerate(val_idx):
        y_true_full = Y[gidx, y_idx_data]
        m_full = M[gidx]
        p_raw = P[gidx]
        p_in = scale_params(p_raw, p_mu, p_std)
        y_pred_full = predict_fields(model, norm, m_full, p_in, device=device, as_numpy=True)[y_idx_pred]

        y_true = center_crop_2d(y_true_full, (H0, W0))
        m = center_crop_2d(m_full, (H0, W0))
        y_pred = center_crop_2d(y_pred_full, (H0, W0))

        metrics = compute_metrics(y_true, y_pred, m)
        sample_metrics.append(metrics)

        v = m > 0.5
        y_true_cat.append(y_true[v])
        y_pred_cat.append(y_pred[v])
        abs_err_cat.append(np.abs(y_pred[v] - y_true[v]))
        vscale = float(np.nanpercentile(np.abs(y_true[v]), 95)) if np.any(v) else error_eps
        pct_map = compute_error_map(y_true, y_pred, m, mode="percent", eps=error_eps)
        pct_err_cat.append(pct_map[v])

        if log_metrics:
            log_true_cat.append(np.log10(np.maximum(y_true[v], log_metrics_eps)))
            log_pred_cat.append(np.log10(np.maximum(y_pred[v], log_metrics_eps)))

        if j < n_examples:
            plot_mesh_triplet(
                path=os.path.join(outdir, f"mesh_example_{j:02d}.png"),
                polys=polys,
                y_true=y_true,
                y_pred=y_pred,
                mask=m,
                y_key=y_key,
                run_name=run_name,
                sample_idx=int(gidx),
                error_mode=error_mode_eff,
                error_eps=error_eps,
                error_rel_floor=error_rel_floor,
                error_scale=vscale,
                error_sign=error_sign,
                log_display=log_display_eff,
                log_display_eps=log_display_eps,
            )

    y_true_cat = np.concatenate(y_true_cat).astype(np.float64, copy=False)
    y_pred_cat = np.concatenate(y_pred_cat).astype(np.float64, copy=False)
    abs_err_cat = np.concatenate(abs_err_cat)
    pct_err_cat = np.concatenate(pct_err_cat)
    if log_metrics:
        log_true_cat = np.concatenate(log_true_cat)
        log_pred_cat = np.concatenate(log_pred_cat)

    rng = np.random.default_rng(seed)
    take = min(scatter_points, y_true_cat.size)
    ii = rng.choice(y_true_cat.size, size=take, replace=False)

    pearson_global = float(np.corrcoef(y_true_cat, y_pred_cat)[0, 1])
    spearman_global = float(spearmanr(y_true_cat, y_pred_cat).correlation)
    mae_global = float(np.mean(np.abs(y_pred_cat - y_true_cat)))
    rmse_global = float(np.sqrt(np.mean(np.square(y_pred_cat - y_true_cat))))
    p90_abs_err = float(np.percentile(abs_err_cat, 90))
    p95_abs_err = float(np.percentile(abs_err_cat, 95))
    p90_pct_err = float(np.percentile(pct_err_cat, 90))
    p95_pct_err = float(np.percentile(pct_err_cat, 95))

    plot_err_map = compute_error_map(
        y_true_cat,
        y_pred_cat,
        np.ones_like(y_true_cat),
        mode=error_mode_eff,
        eps=error_eps,
        scale=float(np.percentile(np.abs(y_true_cat), 95)),
        rel_floor=error_rel_floor,
    )
    plot_err_vals = plot_err_map[np.isfinite(plot_err_map)]
    p90_plot_err = float(np.percentile(plot_err_vals, 90))
    p95_plot_err = float(np.percentile(plot_err_vals, 95))

    metrics_out = {
        "y_key": y_key,
        "error_mode_used": error_mode_eff,
        "error_sign_used": error_sign,
        "display_scale_used": "log10" if log_display_eff else "linear",
        "n_val_samples": int(len(val_idx)),
        "n_valid_points": int(y_true_cat.size),
        "global_mae": mae_global,
        "global_rmse": rmse_global,
        "global_pearson": pearson_global,
        "global_spearman": spearman_global,
        "p90_plot_error": p90_plot_err,
        "p95_plot_error": p95_plot_err,
        "p90_abs_error": p90_abs_err,
        "p95_abs_error": p95_abs_err,
        "p90_percent_error": p90_pct_err,
        "p95_percent_error": p95_pct_err,
        "mean_sample_pearson": float(np.nanmean([m["pearson"] for m in sample_metrics])),
        "mean_sample_spearman": float(np.nanmean([m["spearman"] for m in sample_metrics])),
    }
    if log_metrics:
        metrics_out.update(compute_log_metrics(y_true_cat, y_pred_cat, np.ones_like(y_true_cat), eps=log_metrics_eps))

    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(metrics_out, f, indent=2)

    # Scatter
    fig = plt.figure(figsize=(6, 5))
    plt.hexbin(y_true_cat[ii], y_pred_cat[ii], gridsize=80, bins="log", mincnt=1)
    lo = float(np.percentile(y_true_cat[ii], 1))
    hi = float(np.percentile(y_true_cat[ii], 99))
    plt.plot([lo, hi], [lo, hi], "w--", linewidth=1.5, label="y=x")
    plt.xlabel(f"True {y_key}")
    plt.ylabel(f"Predicted {y_key}")
    plt.title(f"Scatter (Pearson={pearson_global:.4f}, Spearman={spearman_global:.4f})")
    plt.colorbar(label="log10(count)")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "scatter_true_vs_pred.png"), dpi=220)
    plt.close(fig)

    if log_metrics:
        take_l = min(scatter_points, log_true_cat.size)
        iil = rng.choice(log_true_cat.size, size=take_l, replace=False)
        fig = plt.figure(figsize=(6, 5))
        plt.hexbin(log_true_cat[iil], log_pred_cat[iil], gridsize=80, bins="log", mincnt=1)
        lo = float(np.percentile(log_true_cat[iil], 1))
        hi = float(np.percentile(log_true_cat[iil], 99))
        plt.plot([lo, hi], [lo, hi], "w--", linewidth=1.5, label="y=x")
        plt.xlabel(f"log10 True {y_key}")
        plt.ylabel(f"log10 Predicted {y_key}")
        plt.title(
            f"log10 Scatter (Pearson={metrics_out['log_pearson']:.4f}, "
            f"Spearman={metrics_out['log_spearman']:.4f})"
        )
        plt.colorbar(label="log10(count)")
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "scatter_log10_true_vs_pred.png"), dpi=220)
        plt.close(fig)

    # Error histogram
    if error_mode_eff == "abs":
        hvals = abs_err_cat
        hlabel = f"|Error| ({y_key})"
        hname = "abs_error_hist.png"
        htitle = f"Abs Error Histogram (P90={p90_abs_err:.3g}, P95={p95_abs_err:.3g})"
    elif error_mode_eff == "percent":
        hvals = pct_err_cat
        hlabel = "Percent Error [%]"
        hname = "percent_error_hist.png"
        htitle = f"Percent Error Histogram (P90={p90_pct_err:.3g}, P95={p95_pct_err:.3g})"
    elif error_mode_eff == "percent_robust":
        scale_hist = max(float(np.percentile(np.abs(y_true_cat), 95)), error_eps)
        denom = np.maximum(np.abs(y_true_cat), error_rel_floor * scale_hist)
        denom = np.maximum(denom, error_eps)
        hvals = 100.0 * abs_err_cat / denom
        hlabel = "Percent Error Robust [%]"
        hname = "percent_robust_error_hist.png"
        htitle = (
            f"Percent Robust Histogram (floor={100.0*error_rel_floor:.1f}%*P95|truth|, "
            f"P90={float(np.percentile(hvals,90)):.3g}, P95={float(np.percentile(hvals,95)):.3g})"
        )
    elif error_mode_eff == "scaled_abs":
        scale_hist = max(float(np.percentile(np.abs(y_true_cat), 95)), error_eps)
        hvals = 100.0 * abs_err_cat / scale_hist
        hlabel = "Scaled Abs Error [%]"
        hname = "scaled_abs_error_hist.png"
        htitle = (
            f"Scaled Abs Error Histogram (scale=P95|truth|, "
            f"P90={float(np.percentile(hvals,90)):.3g}, P95={float(np.percentile(hvals,95)):.3g})"
        )
    else:
        sm = compute_error_map(y_true_cat, y_pred_cat, np.ones_like(y_true_cat), mode="smape", eps=error_eps)
        hvals = sm[np.isfinite(sm)]
        hlabel = "sMAPE [%]"
        hname = "smape_error_hist.png"
        htitle = f"sMAPE Histogram (P90={float(np.percentile(hvals,90)):.3g}, P95={float(np.percentile(hvals,95)):.3g})"

    fig = plt.figure(figsize=(6, 4))
    plt.hist(hvals[np.isfinite(hvals)], bins=80, alpha=0.9)
    plt.xlabel(hlabel)
    plt.ylabel("Count")
    plt.title(htitle)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, hname), dpi=220)
    plt.close(fig)

    # Per-sample correlation histogram
    fig = plt.figure(figsize=(6, 4))
    sample_pearsons = np.array([m["pearson"] for m in sample_metrics], dtype=float)
    plt.hist(sample_pearsons[np.isfinite(sample_pearsons)], bins=30, alpha=0.9)
    plt.xlabel("Per-sample Pearson")
    plt.ylabel("Count")
    plt.title("Distribution of Per-Sample Correlation")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "per_sample_pearson_hist.png"), dpi=220)
    plt.close(fig)

    # Parameter response
    if P.shape[1] > 0:
        sname = sweep_param if sweep_param is not None else p_keys[0]
        if sname in p_keys:
            k = p_keys.index(sname)
            p_med = np.median(P, axis=0).astype(np.float32)
            p_lo = float(np.percentile(P[:, k], 5))
            p_hi = float(np.percentile(P[:, k], 95))
            xs = np.linspace(p_lo, p_hi, 25).astype(np.float32)
            mref_full = M[val_idx[0]]
            ys = []
            for xv in xs:
                p_cur = p_med.copy()
                p_cur[k] = xv
                p_in = scale_params(p_cur, p_mu, p_std)
                pred = predict_fields(model, norm, mref_full, p_in, device=device, as_numpy=True)[y_idx_pred]
                pred = center_crop_2d(pred, (H0, W0))
                mref = center_crop_2d(mref_full, (H0, W0))
                ys.append(float(np.nanmean(np.where(mref > 0.5, pred, np.nan))))
            fig = plt.figure(figsize=(6, 4))
            plt.plot(xs, np.array(ys), "-o", markersize=3)
            plt.xlabel(sname)
            plt.ylabel(f"Mean predicted {y_key} (masked)")
            plt.title("1D Parameter Response (model sanity)")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"sweep_{sname}.png"), dpi=220)
            plt.close(fig)

    return metrics_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=None)
    ap.add_argument("--npz_path", default=None, help="Alias for --npz")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base-dir", required=True, help="SOLPS run_* directory base for mesh polygons")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--y-key", default="Te")
    ap.add_argument("--all-fields", action="store_true", help="Evaluate all fields available in checkpoint+dataset.")
    ap.add_argument(
        "--paper-grid",
        action="store_true",
        help="Create one large 3-column (truth/pred/error) figure for all selected fields.",
    )
    ap.add_argument("--paper-grid-k", type=int, default=0, help="Use k-th validation sample for paper grid.")
    ap.add_argument("--paper-grid-path", default=None, help="Output path for paper grid figure.")
    ap.add_argument("--paper-grid-rows", type=int, default=3, help="Number of rows in compact all-fields paper grid.")
    ap.add_argument(
        "--paper-grid-split-groups",
        action="store_true",
        help="Also write separate paper grids for plasma fields and source fields.",
    )
    ap.add_argument("--outdir", default="outputs/paper_eval_mesh")
    ap.add_argument("--split", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-examples", type=int, default=3)
    ap.add_argument("--scatter-points", type=int, default=50000)
    ap.add_argument(
        "--error-mode",
        choices=["auto", "abs", "percent", "percent_robust", "smape", "scaled_abs"],
        default="auto",
    )
    ap.add_argument("--error-sign", choices=["absolute", "signed"], default="absolute")
    ap.add_argument("--error-eps", type=float, default=1e-3)
    ap.add_argument(
        "--error-rel-floor",
        type=float,
        default=0.02,
        help="For percent_robust: denominator floor as fraction of P95(|truth|).",
    )
    ap.add_argument("--log-display", choices=["auto", "off", "on"], default="auto",
                    help="Color scale for truth/pred maps: auto(log10 for ne/ni/Sp), off(linear), on(log10).")
    ap.add_argument("--log-display-eps", type=float, default=1e-12)
    ap.add_argument("--log-metrics", action="store_true")
    ap.add_argument("--log-metrics-eps", type=float, default=1e-12)
    ap.add_argument("--sweep-param", type=str, default=None)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    device = pick_device()
    print("Device:", device)
    npz_path = args.npz if args.npz is not None else args.npz_path
    if npz_path is None:
        raise ValueError("Provide --npz (or --npz_path).")
    model, norm, (p_mu, p_std) = load_checkpoint(args.ckpt, device)
    Y, y_keys_data, M, P, p_keys = load_npz_all(npz_path)
    _, val_idx = split_indices(Y.shape[0], split=args.split, seed=args.seed)
    run_name, grid, polys = load_mesh_polygons(args.base_dir, args.run_name)
    mesh_hw = grid.shape[:2]

    # channel mapping
    if hasattr(norm, "y_keys"):
        y_keys_pred = [str(k) for k in norm.y_keys]
    else:
        y_keys_pred = [y_keys_data[0]]
    data_map = {k: i for i, k in enumerate(y_keys_data)}
    pred_map = {k: i for i, k in enumerate(y_keys_pred)}

    if args.all_fields:
        fields = [k for k in y_keys_data if k in pred_map]
    else:
        fields = [args.y_key]
    missing = [k for k in fields if k not in data_map or k not in pred_map]
    if missing:
        raise KeyError(f"Missing requested fields in dataset/checkpoint: {missing}")

    summary_rows = []
    for k in fields:
        print(f"[field] {k}")
        out_k = os.path.join(args.outdir, k) if len(fields) > 1 else args.outdir
        metrics = evaluate_field(
            outdir=out_k,
            y_key=k,
            y_idx_data=data_map[k],
            y_idx_pred=pred_map[k],
            Y=Y, M=M, P=P, p_keys=p_keys,
            val_idx=val_idx,
            model=model, norm=norm, p_mu=p_mu, p_std=p_std, device=device,
            mesh_hw=mesh_hw, polys=polys, run_name=run_name,
            n_examples=args.n_examples, scatter_points=args.scatter_points,
            error_mode=args.error_mode, error_eps=args.error_eps,
            error_rel_floor=args.error_rel_floor,
            error_sign=args.error_sign,
            log_display_mode=args.log_display, log_display_eps=args.log_display_eps,
            log_metrics=args.log_metrics, log_metrics_eps=args.log_metrics_eps,
            sweep_param=args.sweep_param, seed=args.seed,
        )
        print(
            f"  mode={metrics['error_mode_used']} "
            f"sign={metrics['error_sign_used']} "
            f"display={metrics['display_scale_used']} "
            f"plot_p90={metrics['p90_plot_error']:.3g} plot_p95={metrics['p95_plot_error']:.3g}"
        )
        summary_rows.append(metrics)

    if args.paper_grid:
        if len(val_idx) == 0:
            raise RuntimeError("No validation samples available for --paper-grid.")
        kk = int(np.clip(args.paper_grid_k, 0, len(val_idx) - 1))
        gidx = int(val_idx[kk])
        def _plot_grid(grid_fields, out_path):
            if len(grid_fields) == 0:
                return
            plot_all_fields_grid(
                path=out_path,
                fields=grid_fields,
                data_map=data_map,
                pred_map=pred_map,
                Y=Y,
                M=M,
                P=P,
                model=model,
                norm=norm,
                p_mu=p_mu,
                p_std=p_std,
                device=device,
                gidx=gidx,
                mesh_hw=mesh_hw,
                polys=polys,
                run_name=run_name,
                error_mode=args.error_mode,
                error_eps=args.error_eps,
                error_rel_floor=args.error_rel_floor,
                error_sign=args.error_sign,
                log_display_mode=args.log_display,
                log_display_eps=args.log_display_eps,
                paper_grid_rows=args.paper_grid_rows,
            )
            print("Saved:", out_path)

        grid_path = args.paper_grid_path
        if grid_path is None:
            grid_path = os.path.join(args.outdir, "paper_grid_all_fields.png")
        _plot_grid(fields, grid_path)

        if args.paper_grid_split_groups:
            plasma_keys = [k for k in ["Te", "Ti", "ne", "ni", "ua"] if k in fields]
            source_keys = [k for k in ["Sp", "Qp", "Qe", "Qi", "Sm"] if k in fields]
            _plot_grid(plasma_keys, os.path.join(args.outdir, "paper_grid_plasma_fields.png"))
            _plot_grid(source_keys, os.path.join(args.outdir, "paper_grid_source_fields.png"))

    if len(summary_rows) > 1:
        cols = [
            "y_key", "global_mae", "global_rmse", "global_pearson", "global_spearman",
            "error_mode_used", "error_sign_used", "display_scale_used", "p90_plot_error", "p95_plot_error",
            "p90_abs_error", "p95_abs_error", "p90_percent_error", "p95_percent_error",
        ]
        if args.log_metrics:
            cols += ["log_mae", "log_rmse", "log_pearson", "log_spearman"]
        with open(os.path.join(args.outdir, "summary.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in summary_rows:
                w.writerow({c: r.get(c) for c in cols})
        with open(os.path.join(args.outdir, "summary.json"), "w") as f:
            json.dump(summary_rows, f, indent=2)
        print("Saved:", os.path.join(args.outdir, "summary.csv"))
    print("Saved diagnostics to:", args.outdir)


if __name__ == "__main__":
    main()
