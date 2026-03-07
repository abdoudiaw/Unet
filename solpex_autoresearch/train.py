"""
SOLPEx autoresearch train — wired to real models and data.

Trains a conditional UNet (forward: params -> fields) with staged
cycle consistency (encode -> z -> inverse MLP -> params -> forward)
and parameter recovery losses.
"""

import os
import sys
import time
import csv
import subprocess

import numpy as np
import torch
import torch.nn.functional as F

# Ensure solpex package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from solpex.data import (
    make_loaders,
    MaskedLogStandardizer,
    MaskedSymLogStandardizer,
)
from solpex.models import UNet, bottleneck_to_z
from solpex.latent import ZToParam, extract_z_dataset
from solpex.losses import masked_weighted_loss, edge_weights, mae_norm
from prepare import MetricBundle, compute_val_score, print_summary

# ---------------------------------------------------------------------------
# Tunable weights (agent edits these)
# ---------------------------------------------------------------------------

LAMBDA_W = 1.0
LAMBDA_G = 0.2

ALPHA_TARGET = 0.05   # cycle weight target
BETA_TARGET = 0.05    # parameter loss weight target

FORWARD_ONLY_EPOCHS = 20
CYCLE_RAMP_EPOCHS = 40
PARAM_RAMP_EPOCHS = 40
TOTAL_EPOCHS = 120

ALPHA_EVAL = 0.1
BETA_EVAL = 0.1

# Forward loss controls
BASE_LOSS = "huber"
HUBER_BETA = 0.05
GRAD_MODE = "vector"
GRAD_BASE = "l1"
MULTISCALE = 1
MS_WEIGHT = 0.5
GRAD_DS = 2

# Grad loss warmup (ramps LAMBDA_G from 0 to full)
GRAD_WARMUP_START = 5
GRAD_WARMUP_END = 30

# Architecture
BASE_CH = 32
Z_DIM = 64
FILM_HIDDEN = 128

# Optimizer
LR = 3e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 4
CLIP_GRAD = 1.0

# Data
NPZ_PATH = os.environ.get("NPZ_PATH", os.path.join(os.path.dirname(__file__), "..", "solps.npz"))
Y_KEYS = ["Te", "Ti", "ne", "ua", "Sp", "Qe", "Qi", "Sm"]  # drop ni (=ne for D2-only)
PARAM_TRANSFORM = "throughput_ratio"
INPUTS_MODE = "params"

SAVE_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
RESULTS_TSV = os.path.join(os.path.dirname(__file__), "results.tsv")

# Normalizer bank (must match pipeline)
NORMS_BANK = {
    "Te": MaskedLogStandardizer(eps=1e-2),
    "Ti": MaskedLogStandardizer(eps=1e-2),
    "ne": MaskedLogStandardizer(eps=1e16),
    "ua": MaskedSymLogStandardizer(scale=5e3),
    "Sp": MaskedLogStandardizer(eps=1e10),
    "Qe": MaskedSymLogStandardizer(scale=1e2),
    "Qi": MaskedSymLogStandardizer(scale=1e2),
    "Sm": MaskedSymLogStandardizer(scale=1e-2),
}


def ramp(epoch: int, start: int, length: int, target: float) -> float:
    if epoch < start:
        return 0.0
    if length <= 0:
        return target
    frac = min((epoch - start) / length, 1.0)
    return target * frac


def mse_masked(pred, target, mask):
    diff2 = (pred - target) ** 2
    if mask.shape[1] == 1 and pred.shape[1] != 1:
        mask = mask.expand_as(pred)
    num = (diff2 * mask).sum()
    den = mask.sum().clamp_min(1e-8)
    return num / den


def train_one_run() -> MetricBundle:
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    print(f"[autoresearch] device={device}", flush=True)

    os.makedirs(SAVE_DIR, exist_ok=True)

    # ---- Data ----
    print("[autoresearch] loading data...", flush=True)
    train_loader, val_loader, norm, P, (H, W), param_scaler, param_transform, param_keys = make_loaders(
        npz_path=NPZ_PATH,
        inputs_mode=INPUTS_MODE,
        batch_size=BATCH_SIZE,
        y_keys=Y_KEYS,
        norms_by_name=NORMS_BANK,
        param_transform=PARAM_TRANSFORM,
    )
    C_out = len(Y_KEYS)

    # in_ch: mask channel (+ optional R,Z geom channels from make_loaders)
    in_ch = train_loader.dataset[0]["x"].shape[0]

    # ---- Models ----
    model = UNet(
        in_ch=in_ch, out_ch=C_out, base=BASE_CH,
        P=P, film_hidden=FILM_HIDDEN, z_dim=Z_DIM,
    ).to(device)

    # Inverse MLP: z_dim -> P (number of scaled params)
    z_dim_actual = Z_DIM if Z_DIM > 0 else BASE_CH * 8
    inverse_mlp = ZToParam(z_dim=z_dim_actual, P=P, hidden=(256, 256)).to(device)
    n_unet = sum(p.numel() for p in model.parameters())
    n_inv = sum(p.numel() for p in inverse_mlp.parameters())
    print(f"[autoresearch] UNet params={n_unet:,}  inverse params={n_inv:,}  P={P} z_dim={z_dim_actual}", flush=True)
    print(f"[autoresearch] in_ch={in_ch} out_ch={C_out} H={H} W={W} N_train={len(train_loader.dataset)} N_val={len(val_loader.dataset)}", flush=True)

    # ---- Optimizer ----
    all_params = list(model.parameters()) + list(inverse_mlp.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6,
    )

    # AMP
    use_amp = use_cuda
    amp_dtype = torch.bfloat16 if (use_amp and getattr(torch.cuda, "is_bf16_supported", lambda: False)()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Edge weight map
    m0 = train_loader.dataset[0]["mask"].numpy()[0]  # (H,W)
    w_map = torch.from_numpy(edge_weights(m0, sigma_px=3.0)).to(device).unsqueeze(0).unsqueeze(0)

    # ---- Checkpoint resume ----
    ckpt_path = os.path.join(SAVE_DIR, "autoresearch_best.pt")
    best_val = float("inf")
    start_epoch = 0

    # ---- Training loop ----
    num_steps = 0
    fwd_score_sum = 0.0
    cycle_score_sum = 0.0
    param_score_sum = 0.0

    for epoch in range(start_epoch, start_epoch + TOTAL_EPOCHS):
        # Schedule weights
        alpha_train = ramp(epoch, FORWARD_ONLY_EPOCHS, CYCLE_RAMP_EPOCHS, ALPHA_TARGET)
        beta_train = ramp(
            epoch,
            FORWARD_ONLY_EPOCHS + CYCLE_RAMP_EPOCHS,
            PARAM_RAMP_EPOCHS,
            BETA_TARGET,
        )
        # Ramp gradient loss
        lam_grad_epoch = ramp(epoch, GRAD_WARMUP_START, GRAD_WARMUP_END - GRAD_WARMUP_START, LAMBDA_G)

        # ---- Train ----
        model.train()
        inverse_mlp.train()

        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            m = batch["mask"].to(device, non_blocking=True)
            params_scaled = batch["params"].to(device, non_blocking=True) if P > 0 else None

            with torch.amp.autocast(device_type="cuda", enabled=use_amp, dtype=amp_dtype):
                # Forward pass with bottleneck
                pred_fwd, bottleneck = model(x, params=params_scaled, return_bottleneck=True)

                # Forward loss
                l_fwd = masked_weighted_loss(
                    pred_fwd, y, m,
                    w=w_map.to(pred_fwd.dtype),
                    lam_grad=lam_grad_epoch,
                    lam_w=LAMBDA_W,
                    base=BASE_LOSS,
                    huber_beta=HUBER_BETA,
                    grad_mode=GRAD_MODE,
                    grad_base=GRAD_BASE,
                    multiscale=MULTISCALE,
                    ms_weight=MS_WEIGHT,
                    grad_ds=GRAD_DS,
                )

                # Inverse: bottleneck -> z -> predicted params
                z = model.project_z(bottleneck)
                pred_params = inverse_mlp(z)

                # Parameter loss (against scaled true params)
                l_param = F.smooth_l1_loss(pred_params, params_scaled) if params_scaled is not None else torch.zeros((), device=device)

                # Cycle consistency: predicted params -> forward model -> field
                l_cycle = torch.zeros((), device=device)
                if alpha_train > 0 and params_scaled is not None:
                    with torch.no_grad():
                        pred_params_detach = pred_params.detach()
                    pred_cycle = model(x, params=pred_params_detach)
                    l_cycle = mse_masked(pred_cycle, y, m)

                l_total = l_fwd + alpha_train * l_cycle + beta_train * l_param

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(l_total).backward()
            if CLIP_GRAD > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(all_params, CLIP_GRAD)
            scaler.step(optimizer)
            scaler.update()

            num_steps += 1

        # ---- Validate ----
        model.eval()
        inverse_mlp.eval()
        va_fwd_sum = va_cycle_sum = va_param_sum = va_px = 0.0

        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device, non_blocking=True)
                y = batch["y"].to(device, non_blocking=True)
                m = batch["mask"].to(device, non_blocking=True)
                params_scaled = batch["params"].to(device, non_blocking=True) if P > 0 else None

                with torch.amp.autocast(device_type="cuda", enabled=use_amp, dtype=amp_dtype):
                    pred_fwd, bottleneck = model(x, params=params_scaled, return_bottleneck=True)
                    l_fwd = masked_weighted_loss(
                        pred_fwd, y, m,
                        w=w_map.to(pred_fwd.dtype),
                        lam_grad=lam_grad_epoch,
                        lam_w=LAMBDA_W,
                        base=BASE_LOSS,
                        huber_beta=HUBER_BETA,
                        grad_mode=GRAD_MODE,
                        grad_base=GRAD_BASE,
                        multiscale=MULTISCALE,
                        ms_weight=MS_WEIGHT,
                        grad_ds=GRAD_DS,
                    )

                    z = model.project_z(bottleneck)
                    pred_params = inverse_mlp(z)
                    l_param = F.smooth_l1_loss(pred_params, params_scaled) if params_scaled is not None else torch.zeros((), device=device)

                    pred_cycle = model(x, params=pred_params)
                    l_cycle = mse_masked(pred_cycle, y, m)

                px = float(m.sum().item())
                va_fwd_sum += float(l_fwd.item()) * px
                va_cycle_sum += float(l_cycle.item()) * px
                va_param_sum += float(l_param.item()) * px
                va_px += px

        va_fwd = va_fwd_sum / max(va_px, 1.0)
        va_cycle = va_cycle_sum / max(va_px, 1.0)
        va_param = va_param_sum / max(va_px, 1.0)

        va_composite = compute_val_score(va_fwd, va_cycle, va_param, ALPHA_EVAL, BETA_EVAL)

        # Accumulate for final average
        fwd_score_sum += va_fwd
        cycle_score_sum += va_cycle
        param_score_sum += va_param

        scheduler.step(va_composite)

        if epoch % 5 == 0 or epoch == start_epoch + TOTAL_EPOCHS - 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:03d} | fwd {va_fwd:.5f} cycle {va_cycle:.5f} "
                f"param {va_param:.5f} | val {va_composite:.5f} | lr {lr_now:.2e} "
                f"| alpha {alpha_train:.3f} beta {beta_train:.3f} lam_g {lam_grad_epoch:.3f}",
                flush=True,
            )

        # Checkpoint best
        if va_composite < best_val:
            best_val = va_composite
            mu, std = param_scaler if param_scaler else (None, None)
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "inverse_mlp": inverse_mlp.state_dict(),
                "opt": optimizer.state_dict(),
                "sched": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_val": float(best_val),
                "param_scaler": (mu, std),
                "param_transform": param_transform,
                "param_keys": param_keys,
                "in_ch": in_ch, "out_ch": C_out, "base": BASE_CH,
                "P": P, "z_dim": Z_DIM, "y_keys": Y_KEYS,
            }
            torch.save(ckpt, ckpt_path)

    # ---- Final scores (average over epochs) ----
    n_epochs = float(max(TOTAL_EPOCHS, 1))
    forward_score = fwd_score_sum / n_epochs
    cycle_score = cycle_score_sum / n_epochs
    param_score = param_score_sum / n_epochs

    training_seconds = time.time() - t0
    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
        if torch.cuda.is_available()
        else 0.0
    )

    val_score = compute_val_score(
        forward_score=forward_score,
        cycle_score=cycle_score,
        param_score=param_score,
        alpha_eval=ALPHA_EVAL,
        beta_eval=BETA_EVAL,
    )

    return MetricBundle(
        val_score=val_score,
        forward_score=forward_score,
        cycle_score=cycle_score,
        param_score=param_score,
        training_seconds=training_seconds,
        peak_vram_mb=peak_vram_mb,
        num_steps=num_steps,
    )


def log_to_tsv(metrics: MetricBundle, status: str = "keep", description: str = ""):
    """Append a row to results.tsv."""
    commit = ""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(__file__),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        commit = "unknown"

    memory_gb = metrics.peak_vram_mb / 1024.0

    row = [commit, f"{metrics.val_score:.6f}", f"{metrics.forward_score:.6f}",
           f"{metrics.cycle_score:.6f}", f"{metrics.param_score:.6f}",
           f"{memory_gb:.3f}", status, description]

    with open(RESULTS_TSV, "a", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(row)


if __name__ == "__main__":
    metrics = train_one_run()
    print_summary(metrics)

    # Auto-log to results.tsv
    status = "keep"
    if not np.isfinite(metrics.val_score):
        status = "crash"
    desc = (
        f"epochs={TOTAL_EPOCHS} base={BASE_CH} lr={LR} "
        f"alpha={ALPHA_TARGET} beta={BETA_TARGET} z_dim={Z_DIM}"
    )
    log_to_tsv(metrics, status=status, description=desc)
    print(f"\nLogged to {RESULTS_TSV} (status={status})")
