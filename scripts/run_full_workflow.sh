#!/usr/bin/env bash
set -euo pipefail

# Full SOLPS-AI workflow (D2-only):
# 1) Train params->(plasma+sources) conditional U-Net
# 2) Train inverse MLP (z->params)
# 3) Inverse + cycle evaluation (NN-warm-started Adam)
# 4) Inverse evaluation (pure Adam, no MLP)
# 5) Train plasma->sources model
#
# Plotting uses MPLBACKEND=Agg — all figures saved to outputs/

# -----------------------------
# User-configurable settings
# -----------------------------
default_if_empty() {
  local v="${1:-}"
  local d="${2:-}"
  if [[ -z "${v}" ]]; then echo "${d}"; else echo "${v}"; fi
}

NPZ_PATH="$(default_if_empty "${NPZ_PATH:-}" "solps.npz")"

# D2-only: drop ni (identical to ne), use throughput/ratio reparameterization
Y_KEYS_FWD="$(default_if_empty "${Y_KEYS_FWD:-}" "Te,Ti,ne,ua,Sp,Qe,Qi,Sm")"
PARAM_TRANSFORM="$(default_if_empty "${PARAM_TRANSFORM:-}" "throughput_ratio")"

FWD_CHANNEL_WEIGHTS="$(default_if_empty "${FWD_CHANNEL_WEIGHTS:-}" "Te:1.0,Ti:1.0,ne:1.2,ua:1.5,Sp:1.2,Qe:1.2,Qi:1.2,Sm:1.2")"
SWEEP_TRIALS="$(default_if_empty "${SWEEP_TRIALS:-}" "base=32,lr=3e-4,batch=4; base=48,lr=2e-4,batch=4; base=48,lr=1e-4,batch=4")"
EPOCHS_FWD="$(default_if_empty "${EPOCHS_FWD:-}" "450")"
EARLY_STOP_PATIENCE_FWD="$(default_if_empty "${EARLY_STOP_PATIENCE_FWD:-}" "80")"

# Latent projection & cycle consistency
Z_DIM="$(default_if_empty "${Z_DIM:-}" "64")"
CYCLE_INVERSE="$(default_if_empty "${CYCLE_INVERSE:-}" "true")"
LAM_CYCLE="$(default_if_empty "${LAM_CYCLE:-}" "0.1")"
LAM_GRAD="$(default_if_empty "${LAM_GRAD:-}" "0.1")"
LAM_GRAD_WARMUP_END="$(default_if_empty "${LAM_GRAD_WARMUP_END:-}" "80")"

# Inverse evaluation
INV_N_CASES="$(default_if_empty "${INV_N_CASES:-}" "40")"
INV_STEPS="$(default_if_empty "${INV_STEPS:-}" "1200")"
INV_LR="$(default_if_empty "${INV_LR:-}" "1e-2")"
INV_N_RESTARTS="$(default_if_empty "${INV_N_RESTARTS:-}" "5")"
INV_INIT="$(default_if_empty "${INV_INIT:-}" "noisy_true")"
INV_NOISE_STD="$(default_if_empty "${INV_NOISE_STD:-}" "0.2")"
INV_FIELDS="$(default_if_empty "${INV_FIELDS:-}" "Te,Ti,ne,ua,Sp,Qe,Qi,Sm")"
INV_PURE_N_CASES="$(default_if_empty "${INV_PURE_N_CASES:-}" "20")"
INV_PURE_STEPS="$(default_if_empty "${INV_PURE_STEPS:-}" "400")"

# Plasma -> sources
EPOCHS_SRC="$(default_if_empty "${EPOCHS_SRC:-}" "200")"
EARLY_STOP_PATIENCE_SRC="$(default_if_empty "${EARLY_STOP_PATIENCE_SRC:-}" "30")"
SRC_IN_KEYS="$(default_if_empty "${SRC_IN_KEYS:-}" "Te,Ti,ne,ua")"
SRC_OUT_KEYS="$(default_if_empty "${SRC_OUT_KEYS:-}" "Sp,Qe,Qi,Sm")"
SRC_INCLUDE_PARAMS="$(default_if_empty "${SRC_INCLUDE_PARAMS:-}" "1")"
SRC_BASE="$(default_if_empty "${SRC_BASE:-}" "32")"
SRC_LR="$(default_if_empty "${SRC_LR:-}" "3e-4")"
SRC_BATCH="$(default_if_empty "${SRC_BATCH:-}" "4")"

mkdir -p outputs

echo "=== Workflow configuration ==="
echo "NPZ_PATH=${NPZ_PATH}"
echo "Y_KEYS_FWD=${Y_KEYS_FWD}"
echo "PARAM_TRANSFORM=${PARAM_TRANSFORM}"
echo "FWD_CHANNEL_WEIGHTS=${FWD_CHANNEL_WEIGHTS}"
echo "SWEEP_TRIALS=${SWEEP_TRIALS}"
echo "EPOCHS_FWD=${EPOCHS_FWD}  EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE_FWD}"
echo "Z_DIM=${Z_DIM}  CYCLE_INVERSE=${CYCLE_INVERSE}  LAM_CYCLE=${LAM_CYCLE}"
echo "LAM_GRAD=${LAM_GRAD}  LAM_GRAD_WARMUP_END=${LAM_GRAD_WARMUP_END}"
echo "INV: n_cases=${INV_N_CASES} steps=${INV_STEPS} lr=${INV_LR} restarts=${INV_N_RESTARTS} init=${INV_INIT}"
echo "INV_FIELDS=${INV_FIELDS}"
echo "SRC: in=${SRC_IN_KEYS} out=${SRC_OUT_KEYS} params=${SRC_INCLUDE_PARAMS}"
echo ""

if [[ ! -f "${NPZ_PATH}" ]]; then
  echo "[error] NPZ not found: ${NPZ_PATH}" >&2
  exit 2
fi

# ------------------------------------------------------------------
echo "=== Step 1/5: Train params->(plasma+sources) conditional U-Net ==="
# ------------------------------------------------------------------
MPLBACKEND=Agg \
NPZ_PATH="${NPZ_PATH}" \
Y_KEYS="${Y_KEYS_FWD}" \
CHANNEL_WEIGHTS="${FWD_CHANNEL_WEIGHTS}" \
PARAM_TRANSFORM="${PARAM_TRANSFORM}" \
SWEEP=1 \
EPOCHS="${EPOCHS_FWD}" \
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE_FWD}" \
SWEEP_TRIALS="${SWEEP_TRIALS}" \
Z_DIM="${Z_DIM}" \
LAM_GRAD="${LAM_GRAD}" \
LAM_GRAD_WARMUP_END="${LAM_GRAD_WARMUP_END}" \
python run_conditional_unet_pipeline.py

# ------------------------------------------------------------------
echo "=== Step 2/5: Train inverse MLP (z->params) ==="
# ------------------------------------------------------------------
INV_MLP_EPOCHS="$(default_if_empty "${INV_MLP_EPOCHS:-}" "400")"
INV_MLP_HIDDEN="$(default_if_empty "${INV_MLP_HIDDEN:-}" "128,128")"
INV_CYCLE_FLAGS=""
if [[ "${CYCLE_INVERSE}" == "true" ]]; then
  INV_CYCLE_FLAGS="--cycle --lam-cycle ${LAM_CYCLE} --use-layernorm"
fi
python train_inverse_mlp.py \
  --npz "${NPZ_PATH}" \
  --ckpt outputs/cond_unet.pt \
  --out outputs/inverse_mlp.pt \
  --epochs "${INV_MLP_EPOCHS}" \
  --hidden "${INV_MLP_HIDDEN}" \
  ${INV_CYCLE_FLAGS}

# ------------------------------------------------------------------
echo "=== Step 3/5: Inverse + cycle evaluation (NN-warm-started Adam) ==="
# ------------------------------------------------------------------
MPLBACKEND=Agg python eval_inverse_cycle_conditional_unet.py \
  --npz "${NPZ_PATH}" \
  --ckpt outputs/cond_unet.pt \
  --inverse-ckpt outputs/inverse_mlp.pt \
  --optimizer adam \
  --n-cases "${INV_N_CASES}" \
  --steps "${INV_STEPS}" \
  --lr "${INV_LR}" \
  --n-restarts "${INV_N_RESTARTS}" \
  --init nn \
  --noise-std "${INV_NOISE_STD}" \
  --fields "${INV_FIELDS}" \
  --out-csv outputs/inverse_cycle_metrics.csv \
  --out-plot outputs/inverse_param_correlation.png \
  --out-param-corr-csv outputs/inverse_param_correlation.csv

# ------------------------------------------------------------------
echo "=== Step 4/5: Inverse evaluation (pure Adam, no MLP) ==="
# ------------------------------------------------------------------
MPLBACKEND=Agg python eval_inverse_cycle_conditional_unet.py \
  --npz "${NPZ_PATH}" \
  --ckpt outputs/cond_unet.pt \
  --optimizer adam \
  --n-cases "${INV_PURE_N_CASES}" \
  --steps "${INV_PURE_STEPS}" \
  --fields "${INV_FIELDS}" \
  --out-csv outputs/inverse_cycle_metrics_pure.csv \
  --out-plot outputs/inverse_param_correlation_pure.png \
  --out-param-corr-csv outputs/inverse_param_correlation_pure.csv

# ------------------------------------------------------------------
echo "=== Step 5/5: Train plasma->sources model ==="
# ------------------------------------------------------------------
MPLBACKEND=Agg \
NPZ_PATH="${NPZ_PATH}" \
IN_KEYS="${SRC_IN_KEYS}" \
OUT_KEYS="${SRC_OUT_KEYS}" \
INCLUDE_PARAMS="${SRC_INCLUDE_PARAMS}" \
EPOCHS="${EPOCHS_SRC}" \
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE_SRC}" \
BASE="${SRC_BASE}" \
LR="${SRC_LR}" \
BATCH_SIZE="${SRC_BATCH}" \
python run_source_from_plasma_pipeline.py

echo ""
echo "=== Workflow complete ==="
echo "Key outputs:"
echo "  outputs/cond_unet.pt                       -- forward model"
echo "  outputs/inverse_mlp.pt                     -- inverse MLP"
echo "  outputs/inverse_cycle_metrics.csv           -- NN-warm inverse results"
echo "  outputs/inverse_param_correlation.png       -- param recovery scatter"
echo "  outputs/inverse_cycle_metrics_pure.csv      -- pure-Adam inverse results"
echo "  outputs/inverse_param_correlation_pure.png  -- pure-Adam scatter"
echo "  outputs/source_from_plasma.pt               -- plasma->sources model"
