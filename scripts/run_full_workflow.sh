#!/usr/bin/env bash
set -euo pipefail

# Full SOLPS-AI workflow:
# 1) Train params->(plasma+sources) model (conditional U-Net sweep)
# 2) Evaluate/plot params->fields model
# 3) Inverse + cycle evaluation
# 4) Train plasma->sources model (missing step)
# 5) Evaluate/plot plasma->sources model

# -----------------------------
# User-configurable settings
# -----------------------------
default_if_empty() {
  # usage: default_if_empty "$VALUE" "fallback"
  local v="${1:-}"
  local d="${2:-}"
  if [[ -z "${v}" ]]; then
    echo "${d}"
  else
    echo "${v}"
  fi
}

NPZ_PATH="$(default_if_empty "${NPZ_PATH:-}" "data/solps_native_all_qc.npz")"
BASE_DIR="$(default_if_empty "${BASE_DIR:-}" "/content/drive/MyDrive/SOLPS_DB")"

# Step 1: params -> all fields
Y_KEYS_FWD="$(default_if_empty "${Y_KEYS_FWD:-}" "Te,Ti,ne,ni,ua,Sp,Qe,Qi,Sm")"
FWD_CHANNEL_WEIGHTS="$(default_if_empty "${FWD_CHANNEL_WEIGHTS:-}" "Te:1.0,Ti:1.0,ne:1.2,ni:1.2,ua:1.5,Sp:1.2,Qe:1.2,Qi:1.2,Sm:1.2")"
SWEEP_TRIALS="$(default_if_empty "${SWEEP_TRIALS:-}" "base=32,lr=3e-4,batch=4; base=48,lr=2e-4,batch=4; base=48,lr=1e-4,batch=4")"
EPOCHS_FWD="$(default_if_empty "${EPOCHS_FWD:-}" "450")"
EARLY_STOP_PATIENCE_FWD="$(default_if_empty "${EARLY_STOP_PATIENCE_FWD:-}" "80")"

# Step 3: inverse/cycle
INV_N_CASES="$(default_if_empty "${INV_N_CASES:-}" "40")"
INV_STEPS="$(default_if_empty "${INV_STEPS:-}" "1200")"
INV_LR="$(default_if_empty "${INV_LR:-}" "1e-2")"
INV_N_RESTARTS="$(default_if_empty "${INV_N_RESTARTS:-}" "5")"
INV_INIT="$(default_if_empty "${INV_INIT:-}" "noisy_true")"
INV_NOISE_STD="$(default_if_empty "${INV_NOISE_STD:-}" "0.2")"
INV_FIELDS="$(default_if_empty "${INV_FIELDS:-}" "Te,Ti,ne,ni,ua,Sp,Qe,Qi,Sm")"

# Step 4: plasma -> sources (missing step from your list)
EPOCHS_SRC="$(default_if_empty "${EPOCHS_SRC:-}" "200")"
EARLY_STOP_PATIENCE_SRC="$(default_if_empty "${EARLY_STOP_PATIENCE_SRC:-}" "30")"
SRC_IN_KEYS="$(default_if_empty "${SRC_IN_KEYS:-}" "Te,Ti,ne,ni,ua")"
SRC_OUT_KEYS="$(default_if_empty "${SRC_OUT_KEYS:-}" "Sp,Qe,Qi,Sm")"
SRC_INCLUDE_PARAMS="$(default_if_empty "${SRC_INCLUDE_PARAMS:-}" "1")"
SRC_BASE="$(default_if_empty "${SRC_BASE:-}" "32")"
SRC_LR="$(default_if_empty "${SRC_LR:-}" "3e-4")"
SRC_BATCH="$(default_if_empty "${SRC_BATCH:-}" "4")"

mkdir -p outputs

echo "=== Workflow configuration ==="
echo "NPZ_PATH=${NPZ_PATH}"
echo "BASE_DIR=${BASE_DIR}"
echo "Y_KEYS_FWD=${Y_KEYS_FWD}"
echo "FWD_CHANNEL_WEIGHTS=${FWD_CHANNEL_WEIGHTS}"
echo "SWEEP_TRIALS=${SWEEP_TRIALS}"
echo "EPOCHS_FWD=${EPOCHS_FWD} EARLY_STOP_PATIENCE_FWD=${EARLY_STOP_PATIENCE_FWD}"
echo "INV_N_CASES=${INV_N_CASES} INV_STEPS=${INV_STEPS}"
echo "INV_LR=${INV_LR} INV_N_RESTARTS=${INV_N_RESTARTS} INV_INIT=${INV_INIT} INV_NOISE_STD=${INV_NOISE_STD}"
echo "INV_FIELDS=${INV_FIELDS}"
echo "SRC_IN_KEYS=${SRC_IN_KEYS}"
echo "SRC_OUT_KEYS=${SRC_OUT_KEYS}"
echo "SRC_INCLUDE_PARAMS=${SRC_INCLUDE_PARAMS}"
echo "EPOCHS_SRC=${EPOCHS_SRC} EARLY_STOP_PATIENCE_SRC=${EARLY_STOP_PATIENCE_SRC}"
echo "SRC_BASE=${SRC_BASE} SRC_LR=${SRC_LR} SRC_BATCH=${SRC_BATCH}"

if [[ ! -f "${NPZ_PATH}" ]]; then
  echo "[error] NPZ not found: ${NPZ_PATH}" >&2
  exit 2
fi

echo "=== Step 1/5: Train params->(plasma+sources) conditional U-Net ==="
MPLBACKEND=Agg \
NPZ_PATH="${NPZ_PATH}" \
Y_KEYS="${Y_KEYS_FWD}" \
CHANNEL_WEIGHTS="${FWD_CHANNEL_WEIGHTS}" \
SWEEP=1 \
EPOCHS="${EPOCHS_FWD}" \
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE_FWD}" \
SWEEP_TRIALS="${SWEEP_TRIALS}" \
python run_conditional_unet_pipeline.py

echo "=== Step 2/5: Plot/evaluate params->(plasma+sources) model ==="
MPLBACKEND=Agg python plot_paper_evaluation_mesh.py \
  --npz "${NPZ_PATH}" \
  --ckpt outputs/cond_unet.pt \
  --base-dir "${BASE_DIR}" \
  --all-fields \
  --paper-grid \
  --paper-grid-rows 3 \
  --paper-grid-split-groups \
  --paper-grid-k 0 \
  --error-mode abs \
  --error-sign absolute \
  --log-display auto \
  --log-metrics \
  --outdir outputs/paper_eval_mesh_all_abs

echo "=== Step 3/5: Inverse + cycle evaluation (fields->params->fields) ==="
MPLBACKEND=Agg python eval_inverse_cycle_conditional_unet.py \
  --npz "${NPZ_PATH}" \
  --ckpt outputs/cond_unet.pt \
  --n-cases "${INV_N_CASES}" \
  --steps "${INV_STEPS}" \
  --lr "${INV_LR}" \
  --n-restarts "${INV_N_RESTARTS}" \
  --init "${INV_INIT}" \
  --noise-std "${INV_NOISE_STD}" \
  --fields "${INV_FIELDS}" \
  --out-csv outputs/inverse_cycle_metrics.csv \
  --out-plot outputs/inverse_param_correlation.png \
  --out-param-corr-csv outputs/inverse_param_correlation.csv

echo "=== Step 4/5: Train plasma->sources model (Te/Ti/ne/ni/ua -> Qp/Sp/Qe/Qi/Sm) ==="
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

echo "=== Step 5/5: Plot/evaluate plasma->sources model ==="
MPLBACKEND=Agg python eval_source_from_plasma_mesh.py \
  --npz "${NPZ_PATH}" \
  --ckpt outputs/source_from_plasma.pt \
  --base-dir "${BASE_DIR}" \
  --all-fields \
  --error-mode abs \
  --error-sign absolute \
  --log-display auto \
  --paper-grid \
  --outdir outputs/source_eval_mesh_abs

echo "=== Workflow complete ==="
echo "Key outputs:"
echo "  outputs/cond_unet.pt"
echo "  outputs/paper_eval_mesh_all_abs/"
echo "  outputs/inverse_cycle_metrics.csv"
echo "  outputs/inverse_param_correlation.png"
echo "  outputs/inverse_param_correlation.csv"
echo "  outputs/source_from_plasma.pt"
echo "  outputs/source_eval_mesh_abs/"
