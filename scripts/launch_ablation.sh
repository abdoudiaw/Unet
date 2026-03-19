#!/bin/bash
# Launch UNet ablation sweep for SOLPEx paper.
#
# Usage:
#   bash scripts/launch_ablation.sh                    # full sweep, auto-detect device
#   bash scripts/launch_ablation.sh --subset arch      # architecture only
#   bash scripts/launch_ablation.sh --device cpu       # force CPU
#   bash scripts/launch_ablation.sh --epochs 100       # fewer epochs
#   bash scripts/launch_ablation.sh --tags "baseline_d3_b32,depth4_b32"  # specific runs

set -euo pipefail

# ---- Defaults (override via flags) ----
NPZ="${NPZ:-solps.npz}"
EPOCHS="${EPOCHS:-200}"
DEVICE="${DEVICE:-auto}"
SUBSET="${SUBSET:-all}"
EARLY_STOP="${EARLY_STOP:-30}"
Y_KEYS="${Y_KEYS:-Te,Ti,ne,ni,ua,Sp,Qe,Qi,Sm}"
PARAM_TRANSFORM="${PARAM_TRANSFORM:-throughput_ratio}"
EXTRA_ARGS=""

# ---- Parse CLI flags ----
while [[ $# -gt 0 ]]; do
    case $1 in
        --npz)          NPZ="$2"; shift 2;;
        --epochs)       EPOCHS="$2"; shift 2;;
        --device)       DEVICE="$2"; shift 2;;
        --subset)       SUBSET="$2"; shift 2;;
        --early-stop)   EARLY_STOP="$2"; shift 2;;
        --y-keys)       Y_KEYS="$2"; shift 2;;
        --tags)         EXTRA_ARGS="$EXTRA_ARGS --tags $2"; shift 2;;
        *)              EXTRA_ARGS="$EXTRA_ARGS $1"; shift;;
    esac
done

# ---- Auto-detect device ----
if [ "$DEVICE" = "auto" ]; then
    if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        DEVICE="cuda"
    elif python3 -c "import torch; assert torch.backends.mps.is_available()" 2>/dev/null; then
        DEVICE="mps"
    else
        DEVICE="cpu"
    fi
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="outputs/ablation"
OUT_CSV="$OUT_DIR/results.csv"

echo "============================================"
echo "  SOLPEx UNet Ablation Sweep"
echo "============================================"
echo "  NPZ:       $NPZ"
echo "  Epochs:    $EPOCHS"
echo "  Device:    $DEVICE"
echo "  Subset:    $SUBSET"
echo "  Y-keys:    $Y_KEYS"
echo "  Early stop: $EARLY_STOP"
echo "  Output:    $OUT_CSV"
echo "============================================"

mkdir -p "$OUT_DIR"

python3 "$SCRIPT_DIR/run_ablation_sweep.py" \
    --npz "$NPZ" \
    --epochs "$EPOCHS" \
    --device "$DEVICE" \
    --subset "$SUBSET" \
    --early-stop "$EARLY_STOP" \
    --y-keys "$Y_KEYS" \
    --param-transform "$PARAM_TRANSFORM" \
    --out-dir "$OUT_DIR" \
    --out-csv "$OUT_CSV" \
    $EXTRA_ARGS

echo ""
echo "Done. Results: $OUT_CSV"
