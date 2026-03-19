#!/bin/bash
# Train MLP ensemble on 2D SOLPS cell data.
#
# Usage:
#   bash mlp/launch_mlp_2d.sh --npz solps.npz                  # build DB + train
#   bash mlp/launch_mlp_2d.sh --npz solps.npz --device cuda    # GPU
#   bash mlp/launch_mlp_2d.sh --data mlp/dataset_2d.npz        # skip DB build, just train

set -euo pipefail

NPZ="${NPZ:-}"
DATA="${DATA:-mlp/dataset_2d.npz}"
BFIELD="${BFIELD:-}"
DEVICE="${DEVICE:-cpu}"
OUTPUT="${OUTPUT:-mlp/ensemble_2d.pt}"
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --npz)     NPZ="$2"; shift 2;;
        --data)    DATA="$2"; shift 2;;
        --bfield)  BFIELD="$2"; shift 2;;
        --device)  DEVICE="$2"; shift 2;;
        --output)  OUTPUT="$2"; shift 2;;
        *)         EXTRA_ARGS="$EXTRA_ARGS $1"; shift;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Step 1: Build database if npz provided and dataset doesn't exist
if [ -n "$NPZ" ]; then
    echo "=== Building 2D cell database ==="
    BFIELD_FLAG=""
    if [ -n "$BFIELD" ]; then
        BFIELD_FLAG="--bfield $BFIELD"
    fi
    python3 "$SCRIPT_DIR/build_2d_database.py" --npz "$NPZ" --out "$DATA" $BFIELD_FLAG
    echo ""
fi

# Step 2: Train ensemble
echo "=== Training MLP ensemble ==="
python3 "$SCRIPT_DIR/train_2d.py" \
    --data "$DATA" \
    --output "$OUTPUT" \
    --device "$DEVICE" \
    --results-csv mlp/results_2d.csv \
    $EXTRA_ARGS

echo ""
echo "Done. Model: $OUTPUT"
