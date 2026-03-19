#!/bin/bash -l
#SBATCH -p regular
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH -J solpex_gnn
#SBATCH -e gnn_%j.err
#SBATCH -o gnn_%j.out
#SBATCH -C gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --account=m5186

set -euo pipefail

module load python
module load pytorch/2.0  # adjust to available pytorch module on Perlmutter

# Activate your venv (create once: python -m venv ~/.venvs/solpex-gnn)
source ~/.venvs/solpex-gnn/bin/activate

# ---- Paths ----
REPO="${HOME}/SOLPEx"
DATA="${SCRATCH}/solpex_data/coupling_dataset.npz"
OUTDIR="${SCRATCH}/solpex_gnn_out"
mkdir -p "${OUTDIR}"

# ---- Pull latest code ----
cd "${REPO}"
git pull --ff-only 2>/dev/null || true

# ---- Which model to train ----
MODEL="${MODEL:-conditional}"  # "conditional" or "eirene"

if [ "${MODEL}" = "conditional" ]; then
    echo "=== Training Conditional GNN (paper surrogate) ==="
    python gnn/train_gnn.py \
        --data "${DATA}" \
        --output "${OUTDIR}/cond_gnn.pt" \
        --device cuda \
        --hidden "${HIDDEN:-128}" \
        --n-layers "${NLAYERS:-6}" \
        --dropout "${DROPOUT:-0.1}" \
        --epochs "${EPOCHS:-200}" \
        --lr "${LR:-3e-4}" \
        --batch-size "${BATCH:-8}" \
        --patience "${PATIENCE:-30}" \
        --results-csv "${OUTDIR}/cond_gnn_results.csv"

elif [ "${MODEL}" = "eirene" ]; then
    echo "=== Training EIRENE-replacement GNN ==="
    python gnn/train_eirene_gnn.py \
        --data "${DATA}" \
        --output "${OUTDIR}/eirene_gnn.pt" \
        --device cuda \
        --hidden "${HIDDEN:-128}" \
        --n-layers "${NLAYERS:-6}" \
        --dropout "${DROPOUT:-0.1}" \
        --epochs "${EPOCHS:-500}" \
        --lr "${LR:-1e-3}" \
        --batch-size "${BATCH:-16}" \
        --patience "${PATIENCE:-100}" \
        --results-csv "${OUTDIR}/eirene_gnn_results.csv"

else
    echo "Unknown MODEL=${MODEL}. Use 'conditional' or 'eirene'."
    exit 1
fi

echo ""
echo "Output: ${OUTDIR}"
ls -lh "${OUTDIR}"
