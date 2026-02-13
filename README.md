# SOLARIS: Scrape-Off Layer AI for Reconstruction and Integrated Surrogates
 
Mask-aware surrogate models for SOLPS-ITER edge plasma data.

Current project workflow supports:

1. **Forward model**  
   `params + mask -> Te,Ti,ne,ni,ua,Sp,Qe,Qi,Sm`  
   Script: `scripts/run_conditional_unet_pipeline.py`

2. **Inverse + cycle evaluation**  
   `target fields -> recovered params -> forward reconstruction`  
   Script: `scripts/eval_inverse_cycle_conditional_unet.py`

3. **Plasma-to-sources model**  
   `Te,Ti,ne,ni,ua (+optional params) + mask -> Sp,Qe,Qi,Sm`  
   Script: `scripts/run_source_from_plasma_pipeline.py`

4. **Mesh-native paper evaluation/plots**  
   Script: `scripts/plot_paper_evaluation_mesh.py`  
   Source-model evaluator: `scripts/eval_source_from_plasma_mesh.py`

5. **One-shot full workflow runner**  
   Script: `scripts/run_full_workflow.sh`

## Install

```bash
pip install -e .
```

## Data Expectations

The training/eval scripts expect an `.npz` dataset with:

- `Y`: `(N,C,H,W)` fields
- `y_keys`: field names matching channels in `Y`
- `mask`: `(N,H,W)` or `(H,W)`
- `params` (or `X`): `(N,P)` scalar inputs

Typical current field set:
`Te,Ti,ne,ni,ua,Sp,Qe,Qi,Sm`

Note: some datasets may not include `Qp`. Scripts now handle missing keys by explicit key selection.

## Quick Start (Recommended)

From `scripts/`:

```bash
NPZ_PATH=data/solps_native_all_qc.npz \
BASE_DIR="/content/drive/MyDrive/SOLPS_DB" \
./run_full_workflow.sh
```

This runs:

1. Forward-model sweep training
2. Forward-model mesh plots + paper grids
3. Inverse/cycle evaluation
4. Plasma->sources training
5. Plasma->sources mesh plots + paper grid

## Key Outputs

- `outputs/cond_unet.pt`
- `outputs/paper_eval_mesh_all_abs/`
- `outputs/inverse_cycle_metrics.csv`
- `outputs/inverse_param_correlation.png`
- `outputs/inverse_param_correlation.csv`
- `outputs/source_from_plasma.pt`
- `outputs/source_eval_mesh_abs/`

## Important Plot Notes

- Use **absolute error** as primary map (`--error-mode abs`), especially for signed channels.
- `plot_paper_evaluation_mesh.py` supports:
  - `--paper-grid`
  - `--paper-grid-rows 3`
  - `--paper-grid-split-groups` (separate plasma/source figures)
  - `--log-display auto` (log display for density-like/source-like fields)
  - `--log-metrics` (quantitative log-space metrics)

## Current Scope / Next Step

Current deployment models are direct conditional U-Nets (no mandatory latent bottleneck).
Planned next extension: lightweight manifold regularizer (frozen encoder latent consistency) as an ablation on top of this baseline.
