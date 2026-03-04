# SOLPEx

ML surrogate workflow for SOLPS-ITER edge plasma predictions. Mask-aware neural network surrogates map control parameters to 2D plasma fields and back (inverse).

## Project structure

- `solpex/` — Main Python package
  - `models.py` — UNet architecture (encoder-decoder + skip connections, GroupNorm, SiLU) and latent MLPs (ParamToZ, ZToParam)
  - `data.py` — SOLPSDataset, MaskedLogStandardizer, DataLoader creation, coordinate normalization
  - `train.py` — Training loop (AMP, gradient accumulation, ReduceLROnPlateau, early stopping)
  - `losses.py` — masked_weighted_loss (Huber/L1/MSE + edge weights + Sobel gradient + multi-scale)
  - `latent.py` — Latent space training (param2z, z2param, L-BFGS inverse)
  - `predict.py` — Checkpoint loading and inference
  - `plotting.py` — Visualization utilities
  - `utils.py` — Device selection, parameter scaling, geometry I/O
- `scripts/` — Pipelines and evaluation
  - `run_full_workflow.sh` — Master 6-step orchestration
  - `run_conditional_unet_pipeline.py` — Step 1: train forward model (params -> fields)
  - `plot_paper_evaluation_mesh.py` — Step 2: evaluate forward model
  - `train_inverse_mlp.py` — Step 3: train inverse mapper (z -> params)
  - `eval_inverse_cycle_conditional_unet.py` — Step 4: inverse + cycle evaluation
  - `run_source_from_plasma_pipeline.py` — Step 5: train plasma -> sources model
  - `eval_source_from_plasma_mesh.py` — Step 6: evaluate sources model
- `solps.npz` — Binary dataset (~52 MB)

## Conventions

- **Field names**: Te, Ti, ne, ni (electron/ion temp & density), ua (velocity), Sp, Qe, Qi, Sm (sources)
- **Tensor shapes**: fields `(B, C, H, W)`, mask `(B, 1, H, W)`, params `(B, P)`, latent `(B, z_dim)`
- **Normalization**: fields use log-transform (`log(y + eps)`) with masked stats; params use `(p - mu) / std`
- **Checkpoints**: dict with keys `model`, `norm`, `param_scaler`, `in_ch`, `out_ch`, `base`, `dropout`

## Tech stack

- Python >=3.9, PyTorch >=2.2, NumPy, SciPy, h5py, Matplotlib
- Install: `pip install -e .`

## Running

```bash
# Full 6-step workflow
cd scripts && ./run_full_workflow.sh

# Individual steps
python scripts/run_conditional_unet_pipeline.py
python scripts/train_inverse_mlp.py
```

Configure via env vars: `NPZ_PATH`, `BASE_DIR`, `Y_KEYS_FWD`, `CHANNEL_WEIGHTS`, `EPOCHS_FWD`, `SWEEP_TRIALS`, etc.
