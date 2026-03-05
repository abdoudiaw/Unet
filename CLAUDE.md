# SOLPEx

ML surrogate workflow for SOLPS-ITER edge plasma predictions. Mask-aware neural network surrogates map control parameters to 2D plasma fields and back (inverse).

## Project structure

- `solpex/` — Main Python package
  - `models.py` — UNet architecture (encoder-decoder + skip connections, GroupNorm, SiLU) and latent MLPs (ParamToZ, ZToParam)
  - `data.py` — SOLPSDataset, MaskedLogStandardizer, DataLoader creation, coordinate normalization, parameter transforms
  - `train.py` — Training loop (AMP, gradient accumulation, ReduceLROnPlateau, early stopping)
  - `losses.py` — masked_weighted_loss (Huber/L1/MSE + edge weights + Sobel gradient + multi-scale)
  - `latent.py` — Latent space training (param2z, z2param, L-BFGS inverse)
  - `predict.py` — Checkpoint loading and inference
  - `plotting.py` — Visualization utilities
  - `utils.py` — Device selection, parameter scaling, geometry I/O
- `scripts/` — Pipelines and evaluation
  - `run_full_workflow.sh` — Master 5-step orchestration
  - `run_conditional_unet_pipeline.py` — Step 1: train forward model (params -> fields)
  - `train_inverse_mlp.py` — Step 2: train inverse mapper (z -> params)
  - `eval_inverse_cycle_conditional_unet.py` — Step 3-4: inverse + cycle evaluation
  - `run_source_from_plasma_pipeline.py` — Step 5: train plasma -> sources model
- `solps.npz` — Binary dataset (~52 MB)

## D2-only conventions

- **Field names**: Te, Ti, ne (ne=ni so ni is dropped), ua (velocity), Sp, Qe, Qi, Sm (sources)
- **Parameter transform** (`PARAM_TRANSFORM=throughput_ratio`): maps raw `(Gamma_D2, n_core)` to `(log10(Gamma_D2+n_core), n_core/(Gamma_D2+n_core))` for better conditioning. Stored in checkpoints; inverse scripts auto-recover physical params.
- **Tensor shapes**: fields `(B, C, H, W)`, mask `(B, 1, H, W)`, params `(B, P)`, latent `(B, z_dim)`
- **Normalization**: fields use log-transform (`log(y + eps)`) with masked stats; params use `(p - mu) / std`
- **Checkpoints**: dict with keys `model`, `norm`, `param_scaler`, `param_transform`, `param_keys`, `in_ch`, `out_ch`, `base`, `dropout`

## Tech stack

- Python >=3.9, PyTorch >=2.2, NumPy, SciPy, h5py, Matplotlib
- Install: `pip install -e .`

## Running

```bash
# Full workflow (D2-only defaults: no ni, throughput/ratio transform)
cd scripts && ./run_full_workflow.sh

# Override defaults
NPZ_PATH=../solps.npz EPOCHS_FWD=100 ./run_full_workflow.sh

# Individual steps
PARAM_TRANSFORM=throughput_ratio Y_KEYS=Te,Ti,ne,ua,Sp,Qe,Qi,Sm python scripts/run_conditional_unet_pipeline.py
python scripts/train_inverse_mlp.py --npz solps.npz --ckpt outputs/cond_unet.pt
```

Configure via env vars: `NPZ_PATH`, `Y_KEYS_FWD`, `PARAM_TRANSFORM`, `CHANNEL_WEIGHTS`, `EPOCHS_FWD`, `SWEEP_TRIALS`, etc.

## Inverse optimization

- `eval_inverse_cycle_conditional_unet.py` — optimizes params to match target fields via Adam or L-BFGS
- Supports early stopping (`--early-stop-patience`, `--early-stop-tol`) to skip plateaued runs
- NN-warm-start (`--init nn --inverse-ckpt`) uses trained inverse MLP for initial guess, then refines with optimizer
- Multiple restarts (`--n-restarts`) with best-fit selection
- When `param_transform` is set in the checkpoint, inverse scripts automatically transform dataset params to match and invert recovered params back to physical space for reporting
