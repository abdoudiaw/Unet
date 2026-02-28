# SOLPEx: Edge Plasma Surrogate Workflow

Mask-aware ML surrogate models for SOLPS-ITER edge plasma predictions.

## Install

```bash
pip install -e .
```

## Workflow

1. **Forward model** — `scripts/run_conditional_unet_pipeline.py`
2. **Inverse + cycle evaluation** — `scripts/eval_inverse_cycle_conditional_unet.py`
3. **Plasma-to-sources model** — `scripts/run_source_from_plasma_pipeline.py`
4. **Mesh-native evaluation/plots** — `scripts/plot_paper_evaluation_mesh.py`
