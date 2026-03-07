# SOLPEx Autoresearch Program

Goal: Improve inverse/reverse parameter recovery while preserving or improving forward-model accuracy.

## Scope

- Editable file: `solpex_autoresearch/train.py`
- Read-only files during loop: `solpex_autoresearch/prepare.py` and evaluator logic
- No dependency changes during experiment loop

## Primary Objective

Minimize:

`val_score = forward_score + alpha_eval * cycle_score + beta_eval * param_score`

Lower is better.

Rules:
- `forward_score` is primary and should not regress.
- `cycle_score` and `param_score` should improve without harming forward.

## Loss Design

Keep your current forward losses as baseline, then add reverse terms gradually:

`L_fwd = L_base + lambda_w * L_edge + lambda_g * L_grad`

`L_total = L_fwd + alpha_train * L_cycle + beta_train * L_param`

Where:
- `L_base`: masked robust pointwise loss (Huber recommended)
- `L_edge`: boundary-weighted masked loss
- `L_grad`: masked gradient loss (Sobel/finite-difference)
- `L_cycle`: masked field mismatch after round-trip `F(G(y))`
- `L_param`: parameter reconstruction loss `G(y)` vs true parameters (if labels available)

## Scheduling

- Warm start forward only: `alpha_train=0`, `beta_train=0`
- Ramp cycle term first
- Ramp parameter term second
- Keep forward dominant throughout

Suggested starting values:
- `lambda_g = 0.2` with warmup (epoch 20 -> 60)
- `alpha_train` target: `0.05`
- `beta_train` target: `0.05` (if supervised inverse labels available)

## Acceptance Criteria

Keep a run only if:
1. `forward_score` improves or stays within +0.2% of best
2. and at least one reverse metric (`cycle_score` or `param_score`) improves by >=1% relative

Otherwise discard.

Hard reject:
- NaN/inf
- unstable inverse optimization
- clear forward degradation

## Required End-of-Run Summary

Print exactly:

---
val_score:          <float>
forward_score:      <float>
cycle_score:        <float>
param_score:        <float>
training_seconds:   <float>
peak_vram_mb:       <float>
num_steps:          <int>

## Logging

Append to `solpex_autoresearch/results.tsv`:

`commit	val_score	forward_score	cycle_score	param_score	memory_gb	status	description`

`status` is one of:
- `keep`
- `discard`
- `crash`

## Experiment Loop

1. Edit only `solpex_autoresearch/train.py` with one idea
2. Commit
3. Run `uv run solpex_autoresearch/train.py > solpex_autoresearch/run.log 2>&1`
4. Parse summary metrics
5. Log to `results.tsv`
6. Keep/discard by criteria above
7. Repeat

