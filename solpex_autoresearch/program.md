# SOLPEx Autoresearch Program

This is an autonomous ML research loop. You are the researcher. You edit code, run experiments, evaluate results, keep or discard, and repeat — forever, until manually stopped.

## Setup

To set up a new experiment run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar7`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current HEAD.
3. **Read the in-scope files**:
   - `solpex_autoresearch/program.md` — this file (your instructions)
   - `solpex_autoresearch/train.py` — the file you modify
   - `solpex_autoresearch/prepare.py` — fixed contracts (do NOT modify)
4. **Verify data exists**: Check that `solps.npz` exists in the repo root. If not, tell the human.
5. **Initialize results.tsv**: If `results.tsv` only has a header, run the baseline first (train.py as-is) and record it.
6. **Confirm and go**: Confirm setup looks good, then kick off the experiment loop.

## Scope

- **Editable file**: `solpex_autoresearch/train.py` — this is the ONLY file you modify. Everything is fair game: loss weights, schedule, architecture knobs, optimizer settings, model size, training strategy.
- **Read-only files**: `solpex_autoresearch/prepare.py` and `solpex/` package. Do NOT modify these.
- **No dependency changes** during the experiment loop.

## The Model

SOLPEx trains a conditional UNet (forward model: control parameters -> 2D plasma fields) jointly with an inverse MLP (latent z -> control parameters). The training uses:

- **Forward loss**: masked Huber + edge weights + Sobel gradient + multiscale
- **Cycle loss**: encode field -> bottleneck -> z -> inverse MLP -> predicted params -> forward model -> reconstructed field vs original
- **Parameter loss**: inverse MLP output vs true scaled parameters
- **Staged schedule**: forward-only warmup, then ramp cycle, then ramp param loss

Data: `solps.npz` — 485 SOLPS-ITER simulations, 8 plasma field channels (Te, Ti, ne, ua, Sp, Qe, Qi, Sm), 5 control parameters, 104x40 grid with mask.

## Primary Objective

Minimize:

```
val_score = forward_score + 0.1 * cycle_score + 0.1 * param_score
```

Lower is better.

Rules:
- `forward_score` is primary and should not regress
- `cycle_score` and `param_score` should improve without harming forward
- All scores are averaged over validation epochs

## Running an Experiment

```bash
cd solpex_autoresearch
python3 -u train.py > run.log 2>&1
```

Always redirect output. Do NOT let it flood your context. A run takes ~15-20 minutes on a single GPU.

Extract results:
```bash
grep "^val_score:\|^forward_score:\|^cycle_score:\|^param_score:\|^peak_vram_mb:" run.log
```

If grep is empty, the run crashed. Run `tail -n 50 run.log` to read the traceback and attempt a fix.

## Output Format

The script prints a summary at the end:

```
---
val_score:          <float>
forward_score:      <float>
cycle_score:        <float>
param_score:        <float>
training_seconds:   <float>
peak_vram_mb:       <float>
num_steps:          <int>
```

The script also auto-appends to `results.tsv`.

## Logging Results

`results.tsv` is tab-separated with columns:

```
commit	val_score	forward_score	cycle_score	param_score	memory_gb	status	description
```

The script logs automatically, but verify the entry is correct. If a run crashed, manually add an entry with status `crash`.

Status values: `keep`, `discard`, `crash`

## Acceptance Criteria

Keep a run only if:
1. `forward_score` improves or stays within +0.2% of best
2. AND at least one reverse metric (`cycle_score` or `param_score`) improves by >=1% relative

Otherwise discard.

Hard reject:
- NaN/inf in any metric
- Clear forward degradation (>0.5% worse)
- Unstable training (loss oscillating wildly)

## The Experiment Loop

LOOP FOREVER:

1. Look at `results.tsv` and the current state of `train.py` to decide what to try next
2. Edit `train.py` with ONE experimental idea (change one thing at a time)
3. `git add solpex_autoresearch/train.py && git commit -m "experiment: <description>"`
4. Run: `python3 -u train.py > run.log 2>&1`
5. Read results: `grep "^val_score:\|^forward_score:" run.log`
6. If grep is empty, check `tail -n 50 run.log` for crash info. Fix if trivial, else log as crash and move on.
7. Verify the `results.tsv` entry
8. If val_score improved (lower): keep the commit, this is the new baseline
9. If val_score is equal or worse: `git reset --hard HEAD~1` to revert to the previous best
10. Go to step 1

## What to Try

Ideas roughly ordered by expected impact:

- **More epochs**: TOTAL_EPOCHS is 120, cycle/param losses may not have converged. Try 200, 300.
- **Stronger inverse**: increase ALPHA_TARGET, BETA_TARGET (e.g. 0.1, 0.15)
- **Learning rate**: try LR=1e-3 with cosine schedule, or warmup+cosine
- **Architecture**: increase BASE_CH (32->48), Z_DIM (64->128), deeper inverse MLP
- **Optimizer**: try different weight decay, grad clip values, or SGD for inverse MLP
- **Loss function**: try L1 instead of Huber for forward, different HUBER_BETA
- **Schedule**: overlap cycle and param ramps instead of sequential, or start them earlier
- **Gradient loss**: try magnitude mode instead of vector, different GRAD_DS
- **Multiscale**: try MULTISCALE=2 for deeper pyramid
- **Batch size**: try BATCH_SIZE=8 or 16 (may need to adjust LR)
- **Detach strategies**: currently cycle path detaches pred_params; try not detaching (end-to-end)
- **Channel weights**: weight important fields (Te, ne) higher than source terms

If you run out of ideas: re-read `train.py` for new angles, try combining previous near-misses, try more radical changes.

## NEVER STOP

Once the experiment loop has begun (after initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human may be away from the computer and expects you to continue working indefinitely until manually stopped. You are autonomous. The loop runs until the human interrupts you, period.

## Crashes

If a run crashes: use your judgment. If it's a typo or easy fix, fix and re-run (same commit). If the idea is fundamentally broken (OOM, architecture mismatch), log as crash, revert, and move on.

If you get stuck on the same error for more than 2 attempts, revert and try something else.
