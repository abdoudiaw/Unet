from pathlib import Path

import argparse
import importlib.util
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.collections import PolyCollection
import numpy as np

from solpex.predict import load_checkpoint, predict_fields, scale_params
from solpex.utils import pick_device


def pick_sample_params(npz):
  # Prefer params from dataset to stay in-distribution.
  for key in ("params", "X_params", "p"):
      if key in npz.files:
          arr = npz[key]
          if arr.ndim == 2 and arr.shape[0] > 0:
              return arr[0]
          if arr.ndim == 1:
              return arr
  # Fallback: your manual values
  return np.array([1e21, 1.05e7, 7.5e20, 0.3, 1.0], dtype=np.float32)


def pick_mask(npz):
  if "mask" not in npz.files:
      raise KeyError("solps.npz is missing key 'mask'")
  m = npz["mask"]
  if m.ndim == 3:
      # Typical shapes: (N,H,W) or (1,H,W)
      m2 = m[0]
  elif m.ndim == 2:
      m2 = m
  else:
      raise ValueError(f"Unexpected mask shape: {m.shape}")

  # Ensure numeric binary-ish mask
  m2 = np.asarray(m2)
  if m2.dtype != np.float32 and m2.dtype != np.float64:
      m2 = m2.astype(np.float32)

  return m2


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--b2fgmtry",
      type=str,
      default="",
      help="Optional path to SOLPS b2fgmtry; enables MATLAB-style polygon plotting.",
  )
  args = parser.parse_args()

  here = Path(__file__).resolve().parent
  ckpt_path = here / "cond_unet.pt"
  npz_path = Path("/Users/42d/Downloads/scripts/data/solps.npz")

  if not ckpt_path.exists():
      raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
  if not npz_path.exists():
      raise FileNotFoundError(f"Missing dataset: {npz_path}")

  device = pick_device()
  model, norm, (p_mu, p_std) = load_checkpoint(str(ckpt_path), device)

  data = np.load(npz_path)
  mask = pick_mask(data)
  params = pick_sample_params(data).astype(np.float32)
  params_in = scale_params(params, p_mu, p_std).astype(np.float32)

  print("Using files:")
  print("  ckpt:", ckpt_path)
  print("  npz :", npz_path)
  print("mask shape:", mask.shape, "min/max:", float(mask.min()), float(mask.max()))
  print("mask nonzero:", int(np.count_nonzero(mask)), "/", int(mask.size))

  print("params (raw):", params)
  print("params (scaled):", params_in)

  fields = predict_fields(model, norm, mask, params_in, device=device, as_numpy=True)
  # channel order: Te, Ti, ne, ni, ua, Sp, Qe, Qi, Sm

  te = fields[0]
  ne = fields[2]
  valid = mask > 0.5
  te_plot = np.where(valid, te, np.nan)
  ne_plot = np.where(valid, ne, np.nan)

  print("Te shape:", te.shape, "min/max:", float(te.min()), float(te.max()))
  print("ne shape:", ne.shape, "min/max:", float(ne.min()), float(ne.max()))

if __name__ == "__main__":
  main()
