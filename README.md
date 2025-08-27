# SOLPS-AI — U-Net Surrogate for 2D SOLPS-ITER Profiles

![status](https://img.shields.io/badge/status-experimental-8a2be2)
![python](https://img.shields.io/badge/python-≥3.9-3776ab?logo=python)
![pytorch](https://img.shields.io/badge/PyTorch-≥2.1-ee4c2c?logo=pytorch)
![license](https://img.shields.io/badge/physics-aware-darkgreen)

**Goal:** learn fast, mask-aware surrogates that map **inputs** (puff rates, power, transport coeffs) to **full 2D plasma fields** (starting with \(T_e\)); later extend to **EIRENE** outputs.

> 🔵 Inputs → 🟣 U-Net → 🟡 2D \(T_e\) (eV)

---

## ✨ Features

- **U-Net encoder–decoder** with skip connections (pixel-wise regression).
- **Physics-aware loss:** masked MSE + edge weighting + ∇ penalty; optional eV-space term.
- **Mixed precision (AMP)**, **ReduceLROnPlateau** scheduler, best-ckpt saving.
- **Training history & plots** (normalized MSE/MAE + physical MAE/RMSE in eV).

---

## 📦 Install

```bash
# from repo root
pip install -e .

