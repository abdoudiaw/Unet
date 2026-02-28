# Copyright 2025-2026 Oak Ridge National Laboratory
# @authors: Abdourahmane (Abdou) Diaw - diawa@ornl.gov
#
# SPDX-License-Identifier: MIT

#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


def add_box(ax, x, y, w, h, text, fc="#ffffff", ec="#222222", lw=1.8, fs=12, ls="-"):
    box = Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls)
    ax.add_patch(box)
    ax.text(x + 0.5 * w, y + 0.5 * h, text, ha="center", va="center", fontsize=fs)
    return box


def add_arrow(ax, x0, y0, x1, y1, text=None, fs=11, color="#222222", style="-|>"):
    arr = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle=style,
        mutation_scale=12,
        linewidth=1.6,
        color=color,
    )
    ax.add_patch(arr)
    if text:
        ax.text(0.5 * (x0 + x1), 0.5 * (y0 + y1) + 0.03, text, ha="center", va="bottom", fontsize=fs, color=color)
    return arr


def panel_a(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.01, 0.97, "A", fontsize=22, fontweight="bold", va="top")
    ax.text(0.5, 0.96, "Forward Surrogate", fontsize=18, ha="center", va="top")

    add_box(ax, 0.05, 0.58, 0.20, 0.23, "Inputs\nx: controls\n+ mask/geom", fc="#f3f7ff", fs=12)
    add_box(ax, 0.34, 0.56, 0.30, 0.27, "Conditional\nU-Net\nF(x)", fc="#e9f2ff", fs=14)
    add_box(ax, 0.73, 0.58, 0.20, 0.23, "Predicted fields\n$\\hat{y}$", fc="#eef9f1", fs=12)

    add_arrow(ax, 0.25, 0.695, 0.34, 0.695)
    add_arrow(ax, 0.64, 0.695, 0.73, 0.695)

    add_box(ax, 0.73, 0.22, 0.20, 0.18, "Ground truth\ny", fc="#fff6e8", fs=12)
    add_box(ax, 0.40, 0.20, 0.18, 0.14, "$\\mathcal{L}_{field}$", fc="#ffffff", ec="#444444", fs=14)

    add_arrow(ax, 0.78, 0.58, 0.52, 0.34, text="compare")
    add_arrow(ax, 0.83, 0.40, 0.58, 0.34, text="compare")

    ax.text(
        0.5,
        0.05,
        "Supervised training: predict plasma/source fields from control parameters",
        fontsize=11,
        ha="center",
        color="#333333",
    )


def panel_b(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.01, 0.97, "B", fontsize=22, fontweight="bold", va="top")
    ax.text(0.5, 0.96, "Inverse + Cycle Consistency", fontsize=18, ha="center", va="top")

    add_box(ax, 0.07, 0.61, 0.20, 0.22, "Target fields\n$y^*$", fc="#fff6e8", fs=12)
    add_box(ax, 0.35, 0.59, 0.28, 0.26, "Inverse model\nG($y^*$)", fc="#f5edff", fs=14)
    add_box(ax, 0.70, 0.61, 0.22, 0.22, "Estimated controls\n$\\hat{x}$", fc="#f3f7ff", fs=12)

    add_arrow(ax, 0.27, 0.72, 0.35, 0.72)
    add_arrow(ax, 0.63, 0.72, 0.70, 0.72)

    add_box(
        ax,
        0.36,
        0.24,
        0.28,
        0.20,
        "Frozen Forward U-Net\n$\\hat{y}_{cycle}=F(\\hat{x})$",
        fc="#e9f2ff",
        ec="#2a5ea8",
        ls="--",
        fs=12,
    )
    add_arrow(ax, 0.81, 0.61, 0.50, 0.44)

    add_box(ax, 0.07, 0.20, 0.20, 0.18, "True controls\nx", fc="#f3f7ff", fs=12)
    add_box(ax, 0.72, 0.20, 0.20, 0.18, "Cycle target\n$y^*$", fc="#fff6e8", fs=12)

    add_box(ax, 0.30, 0.05, 0.17, 0.11, "$\\mathcal{L}_{param}$", fc="#ffffff", ec="#444444", fs=13)
    add_box(ax, 0.54, 0.05, 0.17, 0.11, "$\\mathcal{L}_{cycle}$", fc="#ffffff", ec="#444444", fs=13)

    add_arrow(ax, 0.72, 0.61, 0.39, 0.16, text="compare")
    add_arrow(ax, 0.17, 0.38, 0.39, 0.16, text="compare")
    add_arrow(ax, 0.50, 0.24, 0.62, 0.16, text="compare")
    add_arrow(ax, 0.82, 0.20, 0.62, 0.16, text="compare")

    ax.text(
        0.5,
        0.005,
        "Train inverse model with parameter and cycle losses; forward surrogate kept fixed",
        fontsize=11,
        ha="center",
        color="#333333",
    )


def main():
    ap = argparse.ArgumentParser(description="Draw SOLPEx model architecture diagram.")
    ap.add_argument("--out", default="outputs/solaris_model_architecture.png", help="Output PNG path.")
    ap.add_argument("--svg", action="store_true", help="Also save SVG.")
    ap.add_argument("--dpi", type=int, default=300, help="PNG DPI.")
    args = ap.parse_args()

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
        }
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7), dpi=args.dpi, gridspec_kw={"wspace": 0.06})
    panel_a(ax1)
    panel_b(ax2)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved {out}")

    if args.svg:
        svg_path = out.with_suffix(".svg")
        fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.02)
        print(f"Saved {svg_path}")

    plt.show()


if __name__ == "__main__":
    main()
