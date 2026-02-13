import argparse
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def draw_block(ax, x, y, w, h, label, fc="#EAF0FF", ec="#2F4F8F", lw=1.6, ls="-", fs=11):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.01,rounding_size=0.015",
        linewidth=lw, edgecolor=ec, facecolor=fc, linestyle=ls,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, color="#16233F")


def draw_arrow(ax, x0, y0, x1, y1, color="#1E2A44", lw=1.4, style="-|>"):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=11, linewidth=lw, color=color)
    ax.add_patch(a)


def make_panel_a(ax, include_latent_prior=False):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.01, 0.98, "A", fontsize=18, fontweight="bold", va="top")
    ax.text(0.08, 0.95, "Training Workflow", fontsize=13, fontweight="bold", va="top")

    draw_block(ax, 0.04, 0.66, 0.22, 0.20, "SOLPS Dataset\nparams + mask + fields", fc="#F3F6FB")
    draw_block(ax, 0.34, 0.66, 0.27, 0.20, "Model 1 (Forward)\nparams + mask ->\nTe,Ti,ne,ni,ua,Sp,Qe,Qi,Sm")
    draw_block(ax, 0.69, 0.66, 0.27, 0.20, "Forward Losses\nmasked Huber + grad\n(+ early-stop/sweep)", fc="#FFF5E8", ec="#8A5A15")

    draw_block(ax, 0.34, 0.35, 0.27, 0.20, "Model 2 (Closure)\nTe,Ti,ne,ni,ua (+params) + mask ->\nSp,Qe,Qi,Sm")
    draw_block(ax, 0.69, 0.35, 0.27, 0.20, "Closure Losses\nmasked Huber + grad", fc="#FFF5E8", ec="#8A5A15")

    draw_block(ax, 0.04, 0.07, 0.22, 0.20, "Inverse/Cycle Eval\ntarget fields -> recovered params\n-> forward reconstruction", fc="#EEF9F1", ec="#1E7A4D")
    draw_block(ax, 0.34, 0.07, 0.27, 0.20, "Diagnostics\nparam correlation\ncycle MAE/RMSE", fc="#EEF9F1", ec="#1E7A4D")
    draw_block(ax, 0.69, 0.07, 0.27, 0.20, "Mesh Paper Plots\ntruth/pred/error\n(all fields grid)", fc="#EEF9F1", ec="#1E7A4D")

    draw_arrow(ax, 0.26, 0.76, 0.34, 0.76)
    draw_arrow(ax, 0.61, 0.76, 0.69, 0.76)
    draw_arrow(ax, 0.26, 0.45, 0.34, 0.45)
    draw_arrow(ax, 0.61, 0.45, 0.69, 0.45)
    draw_arrow(ax, 0.26, 0.17, 0.34, 0.17)
    draw_arrow(ax, 0.61, 0.17, 0.69, 0.17)

    draw_arrow(ax, 0.17, 0.66, 0.17, 0.27, lw=1.2)
    draw_arrow(ax, 0.48, 0.66, 0.48, 0.27, lw=1.2)
    draw_arrow(ax, 0.82, 0.66, 0.82, 0.27, lw=1.2)

    if include_latent_prior:
        draw_block(
            ax, 0.39, 0.88, 0.22, 0.09,
            "Optional Latent Prior\n(future: manifold regularizer)",
            fc="#FFFFFF", ec="#405B93", ls="--", fs=9,
        )
        draw_arrow(ax, 0.50, 0.88, 0.50, 0.86, lw=1.0)


def make_panel_b(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.01, 0.98, "B", fontsize=18, fontweight="bold", va="top")
    ax.text(0.08, 0.95, "Inference + Consistency Checks", fontsize=13, fontweight="bold", va="top")

    draw_block(ax, 0.05, 0.68, 0.26, 0.20, "Design Query\ninput params + mask", fc="#F3F6FB")
    draw_block(ax, 0.38, 0.68, 0.24, 0.20, "Forward Model", fc="#EAF0FF")
    draw_block(ax, 0.69, 0.68, 0.26, 0.20, "Predicted Fields\nplasma + sources", fc="#EAF0FF")

    draw_block(ax, 0.69, 0.38, 0.26, 0.20, "Target/Truth Fields\n(validation case)", fc="#F6F6F6", ec="#656565")
    draw_block(ax, 0.38, 0.38, 0.24, 0.20, "Inverse (optimize params)", fc="#EEF9F1", ec="#1E7A4D")
    draw_block(ax, 0.05, 0.38, 0.26, 0.20, "Recovered Params", fc="#EEF9F1", ec="#1E7A4D")

    draw_block(ax, 0.05, 0.08, 0.90, 0.20, "Final Reporting: per-field MAE/RMSE, correlation, cycle consistency,\nmesh truth/pred/error (absolute primary, signed residual secondary)", fc="#FFF5E8", ec="#8A5A15")

    draw_arrow(ax, 0.31, 0.78, 0.38, 0.78)
    draw_arrow(ax, 0.62, 0.78, 0.69, 0.78)
    draw_arrow(ax, 0.82, 0.68, 0.82, 0.58)
    draw_arrow(ax, 0.69, 0.48, 0.62, 0.48)
    draw_arrow(ax, 0.38, 0.48, 0.31, 0.48)
    draw_arrow(ax, 0.18, 0.58, 0.18, 0.68)
    draw_arrow(ax, 0.18, 0.38, 0.18, 0.28)
    draw_arrow(ax, 0.50, 0.38, 0.50, 0.28)
    draw_arrow(ax, 0.82, 0.38, 0.82, 0.28)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--name", default="workflow_overview")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--formats", default="png,pdf,svg", help="Comma-separated, e.g. png,pdf")
    ap.add_argument("--include-latent-prior", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    fmts = [f.strip().lower() for f in args.formats.split(",") if f.strip()]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    make_panel_a(axes[0], include_latent_prior=args.include_latent_prior)
    make_panel_b(axes[1])

    fig.suptitle(
        "SOLPS-AI Surrogate Workflow: Forward, Inverse/Cycle, and Plasma-to-Source Modeling",
        fontsize=14, y=1.02
    )
    for ext in fmts:
        out = os.path.join(args.outdir, f"{args.name}.{ext}")
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        print("Saved:", out)
    plt.close(fig)


if __name__ == "__main__":
    main()

