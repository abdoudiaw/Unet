"""
SOLPEx autoresearch scaffolding.

This file defines fixed contracts for:
- metric computation
- summary formatting

Keep this file stable during autoresearch loops.
"""

from dataclasses import dataclass


@dataclass
class MetricBundle:
    val_score: float
    forward_score: float
    cycle_score: float
    param_score: float
    training_seconds: float
    peak_vram_mb: float
    num_steps: int


def compute_val_score(forward_score: float, cycle_score: float, param_score: float,
                      alpha_eval: float = 0.1, beta_eval: float = 0.1) -> float:
    """Composite validation score (lower is better)."""
    return forward_score + alpha_eval * cycle_score + beta_eval * param_score


def print_summary(metrics: MetricBundle) -> None:
    """Required output format for parsers and logs."""
    print("---")
    print(f"val_score:          {metrics.val_score:.6f}")
    print(f"forward_score:      {metrics.forward_score:.6f}")
    print(f"cycle_score:        {metrics.cycle_score:.6f}")
    print(f"param_score:        {metrics.param_score:.6f}")
    print(f"training_seconds:   {metrics.training_seconds:.1f}")
    print(f"peak_vram_mb:       {metrics.peak_vram_mb:.1f}")
    print(f"num_steps:          {metrics.num_steps}")

