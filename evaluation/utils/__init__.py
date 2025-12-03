"""Unified import surface for evaluation utilities.

Re-exports a minimal set used across the repo to replace eval_depth_utils.
"""

from .models import load_mean_model, load_variance_model, get_predictions
from .common import visualize_results, ensure_dir, timestamped_dir, write_csv
from .metrics import compute_ece, compute_euc, compute_crps, compute_nll, compute_ause_rmse
from .ood import plot_ood_analysis

__all__ = [
    "load_mean_model",
    "load_variance_model",
    "get_predictions",
    "visualize_results",
    "ensure_dir",
    "timestamped_dir",
    "write_csv",
    "compute_ece",
    "compute_euc",
    "compute_crps",
    "compute_nll",
    "compute_ause_rmse",
    "plot_ood_analysis",
]


