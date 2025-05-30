#!/usr/bin/env python
"""
Script to compare different post-hoc models and evaluate them on the Apolloscape dataset.
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

from dataloaders.simple_depth import DepthDataset
from models.base_ensemble_model import BaseEnsemble
from models.post_hoc_ensemble_model import PostHocEnsemble
from networks.unet_model import UNet
from dataloaders.apolloscape_depth import ApolloscapeDepthDataset
from predictors.mse import predict_mse
from predictors.gaussian import post_hoc_predict_gaussian

# Set up matplotlib style
plt.rcParams.update({
    "font.size": 12,
    "text.usetex": False,
    "font.family": "stixgeneral",
    "mathtext.fontset": "stix",
})

MODEL_ORDER = [
    'edgy_depth_01', 'edgy_depth_01_aug',
    'edgy_depth_05', 'edgy_depth_05_aug',
    'edgy_depth', 'edgy_depth_aug'
]

# --- LOG PARSING ---
def load_all_metrics_from_logs(log_dir):
    """Aggregate all metrics from all log files in a directory."""
    metrics = {}
    for log_file in Path(log_dir).glob("*.pkl"):
        with open(log_file, 'rb') as f:
            metrics = pickle.load(f)
    return metrics

# --- PLOTTING ---
def plot_metrics_across_models(nyu_metrics, apollo_metrics, save_dir, title_prefix=""):
    os.makedirs(save_dir, exist_ok=True)
    metrics_to_plot = ['nll', 'ece', 'euc', 'avg_var']
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()
    base_models = ['edgy_depth_01', 'edgy_depth_05', 'edgy_depth']
    aug_models = ['edgy_depth_01_aug', 'edgy_depth_05_aug', 'edgy_depth_aug']
    x = np.arange(len(base_models))
    for i, metric in enumerate(metrics_to_plot):
        ax = axs[i]
        # ID (NYU, original models)
        nyu_vals = [nyu_metrics.get(model, {}).get(metric, []) for model in base_models]
        nyu_vals = [np.array([val.cpu() if hasattr(val, 'cpu') else val for val in vals], dtype=np.float32) for vals in nyu_vals]
        nyu_means = [np.mean(vals) if len(vals) > 0 else np.nan for vals in nyu_vals]
        nyu_stds = [np.std(vals) if len(vals) > 0 else 0 for vals in nyu_vals]
        ax.plot(x, nyu_means, marker='o', label='ID (NYU)', color='blue')
        ax.fill_between(x, np.array(nyu_means) - np.array(nyu_stds), np.array(nyu_means) + np.array(nyu_stds), alpha=0.3, color='blue')
        # OOD (Apolloscape, original models)
        ood_vals = [apollo_metrics.get(model, {}).get(metric, []) for model in base_models]
        ood_vals = [np.array([val.cpu() if hasattr(val, 'cpu') else val for val in vals], dtype=np.float32) for vals in ood_vals]
        ood_means = [np.mean(vals) if len(vals) > 0 else np.nan for vals in ood_vals]
        ood_stds = [np.std(vals) if len(vals) > 0 else 0 for vals in ood_vals]
        ax.plot(x, ood_means, marker='s', label='OOD', color='red')
        ax.fill_between(x, np.array(ood_means) - np.array(ood_stds), np.array(ood_means) + np.array(ood_stds), alpha=0.3, color='red')
        # OOD-aug (Apolloscape, augmented models)
        ood_aug_vals = [apollo_metrics.get(model, {}).get(metric, []) for model in aug_models]
        ood_aug_vals = [np.array([val.cpu() if hasattr(val, 'cpu') else val for val in vals], dtype=np.float32) for vals in ood_aug_vals]
        ood_aug_means = [np.mean(vals) if len(vals) > 0 else np.nan for vals in ood_aug_vals]
        ood_aug_stds = [np.std(vals) if len(vals) > 0 else 0 for vals in ood_aug_vals]
        ax.plot(x, ood_aug_means, marker='^', label='OOD-aug', color='green')
        ax.fill_between(x, np.array(ood_aug_means) - np.array(ood_aug_stds), np.array(ood_aug_means) + np.array(ood_aug_stds), alpha=0.3, color='green')
        ax.set_xticks(x)
        ax.set_xticklabels(base_models, rotation=30)
        ax.set_title(f'{metric} (mean ± std)')
        ax.set_ylabel(metric)
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{title_prefix}combined_metrics.png"))
    plt.close()

# --- MODEL LOADING ---
def load_base_model(model_path, device):
    """Load the base model from the given path"""
    base_model = BaseEnsemble(
        model_class=UNet,
        model_params={"in_channels": 3, "out_channels": [1], "drop_prob": 0.2},
        n_models=5,
        device=device,
        infer=predict_mse
    )
    base_model.load(model_path)
    return base_model

def load_post_hoc_model(model_path, device):
    """Load a post-hoc model from the given path"""
    post_hoc_model = PostHocEnsemble(
        model_class=UNet,
        model_params={"in_channels": 4, "out_channels": [1], "drop_prob": 0.3},
        n_models=1,
        device=device,
        infer=post_hoc_predict_gaussian
    )
    post_hoc_model.load(model_path)
    return post_hoc_model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    base_model_path = "results/pretrained/base_mse_ensemble.pth"
    base_model = load_base_model(base_model_path, device)
    model_paths = {
        'edgy_depth_01': "results/edgy_depth_01/checkpoints/variance_ensemble.pt",
        'edgy_depth_01_aug': "results/edgy_depth_01_aug/checkpoints/variance_ensemble.pt",
        'edgy_depth_05': "results/edgy_depth_05/checkpoints/variance_ensemble.pt",
        'edgy_depth_05_aug': "results/edgy_depth_05_aug/checkpoints/variance_ensemble.pt",
        'edgy_depth': "results/edgy_depth/checkpoints/variance_ensemble.pt",
        'edgy_depth_aug': "results/edgy_depth_aug/checkpoints/variance_ensemble.pt"
    }
    print("Loading NYU logs...")
    nyu_metrics = {}
    for name in MODEL_ORDER:
        log_dir = f"results/{name}/figs"
        if os.path.exists(log_dir):
            nyu_metrics[name] = load_all_metrics_from_logs(log_dir)
        else:
            nyu_metrics[name] = {}
    print("Loading Apolloscape dataset...")
    apolloscape_dataset = DepthDataset(img_size=(128, 160), flip=True)# ApolloscapeDepthDataset()
    _, test_loader = apolloscape_dataset.get_dataloaders(batch_size=16, shuffle=False)
    print("Evaluating models on Apolloscape...")
    apollo_metrics = {}
    for name in MODEL_ORDER:
        print(f"Evaluating {name}...")
        if os.path.exists(model_paths[name]):
            model = load_post_hoc_model(model_paths[name], device)
            eval_result = model.evaluate(test_loader, base_model)
            metrics = eval_result['metrics']
            apollo_metrics[name] = {k: [float(v)] for k, v in metrics.items()}
        else:
            apollo_metrics[name] = {}
    print("Plotting combined metrics (NYU & Apolloscape)...")
    plot_metrics_across_models(nyu_metrics, apollo_metrics, "results/post_hoc_comparison", title_prefix="combined_")
    print("Done! Results saved to results/post_hoc_comparison/")

if __name__ == "__main__":
    main() 