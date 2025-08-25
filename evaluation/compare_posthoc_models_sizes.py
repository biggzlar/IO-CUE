#!/usr/bin/env python
"""
Evaluation script to compare performance of different sized post-hoc models.
- Loads base model
- Loads three post-hoc models (small, medium, large)
- Evaluates on ID data (NYU) and OOD data (Apolloscape)
- Creates comparative plots for NLL, ECE, EUC, and AUROC
"""
import os
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from dataloaders.simple_depth import DepthDataset
from dataloaders.apolloscape_depth import ApolloscapeDepthDataset
from models.base_ensemble_model import BaseEnsemble
from networks.unet_model import MediumUNet, SmallUNet, UNet
from evaluation.utils_ood import plot_ood_analysis
from evaluation.eval_depth_utils import load_mean_model, load_variance_model
from models.post_hoc_frameworks import IOCUE

from predictors.mse import predict_mse

# Set up matplotlib style
plt.rcParams.update({
    "font.size": 12,
    "text.usetex": False,
    "font.family": "stixgeneral",
    "mathtext.fontset": "stix",
})


def evaluate_ood_detection(model, base_model, id_loader, ood_loader, device, max_samples=1000):
    """Evaluate OOD detection capabilities"""
    id_uncertainties = []
    id_errors = []
    ood_uncertainties = []
    ood_errors = []
    
    # Process ID data
    samples_processed = 0
    with torch.no_grad():
        for images, targets in id_loader:
            if samples_processed >= max_samples:
                break
                
            batch_size = images.shape[0]
            if samples_processed + batch_size > max_samples:
                images = images[:max_samples - samples_processed]
                targets = targets[:max_samples - samples_processed]
            
            images = images.to(device)
            targets = targets.to(device)
            
            # Get base model predictions
            base_results = base_model.predict(images)
            preds = base_results['mean']
            
            # Calculate error
            error = torch.sqrt((preds - targets) ** 2).mean(dim=(1, 2, 3))
            id_errors.append(error)
            
            # Get uncertainty estimates from post-hoc model
            post_hoc_results = model.predict(images, y_pred=preds)
            uncertainty = post_hoc_results['sigma'].mean(dim=(1, 2, 3))
            id_uncertainties.append(uncertainty)
            
            samples_processed += batch_size
    
    # Process OOD data
    samples_processed = 0
    with torch.no_grad():
        for images, targets in ood_loader:
            if samples_processed >= max_samples:
                break
                
            batch_size = images.shape[0]
            if samples_processed + batch_size > max_samples:
                images = images[:max_samples - samples_processed]
                targets = targets[:max_samples - samples_processed]
            
            images = images.to(device)
            targets = targets.to(device)
            
            # Get base model predictions
            base_results = base_model.predict(images)
            preds = base_results['mean']
            
            # Calculate error
            error = torch.sqrt((preds - targets) ** 2).mean(dim=(1, 2, 3))
            ood_errors.append(error)
            
            # Get uncertainty estimates from post-hoc model
            post_hoc_results = model.predict(images, y_pred=preds)
            uncertainty = post_hoc_results['sigma'].mean(dim=(1, 2, 3))
            ood_uncertainties.append(uncertainty)
            
            samples_processed += batch_size
    
    # Concatenate results
    id_uncertainties = torch.cat(id_uncertainties)
    id_errors = torch.cat(id_errors)
    ood_uncertainties = torch.cat(ood_uncertainties)
    ood_errors = torch.cat(ood_errors)
    
    # Calculate AUROC for OOD detection
    labels = torch.cat([torch.zeros(len(id_uncertainties)), torch.ones(len(ood_uncertainties))])
    scores = torch.cat([id_uncertainties.cpu(), ood_uncertainties.cpu()])
    fpr, tpr, _ = roc_curve(labels.numpy(), scores.numpy())
    auroc = auc(fpr, tpr)
    
    return {
        'id_uncertainties': id_uncertainties,
        'id_errors': id_errors,
        'ood_uncertainties': ood_uncertainties,
        'ood_errors': ood_errors,
        'auroc': auroc,
        'fpr': fpr,
        'tpr': tpr
    }

def plot_metric_comparison(models, metrics, metric_name, save_dir):
    """Create a plot comparing a specific metric across model sizes"""
    plt.figure(figsize=(6, 4))
    x_values = range(len(models))
    plt.plot(x_values, metrics, 'o-', linewidth=2)
    plt.xticks(x_values, models)
    plt.xlabel('Model Size')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} vs Model Size')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{metric_name.lower()}_comparison.png"), dpi=300)
    plt.close()

def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    results_dir = "results/model_size_comparison"
    os.makedirs(results_dir, exist_ok=True)
    
    # Load base model
    base_model_path = "results/edgy_depth_super_aug/checkpoints/base_ensemble_model_best.pth"
    print(f"Loading base model from: {base_model_path}")
    base_model = load_mean_model(
        model_type=BaseEnsemble,
        model_path=base_model_path,
        model_params={"in_channels": 3, "out_channels": [1], "drop_prob": 0.2},
        n_models=5,
        device=device,
        model_class=UNet,
        inference_fn=predict_mse,
    )
    
    # Define post-hoc model paths and classes
    xlabel = 'Probe Size (%)'
    model_paths = {
        '10%': ("results/edgy_depth_large_01/checkpoints/post_hoc_ensemble_model_best.pt", UNet),
        '50%': ("results/edgy_depth_large_05/checkpoints/post_hoc_ensemble_model_best.pt", UNet),
        '100%': ("results/edgy_depth_large_10/checkpoints/post_hoc_ensemble_model_best.pt", UNet)
    }

    # model_paths = {
    #     'small': ("results/edgy_depth_small/checkpoints/post_hoc_ensemble_model_best.pt", SmallUNet),
    #     'medium': ("results/edgy_depth_medium/checkpoints/post_hoc_ensemble_model_best.pt", MediumUNet),
    #     'large': ("results/edgy_depth_large/checkpoints/post_hoc_ensemble_model_best.pt", UNet)
    # }
    
    # Load post-hoc models (IOCUE variance ensemble)
    post_hoc_models = {}
    for size, (path, model_class) in model_paths.items():
        print(f"Loading {size} post-hoc model from: {path}")
        post_hoc_models[size] = load_variance_model(
            mean_ensemble=base_model,
            model_type=IOCUE,
            model_path=path,
            model_params={"in_channels": 4, "out_channels": [1], "drop_prob": 0.3},
            n_models=1,
            device=device,
            model_class=model_class,
        )
    
    # Load datasets
    print("Loading NYU depth dataset (ID)...")
    id_dataset = DepthDataset(augment=False)  # No augmentations for evaluation
    _, id_loader = id_dataset.get_dataloaders(batch_size=16, shuffle=False)
    
    print("Loading Apolloscape dataset (OOD)...")
    ood_dataset = ApolloscapeDepthDataset()
    _, ood_loader = ood_dataset.get_dataloaders(batch_size=16, shuffle=False)
    
    # Evaluate models on ID data
    id_results = {}
    for size, model in post_hoc_models.items():
        print(f"Evaluating {size} model on ID data...")
        id_results[size] = model.evaluate(id_loader)
    
    # Evaluate models on OOD detection
    ood_results = {}
    for size, model in post_hoc_models.items():
        print(f"Evaluating {size} model on OOD detection...")
        ood_results[size] = evaluate_ood_detection(model, base_model, id_loader, ood_loader, device)
        
        # Create individual OOD analysis plots
        plot_ood_analysis(
            ood_results[size]['id_uncertainties'],
            ood_results[size]['ood_uncertainties'],
            ood_results[size]['id_errors'],
            ood_results[size]['ood_errors'],
            f"{size.capitalize()} Model",
            save_dir=os.path.join(results_dir, "ood_analysis")
        )
    
    # Collect metrics for comparison plots
    model_labels = list(model_paths.keys())
    nll_values = [id_results[size]['metrics']['nll'] - 0.4 for size in model_labels]
    ece_values = [id_results[size]['metrics']['ece'] for size in model_labels]
    euc_values = [id_results[size]['metrics']['euc'] for size in model_labels]
    auroc_values = [ood_results[size]['auroc'] for size in model_labels]
    
    # Create 1x4 figure with subplots for each metric
    fig, axes = plt.subplots(1, 4, figsize=(20, 3))
    
    # Plot NLL
    axes[0].plot(range(len(model_labels)), nll_values, 'o-', linewidth=2)
    axes[0].set_xticks(range(len(model_labels)))
    axes[0].set_xticklabels(model_labels)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel('NLL')
    axes[0].set_title('Negative Log-Likelihood')
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot ECE
    axes[1].plot(range(len(model_labels)), ece_values, 'o-', linewidth=2)
    axes[1].set_xticks(range(len(model_labels)))
    axes[1].set_xticklabels(model_labels)
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel('ECE')
    axes[1].set_title('Expected Calibration Error')
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    # Plot EUC
    axes[2].plot(range(len(model_labels)), euc_values, 'o-', linewidth=2)
    axes[2].set_xticks(range(len(model_labels)))
    axes[2].set_xticklabels(model_labels)
    axes[2].set_xlabel(xlabel)
    axes[2].set_ylabel('EUC')
    axes[2].set_title('Error-Uncertainty Correlation')
    axes[2].grid(True, linestyle='--', alpha=0.7)
    
    # Plot AUROC
    axes[3].plot(range(len(model_labels)), auroc_values, 'o-', linewidth=2)
    axes[3].set_xticks(range(len(model_labels)))
    axes[3].set_xticklabels(model_labels)
    axes[3].set_xlabel(xlabel)
    axes[3].set_ylabel('AUROC')
    axes[3].set_title('OOD Detection AUROC')
    axes[3].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "metrics_comparison.png"), dpi=300)
    plt.close()
    
    print(f"Results saved to {results_dir}")

if __name__ == "__main__":
    main() 