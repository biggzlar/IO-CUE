import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

from evaluation.metrics import compute_ece, compute_nll, compute_euc

# Add current directory to path
sys.path.append(os.getcwd())

from dataloaders.simple_depth import DepthDataset
from models.base_ensemble_model import BaseEnsemble
from models.post_hoc_ensemble_model import PostHocEnsemble
from models.ttda_model import TTDAModel
from networks.unet_model import UNet
from evaluation.eval_depth_utils import get_predictions

from predictors.bayescap import predict_bayescap
from predictors.generalized_gaussian import post_hoc_predict_gen_gaussian
from predictors.mse import predict_mse
from predictors.gaussian import post_hoc_predict_gaussian, predict_gaussian
# Set up matplotlib style
plt.rcParams.update({
    "font.size": 32,
    "text.usetex": False,
    "font.family": "stixgeneral",
    "mathtext.fontset": "stix",
})

def custom_infer(preds):
    """Custom inference function for models that may output 1 or 2 channels"""
    if preds.shape[1] == 2:
        # Model outputs both mean and log_sigma
        mu, log_sigma = torch.split(preds, 1, dim=1)
        sigma = torch.exp(log_sigma)
        return {"mean": mu, "sigma": sigma, "log_sigma": log_sigma}
    else:
        # Model only outputs mean
        mean = preds
        log_sigma = torch.ones_like(mean) * -3  # Default small uncertainty
        sigma = torch.exp(log_sigma)
        return {"mean": mean, "sigma": sigma, "log_sigma": log_sigma}

def load_base_model(model_path, device):
    """Load the base model from the given path"""
    infer = predict_mse

    base_model = BaseEnsemble(
        model_class=UNet,
        model_params={"in_channels": 3, "out_channels": [1], "drop_prob": 0.2},
        n_models=5,
        device=device,
        infer=infer
    )
    base_model.load(model_path)
    return base_model

def load_gaussian_base_model(model_path, device):
    """Load the base model from the given path"""
    infer = predict_gaussian

    base_model = BaseEnsemble(
        model_class=UNet,
        model_params={"in_channels": 3, "out_channels": [1, 1], "drop_prob": 0.2},
        n_models=5,
        device=device,
        infer=infer
    )
    base_model.load(model_path)
    return base_model

def load_bayescap_model(model_path, device):
    """Load the BayesCap post-hoc model"""
    infer = predict_bayescap

    # BayesCap model has 1 input channel and 3 output heads
    post_hoc_model = PostHocEnsemble(
        model_class=UNet,
        model_params={"in_channels": 1, "out_channels": [1, 1, 1], "drop_prob": 0.2},
        n_models=1,
        device=device,
        infer=infer
    )
    post_hoc_model.is_bayescap = True
    post_hoc_model.load(model_path)
    return post_hoc_model

def load_gaussian_model(model_path, device):
    """Load the Gaussian post-hoc model"""
    infer = post_hoc_predict_gaussian

    post_hoc_model = PostHocEnsemble(
        model_class=UNet,
        model_params={"in_channels": 4, "out_channels": [1], "drop_prob": 0.2},
        n_models=1,
        device=device,
        infer=infer
    )
    post_hoc_model.load(model_path)
    return post_hoc_model

def load_gen_gaussian_model(model_path, device):
    """Load the Generative Gaussian post-hoc model"""
    infer = post_hoc_predict_gen_gaussian

    # Gen Gaussian model has 3 input channels and 2 output heads
    post_hoc_model = PostHocEnsemble(
        model_class=UNet,
        model_params={"in_channels": 4, "out_channels": [1, 1], "drop_prob": 0.2},
        n_models=1,
        device=device,
        infer=infer
    )
    post_hoc_model.load(model_path)
    return post_hoc_model

def load_ttda_model(model_path, device):
    """Load TTDA model from the given path"""
    infer = predict_mse

    ttda_model = TTDAModel(
        model_class=UNet,
        model_params={"in_channels": 3, "out_channels": [1], "drop_prob": 0.2},
        device=device,
        infer=infer
    )
    ttda_model.load(model_path)
    return ttda_model

def plot_uncertainty_comparison(test_loader, base_model, bayescap_model, edgy_model, gen_gaussian_model, gaussian_model, ttda_model, n_samples=5, max_eval_samples=100, save_path="results/uncertainty_comparison.png"):
    """
    Plot a comparison of different uncertainty estimation methods
    
    Args:
        test_loader: DataLoader for test data
        base_model: Base depth estimation model
        bayescap_model: BayesCap post-hoc model
        edgy_model: Edgy depth (Gaussian) model
        gen_gaussian_model: Generative Gaussian model
        ttda_model: TTDA model for uncertainty estimation
        n_samples: Number of samples to visualize
        max_eval_samples: Maximum number of samples to use for metric evaluation
        save_path: Path to save the visualization
    """
    # Step 1: Compute metrics on the test set
    print(f"Computing metrics on the test set...")
    
    # Initialize lists to accumulate predictions and ground truths for visualization
    all_images = []
    all_mu = []
    all_depths = []
    all_base_sigma = []
    all_ttda_std = []
    all_bayescap_sigma = []
    all_gaussian_sigma = []
    all_gen_gaussian_sigma = []
    all_error_batch = []
    
    # Process batches from the test loader, just for visualization data
    samples_processed = 0
    with torch.no_grad():
        for images, depths in test_loader:
            # Check if we've processed enough samples for visualization
            batch_size = images.shape[0]
            if samples_processed + batch_size > n_samples:
                # Take only what we need to reach n_samples
                take_samples = n_samples - samples_processed
                images = images[:take_samples]
                depths = depths[:take_samples]
                if take_samples <= 0:
                    break
            
            images = images.to(base_model.device)
            depths = depths.to(base_model.device)
            
            # Get base model predictions
            base_results = base_model.predict(images)
            
            # Get Edgy Depth (Gaussian) predictions
            edgy_results = edgy_model.predict(images)
            mu_batch = edgy_results['mean']
            error_batch = torch.abs(mu_batch - depths)
            
            # Get BayesCap predictions
            bayescap_preds = bayescap_model.predict(mu_batch)
            bayescap_sigma = bayescap_preds['sigma']
            
            # Get Gen Gaussian predictions
            gen_gaussian_preds = gen_gaussian_model.predict(images, y_pred=mu_batch)
            gen_gaussian_sigma = gen_gaussian_preds['sigma']
            
            # Get Gaussian predictions
            gaussian_preds = gaussian_model.predict(images, y_pred=mu_batch)
            gaussian_sigma = gaussian_preds['sigma']
            
            # Get TTDA predictions
            ttda_preds = ttda_model.predict(images, return_individual=True)
            ttda_means = ttda_preds['all_means']
            ttda_std = torch.std(ttda_means, dim=0)
            
            # Accumulate results for visualization
            all_images.append(images.cpu())
            all_mu.append(mu_batch.cpu())
            all_depths.append(depths.cpu())
            all_base_sigma.append(base_results['al_sigma'].cpu())
            all_ttda_std.append(ttda_std.cpu())
            all_bayescap_sigma.append(bayescap_sigma.cpu())
            all_gaussian_sigma.append(gaussian_sigma.cpu())
            all_gen_gaussian_sigma.append(gen_gaussian_sigma.cpu())
            all_error_batch.append(error_batch.cpu())
            
            samples_processed += images.shape[0]
            if samples_processed >= n_samples:
                break
    
    # Concatenate all accumulated tensors for visualization
    all_images = torch.cat(all_images, dim=0)
    all_mu = torch.cat(all_mu, dim=0)
    all_depths = torch.cat(all_depths, dim=0)
    all_base_sigma = torch.cat(all_base_sigma, dim=0)
    all_ttda_std = torch.cat(all_ttda_std, dim=0)
    all_bayescap_sigma = torch.cat(all_bayescap_sigma, dim=0)
    all_gaussian_sigma = torch.cat(all_gaussian_sigma, dim=0)
    all_gen_gaussian_sigma = torch.cat(all_gen_gaussian_sigma, dim=0)
    all_error_batch = torch.cat(all_error_batch, dim=0)
    
    # Get evaluation metrics
    print("Evaluating models on the full test set...")
    base_eval = base_model.evaluate(test_loader)
    ttda_eval = ttda_model.evaluate(test_loader)
    
    # PostHoc models need the mean ensemble model
    bayescap_eval = bayescap_model.evaluate(test_loader, edgy_model)
    gaussian_eval = gaussian_model.evaluate(test_loader, edgy_model)
    gen_gaussian_eval = gen_gaussian_model.evaluate(test_loader, edgy_model)

    # import ipdb; ipdb.set_trace()
    
    # print("Computing calibration metrics...")
    # # Calculate calibration metrics for the whole dataset
    # all_uncertainties = []
    # for batch_X, batch_y in test_loader:
    #     batch_X = batch_X.to(base_model.device)
    #     batch_y = batch_y.to(base_model.device)
        
    #     # Get predictions
    #     base_preds = base_model.predict(batch_X)
    #     edgy_preds = edgy_model.predict(batch_X)
    #     mu = edgy_preds['mean']
        
    #     # Get uncertainties
    #     base_sigma = base_preds['al_sigma']
        
    #     ttda_preds = ttda_model.predict(batch_X, return_individual=True)
    #     ttda_std = torch.std(ttda_preds['all_means'], dim=0)
        
    #     bayescap_sigma = bayescap_model.predict(mu)['sigma']
    #     gaussian_sigma = gaussian_model.predict(batch_X, y_pred=mu)['sigma']
    #     gen_gaussian_sigma = gen_gaussian_model.predict(batch_X, y_pred=mu)['sigma']
        
    #     # Store results
    #     all_uncertainties.append({
    #         'mu': mu.cpu(),
    #         'y_true': batch_y.cpu(),
    #         'base': base_sigma.cpu(),
    #         'ttda': ttda_std.cpu(),
    #         'bayescap': bayescap_sigma.cpu(),
    #         'gaussian': gaussian_sigma.cpu(),
    #         'gen_gaussian': gen_gaussian_sigma.cpu()
    #     })
    
    # # Concat all uncertainties
    # concat_mu = torch.cat([d['mu'] for d in all_uncertainties], dim=0)
    # concat_y = torch.cat([d['y_true'] for d in all_uncertainties], dim=0)
    # concat_base = torch.cat([d['base'] for d in all_uncertainties], dim=0)
    # concat_ttda = torch.cat([d['ttda'] for d in all_uncertainties], dim=0)
    # concat_bayescap = torch.cat([d['bayescap'] for d in all_uncertainties], dim=0)
    # concat_gaussian = torch.cat([d['gaussian'] for d in all_uncertainties], dim=0)
    # concat_gen_gaussian = torch.cat([d['gen_gaussian'] for d in all_uncertainties], dim=0)
        
    # # Format: (method_name, uncertainty_value)
    # uncertainty_methods = [
    #     ("Base", concat_base),
    #     ("TTDA", concat_ttda),
    #     ("BayesCap", concat_bayescap),
    #     ("Gaussian", concat_gaussian),
    #     ("Gen Gaussian", concat_gen_gaussian)
    # ]
    
    # # Calculate metrics for each method
    # metrics = {}
    # for method_name, uncertainty in uncertainty_methods:
    #     residuals = torch.abs(concat_mu - concat_y)
    #     ece = compute_ece(residuals, uncertainty)
    #     nll = compute_nll(concat_mu, uncertainty, concat_y)
    #     euc, _ = compute_euc(concat_mu, uncertainty, concat_y)
    #     metrics[method_name] = {
    #         'ECE': float(ece.mean()),
    #         'NLL': float(nll.mean()),
    #         'EUC': float(euc.mean())
    #     }
    
    # # Add RMSE from each model's evaluation
    # metrics['Base']['RMSE'] = base_eval['rmse']
    # metrics['TTDA']['RMSE'] = ttda_eval['rmse']
    # metrics['BayesCap']['RMSE'] = bayescap_eval['metrics']['rmse'].item()
    # metrics['Gaussian']['RMSE'] = gaussian_eval['metrics']['rmse'].item()
    # metrics['Gen Gaussian']['RMSE'] = gen_gaussian_eval['metrics']['rmse'].item()
    
    # # Function to format metrics into a string
    # def format_metrics(method_name):
    #     if method_name not in metrics:
    #         return ""
    #     m = metrics[method_name]
    #     return f"ECE: {m['ECE']:.4f}, NLL: {m['NLL']:.4f},\nEUC: {m['EUC']:.4f}, RMSE: {m['RMSE']:.4f}"
    
    print(f"Creating visualization with {n_samples} samples...")
    
    # Set column titles
    titles = ['RGB Image', 'Ground Truth', 'Base Prediction', 'Error Map', 
             'TTDA', 'BayesCap', 'Gaussian Ensemble', 'Post-Hoc Gaussian']
    
    # Setup the figure for comparison
    fig, axes = plt.subplots(n_samples, len(titles), figsize=(36, 3.5 * n_samples))
    
    # Handle case where n_samples = 1
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    # for i, title in enumerate(titles):
    #     # Add metrics to uncertainty method titles
    #     if i == 4:  # TTDA
    #         metric_str = format_metrics("TTDA")
    #         # axes[0, i].set_title(f"{title}\n{metric_str}")
    #         axes[0, i].set_title(f"{title}")
    #     elif i == 5:  # BayesCap
    #         metric_str = format_metrics("BayesCap")
    #         axes[0, i].set_title(f"{title}")
    #     elif i == 6:  # Gaussian (Base)
    #         metric_str = format_metrics("Base")
    #         axes[0, i].set_title(f"{title}")
    #     elif i == 7:  # Post Hoc Gaussian
    #         metric_str = format_metrics("Gaussian")
    #         axes[0, i].set_title(f"{title}")
    #     elif i == 8:  # Post Hoc Gen Gaussian
    #         metric_str = format_metrics("Gen Gaussian")
    #         axes[0, i].set_title(f"{title}")
    #     else:
    #         axes[0, i].set_title(title)

    error_colormap = 'rainbow'
    # uncertainty_colormap = 'cubehelix'
    uncertainty_colormap = 'rainbow'
    
    # Plot each sample
    for i in range(n_samples):
        # RGB image
        img = all_images[i].permute(1, 2, 0).numpy()
        axes[i, 0].imshow(img)
        axes[i, 0].axis('off')
        if i == 0:
            axes[i, 0].set_title('RGB Image')
        
        # Ground truth depth
        depth = all_depths[i].squeeze().numpy()
        im1 = axes[i, 1].imshow(depth, cmap='viridis')
        im1.set_clim(0, 1)
        axes[i, 1].axis('off')
        if i == 0:
            axes[i, 1].set_title('Ground Truth')
        
        # Base model prediction
        pred = all_mu[i].squeeze().numpy()
        im2 = axes[i, 2].imshow(pred, cmap='viridis')
        im2.set_clim(0, 1)
        axes[i, 2].axis('off')
        if i == 0:
            axes[i, 2].set_title('Base Prediction')
        
        # Error map
        error = all_error_batch[i].squeeze().numpy()
        im3 = axes[i, 3].imshow(error, cmap=error_colormap)
        im3.set_clim(0, .3)
        axes[i, 3].axis('off')
        if i == 0:
            axes[i, 3].set_title('Error Map')
        
        # # TTDA uncertainty
        # ttda_uncertainty = all_ttda_std[i].squeeze().numpy()
        # im4 = axes[i, 4].imshow(ttda_uncertainty, cmap=uncertainty_colormap)
        # im4.set_clim(0, .3)
        # axes[i, 4].axis('off')

        # Edgy Depth uncertainty
        scratch_uncertainty = all_base_sigma[i].squeeze().numpy()
        im4 = axes[i, 4].imshow(scratch_uncertainty, cmap=uncertainty_colormap)
        im4.set_clim(0, .3)
        axes[i, 4].axis('off')
        if i == 0:
            axes[i, 4].set_title('Ensemble')

        # BayesCap uncertainty - handle multi-channel output
        bayescap_uncertainty = all_bayescap_sigma[i]
        if len(bayescap_uncertainty.shape) > 2:
            # If multiple channels, take the first channel
            bayescap_uncertainty = bayescap_uncertainty[0]
        bayescap_uncertainty = bayescap_uncertainty.squeeze().numpy()
        im5 = axes[i, 5].imshow(bayescap_uncertainty, cmap=uncertainty_colormap)
        im5.set_clim(0, .3)
        axes[i, 5].axis('off')
        if i == 0:
            axes[i, 5].set_title('BayesCap')
        
        # Gaussian uncertainty
        gaussian_uncertainty = all_gaussian_sigma[i].squeeze().numpy()
        im6 = axes[i, 6].imshow(gaussian_uncertainty, cmap=uncertainty_colormap)
        im6.set_clim(0, .3)
        axes[i, 6].axis('off')
        if i == 0:
            axes[i, 6].set_title('IO-CUE')
        
        # Gen Gaussian uncertainty - handle multi-channel output
        gen_gaussian_uncertainty = all_gen_gaussian_sigma[i]
        if len(gen_gaussian_uncertainty.shape) > 2:
            # If multiple channels, take the first channel
            gen_gaussian_uncertainty = gen_gaussian_uncertainty[0]
        gen_gaussian_uncertainty = gen_gaussian_uncertainty.squeeze().numpy()
        im7 = axes[i, 7].imshow(gen_gaussian_uncertainty, cmap=uncertainty_colormap)
        im7.set_clim(0, .3)
        axes[i, 7].axis('off')
        if i == 0:
            axes[i, 7].set_title('IO-CUE (GG)')
    
    # Add a colorbar for uncertainty that spans the full height
    cbar_ax = fig.add_axes([0.89, 0.03, 0.02, 0.91])  # [left, bottom, width, height]
    norm = plt.Normalize(0, 0.3)  # Match the uncertainty plot range
    sm = plt.cm.ScalarMappable(cmap=uncertainty_colormap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(r"Error / Uncertainty ($\sigma$)")
    
    plt.tight_layout(rect=[0, 0, 0.89, 1])  # Adjust the main plot to make room for colorbar
    
    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to {save_path}")
    
    return fig

def main():
    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load depth dataset
    dataset = DepthDataset(img_size=(128, 160))
    _, test_loader = dataset.get_dataloaders(batch_size=8)  # Batch size slightly larger than n_samples
    
    # Define model paths (adjust these based on your actual paths)
    base_model_path = "results/base_gaussian/checkpoints/base_ensemble_model_50.pth"
    bayescap_model_path = "results/edgy_depth_bayescap/checkpoints/variance_ensemble.pt"
    edgy_model_path = "results/edgy_depth_super_aug/checkpoints/base_ensemble_model_best.pth"
    gen_gaussian_model_path = "results/edgy_depth_gen_gaussian/checkpoints/variance_ensemble.pt"
    ttda_model_path = "results/edgy_depth_ttda/checkpoints/ttda_model_75.pth"
    # gaussian_model_path = "results/pretrained/post_hoc_gaussian_aug.pth"
    gaussian_model_path = "results/edgy_depth_aug/checkpoints/post_hoc_ensemble_model_best.pt"
    
    # Load models
    print("Loading base model...")
    base_model = load_gaussian_base_model(base_model_path, device)
    
    print("Loading BayesCap model...")
    bayescap_model = load_bayescap_model(bayescap_model_path, device)
    
    print("Loading Edgy Depth model...")
    edgy_model = load_base_model(edgy_model_path, device)
    
    print("Loading Gen Gaussian model...")
    gen_gaussian_model = load_gen_gaussian_model(gen_gaussian_model_path, device)

    print("Loading Gaussian model...")
    gaussian_model = load_gaussian_model(gaussian_model_path, device)
    
    print("Setting up TTDA model...")
    # TTDA uses the base model with test-time augmentations
    # load_ttda_model(ttda_model_path, device)
    ttda_model = TTDAModel(
        model_class=UNet,
        model_params={"in_channels": 3, "out_channels": [1], "drop_prob": 0.2},
        infer=predict_mse,
        device=device
    )
    ttda_model.model.load_state_dict(edgy_model.models[0].state_dict())  # Use the first model from the ensemble
    
    # Create comparison plot
    n_samples = 5
    max_eval_samples = 100  # Evaluate metrics on 100 samples
    save_path = "results/uncertainty_comparison.png"
    
    print(f"Creating comparison plot with {n_samples} samples and metrics on up to {max_eval_samples} samples...")
    plot_uncertainty_comparison(
        test_loader=test_loader,
        base_model=base_model,
        bayescap_model=bayescap_model,
        edgy_model=edgy_model,
        gen_gaussian_model=gen_gaussian_model,
        gaussian_model=gaussian_model,
        ttda_model=ttda_model,
        n_samples=n_samples,
        max_eval_samples=max_eval_samples,
        save_path=save_path
    )
    
    print("Done!")

if __name__ == "__main__":
    main() 