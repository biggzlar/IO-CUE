import torch
import matplotlib.pyplot as plt
import os
import sys

# Add current directory to path
sys.path.append(os.getcwd())

from dataloaders.simple_depth import DepthDataset
from models.base_ensemble_model import BaseEnsemble
from models.ttda_model import TTDAModel
from networks.unet_model import UNet
from evaluation.eval_depth_utils import get_predictions, load_mean_model, load_variance_model
from models.post_hoc_frameworks import IOCUE, BayesCap

from predictors.generalized_gaussian import post_hoc_predict_gen_gaussian
from predictors.mse import predict_mse
from predictors.gaussian import predict_gaussian
# Set up matplotlib style
plt.rcParams.update({
    "font.size": 32,
    "text.usetex": False,
    "font.family": "stixgeneral",
    "mathtext.fontset": "stix",
})


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
            bayescap_preds = bayescap_model.predict(y_pred=mu_batch)
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
    
    print(f"Creating visualization with {n_samples} samples...")
    
    # Set column titles
    titles = ['RGB Image', 'Ground Truth', 'Base Prediction', 'Error Map', 
             'TTDA', 'BayesCap', 'Gaussian Ensemble', 'Post-Hoc Gaussian']
    
    # Setup the figure for comparison
    fig, axes = plt.subplots(n_samples, len(titles), figsize=(36, 3.5 * n_samples))
    
    # Handle case where n_samples = 1
    if n_samples == 1:
        axes = axes.reshape(1, -1)

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
    base_model = load_mean_model(
        model_type=BaseEnsemble,
        model_path=base_model_path,
        model_params={"in_channels": 3, "out_channels": [1, 1], "drop_prob": 0.2},
        n_models=5,
        device=device,
        model_class=UNet,
        inference_fn=predict_gaussian,
    )
    
    print("Loading BayesCap model...")
    # Load edgy (mean) model first using mean loader
    
    print("Loading Edgy Depth model...")
    edgy_model = load_mean_model(
        model_type=BaseEnsemble,
        model_path=edgy_model_path,
        model_params={"in_channels": 3, "out_channels": [1], "drop_prob": 0.2},
        n_models=5,
        device=device,
        model_class=UNet,
        inference_fn=predict_mse,
    )

    bayescap_model = load_variance_model(
        mean_ensemble=edgy_model,
        model_type=BayesCap,
        model_path=bayescap_model_path,
        model_params={"in_channels": 1, "out_channels": [1, 1, 1], "drop_prob": 0.2},
        n_models=1,
        device=device,
        model_class=UNet,
    )
    
    print("Loading Gen Gaussian model...")
    gen_gaussian_model = load_variance_model(
        mean_ensemble=edgy_model,
        model_type=IOCUE,
        model_path=gen_gaussian_model_path,
        model_params={"in_channels": 4, "out_channels": [1, 1], "drop_prob": 0.2},
        n_models=1,
        device=device,
        model_class=UNet,
    )
    # Override inference to generalized Gaussian
    gen_gaussian_model.infer = post_hoc_predict_gen_gaussian

    print("Loading Gaussian model...")
    gaussian_model = load_variance_model(
        mean_ensemble=edgy_model,
        model_type=IOCUE,
        model_path=gaussian_model_path,
        model_params={"in_channels": 4, "out_channels": [1], "drop_prob": 0.2},
        n_models=1,
        device=device,
        model_class=UNet,
    )
    
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