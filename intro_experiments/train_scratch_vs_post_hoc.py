import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to access modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from networks.simple_regression_model import SimpleRegressionModel
from intro_experiments.dataset import generate_data, ground_truth_function, ground_truth_noise
from models.base_ensemble_model import BaseEnsemble
from models.post_hoc_ensemble_model import PostHocEnsemble
from predictors.gaussian import gaussian_nll, gaussian_nll_detached, predict_gaussian, post_hoc_predict_gaussian
from predictors.mse import mse, predict_mse

plt.rcParams.update(
    {
        "font.size": 12,
        "text.usetex": False,
        "font.family": "stixgeneral",
        "mathtext.fontset": "stix",
    }
)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

def main():
    # Fixed parameters
    train_samples = 300
    test_samples = 200
    noise_level = 0.2
    batch_size = 32
    
    base_hidden_dim = 128
    posthoc_hidden_dim = 64
    n_ensemble = 3
    
    base_n_epochs = 200
    posthoc_n_epochs = 200
    
    base_lr = 1e-3
    posthoc_lr = 1e-3
    
    output_name = 'intro_gaussian_nll_vs_mse_posthoc'
    
    ood_lower, ood_upper = 0, 5

    # Generate dataset with periodic component
    train_loader, X_train, y_train, true_std_train = generate_data(
        n_samples=train_samples, 
        x_range=(ood_lower + 1, ood_upper - 1), 
        noise_level=noise_level, 
        batch_size=batch_size
    )
    
    test_loader, X_test, y_test, true_std_test = generate_data(
        n_samples=test_samples, 
        x_range=(ood_lower, ood_upper),  # Slightly wider range to test extrapolation
        noise_level=noise_level, 
        batch_size=batch_size
    )
    
    # Create directory for results if it doesn't exist
    results_dir = os.path.join('intro_experiments', 'results')
    model_dir = os.path.join('intro_experiments', 'models')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Setup base ensemble model (using Gaussian NLL)
    gaussian_base_model_params = {
        'in_channels': 1,
        'hidden_dim': base_hidden_dim,
        'activation': nn.Mish,
        'output_dim': 2,  # mean and log_sigma for Gaussian
    }
    
    # Create Gaussian base ensemble
    gaussian_base_ensemble = BaseEnsemble(
        model_class=SimpleRegressionModel,
        model_params=gaussian_base_model_params,
        infer=predict_gaussian,
        n_models=n_ensemble,
        device=device
    )

    # Setup MSE base ensemble model
    mse_base_model_params = {
        'in_channels': 1,
        'hidden_dim': base_hidden_dim,
        'activation': nn.Mish,
        'output_dim': 1,  # only mean for MSE
    }
    
    # Create MSE base ensemble
    mse_base_ensemble = BaseEnsemble(
        model_class=SimpleRegressionModel,
        model_params=mse_base_model_params,
        infer=predict_mse,
        n_models=n_ensemble,
        device=device
    )
    
    # Setup post-hoc ensemble model
    posthoc_model_params = {
        'in_channels': 1,  # Input dimension for post-hoc model
        'hidden_dim': posthoc_hidden_dim,
        'activation': nn.Mish,
        'output_dim': 1,  # log_sigma for post-hoc Gaussian
    }
    
    # Create post-hoc ensemble
    posthoc_ensemble = PostHocEnsemble(
        model_class=SimpleRegressionModel,
        model_params=posthoc_model_params,
        infer=post_hoc_predict_gaussian,
        n_models=n_ensemble,
        device=device
    )
    
    # Define optimizer parameters
    base_optimizer_params = {
        'lr': base_lr,
        'weight_decay': 1e-4
    }
    
    posthoc_optimizer_params = {
        'lr': posthoc_lr,
        'weight_decay': 1e-4
    }
    
    # Train Gaussian base ensemble model
    print(f"Training Gaussian base ensemble model...")
    gaussian_base_ensemble.optimize(
        results_dir=results_dir,
        model_dir=model_dir,
        train_loader=train_loader,
        test_loader=test_loader,
        n_epochs=base_n_epochs,
        optimizer_type='AdamW',
        optimizer_params=base_optimizer_params,
        criterion=gaussian_nll,
        eval_freq=20
    )

    # Train MSE base ensemble model
    print(f"Training MSE base ensemble model...")
    mse_base_ensemble.optimize(
        results_dir=results_dir,
        model_dir=model_dir,
        train_loader=train_loader,
        test_loader=test_loader,
        n_epochs=base_n_epochs,
        optimizer_type='AdamW',
        optimizer_params=base_optimizer_params,
        criterion=mse,
        eval_freq=20
    )
    
    # Train post-hoc ensemble model (using MSE base model)
    print(f"Training post-hoc ensemble model...")
    posthoc_ensemble.optimize(
        results_dir=results_dir,
        model_dir=model_dir,
        train_loader=train_loader,
        mean_ensemble=mse_base_ensemble,  # Pass the MSE base ensemble for mean predictions
        test_loader=test_loader,
        n_epochs=posthoc_n_epochs,
        optimizer_type='Adam',
        optimizer_params=posthoc_optimizer_params,
        criterion=gaussian_nll_detached,
        eval_freq=20
    )
    
    # Generate predictions for visualization
    x_plot = np.linspace(ood_lower, ood_upper, 500).reshape(-1, 1)
    x_plot_tensor = torch.FloatTensor(x_plot).to(device)
    
    # Get Gaussian base ensemble predictions
    gaussian_preds = gaussian_base_ensemble.predict(x_plot_tensor)
    gaussian_mean = gaussian_preds['mean'].cpu().numpy()
    
    try:
        gaussian_sigma = gaussian_preds['al_sigma'].cpu().numpy()
    except KeyError:
        print("Warning: 'al_sigma' key not found in gaussian_preds. Using 'sigma' instead.")
        gaussian_sigma = gaussian_preds['sigma'].cpu().numpy() if 'sigma' in gaussian_preds else np.zeros_like(gaussian_mean)
    
    # Get MSE base ensemble predictions
    mse_preds = mse_base_ensemble.predict(x_plot_tensor)
    mse_mean = mse_preds['mean'].cpu().numpy()
    
    # Get post-hoc ensemble predictions (with MSE base model)
    posthoc_preds = posthoc_ensemble.predict(x_plot_tensor)
    posthoc_sigma = posthoc_preds['sigma'].cpu().numpy()
    
    # Ground truth function (without noise)
    y_true = np.array([ground_truth_function(x[0]) for x in x_plot])
    
    # True noise level
    true_noise = (noise_level * ground_truth_noise(x_plot)).squeeze()
    
    # Create figure with 3 subplots side-by-side
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    
    # Plot 1: Gaussian base model with its uncertainty (left)
    axs[0].scatter(X_train[:50], y_train[:50], marker='x', color='k')
    axs[0].plot(x_plot, y_true, color='gray', label='Ground Truth')
    axs[0].plot(x_plot, gaussian_mean, color='#8a2be280', label='Gaussian Ensemble Mean')
    axs[0].axvspan(ood_lower, ood_lower + 1, alpha=0.3, color='gray', label='OOD Region', linewidth=0)
    axs[0].axvspan(ood_upper - 1, ood_upper, alpha=0.3, color='gray', linewidth=0)
    axs[0].fill_between(x_plot.flatten(), 
                     gaussian_mean.flatten() - 2 * gaussian_sigma.flatten(), 
                     gaussian_mean.flatten() + 2 * gaussian_sigma.flatten(), 
                     color='#8a2be280', alpha=0.3, label=r'Gaussian Uncertainty $\left( \pm 2\sigma \right)$',
                     linewidth=0)
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    axs[0].set_title('Model trained on Gaussian NLL')
    axs[0].legend(ncol=2, loc='upper center')
    axs[0].set_ylim(-4, 4)
    axs[0].set_xlim(ood_lower, ood_upper)
    
    # Plot 2: MSE base model mean with post-hoc uncertainty (middle)
    axs[1].scatter(X_train[:50], y_train[:50], marker='x', color='k')
    axs[1].plot(x_plot, y_true, color='gray', label='Ground Truth')
    axs[1].plot(x_plot, mse_mean, color='#8a2be280', label='MSE Ensemble Mean')
    axs[1].axvspan(ood_lower, ood_lower + 1, alpha=0.3, color='gray', label='OOD Region', linewidth=0)
    axs[1].axvspan(ood_upper - 1, ood_upper, alpha=0.3, color='gray', linewidth=0)
    axs[1].fill_between(x_plot.flatten(), 
                     mse_mean.flatten() - 2 * posthoc_sigma.flatten(), 
                     mse_mean.flatten() + 2 * posthoc_sigma.flatten(), 
                     color='#8a2be280', alpha=0.3, label=r'Post-hoc Uncertainty $\left( \pm 2\sigma \right)$',
                     linewidth=0)
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('y')
    axs[1].set_title('Model trained on MSE with Post-hoc Uncertainty')
    axs[1].legend(ncol=2, loc='upper center')
    axs[1].set_ylim(-4, 4)
    axs[1].set_xlim(ood_lower, ood_upper)
    
    # Plot 3: Comparison of noise predictions (right)
    axs[2].plot(x_plot, true_noise, 'k-', label='Ground Truth Noise (σ)')
    axs[2].plot(x_plot, gaussian_sigma, 'g-', label='Gaussian Predicted Noise (σ)')
    axs[2].plot(x_plot, posthoc_sigma, 'r-', label='Post-hoc Predicted Noise (σ)')
    
    # Plot OOD regions
    axs[2].axvspan(ood_lower, ood_lower + 1, alpha=0.3, color='gray', label='OOD Region', linewidth=0)
    axs[2].axvspan(ood_upper - 1, ood_upper, alpha=0.3, color='gray', linewidth=0)
    
    axs[2].set_xlabel('x')
    axs[2].set_ylabel('Standard Deviation (σ)')
    axs[2].set_title('Uncertainty prediction comparison vs. ground truth')
    axs[2].set_ylim(-0.1, 2.)
    axs[2].set_xlim(ood_lower, ood_upper)
    axs[2].legend(ncol=2, loc='upper center')
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(results_dir, f'{output_name}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training complete. Results saved to intro_experiments/results/{output_name}.png")

if __name__ == "__main__":
    main() 