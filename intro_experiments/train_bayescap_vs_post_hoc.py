import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import argparse

# Add parent directory to path to access modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from networks.simple_regression_model import SimpleRegressionModel
from intro_experiments.dataset import generate_data, ground_truth_function, ground_truth_noise
from intro_experiments.predictor_wrappers import LossWrapper, PredictorWrapper

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
device = "cpu"

def train_base_model(model, train_loader, loss_wrapper, n_epochs=500, lr=1e-3):
    """Train the base regression model"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-1)
    
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass
            outputs = model(X_batch)
            loss = loss_wrapper.calculate_loss(y_batch, outputs)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print progress
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {total_loss/len(train_loader):.4f}')
    
    return model

def train_posthoc_model(model, base_model, loss_wrapper, X, y, n_epochs=500, lr=1e-3, is_bayescap=False):
    """Train the post-hoc uncertainty model"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    X_tensor = torch.FloatTensor(X).to(device)
    y_tensor = torch.FloatTensor(y).to(device)
    
    model.train()
    for epoch in range(n_epochs):
        # Get base model predictions
        with torch.no_grad():
            base_outputs = base_model(X_tensor)
            # Base model uses MSE loss, so it only outputs mean
            base_means = base_outputs
        
        # Forward pass for post-hoc model
        if is_bayescap:
            # BayesCap uses base model predictions as inputs
            # This is a key architectural difference from the Gaussian model
            outputs = model(base_means)
            # Use the loss wrapper for post-hoc loss types
            loss = loss_wrapper.calculate_loss(y_tensor, y_pred=base_means, params=outputs, epoch=epoch, n_epochs=n_epochs)
        else:
            # Gaussian model uses original inputs (not base model predictions)
            outputs = model(X_tensor)
            # Use the loss wrapper for post-hoc loss types
            loss = loss_wrapper.calculate_loss(y_tensor, y_pred=base_means, params=outputs)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print progress
        if (epoch+1) % 100 == 0:
            print(f'Post-hoc Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')
    
    return model

def main(args):
    ood_lower, ood_upper = 0, 5

    # Generate dataset with periodic component
    train_loader, X_train, y_train, true_std_train = generate_data(
        n_samples=args.train_samples, 
        x_range=(ood_lower + 1, ood_upper - 1), 
        noise_level=args.noise_level, 
        batch_size=args.batch_size
    )
    
    test_loader, X_test, y_test, true_std_test = generate_data(
        n_samples=args.test_samples, 
        x_range=(ood_lower, ood_upper),  # Slightly wider range to test extrapolation
        noise_level=args.noise_level, 
        batch_size=args.batch_size
    )
    
    # Create directory for results if it doesn't exist
    os.makedirs(os.path.join('intro_experiments', 'results'), exist_ok=True)

    # Setup base model with MSE loss (not Gaussian NLL)
    base_loss_wrapper = LossWrapper("mse")
    base_output_dims = base_loss_wrapper.output_dims
    base_model = SimpleRegressionModel(in_channels=1, hidden_dim=args.base_hidden_dim, output_dim=base_output_dims, activation=nn.Mish).to(device)
    
    # Setup post-hoc models
    bayescap_loss_wrapper = LossWrapper("bayescap", post_hoc=True)
    gaussian_loss_wrapper = LossWrapper("gaussian", post_hoc=True)
    
    bayescap_output_dims = bayescap_loss_wrapper.output_dims
    gaussian_output_dims = gaussian_loss_wrapper.output_dims
    
    # BayesCap input is the base model's output (mean prediction)
    # The default assumption would be to use the original data as input
    bayescap_model = SimpleRegressionModel(in_channels=base_output_dims, hidden_dim=args.posthoc_hidden_dim, output_dim=bayescap_output_dims, activation=nn.ReLU).to(device)
    gaussian_model = SimpleRegressionModel(in_channels=1, hidden_dim=args.posthoc_hidden_dim, output_dim=gaussian_output_dims, activation=nn.Mish).to(device)

    # Setup predictors for obtaining uncertainties
    bayescap_predictor = PredictorWrapper("bayescap")
    gaussian_predictor = PredictorWrapper("gaussian")

    # Train base model
    print(f"Training base model with MSE loss...")
    base_model = train_base_model(
        model=base_model, 
        train_loader=train_loader, 
        n_epochs=args.n_epochs, 
        lr=args.base_lr,
        loss_wrapper=base_loss_wrapper
    )
    
    # Train Bayescap post-hoc model
    print(f"Training Bayescap post-hoc model...")
    bayescap_model = train_posthoc_model(
        model=bayescap_model, 
        base_model=base_model, 
        X=X_train, 
        y=y_train, 
        n_epochs=args.n_epochs, 
        lr=args.posthoc_lr,
        loss_wrapper=bayescap_loss_wrapper,
        is_bayescap=True  # Flag to indicate BayesCap model
    )
    
    # Train Gaussian post-hoc model
    print(f"Training Gaussian post-hoc model...")
    gaussian_model = train_posthoc_model(
        model=gaussian_model, 
        base_model=base_model, 
        X=X_train, 
        y=y_train, 
        n_epochs=args.n_epochs, 
        lr=args.posthoc_lr,
        loss_wrapper=gaussian_loss_wrapper,
        is_bayescap=False  # Flag to indicate not BayesCap model
    )
    
    # Generate predictions for visualization
    x_plot = np.linspace(ood_lower, ood_upper, 500).reshape(-1, 1)
    x_plot_tensor = torch.FloatTensor(x_plot).to(device)
    
    with torch.no_grad():
        # Get base model predictions (MSE model only outputs mean)
        base_mean_tensor = base_model(x_plot_tensor)
        base_mean = base_mean_tensor.cpu().numpy()
        
        # Get BayesCap predictions
        # Important: BayesCap uses base model outputs as input, not original data
        bayescap_outputs = bayescap_model(base_mean_tensor)
        bayescap_info = bayescap_predictor.predict_from_outputs(bayescap_outputs)
        bayescap_sigma = bayescap_info["sigma"].cpu().numpy()
        
        # Get Gaussian predictions (uses original data as input)
        gaussian_outputs = gaussian_model(x_plot_tensor)
        gaussian_info = gaussian_predictor.predict_from_outputs(gaussian_outputs)
        gaussian_sigma = gaussian_info["sigma"].cpu().numpy()

    # Ground truth function (without noise)
    y_true = np.array([ground_truth_function(x[0]) for x in x_plot])
    
    # Calculate true noise (standard deviation)
    true_noise = (args.noise_level * ground_truth_noise(x_plot)).squeeze()
    
    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    
    # Plot 1: Base model mean with Bayescap uncertainty (top left)
    axs[0].scatter(X_train[:50], y_train[:50], marker='x', color='k')
    axs[0].plot(x_plot, y_true, color='gray', label='Ground Truth')
    axs[0].plot(x_plot, base_mean, color='#8a2be280', label='Base Model Mean')
    axs[0].axvspan(ood_lower, ood_lower + 1, alpha=0.3, color='gray', label='OOD Region', linewidth=0)
    axs[0].axvspan(ood_upper - 1, ood_upper, alpha=0.3, color='gray', linewidth=0)
    axs[0].fill_between(x_plot.flatten(), 
        base_mean.flatten() - 2 * bayescap_sigma.flatten(), 
        base_mean.flatten() + 2 * bayescap_sigma.flatten(), 
        color='#8a2be280', alpha=0.3, label=r'Bayescap Uncertainty $\left( \pm 2\sigma \right)$',
        linewidth=0)
    
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    axs[0].set_title('(BayesCap) Estimated on base model outputs')
    axs[0].legend(ncol=2, loc='upper center')
    axs[0].set_ylim(-4, 4)
    axs[0].set_xlim(ood_lower, ood_upper)
    # axs[0].grid(True, alpha=0.3)
    
    # Plot 2: Base model mean with Gaussian uncertainty (top right)
    axs[1].scatter(X_train[:50], y_train[:50], marker='x', color='k')
    axs[1].plot(x_plot, y_true, color='gray', label='Ground Truth')
    axs[1].plot(x_plot, base_mean, color='#8a2be280', label='Base Model Mean')
    axs[1].axvspan(ood_lower, ood_lower + 1, alpha=0.3, color='gray', label='OOD Region', linewidth=0)
    axs[1].axvspan(ood_upper - 1, ood_upper, alpha=0.3, color='gray', linewidth=0)
    axs[1].fill_between(x_plot.flatten(), 
        base_mean.flatten() - 2 * gaussian_sigma.flatten(), 
        base_mean.flatten() + 2 * gaussian_sigma.flatten(), 
        color='#8a2be280', alpha=0.3, label=r'Post-hoc Uncertainty $\left( \pm 2\sigma \right)$',
        linewidth=0)
    
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('y')
    axs[1].set_title('(Ours) Estimated on base model inputs')
    axs[1].legend(ncol=2, loc='upper center')
    axs[1].set_ylim(-4, 4)
    axs[1].set_xlim(ood_lower, ood_upper)
    
    # Plot 4: Comparison of noise predictions (bottom right)
    axs[2].plot(x_plot, true_noise, 'k-', label='Ground Truth Noise (σ)')
    axs[2].plot(x_plot, bayescap_sigma, 'g-', label='Bayescap Predicted Noise (σ)')
    axs[2].plot(x_plot, gaussian_sigma, 'r-', label='Gaussian Predicted Noise (σ)')
    
    # Pl2egions covered by training data
    axs[2].axvspan(ood_lower, ood_lower + 1, alpha=0.3, color='gray', label='OOD Region', linewidth=0)
    axs[2].axvspan(ood_upper - 1, ood_upper, alpha=0.3, color='gray', linewidth=0)
    
    axs[2].set_xlabel('x')
    axs[2].set_ylabel('Standard Deviation (σ)')
    axs[2].set_title('Uncertainty prediction comparison vs. ground truth')
    axs[2].set_ylim(-0.1, 2.)
    axs[2].set_xlim(ood_lower, ood_upper)
    axs[2].legend(ncol=2, loc='upper center')
    # axs[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join('intro_experiments', 'results', f'{args.output_name}_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training complete. Results saved to intro_experiments/results/{args.output_name}_comparison.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and visualize multiple post-hoc models')
    
    # Dataset parameters
    parser.add_argument('--train_samples', type=int, default=300, help='Number of training samples')
    parser.add_argument('--test_samples', type=int, default=200, help='Number of test samples')
    parser.add_argument('--noise_level', type=float, default=0.2, help='Noise level for the dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    
    # Model parameters
    parser.add_argument('--base_hidden_dim', type=int, default=128, help='Hidden dimension for the base model')
    parser.add_argument('--posthoc_hidden_dim', type=int, default=64, help='Hidden dimension for the post-hoc models')
    
    # Training parameters
    parser.add_argument('--n_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--base_lr', type=float, default=1e-3, help='Learning rate for base model')
    parser.add_argument('--posthoc_lr', type=float, default=1e-3, help='Learning rate for post-hoc models')
    
    # Output parameters
    parser.add_argument('--output_name', type=str, default='intro_bayescap_gaussian', help='Base name for output files')
    
    args = parser.parse_args()
    main(args) 