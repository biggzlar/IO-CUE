import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import argparse
from pathlib import Path

# Add parent directory to path to access modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from networks.simple_regression_model import SimpleRegressionModel
from dataloaders.uci_datasets import UCIDatasets
from models.base_ensemble_model import BaseEnsemble
from models.post_hoc_ensemble_model import PostHocEnsemble
# Import with alias to avoid any potential conflicts
from predictors.gaussian import gaussian_nll as gaussian_nll_loss
from predictors.gaussian import gaussian_nll_detached, predict_gaussian, post_hoc_predict_gaussian
from predictors.mse import mse, predict_mse
from predictors.bayescap import bayescap_loss, predict_bayescap
from predictors.generalized_gaussian import gen_gaussian_nll, predict_gen_gaussian, post_hoc_predict_gen_gaussian
from evaluation.metrics import compute_ece, compute_nll, compute_euc

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

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train and evaluate models on UCI datasets')
    parser.add_argument('dataset', type=str, help='Name of the UCI dataset to use')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--hidden_dim', type=int, default=100, help='Hidden dimension size')
    parser.add_argument('--n_ensemble', type=int, default=5, help='Number of ensemble members')
    parser.add_argument('--run_id', type=int, default=0, help='Run ID to attach to metrics output file')
    parser.add_argument('--lr', type=float, default=5e-3, help='Learning rate')
    parser.add_argument('--eval_freq', type=int, default=20, help='Evaluation frequency (epochs)')
    parser.add_argument('--device', type=str, default='cuda:2', help='Device to use for training')
    args = parser.parse_args()
    
    # Device configuration
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Load UCI dataset
    dataset = UCIDatasets(args.dataset, batch_size=args.batch_size)
    train_loader = dataset.get_train_loader()
    test_loader = dataset.get_test_loader()
    
    n_epochs = 100

    # Get feature dimension
    for batch_x, _ in train_loader:
        input_dim = batch_x.shape[1]
        break
    
    # Create results directory
    results_dir = Path(f"results/uci_benchmarks/{args.dataset}")
    results_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(f"intro_experiments/models/{args.dataset}")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup models
    
    # 1. MSE base model
    mse_base_model_params = {
        'in_channels': input_dim,
        'hidden_dim': args.hidden_dim,
        'output_dim': 1,  # only mean for MSE
        'activation': nn.ReLU,
    }
    
    mse_base_ensemble = BaseEnsemble(
        model_class=SimpleRegressionModel,
        model_params=mse_base_model_params,
        infer=predict_mse,
        n_models=8,
        device=device
    )
    
    # 2. Gaussian base model
    gaussian_base_model_params = {
        'in_channels': input_dim,
        'hidden_dim': args.hidden_dim,
        'output_dim': 2,  # mean and log_sigma for Gaussian
        'activation': nn.ReLU,
    }
    
    gaussian_base_ensemble = BaseEnsemble(
        model_class=SimpleRegressionModel,
        model_params=gaussian_base_model_params,
        infer=predict_gaussian,
        n_models=1,
        device=device
    )
    
    # 3. BayesCap post-hoc model
    bayescap_posthoc_model_params = {
        'in_channels': 1,  # Input is just the mean prediction
        'hidden_dim': args.hidden_dim // 2,
        'output_dim': 3,  # mu_tilde, one_over_alpha, beta for BayesCap
        'activation': nn.Tanh,
    }
    
    bayescap_posthoc_ensemble = PostHocEnsemble(
        model_class=SimpleRegressionModel,
        model_params=bayescap_posthoc_model_params,
        infer=predict_bayescap,
        n_models=1,
        device=device
    )
    # # Set the in_channels parameter used in the predict method
    # bayescap_posthoc_ensemble.model_params['in_channels'] = 1
    
    # 4. Gaussian post-hoc model
    gaussian_posthoc_model_params = {
        'in_channels': input_dim,  # Input dimension + 1 for the mean prediction
        'hidden_dim': args.hidden_dim // 2,
        'output_dim': 1,  # log_sigma for post-hoc Gaussian
        'activation': nn.Tanh,
    }
    
    gaussian_posthoc_ensemble = PostHocEnsemble(
        model_class=SimpleRegressionModel,
        model_params=gaussian_posthoc_model_params,
        infer=post_hoc_predict_gaussian,
        n_models=1,
        device=device
    )
    # # Set the in_channels parameter used in the predict method
    # gaussian_posthoc_ensemble.model_params['in_channels'] = 3
    
    # Define optimizer parameters
    optimizer_params = {
        'lr': args.lr,
        'weight_decay': 1e-4
    }
    
    # Train MSE base ensemble model
    print(f"\nTraining MSE base ensemble model...")
    mse_base_ensemble.optimize(
        results_dir=results_dir,
        model_dir=model_dir,
        train_loader=train_loader,
        test_loader=test_loader,
        n_epochs=n_epochs,
        optimizer_type='Adam',
        optimizer_params=optimizer_params,
        scheduler_type='CosineAnnealingLR',
        scheduler_params={'T_max': n_epochs},
        criterion=mse,
        eval_freq=args.eval_freq
    )
    
    # Train Gaussian base ensemble model
    print(f"\nTraining Gaussian base ensemble model...")
    gaussian_base_ensemble.optimize(
        results_dir=results_dir,
        model_dir=model_dir,
        train_loader=train_loader,
        test_loader=test_loader,
        n_epochs=n_epochs,
        optimizer_type='Adam',
        optimizer_params={'lr': 1e-4, 'weight_decay': 1e-1},
        scheduler_type='CosineAnnealingLR',
        scheduler_params={'T_max': n_epochs},
        criterion=gaussian_nll_loss,
        eval_freq=args.eval_freq
    )
    
    # Train BayesCap post-hoc ensemble model
    print(f"\nTraining BayesCap post-hoc ensemble model...")
    bayescap_posthoc_ensemble.optimize(
        results_dir=results_dir,
        model_dir=model_dir,
        train_loader=train_loader,
        mean_ensemble=mse_base_ensemble,  # Use MSE base model for mean predictions
        test_loader=test_loader,
        n_epochs=n_epochs,
        optimizer_type='AdamW',
        optimizer_params=optimizer_params,
        scheduler_type='CosineAnnealingLR',
        scheduler_params={'T_max': n_epochs},
        criterion=bayescap_loss,
        eval_freq=args.eval_freq,
        is_bayescap=True
    )
    
    # Train Gaussian post-hoc ensemble model
    print(f"\nTraining Gaussian post-hoc ensemble model...")
    gaussian_posthoc_ensemble.optimize(
        results_dir=results_dir,
        model_dir=model_dir,
        train_loader=train_loader,
        mean_ensemble=mse_base_ensemble,  # Use MSE base model for mean predictions
        test_loader=test_loader,
        n_epochs=n_epochs,
        optimizer_type='AdamW',
        optimizer_params={'lr': 1e-4, 'weight_decay': 1e-1},
        scheduler_type='CosineAnnealingLR',
        scheduler_params={'T_max': n_epochs},
        criterion=gaussian_nll_detached,
        eval_freq=args.eval_freq
    )
    # gaussian_posthoc_ensemble.load(f"{model_dir}/gaussian_posthoc_ensemble_model_best.pt")
    
    # Generate plots for visualization
    # Get test data for plotting
    X_test_all = []
    y_test_all = []
    
    for X_batch, y_batch in test_loader:
        X_test_all.append(X_batch)
        y_test_all.append(y_batch)
    
    X_test_tensor = torch.cat(X_test_all, dim=0).to(device)
    y_test_tensor = torch.cat(y_test_all, dim=0).to(device)
    
    # Choose the first feature for plotting
    plot_feature_idx = 0
    X_test_np = X_test_tensor.cpu().numpy()
    y_test_np = y_test_tensor.cpu().numpy()
    
    # Sort data by the selected feature for cleaner plots
    sort_indices = np.argsort(X_test_np[:, plot_feature_idx])
    X_test_sorted = X_test_np[sort_indices]
    y_test_sorted = y_test_np[sort_indices]
    X_plot_feature = X_test_sorted[:, plot_feature_idx]
    
    # Get predictions from all models
    mse_preds = mse_base_ensemble.predict(X_test_tensor)
    gaussian_preds = gaussian_base_ensemble.predict(X_test_tensor)
    
    # For BayesCap, only pass the mean prediction since it's specialized for that
    bayescap_preds = bayescap_posthoc_ensemble.predict(mse_preds['mean'])
    
    # For Gaussian post-hoc, pass both X and the mean prediction from MSE base
    gaussian_posthoc_preds = gaussian_posthoc_ensemble.predict(X_test_tensor, y_pred=mse_preds['mean'])
    
    # Extract means and uncertainties
    mse_mean = mse_preds['mean'].cpu().numpy()[sort_indices].squeeze()
    mse_ep_sigma = mse_preds['ep_sigma'].cpu().numpy()[sort_indices].squeeze()
    
    gaussian_mean = gaussian_preds['mean'].cpu().numpy()[sort_indices].squeeze()
    gaussian_al_sigma = gaussian_preds['al_sigma'].cpu().numpy()[sort_indices].squeeze()
    
    bayescap_sigma = bayescap_preds['sigma'].cpu().numpy()[sort_indices].squeeze()
    
    gaussian_posthoc_sigma = gaussian_posthoc_preds['sigma'].cpu().numpy()[sort_indices].squeeze()
    
    # Ensure y_test is 1D for plotting
    y_test_sorted = y_test_sorted.squeeze()


    # Inverse transform y_test
    y_test_sorted = y_test_sorted * dataset.y_train_scale.item() + dataset.y_train_mu.item()
    mse_mean = mse_mean * dataset.y_train_scale.item() + dataset.y_train_mu.item()
    gaussian_mean = gaussian_mean * dataset.y_train_scale.item() + dataset.y_train_mu.item()

    bayescap_sigma = bayescap_sigma * dataset.y_train_scale.item()
    gaussian_posthoc_sigma = gaussian_posthoc_sigma * dataset.y_train_scale.item()
    mse_ep_sigma = mse_ep_sigma * dataset.y_train_scale.item()
    gaussian_al_sigma = gaussian_al_sigma * dataset.y_train_scale.item()
    
    # Create figures for each model
    # 1. MSE Base Model
    plt.figure(figsize=(10, 6))
    plt.scatter(X_plot_feature, y_test_sorted, s=10, color='gray', alpha=0.5, label='Test Data')
    plt.plot(X_plot_feature, mse_mean, color='blue', label='MSE Prediction')
    plt.fill_between(X_plot_feature, 
                    mse_mean - 2 * mse_ep_sigma, 
                    mse_mean + 2 * mse_ep_sigma, 
                    color='blue', alpha=0.2, label='Uncertainty (±2σ)')
    plt.xlabel(f'Feature {plot_feature_idx}')
    plt.ylabel('Target')
    plt.title(f'MSE Base Model - {args.dataset.capitalize()}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / f"mse_base_plot.png")
    
    # 2. Gaussian Base Model
    plt.figure(figsize=(10, 6))
    plt.scatter(X_plot_feature, y_test_sorted, s=10, color='gray', alpha=0.5, label='Test Data')
    plt.plot(X_plot_feature, gaussian_mean, color='red', label='Gaussian Prediction')
    plt.fill_between(X_plot_feature, 
                    gaussian_mean - 2 * gaussian_al_sigma, 
                    gaussian_mean + 2 * gaussian_al_sigma, 
                    color='red', alpha=0.2, label='Uncertainty (±2σ)')
    plt.xlabel(f'Feature {plot_feature_idx}')
    plt.ylabel('Target')
    plt.title(f'Gaussian Base Model - {args.dataset.capitalize()}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / f"gaussian_base_plot.png")
    
    # 3. BayesCap Post-hoc Model
    plt.figure(figsize=(10, 6))
    plt.scatter(X_plot_feature, y_test_sorted, s=10, color='gray', alpha=0.5, label='Test Data')
    plt.plot(X_plot_feature, mse_mean, color='green', label='MSE Base + BayesCap Prediction')
    plt.fill_between(X_plot_feature, 
                    mse_mean - 2 * bayescap_sigma, 
                    mse_mean + 2 * bayescap_sigma, 
                    color='green', alpha=0.2, label='Uncertainty (±2σ)')
    plt.xlabel(f'Feature {plot_feature_idx}')
    plt.ylabel('Target')
    plt.title(f'BayesCap Post-hoc Model - {args.dataset.capitalize()}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / f"bayescap_posthoc_plot.png")
    
    # 4. Gaussian Post-hoc Model
    plt.figure(figsize=(10, 6))
    plt.scatter(X_plot_feature, y_test_sorted, s=10, color='gray', alpha=0.5, label='Test Data')
    plt.plot(X_plot_feature, mse_mean, color='purple', label='MSE Base + Gaussian Post-hoc')
    plt.fill_between(X_plot_feature, 
                    mse_mean - 2 * gaussian_posthoc_sigma, 
                    mse_mean + 2 * gaussian_posthoc_sigma, 
                    color='purple', alpha=0.2, label='Uncertainty (±2σ)')
    plt.xlabel(f'Feature {plot_feature_idx}')
    plt.ylabel('Target')
    plt.title(f'Gaussian Post-hoc Model - {args.dataset.capitalize()}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / f"gaussian_posthoc_plot.png")
    
    # Save numerical results
    # Convert numpy arrays to tensors for metrics computation
    mse_mean_tensor = torch.tensor(mse_mean)
    gaussian_mean_tensor = torch.tensor(gaussian_mean)
    y_test_tensor_sorted = torch.tensor(y_test_sorted)
    
    # Prepare uncertainty tensors - these need to be converted to the right format
    mse_ep_sigma_tensor = torch.tensor(mse_ep_sigma)
    gaussian_al_sigma_tensor = torch.tensor(gaussian_al_sigma)
    bayescap_sigma_tensor = torch.tensor(bayescap_sigma)
    gaussian_posthoc_sigma_tensor = torch.tensor(gaussian_posthoc_sigma)
    
    # Compute metrics for MSE base model
    mse_residuals = torch.abs(mse_mean_tensor - y_test_tensor_sorted)
    mse_rmse = np.sqrt(np.mean((mse_mean - y_test_sorted) ** 2))
    mse_ece = compute_ece(mse_residuals, mse_ep_sigma_tensor)
    mse_nll = compute_nll(mse_mean_tensor, mse_ep_sigma_tensor**2, y_test_tensor_sorted).mean().item()
    mse_euc, _ = compute_euc(mse_mean_tensor, mse_ep_sigma_tensor, y_test_tensor_sorted)
    
    # Compute metrics for Gaussian base model
    gaussian_residuals = torch.abs(gaussian_mean_tensor - y_test_tensor_sorted)
    gaussian_rmse = np.sqrt(np.mean((gaussian_mean - y_test_sorted) ** 2))
    gaussian_ece = compute_ece(gaussian_residuals, gaussian_al_sigma_tensor)
    gaussian_nll = compute_nll(gaussian_mean_tensor, gaussian_al_sigma_tensor**2, y_test_tensor_sorted).mean().item()
    gaussian_euc, _ = compute_euc(gaussian_mean_tensor, gaussian_al_sigma_tensor, y_test_tensor_sorted)
    
    # Compute metrics for BayesCap post-hoc model
    bayescap_rmse = np.sqrt(np.mean((mse_mean - y_test_sorted) ** 2))  # Same mean as MSE base
    bayescap_ece = compute_ece(mse_residuals, bayescap_sigma_tensor)
    bayescap_nll = compute_nll(mse_mean_tensor, bayescap_sigma_tensor**2, y_test_tensor_sorted).mean().item()
    bayescap_euc, _ = compute_euc(mse_mean_tensor, bayescap_sigma_tensor, y_test_tensor_sorted)
    
    # Compute metrics for Gaussian post-hoc model
    gaussian_posthoc_rmse = np.sqrt(np.mean((mse_mean - y_test_sorted) ** 2))  # Same mean as MSE base
    gaussian_posthoc_ece = compute_ece(mse_residuals, gaussian_posthoc_sigma_tensor)
    gaussian_posthoc_nll = compute_nll(mse_mean_tensor, gaussian_posthoc_sigma_tensor**2, y_test_tensor_sorted).mean().item()
    gaussian_posthoc_euc, _ = compute_euc(mse_mean_tensor, gaussian_posthoc_sigma_tensor, y_test_tensor_sorted)
    
    results = {
        'mse_base': {
            'rmse': mse_rmse,
            'ece': mse_ece.item(),
            'nll': mse_nll,
            'euc': mse_euc,
            'min_nll': mse_base_ensemble.min_nll,
        },
        'gaussian_base': {
            'rmse': gaussian_rmse,
            'ece': gaussian_ece.item(),
            'nll': gaussian_nll,
            'euc': gaussian_euc,
            'min_nll': gaussian_base_ensemble.min_nll,
        },
        'bayescap_posthoc': {
            'rmse': bayescap_rmse,
            'ece': bayescap_ece.item(),
            'nll': bayescap_nll,
            'euc': bayescap_euc,
            'min_nll': bayescap_posthoc_ensemble.min_nll,
        },
        'gaussian_posthoc': {
            'rmse': gaussian_posthoc_rmse,
            'ece': gaussian_posthoc_ece.item(),
            'nll': gaussian_posthoc_nll,
            'euc': gaussian_posthoc_euc,
            'min_nll': gaussian_posthoc_ensemble.min_nll,
        }
    }
    
    # Save results to file
    with open(results_dir / f"metrics_{args.run_id}.txt", 'w') as f:
        f.write(f"Results for {args.dataset} dataset:\n\n")
        
        f.write("MSE Base Model:\n")
        f.write(f"  RMSE: {results['mse_base']['rmse']:.4f}\n")
        f.write(f"  ECE: {results['mse_base']['ece']:.4f}\n")
        f.write(f"  NLL: {results['mse_base']['nll']:.4f}\n")
        f.write(f"  EUC: {results['mse_base']['euc']:.4f}\n\n")
        
        f.write("Gaussian Base Model:\n")
        f.write(f"  RMSE: {results['gaussian_base']['rmse']:.4f}\n")
        f.write(f"  ECE: {results['gaussian_base']['ece']:.4f}\n")
        f.write(f"  NLL: {results['gaussian_base']['nll']:.4f}\n")
        f.write(f"  EUC: {results['gaussian_base']['euc']:.4f}\n\n")
        
        f.write("BayesCap Post-hoc Model:\n")
        f.write(f"  RMSE: {results['bayescap_posthoc']['rmse']:.4f}\n")
        f.write(f"  ECE: {results['bayescap_posthoc']['ece']:.4f}\n")
        f.write(f"  NLL: {results['bayescap_posthoc']['nll']:.4f}\n")
        f.write(f"  EUC: {results['bayescap_posthoc']['euc']:.4f}\n\n")
        
        f.write("Gaussian Post-hoc Model:\n")
        f.write(f"  RMSE: {results['gaussian_posthoc']['rmse']:.4f}\n")
        f.write(f"  ECE: {results['gaussian_posthoc']['ece']:.4f}\n")
        f.write(f"  NLL: {results['gaussian_posthoc']['nll']:.4f}\n")
        f.write(f"  EUC: {results['gaussian_posthoc']['euc']:.4f}\n")
    
    print(f"\nResults saved to {results_dir}")

if __name__ == "__main__":
    main() 