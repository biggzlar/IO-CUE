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
from models.base_ensemble_model import BaseEnsemble
from models.post_hoc_frameworks import ICUE
from predictors.gaussian import gaussian_nll, predict_gaussian
from predictors.mse import mse, predict_mse
from intro_experiments.train_bayescap_vs_post_hoc import (
    train_base_model as train_single_base_model,
    train_posthoc_model as train_single_posthoc_model,
)

plt.rcParams.update(
    {
        "font.size": 12,
        "text.usetex": False,
        "font.family": "stixgeneral",
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

# Set random seeds for reproducibility
# torch.manual_seed(42)
# np.random.seed(42)

# Device configuration
device = "cpu"




def main(args):
    ood_lower, ood_upper = 0, 5

    # Generate dataset
    train_loader, X_train, y_train, true_std_train = generate_data(
        n_samples=args.train_samples,
        x_range=(ood_lower + 1, ood_upper - 1),
        noise_level=args.noise_level,
        batch_size=args.batch_size,
    )

    test_loader, X_test, y_test, true_std_test = generate_data(
        n_samples=args.test_samples,
        x_range=(ood_lower, ood_upper),
        noise_level=args.noise_level,
        batch_size=args.batch_size,
    )

    # Prepare directories
    results_dir = os.path.join('intro_experiments', 'results')
    model_dir = os.path.join('intro_experiments', 'models')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # -----------------------
    # Single-model BayesCap pipeline
    # -----------------------
    base_loss_wrapper = LossWrapper("mse")
    base_output_dims = base_loss_wrapper.output_dims
    base_model = SimpleRegressionModel(
        in_channels=1,
        hidden_dim=args.base_hidden_dim,
        output_dim=base_output_dims,
        activation=nn.Mish,
    ).to(device)

    bayescap_loss_wrapper = LossWrapper("bayescap", post_hoc=True)
    # gaussian_posthoc_loss_wrapper = LossWrapper("gaussian", post_hoc=True)

    bayescap_output_dims = bayescap_loss_wrapper.output_dims
    # gaussian_posthoc_output_dims = gaussian_posthoc_loss_wrapper.output_dims

    bayescap_model = SimpleRegressionModel(
        in_channels=base_output_dims,
        hidden_dim=args.posthoc_hidden_dim,
        output_dim=bayescap_output_dims,
        activation=nn.ReLU,
    ).to(device)

    # gaussian_posthoc_model = SimpleRegressionModel(
    #     in_channels=1,
    #     hidden_dim=args.posthoc_hidden_dim,
    #     output_dim=gaussian_posthoc_output_dims,
    #     activation=nn.Mish,
    # ).to(device)

    bayescap_predictor = PredictorWrapper("bayescap")
    # gaussian_posthoc_predictor = PredictorWrapper("gaussian")

    print("Training single-model base (MSE)...")
    base_model = train_single_base_model(
        model=base_model,
        train_loader=train_loader,
        loss_wrapper=base_loss_wrapper,
        n_epochs=args.n_epochs,
        lr=args.base_lr,
    )

    print("Training BayesCap post-hoc model...")
    bayescap_model = train_single_posthoc_model(
        model=bayescap_model,
        base_model=base_model,
        loss_wrapper=bayescap_loss_wrapper,
        X=X_train,
        y=y_train,
        n_epochs=args.n_epochs,
        lr=args.posthoc_lr,
        is_bayescap=True,
    )

    # print("Training Gaussian post-hoc model (single-model pipeline)...")
    # gaussian_posthoc_model = train_single_posthoc_model(
    #     model=gaussian_posthoc_model,
    #     base_model=base_model,
    #     loss_wrapper=gaussian_posthoc_loss_wrapper,
    #     X=X_train,
    #     y=y_train,
    #     n_epochs=args.n_epochs,
    #     lr=args.posthoc_lr,
    #     is_bayescap=False,
    # )

    # -----------------------
    # Ensemble pipelines: Gaussian-NLL and MSE + ICUE
    # -----------------------
    gaussian_base_model_params = {
        'in_channels': 1,
        'hidden_dim': args.base_hidden_dim,
        'activation': nn.Mish,
        'output_dim': 2,
    }

    mse_base_model_params = {
        'in_channels': 1,
        'hidden_dim': args.base_hidden_dim,
        'activation': nn.Mish,
        'output_dim': 1,
    }

    gaussian_base_ensemble = BaseEnsemble(
        model_class=SimpleRegressionModel,
        model_params=gaussian_base_model_params,
        infer=predict_gaussian,
        n_models=args.n_ensemble,
        device=device,
    )

    mse_base_ensemble = BaseEnsemble(
        model_class=SimpleRegressionModel,
        model_params=mse_base_model_params,
        infer=predict_mse,
        n_models=args.n_ensemble,
        device=device,
    )

    posthoc_model_params = {
        'in_channels': 1,
        'hidden_dim': args.posthoc_hidden_dim,
        'activation': nn.Mish,
        'output_dim': 1,
    }

    base_optimizer_params = {
        'lr': args.base_lr,
        'weight_decay': 1e-4,
    }

    posthoc_optimizer_params = {
        'lr': args.posthoc_lr,
        'weight_decay': 1e-4,
    }

    print("Training Gaussian-NLL base ensemble...")
    gaussian_base_ensemble.optimize(
        results_dir=results_dir,
        model_dir=model_dir,
        train_loader=train_loader,
        test_loader=test_loader,
        n_epochs=args.n_epochs,
        optimizer_type='AdamW',
        optimizer_params=base_optimizer_params,
        criterion=gaussian_nll,
        eval_freq=20,
    )

    print("Training MSE base ensemble...")
    mse_base_ensemble.optimize(
        results_dir=results_dir,
        model_dir=model_dir,
        train_loader=train_loader,
        test_loader=test_loader,
        n_epochs=args.n_epochs,
        optimizer_type='AdamW',
        optimizer_params=base_optimizer_params,
        criterion=mse,
        eval_freq=20,
    )

    print("Training ICUE post-hoc ensemble (on MSE ensemble)...")
    icue_ensemble = ICUE(
        mean_ensemble=mse_base_ensemble,
        model_class=SimpleRegressionModel,
        model_params=posthoc_model_params,
        n_models=args.n_ensemble,
        device=device,
    )

    # # Experiment: Add guaranteed, random OOD data.
    # random_data = torch.randn(100, 1).to(device) * 8. + 4.
    # mask = torch.logical_or((random_data > 4.), (random_data < 1.))
    # random_data = random_data[mask.squeeze(-1)]
    # random_targets = torch.randn(len(random_data), 1).to(device)
    # random_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(random_data, random_targets), batch_size=32, shuffle=True)
    
    # # Extract all data from both datasets and merge into a single dataloader
    # train_X = torch.cat([train_loader.dataset.X, random_loader.dataset.tensors[0]], dim=0)
    # train_y = torch.cat([train_loader.dataset.y, random_loader.dataset.tensors[1]], dim=0)
    # combined_dataset = torch.utils.data.TensorDataset(train_X, train_y)
    # train_loader = torch.utils.data.DataLoader(
    #     combined_dataset, 
    #     batch_size=train_loader.batch_size, 
    #     shuffle=True
    # )


    icue_ensemble.optimize(
        results_dir=results_dir,
        model_dir=model_dir,
        train_loader=train_loader,
        test_loader=test_loader,
        n_epochs=args.n_epochs,
        optimizer_type='Adam',
        optimizer_params=posthoc_optimizer_params,
        eval_freq=20,
    )

    # -----------------------
    # Inference for plotting
    # -----------------------
    x_plot = np.linspace(ood_lower, ood_upper, 500).reshape(-1, 1)
    x_plot_tensor = torch.FloatTensor(x_plot).to(device)

    with torch.no_grad():
        # Single-model base mean
        base_mean_tensor = base_model(x_plot_tensor)
        base_mean = base_mean_tensor.cpu().numpy()

        # BayesCap sigma (single-model)
        bayescap_outputs = bayescap_model(base_mean_tensor)
        bayescap_info = bayescap_predictor.predict_from_outputs(bayescap_outputs)
        bayescap_sigma = bayescap_info["sigma"].cpu().numpy()

        # # Gaussian post-hoc sigma (single-model)
        # gaussian_posthoc_outputs = gaussian_posthoc_model(x_plot_tensor)
        # gaussian_posthoc_info = gaussian_posthoc_predictor.predict_from_outputs(gaussian_posthoc_outputs)
        # gaussian_posthoc_sigma = gaussian_posthoc_info["sigma"].cpu().numpy()

        # Gaussian ensemble predictions
        gaussian_preds = gaussian_base_ensemble.predict(x_plot_tensor)
        gaussian_mean = gaussian_preds['mean'].cpu().numpy()
        if 'al_sigma' in gaussian_preds:
            gaussian_sigma = gaussian_preds['al_sigma'].cpu().numpy()
        else:
            gaussian_sigma = gaussian_preds['sigma'].cpu().numpy() if 'sigma' in gaussian_preds else np.zeros_like(gaussian_mean)

        # MSE ensemble mean and ICUE sigma
        mse_preds = mse_base_ensemble.predict(x_plot_tensor)
        mse_mean = mse_preds['mean'].cpu().numpy()
        icue_preds = icue_ensemble.predict(x_plot_tensor)
        icue_sigma = icue_preds['sigma'].cpu().numpy()

    # Ground truth (noise-free) and true noise
    y_true = np.array([ground_truth_function(x[0]) for x in x_plot])
    true_noise = (args.noise_level * ground_truth_noise(x_plot)).squeeze()

    # -----------------------
    # Plotting: 1x4 panels (Gaussian NLL, BayesCap, ICUE, Curves)
    # -----------------------
    fig, axs = plt.subplots(1, 4, figsize=(28, 5))

    # (a) Gaussian-NLL ensemble
    axs[0].scatter(X_train[:50], y_train[:50], marker='x', color='k')
    axs[0].plot(x_plot, y_true, color='gray', label='Ground Truth')
    axs[0].plot(x_plot, gaussian_mean, color='#FF6B6B', label='Gaussian Ensemble Mean')
    axs[0].axvspan(ood_lower, ood_lower + 1, alpha=0.3, color='gray', label='OOD Region', linewidth=0)
    axs[0].axvspan(ood_upper - 1, ood_upper, alpha=0.3, color='gray', linewidth=0)
    axs[0].fill_between(
        x_plot.flatten(),
        gaussian_mean.flatten() - 2 * gaussian_sigma.flatten(),
        gaussian_mean.flatten() + 2 * gaussian_sigma.flatten(),
        color='#FF6B6B', alpha=0.3, label=r'Gaussian Uncertainty $\left( \pm 2\sigma \right)$', linewidth=0,
    )
    axs[0].set_xlabel(r'$x$', fontsize=18)
    axs[0].set_ylabel(r'$y$', fontsize=18)
    axs[0].set_title(r'$\mathbf{(a)}$ Model trained on Gaussian NLL')
    axs[0].legend(ncol=2, loc='upper center')
    axs[0].set_ylim(-4, 4)
    axs[0].set_xlim(ood_lower, ood_upper)

    # (b) BayesCap (single model)
    axs[1].scatter(X_train[:50], y_train[:50], marker='x', color='k')
    axs[1].plot(x_plot, y_true, color='gray', label='Ground Truth')
    axs[1].plot(x_plot, base_mean, linewidth=2, color='#20B2AA', label='Base Model Mean')
    axs[1].axvspan(ood_lower, ood_lower + 1, alpha=0.3, color='gray', label='OOD Region', linewidth=0)
    axs[1].axvspan(ood_upper - 1, ood_upper, alpha=0.3, color='gray', linewidth=0)
    axs[1].fill_between(
        x_plot.flatten(),
        base_mean.flatten() - 2 * bayescap_sigma.flatten(),
        base_mean.flatten() + 2 * bayescap_sigma.flatten(),
        color='#20B2AA', alpha=0.3, label=r'Bayescap Uncertainty $\left( \pm 2\sigma \right)$', linewidth=0,
    )
    axs[1].set_xlabel(r'$x$', fontsize=18)
    # axs[1].set_ylabel(r'$y$', fontsize=18)
    axs[1].set_title(r'$\mathbf{(b)}$ BayesCap: Estimated on base model outputs')
    axs[1].legend(ncol=2, loc='upper center')
    axs[1].set_ylim(-4, 4)
    axs[1].set_xlim(ood_lower, ood_upper)

    # (c) ICUE post-hoc on MSE ensemble
    axs[2].scatter(X_train[:50], y_train[:50], marker='x', color='k')
    axs[2].plot(x_plot, y_true, color='gray', label='Ground Truth')
    axs[2].plot(x_plot, mse_mean, color='#8a2be2', label='MSE Ensemble Mean')
    axs[2].axvspan(ood_lower, ood_lower + 1, alpha=0.3, color='gray', label='OOD Region', linewidth=0)
    axs[2].axvspan(ood_upper - 1, ood_upper, alpha=0.3, color='gray', linewidth=0)
    axs[2].fill_between(
        x_plot.flatten(),
        mse_mean.flatten() - 2 * icue_sigma.flatten(),
        mse_mean.flatten() + 2 * icue_sigma.flatten(),
        color='#8a2be2', alpha=0.3, label=r'Post-hoc Uncertainty $\left( \pm 2\sigma \right)$', linewidth=0,
    )
    axs[2].set_xlabel(r'$x$', fontsize=18)
    # axs[2].set_ylabel(r'$y$', fontsize=18)
    axs[2].set_title(r'$\mathbf{(c)}$ Model trained on MSE with Post-hoc Uncertainty')
    axs[2].legend(ncol=2, loc='upper center')
    axs[2].set_ylim(-4, 4)
    axs[2].set_xlim(ood_lower, ood_upper)

    # (d) Uncertainty curves
    axs[3].plot(x_plot, true_noise, 'k--', label=r'Ground Truth Noise ($\sigma$)')
    axs[3].axvspan(ood_lower, ood_lower + 1, alpha=0.3, color='gray', label='OOD Region', linewidth=0)
    axs[3].axvspan(ood_upper - 1, ood_upper, alpha=0.3, color='gray', linewidth=0)
    axs[3].plot(x_plot, gaussian_sigma, linewidth=2, color='#FF6B6B', label=r'Gaussian Predicted Noise ($\sigma$)')
    axs[3].plot(x_plot, bayescap_sigma, linewidth=2, color='#20B2AA', label=r'Bayescap Predicted Noise ($\sigma$)')
    axs[3].plot(x_plot, icue_sigma, linewidth=2, color='#8a2be2', label=r'Post-hoc Predicted Noise ($\sigma$)')
    axs[3].set_xlabel(r'$x$', fontsize=18)
    axs[3].set_ylabel(r'$\sigma$', fontsize=18)
    axs[3].set_title(r'$\mathbf{(d)}$ Uncertainty prediction comparison vs. ground truth')
    axs[3].set_ylim(-0.1, 2.0)
    axs[3].set_xlim(ood_lower, ood_upper)
    axs[3].legend(ncol=2, loc='upper center')

    plt.tight_layout()
    out_path = os.path.join(results_dir, f'{args.output_name}.pdf')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Combined figure saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and visualize BayesCap, Gaussian-NLL ensemble, ICUE post-hoc in one figure')

    # Dataset parameters
    parser.add_argument('--train_samples', type=int, default=300, help='Number of training samples')
    parser.add_argument('--test_samples', type=int, default=200, help='Number of test samples')
    parser.add_argument('--noise_level', type=float, default=0.2, help='Noise level for the dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')

    # Model parameters
    parser.add_argument('--base_hidden_dim', type=int, default=128, help='Hidden dimension for base models')
    parser.add_argument('--posthoc_hidden_dim', type=int, default=64, help='Hidden dimension for post-hoc models')
    parser.add_argument('--n_ensemble', type=int, default=3, help='Number of models in ensembles')

    # Training parameters
    parser.add_argument('--n_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--base_lr', type=float, default=1e-3, help='Learning rate for base models')
    parser.add_argument('--posthoc_lr', type=float, default=1e-3, help='Learning rate for post-hoc models')

    # Output
    parser.add_argument('--output_name', type=str, default='combined_four_panel', help='Output figure file name (without extension)')

    args = parser.parse_args()
    main(args)


