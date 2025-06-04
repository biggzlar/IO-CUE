import numpy as np
import torch
from evaluation.eval_depth_utils import get_predictions, visualize_results
from models import BaseEnsemble, PostHocEnsemble
import os
import argparse
from configs.config_utils import (
    process_config_from_args,
    setup_result_directories
)

# Ensure the results directory exists
os.makedirs('results', exist_ok=True)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Bayesian Crown experiments')
    parser.add_argument('-yc', '--yaml_config', type=str, help='Path to YAML configuration file')
    parser.add_argument('-d', '--device', type=int, help='Device to run on')
    parser.add_argument('--visualize-only', action='store_true', help='Skip training and only create visualizations')
    args = parser.parse_args()
    
    mean_ensemble_path = "results/base_unet/checkpoints/base_ensemble_model_best.pt"

    # Process config from args
    config, config_name = process_config_from_args(args)
    
    # Set random seeds for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)
    
    # Get device
    device = torch.device(f"cuda:{args.device}")
    print(f"Working on device: {device}")

    # Get dataloaders directly from config
    train_loader = config['train_loader']
    test_loader = config['test_loader']
    
    # Extract training parameters from config
    n_ensemble_models = config['n_ensemble_models']
    n_variance_models = config['n_variance_models']
    # n_epochs = config['n_epochs']
    n_epochs = int(60000 / len(train_loader))
    eval_freq = config['eval_freq']
    pair_models = config['pair_models']
    
    # Get model classes - support both old and new config formats
    mean_model_class = config.get('mean_model_class')
    variance_model_class = config.get('variance_model_class')
    
    # Define model parameters
    mean_model_params = config['mean_model_params']
    variance_model_params = config['variance_model_params']
    
    # Setup results directory
    model_dir, results_dir = setup_result_directories(config_name)
    
    # Create mean ensemble
    mean_ensemble = BaseEnsemble(
        model_class=mean_model_class,
        model_params=mean_model_params,
        n_models=n_ensemble_models, 
        device=device,
        infer=config['mean_predictor']
    )
    
    # Create variance ensemble
    sigma_ensemble = PostHocEnsemble(
        model_class=variance_model_class,
        model_params=variance_model_params,
        n_models=n_variance_models, 
        device=device,
        infer=config['variance_predictor']
    )
    
    if not args.visualize_only:
        print("Training models...")
        # Get mean training parameters
        mean_criterion = config['mean_criterion']

        # Load existing base model or train new one as needed
        try:
            mean_ensemble.load(mean_ensemble_path)
            print(f"Loaded mean ensemble from {mean_ensemble_path}")
        except Exception as e:
            print(f"Error loading mean ensemble from {model_dir}/base_ensemble_model_60.pth: {e}")
            mean_ensemble.optimize(
                results_dir=results_dir,
                model_dir=model_dir,
                train_loader=train_loader,
                test_loader=test_loader,
                n_epochs=n_epochs,
                optimizer_type=config['mean_optimizer_type'],
                optimizer_params=config['mean_optimizer_params'],
                scheduler_type=config['mean_scheduler_type'],
                scheduler_params=config['mean_scheduler_params'],
                criterion=mean_criterion,
                eval_freq=eval_freq
            )
    
        # Get variance training parameters
        variance_criterion = config['variance_criterion']
        # Train the variance ensemble with explicit parameters
        sigma_ensemble.optimize(
            results_dir=results_dir,
            model_dir=model_dir,
            train_loader=train_loader,
            test_loader=test_loader,
            mean_ensemble=mean_ensemble,
            n_epochs=n_epochs,
            optimizer_type=config['variance_optimizer_type'],
            optimizer_params=config['variance_optimizer_params'],
            scheduler_type=config['variance_scheduler_type'],
            scheduler_params=config['variance_scheduler_params'],
            pair_models=pair_models,
            criterion=variance_criterion,
            eval_freq=eval_freq
        )
    
        # Final evaluation
        print("\nFinal evaluation on test set:")
        results = mean_ensemble.evaluate(test_loader)
        print(f"Mean Ensemble - Test RMSE: {results['rmse']:.4f}, Test NLL: {results['nll']:.4f}")
        
        results = sigma_ensemble.evaluate(test_loader, mean_ensemble)
        print(f"Variance Ensemble - Test RMSE: {results['metrics']['rmse']:.4f}, Test NLL: {results['metrics']['nll']:.4f}")
        
        # Save model states
        os.makedirs(model_dir, exist_ok=True)
        
        # Save mean ensemble
        mean_ensemble.save(f"{model_dir}/mean_ensemble.pt")
        print(f"Saved mean ensemble to {model_dir}/mean_ensemble.pt")
        
        # Save variance ensemble
        sigma_ensemble.save(f"{model_dir}/variance_ensemble.pt")
        print(f"Saved variance ensemble to {model_dir}/variance_ensemble.pt")
    else:
        # Try to load pre-trained models if available
        if os.path.exists(mean_ensemble_path):
            print(f"Loading mean ensemble from {mean_ensemble_path}")
            mean_ensemble.load(mean_ensemble_path)
        else:
            print("No pre-trained mean ensemble found. Visualizations will use untrained models.")
            
        if os.path.exists(f"{model_dir}/variance_ensemble.pt"):
            print(f"Loading variance ensemble from {model_dir}/variance_ensemble.pt")
            sigma_ensemble.load(f"{model_dir}/variance_ensemble.pt")
        else:
            print("No pre-trained variance ensemble found. Visualizations will use untrained models.")

    # Visualize results
    if config['dataset_name'] == 'simple_depth':
        batch_x, batch_y = next(iter(test_loader))
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        results = sigma_ensemble.evaluate(test_loader, mean_ensemble)
        visualize_results(results, num_samples=5, metric_name="nll", path=f"{results_dir}", suffix="_test")
