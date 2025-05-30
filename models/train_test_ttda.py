import numpy as np
import torch
from evaluation.eval_depth_utils import get_predictions, visualize_results
from models.ttda_model import TTDAModel
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
    parser = argparse.ArgumentParser(description='Run TTDA (Test-Time Data Augmentation) experiments')
    parser.add_argument('-yc', '--yaml_config', type=str, help='Path to YAML configuration file')
    parser.add_argument('--visualize-only', action='store_true', help='Skip training and only create visualizations')
    parser.add_argument('--num-augmentations', type=int, default=10, help='Number of augmentations to apply per sample')
    args = parser.parse_args()
    
    # Process config from args
    config, config_name = process_config_from_args(args)
    
    # Set random seeds for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)
    
    # Get device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Working on device: {device}")

    # Get dataloaders directly from config
    train_loader = config['train_loader']
    test_loader = config['test_loader']
    
    # Extract training parameters from config
    n_epochs = config['n_epochs'] * config['n_ensemble_models']
    eval_freq = config['eval_freq']
    
    # Get model class
    model_class = config.get('mean_model_class')
    
    # Define model parameters - using mean model parameters
    model_params = config['mean_model_params']
    
    # Setup results directory
    model_dir, results_dir = setup_result_directories(f"{config_name}_ttda")
    
    # Create TTDA model with configurable number of augmentations
    ttda_model = TTDAModel(
        model_class=model_class,
        model_params=model_params,
        device=device,
        infer=config['mean_predictor'],
        num_augmentations=args.num_augmentations
    )
    
    print(f"Using {args.num_augmentations} augmentations per sample")
    
    if not args.visualize_only:
        print("Training TTDA model...")
        # Get training parameters
        criterion = config['mean_criterion']

        try:
            # Try to load a pre-trained model
            ttda_model.load(f"{model_dir}/ttda_model_best.pth")
            print(f"Loaded TTDA model from {model_dir}/ttda_model_best.pth")
        except Exception as e:
            print(f"Training a new TTDA model: {e}")
            
            # Train the TTDA model
            ttda_model.train(
                results_dir=results_dir,
                model_dir=model_dir,
                train_loader=train_loader,
                test_loader=test_loader,
                n_epochs=n_epochs,
                optimizer_type=config['mean_optimizer_type'],
                optimizer_params=config['mean_optimizer_params'],
                scheduler_type=config['mean_scheduler_type'],
                scheduler_params=config['mean_scheduler_params'],
                criterion=criterion,
                eval_freq=eval_freq
            )
    
        # Final evaluation
        print("\nFinal evaluation on test set:")
        rmse, nll = ttda_model.evaluate(test_loader)
        print(f"TTDA Model - Test RMSE: {rmse:.4f}, Test NLL: {nll:.4f}")
        
        # Save model state
        os.makedirs(model_dir, exist_ok=True)
        ttda_model.save(f"{model_dir}/ttda_model_best.pth")
        print(f"Saved TTDA model to {model_dir}/ttda_model_best.pth")
    else:
        # Try to load pre-trained model if available
        if os.path.exists(f"{model_dir}/ttda_model_best.pth"):
            print(f"Loading TTDA model from {model_dir}/ttda_model_best.pth")
            ttda_model.load(f"{model_dir}/ttda_model_best.pth")
        else:
            print("No pre-trained TTDA model found. Visualizations will use untrained model.")

    # Visualize results
    if config['dataset_name'] == 'simple_depth':
        batch_x, batch_y = next(iter(test_loader))
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        # Get predictions using the TTDA model
        results = get_predictions(model=ttda_model, images=batch_x, depths=batch_y, device=device)
        
        # Visualize the results
        visualize_results(results, num_samples=6, metric_name="nll", path=f"{results_dir}", suffix="_ttda_test")
    else:
        print("Visualization for this dataset type is not implemented yet.")
        
    print(f"Done! Results saved to '{results_dir}' folder.") 