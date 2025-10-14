"""
Train a base ensemble model for depth estimation or other regression tasks.

This script trains a base ensemble model without using a YAML config or argparse.


Results will be saved to:
- results/BASE_MODEL_NAME/ - Main results directory
- results/BASE_MODEL_NAME/checkpoints/ - Model checkpoints
- results/BASE_MODEL_NAME/configuration.pkl - Saved settings for reproducibility
"""
import os
import torch
import numpy as np
import pickle
from datetime import datetime
from dataloaders.simple_depth import DepthDataset as NYUDepthDataset
from models import BaseEnsemble
from networks.unet_model import UNet
from predictors.gaussian import gaussian_nll, predict_gaussian
from predictors.mse import mse, predict_mse
from predictors.edge_aware_losses import edge_aware_mse_loss

# Set random seeds for reproducibility
np.random.seed(0)
torch.manual_seed(0)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Model configuration
AUGMENTS = {
    "flip": False,
    "colorjitter": False,
    "gaussianblur": False,
    "grayscale": False
}
BASE_MODEL_NAME = "base_unet"
N_MODELS = 5  # Number of models in the ensemble
N_EPOCHS = 64
EVAL_FREQ = 5

# Dataset configuration
IMG_SIZE = (128, 160)
TRAIN_SPLIT = 1.0
BATCH_SIZE = 32

# Model parameters
MODEL_CLASS = UNet
MODEL_PARAMS = {
    "in_channels": 3,
    "out_channels": [1],
    "drop_prob": 0.2
}

# Optimizer configuration
OPTIMIZER_TYPE = "Adam"
OPTIMIZER_PARAMS = {
    "lr": 5.0e-5,
    "weight_decay": 1.0e-5
}

# Scheduler configuration
SCHEDULER_TYPE = "CosineAnnealingLR"
SCHEDULER_PARAMS = {
    "T_max": N_EPOCHS
}

# Loss function
PREDICTOR = predict_mse
CRITERION = edge_aware_mse_loss

def main():
    # Create result directories
    results_dir = os.path.join("results", BASE_MODEL_NAME)
    model_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Save configuration for reproducibility
    config = locals().copy()
    with open(os.path.join(results_dir, "configuration.pkl"), "wb") as f:
        pickle.dump(config, f)
    
    # Load dataset
    dataset = NYUDepthDataset(img_size=IMG_SIZE, augment=True, **AUGMENTS)
    train_loader, test_loader = dataset.get_dataloaders(batch_size=BATCH_SIZE)
    print(f"Dataset loaded: {len(train_loader)} training batches, {len(test_loader)} test batches")
    
    # Create model
    print(f"Creating base ensemble with {N_MODELS} models...")
    base_ensemble = BaseEnsemble(
        model_class=MODEL_CLASS,
        model_params=MODEL_PARAMS,
        n_models=N_MODELS,
        device=DEVICE,
        infer=PREDICTOR
    )
    
    # Train model
    print(f"Training base ensemble for {N_EPOCHS} epochs...")
    base_ensemble.optimize(
        results_dir=results_dir,
        model_dir=model_dir,
        train_loader=train_loader,
        test_loader=test_loader,
        n_epochs=N_EPOCHS,
        optimizer_type=OPTIMIZER_TYPE,
        optimizer_params=OPTIMIZER_PARAMS,
        scheduler_type=SCHEDULER_TYPE,
        scheduler_params=SCHEDULER_PARAMS,
        criterion=CRITERION,
        eval_freq=EVAL_FREQ
    )
    
    # Final evaluation
    print("\nFinal evaluation on test set:")
    results = base_ensemble.evaluate(test_loader)
    print(f"Test RMSE: {results['rmse']:.4f}, Test NLL: {results['nll']:.4f}")
    
    # Save trained model
    model_path = os.path.join(model_dir, "base_ensemble_final.pth")
    base_ensemble.save(model_path)
    print(f"Saved trained model to {model_path}")
    
    return base_ensemble, train_loader, test_loader

if __name__ == "__main__":
    start_time = datetime.now()
    print(f"Started training at {start_time}")
    
    base_ensemble, train_loader, test_loader = main()
    
    end_time = datetime.now()
    training_time = end_time - start_time
    print(f"Training completed in {training_time}")
    print(f"Results saved to results/{BASE_MODEL_NAME}") 