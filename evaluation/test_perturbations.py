from torchvision import transforms
from torchvision.transforms import v2
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import pickle
from dataloaders.simple_depth import DepthDataset as NYUDEPTH_dataset
from dataloaders.hypersim_depth import HyperSimDepthDataset
from dataloaders.apolloscape_depth import ApolloscapeDepthDataset

from models.base_ensemble_model import BaseEnsemble
from models.post_hoc_ensemble_model import PostHocEnsemble
from networks.unet_model import UNet, BabyUNet, MediumUNet
from predictors.gaussian import post_hoc_predict_gaussian, predict_gaussian, gaussian_nll
from predictors.bayescap import predict_bayescap
from predictors.mse import predict_mse, rmse
from evaluation.utils_ood import plot_ood_analysis
from evaluation.eval_depth_utils import load_model


def apply_gaussian_noise(batch, sigma):
    """Apply Gaussian noise to a batch of images"""
    noisy_batch = batch.clone()
    noise = torch.randn_like(noisy_batch) * sigma
    noisy_batch = noisy_batch + noise
    noisy_batch = torch.clamp(noisy_batch, 0, 1)
    return noisy_batch


if __name__ == "__main__":
    # Set parameters
    sigmas = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,]
    n_batches = 8
    
    # Setup dataloader
    id_dataset = NYUDEPTH_dataset(img_size=(128, 160), augment=False)
    _, id_test_loader = id_dataset.get_dataloaders(batch_size=128, shuffle=False)
    
    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load models
    base_model = load_model(
        BaseEnsemble, 
        model_path="results/pretrained/base_gaussian_ensemble.pth", 
        inference_fn=predict_gaussian, 
        model_params={"in_channels": 3, "out_channels": [1, 1], "drop_prob": 0.2}, 
        n_models=5, 
        device=device)
    
    edgy_model = load_model(
        BaseEnsemble, 
        model_path="results/base_unet_depth_model_very_augmented/checkpoints/base_ensemble_model_95.pth", 
        inference_fn=predict_mse, 
        model_params={"in_channels": 3, "out_channels": [1], "drop_prob": 0.2}, 
        n_models=5, 
        device=device)
    
    post_hoc_gaussian_model = load_model(
        PostHocEnsemble, 
        model_path="results/edgy_depth/checkpoints/post_hoc_ensemble_model_best.pt", 
        inference_fn=post_hoc_predict_gaussian, 
        model_params={"in_channels": 4, "out_channels": [1], "drop_prob": 0.3}, 
        n_models=1, 
        device=device,
        model_class=UNet)

    results = {
        "data": {"samples": {}, "clean_samples": None, "clean_targets": None},
        "base_model": {"nlls": [], "rmses": [], "noise_levels": [], "avg_vars": [], "preds": {}, "pred_stds": {}},
        "edgy_model": {"nlls": [], "rmses": [], "noise_levels": [], "avg_vars": [], "preds": {}, "pred_stds": {}},
        "post_hoc_model": {"nlls": [], "rmses": [], "noise_levels": [], "avg_vars": [], "pred_stds": {}},
        "post_hoc_model_clean": {"nlls": [], "rmses": [], "noise_levels": [], "avg_vars": [], "pred_stds": {}}
    }
    
    # Get a sample batch and test predict functions to see what keys are returned
    sample_batch = next(iter(id_test_loader))
    inputs, targets = sample_batch
    inputs = inputs.to(device)
    targets = targets.to(device)
    
    # Test predict functions
    with torch.no_grad():
        # Base model
        base_pred = base_model.predict(inputs)
        print("Base model predict keys:", list(base_pred.keys()))
        
        # Edgy model
        edgy_pred = edgy_model.predict(inputs)
        print("Edgy model predict keys:", list(edgy_pred.keys()))
        
        # Post-hoc model (concatenate input with mean prediction)
        edgy_mean = edgy_pred['mean']
        combined_input = torch.cat([inputs, edgy_mean], dim=1)
        post_hoc_pred = post_hoc_gaussian_model.predict(combined_input)
        print("Post-hoc model predict keys:", list(post_hoc_pred.keys()))
    
    # Iterate over noise levels
    for sigma in tqdm(sigmas, desc="Processing noise levels"):
        # Process specific number of batches
        for batch_idx, (inputs, targets) in enumerate(tqdm(id_test_loader, desc=f"Processing batches for sigma={sigma}")):
            if batch_idx >= n_batches:
                break
                
            inputs = inputs.to(device)
            targets = targets.to(device)

            if results["data"]["clean_samples"] is None:
                # Add clean inputs only once, targets stay clean 
                # so no need to track them continuously
                results["data"]["clean_samples"] = inputs.permute(0, 2, 3, 1).cpu().numpy()
                results["data"]["clean_targets"] = targets.permute(0, 2, 3, 1).cpu().numpy()
            
            # First get clean predictions from edgy model
            with torch.no_grad():
                clean_pred = edgy_model.predict(inputs)
                clean_mean = clean_pred['mean']
            
            # Apply noise to inputs
            noisy_inputs = apply_gaussian_noise(inputs, sigma)
            # Store noisy inputs once per sigma
            if sigma not in results["data"]["samples"]:
                results["data"]["samples"][sigma] = noisy_inputs.permute(0, 2, 3, 1).cpu().numpy()
            
            # Get predictions from all models with noisy inputs
            with torch.no_grad():
                # Base model prediction
                base_pred = base_model.predict(noisy_inputs)
                base_mean = base_pred['mean']
                
                # Get base model variance
                base_std = base_pred['al_sigma']
                edgy_std = edgy_pred['ep_sigma']
                
                # Edgy model prediction with noisy input
                edgy_pred = edgy_model.predict(noisy_inputs)
                edgy_mean = edgy_pred['mean']
                
                # Post-hoc model with noisy input and noisy edgy prediction
                noisy_combined = torch.cat([noisy_inputs, edgy_mean], dim=1)
                post_hoc_pred = post_hoc_gaussian_model.predict(X=noisy_inputs, y_pred=edgy_mean)
                post_hoc_mean = edgy_mean  # Post-hoc only predicts uncertainty, use edgy mean
                
                # Get post-hoc model variance
                post_hoc_std = post_hoc_pred['sigma']
                
                # Post-hoc model with noisy input but clean prediction
                clean_noisy_combined = torch.cat([noisy_inputs, clean_mean], dim=1)
                post_hoc_clean_pred = post_hoc_gaussian_model.predict(X=noisy_inputs, y_pred=clean_mean)
                post_hoc_clean_mean = clean_mean  # Post-hoc only predicts uncertainty, use clean mean
                
                # Get clean post-hoc model variance
                post_hoc_clean_std = post_hoc_clean_pred['sigma']
            
            # Compute metrics
            # Base model
            base_rmse_val = rmse(base_mean, targets).item()
            base_nll_val = gaussian_nll(
                torch.cat([base_mean, torch.log(base_std)], dim=1), 
                targets
            ).item()
            
            # Edgy model (doesn't output variance)
            edgy_rmse_val = rmse(edgy_mean, targets).item()
            edgy_nll_val = gaussian_nll(
                torch.cat([edgy_mean, torch.log(edgy_std)], dim=1), 
                targets
            ).item()

            # Post-hoc model with noisy prediction
            post_hoc_rmse_val = rmse(post_hoc_mean, targets).item()
            post_hoc_nll_val = gaussian_nll(
                torch.cat([post_hoc_mean, torch.log(post_hoc_std)], dim=1), 
                targets
            ).item()
            
            # Post-hoc model with clean prediction
            post_hoc_clean_rmse_val = rmse(post_hoc_clean_mean, targets).item()
            post_hoc_clean_nll_val = gaussian_nll(
                torch.cat([post_hoc_clean_mean, torch.log(torch.sqrt(post_hoc_clean_std))], dim=1), 
                targets
            ).item()
            
            # Store results
            results["base_model"]["rmses"].append(base_rmse_val)
            results["base_model"]["nlls"].append(base_nll_val)
            results["base_model"]["noise_levels"].append(sigma)
            results["base_model"]["avg_vars"].append(base_std.mean().item())
            # Store predictions once per sigma
            if sigma not in results["base_model"]["preds"]:
                results["base_model"]["preds"][sigma] = base_mean.permute(0, 2, 3, 1).cpu().numpy()
                results["base_model"]["pred_stds"][sigma] = base_std.permute(0, 2, 3, 1).cpu().numpy()

            results["edgy_model"]["rmses"].append(edgy_rmse_val)
            results["edgy_model"]["nlls"].append(edgy_nll_val)
            results["edgy_model"]["noise_levels"].append(sigma)
            results["edgy_model"]["avg_vars"].append(edgy_std.mean().item())
            # Store predictions once per sigma
            if sigma not in results["edgy_model"]["preds"]:
                results["edgy_model"]["preds"][sigma] = edgy_mean.permute(0, 2, 3, 1).cpu().numpy()
                results["edgy_model"]["pred_stds"][sigma] = edgy_std.permute(0, 2, 3, 1).cpu().numpy()

            results["post_hoc_model"]["rmses"].append(post_hoc_rmse_val)
            results["post_hoc_model"]["nlls"].append(post_hoc_nll_val)
            results["post_hoc_model"]["noise_levels"].append(sigma)
            results["post_hoc_model"]["avg_vars"].append(post_hoc_std.mean().item())
            # Store predictions once per sigma
            if sigma not in results["post_hoc_model"]["pred_stds"]:
                results["post_hoc_model"]["pred_stds"][sigma] = post_hoc_std.permute(0, 2, 3, 1).cpu().numpy()

            results["post_hoc_model_clean"]["rmses"].append(post_hoc_rmse_val)
            results["post_hoc_model_clean"]["nlls"].append(post_hoc_clean_nll_val)
            results["post_hoc_model_clean"]["noise_levels"].append(sigma)
            results["post_hoc_model_clean"]["avg_vars"].append(post_hoc_clean_std.mean().item())
            # Store predictions once per sigma
            if sigma not in results["post_hoc_model_clean"]["pred_stds"]:
                results["post_hoc_model_clean"]["pred_stds"][sigma] = post_hoc_clean_std.permute(0, 2, 3, 1).cpu().numpy()
    
    # Save results
    os.makedirs("results/input_perturbation", exist_ok=True)
    with open("results/input_perturbation/perturbation_results.pkl", "wb") as f:
        pickle.dump(results, f)
    
    print("Experiment completed. Results saved to results/input_perturbation/perturbation_results.pkl")