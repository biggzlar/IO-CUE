import torch
import torch.nn as nn
import numpy as np
import math
from tqdm import tqdm
from models.model_utils import create_optimizer, create_scheduler
from predictors.gaussian import gaussian_nll
from predictors.mse import mse
import torchvision.transforms.functional as TF
import random
from torchvision import transforms 

class TTDAModel(nn.Module):
    def __init__(self, model_class, model_params, infer, device=None, num_augmentations=8):
        """
        Initialize a single model with Test-Time Data Augmentation (TTDA)
        
        Args:
            model_class: The model class to use (e.g., UNet)
            model_params: Dictionary of parameters to pass to the model constructor
            infer: Inference function to process raw model outputs
            device: Device to run the model on
            num_augmentations: Number of augmentations to apply during test time
        """
        super(TTDAModel, self).__init__()
        self.model_params = model_params
        self.model = model_class(**model_params)
        self.device = device
        
        # Move model to device
        self.model.to(self.device)
        self.infer = infer
        
        # Define rotation angles
        self.rotation_angles = [5, 10, 15, 20, 25, 30, 35, 40]
        
        # Number of augmentations to apply during test time
        self.augmentations = [
            lambda x: x,  # Original
            transforms.RandomHorizontalFlip(p=1.),
            # transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=0.0),
            transforms.ColorJitter(brightness=(1., 2.)),
            transforms.ColorJitter(contrast=(1.5, 3.)),
            transforms.ColorJitter(saturation=(1., 2.)),
            transforms.ColorJitter(hue=0.5),
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.5, 2.0)),
        ]
        
        # Add rotation augmentations
        for angle in self.rotation_angles:
            self.augmentations.append(lambda x, angle=angle: TF.rotate(x, angle))

        self.num_augmentations = len(self.augmentations)
        
        # Minimum NLL for model saving
        self.min_nll = float('inf')

    def augment(self, X, n_augmentations=None):
        """
        Apply different augmentations to the input
        
        Args:
            X: Input tensor [B, C, H, W]
            n_augmentations: Number of augmentations to apply (default: self.num_augmentations)
            
        Returns:
            aug_X: Tensor with shape [B*n_augmentations, C, H, W]
            indices: Original batch indices for each augmented sample
            aug_params: Dictionary containing parameters for each augmentation
        """
        if n_augmentations is None:
            n_augmentations = self.num_augmentations
            
        batch_size = X.shape[0]

        # Copy input X n_augmentations times
        X_aug = X.repeat(self.num_augmentations, 1, 1, 1, 1)
        for i in range(self.num_augmentations):
            X_aug[i] = self.augmentations[i](X_aug[i])

        return X_aug, torch.arange(batch_size), None
        # import matplotlib.pyplot as plt
        # img = X_aug[1].flip(dims=(-1,)).detach().cpu().squeeze().numpy().transpose(1, 2, 0)
        # plt.imsave(img, "poop.png")
        # import ipdb; ipdb.set_trace()


    def optimize(self, results_dir, model_dir, train_loader, test_loader=None, n_epochs=100, 
              optimizer_type='Adam', optimizer_params=None,
              scheduler_type=None, scheduler_params=None, 
              criterion=None, eval_freq=10):
        """
        Train the model on the given data
        
        Args:
            results_dir: Directory to save results
            model_dir: Directory to save model checkpoints
            train_loader: DataLoader for training data
            test_loader: DataLoader for testing/evaluation data (optional)
            n_epochs: Number of epochs to train for
            optimizer_type: Type of optimizer to use ('Adam', 'SGD', etc.)
            optimizer_params: Parameters for the optimizer
            scheduler_type: Type of learning rate scheduler to use
            scheduler_params: Parameters for the scheduler
            criterion: Loss function to use
            eval_freq: Frequency (in epochs) to evaluate on test set
        """
        # Create optimizer
        optimizer = create_optimizer(
            optimizer_type, 
            self.model.parameters(), 
            optimizer_params
        )
        
        # Create scheduler if specified
        if scheduler_type is not None:
            if scheduler_params is None:
                raise ValueError("scheduler_params must be provided when scheduler_type is not None")
            scheduler = create_scheduler(scheduler_type, optimizer, scheduler_params)
        else:
            scheduler = None
        
        pbar = tqdm(range(n_epochs), desc="TTDA")
        for epoch in pbar:
            # Training phase
            self.model.train()
            epoch_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                y_pred = self.model(batch_X)
                loss = criterion(y_pred, batch_y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Track loss
                epoch_loss += loss.item()
            
            # Update learning rate scheduler
            if scheduler is not None:
                scheduler.step()
            
            # Calculate average loss for this epoch
            avg_loss = epoch_loss / len(train_loader)
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
            
            # Evaluate on test set if requested
            if test_loader is not None and (epoch + 1) % eval_freq == 0:
                test_rmse, test_nll = self.evaluate(test_loader)
                print(f"\nEpoch {epoch+1}/{n_epochs} - Test RMSE: {test_rmse:.4f}, Test NLL: {test_nll:.4f}")
                
                if test_nll < self.min_nll:
                    self.min_nll = test_nll
                    self.save(f"{model_dir}/ttda_model_{epoch + 1}.pth")
        
        pbar.close()
    
    def evaluate(self, test_loader):
        """
        Evaluate the model on a test set
        
        Args:
            test_loader: DataLoader containing test data
            
        Returns:
            rmse: Root Mean Squared Error
            nll: Negative Log-Likelihood
        """
        all_means = []
        all_sigmas = []
        all_targets = []
        
        # Collect predictions with TTDA
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                batch_preds = self.predict(batch_X)
                
                # Get predictions
                mean_pred = batch_preds['mean']
                sigma_pred = batch_preds['mean_std']
                
                # Store predictions and targets
                all_means.append(mean_pred)
                all_sigmas.append(sigma_pred)
                all_targets.append(batch_y)
        
        # Combine predictions and targets
        all_means = torch.vstack(all_means)
        all_sigmas = torch.vstack(all_sigmas)
        all_targets = torch.vstack(all_targets)
        
        # Calculate RMSE
        mse_val = torch.mean((all_means - all_targets) ** 2)
        rmse = torch.sqrt(mse_val).item()
        
        # Calculate NLL
        total_sigma = all_sigmas
        nll = gaussian_nll(torch.cat([all_means, total_sigma], dim=1), all_targets, reduce=True)
        results = {
            'rmse': rmse,
            'nll': nll
        }
        return results
    
    def predict(self, X, return_individual=False, debug=False):
        """
        Generate predictions using TTDA
        
        Args:
            X: Input data
            return_individual: Whether to return individual augmentation predictions
            debug: If True, return additional debug information for visualization
            
        Returns:
            Dictionary containing mean, standard deviation, and other statistics
        """
        # Convert input to tensor if it's a numpy array
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Apply augmentations
        with torch.no_grad():
            aug_X, indices, aug_params = self.augment(X)
            
            # Reorganize predictions by original sample
            all_means = []
            all_sigmas = []
            all_log_sigmas = []

            # Get predictions for all augmented samples
            for i in range(self.num_augmentations):
                raw_preds = self.model(aug_X[i])
                preds = self.infer(raw_preds)
            
                if i == 1:
                    # Horizontal flip
                    mean_preds = preds['mean'].flip(dims=(-1,))
                    sigma_preds = preds['sigma'].flip(dims=(-1,))
                    log_sigma_preds = preds['log_sigma'].flip(dims=(-1,))
                elif i >= 8 and i < 8 + len(self.rotation_angles):
                    # Rotation - need to rotate back in the opposite direction
                    rotation_idx = i - 8
                    angle = -self.rotation_angles[rotation_idx]  # Negative angle to rotate back
                    mean_preds = TF.rotate(preds['mean'], angle)
                    sigma_preds = TF.rotate(preds['sigma'], angle)
                    log_sigma_preds = TF.rotate(preds['log_sigma'], angle)
                else:
                    mean_preds = preds['mean']
                    sigma_preds = preds['sigma']
                    log_sigma_preds = preds['log_sigma']

                all_means.append(mean_preds)
                all_sigmas.append(sigma_preds)
                all_log_sigmas.append(log_sigma_preds)
            
            # for i in range(len(all_means)):
            #     print((all_means[0] - all_means[i]).mean(dim=(1, 2, 3)).square())
            # import ipdb; ipdb.set_trace()
            # Calculate mean and variance across augmentations for each sample
            all_means = torch.stack(all_means)
            all_sigmas = torch.stack(all_sigmas)
            all_log_sigmas = torch.stack(all_log_sigmas)

            mean_prediction = all_means.mean(dim=0)
            ep_sigma = all_means.std(dim=0)
            std_prediction = all_sigmas.std(dim=0)
            mean_sigma = all_sigmas.mean(dim=0)
            mean_log_sigma = all_log_sigmas.mean(dim=0)
            std_log_sigma = all_log_sigmas.std(dim=0)
            
            result = {
                'mean': mean_prediction,
                'ep_sigma': ep_sigma,
                'mean_std': mean_sigma,
                'log_sigma_mean': mean_log_sigma,
                'log_sigma_std': std_log_sigma,
                'all_means': all_means,
                'all_sigmas': None,  # Placeholder to match base_ensemble interface
                'all_log_sigmas': None  # Placeholder to match base_ensemble interface
            }
            
            if return_individual:
                result['all_means'] = all_means
            
            if debug:
                # Add additional debug information for visualization
                result['augmented_inputs'] = aug_X
                result['aligned_predictions'] = mean_preds
                result['indices'] = indices
                result['aug_params'] = aug_params
            
            return result
    
    def save(self, path):
        """
        Save the model to disk
        
        Args:
            path: Path to save the model to
        """
        # Create the directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model state
        state_dict = {
            'model_state': self.model.state_dict(),
            'model_params': self.model_params
        }
        torch.save(state_dict, path)
        
    def load(self, path):
        """
        Load the model from disk
        
        Args:
            path: Path to load the model from
        """
        # Load the state dictionary
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        
        # Load state for the model
        self.model.load_state_dict(state_dict['model_state'])
        self.model.to(self.device) 