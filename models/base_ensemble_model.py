import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from dataloaders.dataset_utils import create_bootstrapped_dataloaders
from tqdm import tqdm
from evaluation.utils import compute_ece, compute_euc, compute_crps
from models.model_utils import create_optimizer, create_scheduler, create_model_instances

from predictors.gaussian import gaussian_nll

class BaseEnsemble(nn.Module):
    def __init__(self, model_class, model_params, infer, n_models=5, device=None):
        """
        Initialize an ensemble of models using a model class and parameters
        
        Args:
            model_class: The model class to use for the ensemble
            model_params: Dictionary of parameters to pass to the model constructor
            n_models: Number of models in the ensemble
            device: Device to run the models on
        """
        super(BaseEnsemble, self).__init__()
        self.n_models = n_models
        self.model_params = model_params
        self.return_activations = False
        self.models = create_model_instances(model_class, self.model_params, n_models, return_activations=self.return_activations)
        self.device = device
        
        # Move models to device
        for model in self.models:
            model.to(self.device)

        self.infer = infer
        self.min_rmse = float('inf')
        self.overfit_counter = 0

    def optimize(self, results_dir, model_dir, train_loader, test_loader=None, n_epochs=100, 
              optimizer_type='Adam', optimizer_params=None,
              scheduler_type=None, scheduler_params=None, 
              criterion=None, eval_freq=100):
        """
        Train the ensemble models on the given data
        
        Args:
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
        # Create optimizers, schedulers and dataloaders for each model
        optimizers = []
        schedulers = []
        dataloaders = []
        finished = [False] * self.n_models

        self.min_nll = float('inf')
        
        # Create a bootstrapped dataloader for each model
        for i in range(self.n_models):
            # Add a small randomization to learning rate for better ensemble diversity
            current_optimizer_params = optimizer_params.copy()
            if 'lr' in current_optimizer_params:
                current_optimizer_params['lr'] += np.random.rand() * (current_optimizer_params['lr'] / 8.)
            
            # Create optimizer using utility function
            optimizer = create_optimizer(
                optimizer_type, 
                self.models[i].parameters(), 
                current_optimizer_params
            )
            optimizers.append(optimizer)
            
            # Create scheduler if specified using utility function
            if scheduler_type is not None:
                if scheduler_params is None:
                    raise ValueError("scheduler_params must be provided when scheduler_type is not None")
                scheduler = create_scheduler(scheduler_type, optimizer, scheduler_params)
                schedulers.append(scheduler)
            else:
                schedulers.append(None)
            
        dataloaders = create_bootstrapped_dataloaders(train_loader, self.n_models)
        
        # Train all models concurrently
        dataloader_iters = [iter(dataloaders[i]) for i in range(self.n_models)]
        epoch_losses = [0.0 for _ in range(self.n_models)]

        pbar = tqdm(range(n_epochs), desc="Base")
        for epoch in pbar:
            # Reset epoch metrics
            for i in range(self.n_models):
                epoch_losses[i] = 0.0
                finished[i] = False
                self.models[i].train()

            # Train each model for one epoch
            while not all(finished):
                for i in range(self.n_models):
                    if finished[i]:
                        continue
                    
                    # Get a batch of data
                    try:
                        batch_X, batch_y = next(dataloader_iters[i])
                    except StopIteration:
                        finished[i] = True
                        # If dataloader is exhausted, create a new iterator
                        dataloader_iters[i] = iter(dataloaders[i])
                        continue
                    
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    optimizers[i].zero_grad()

                    # Forward pass
                    y_pred = self.models[i](batch_X)
                    loss = criterion(y_pred, batch_y)
                    
                    # Backward pass
                    loss.backward()
                    optimizers[i].step()
                    
                    # Track losses
                    epoch_losses[i] += loss.item()
            
            # Update learning rate schedulers
            for i in range(len(schedulers)):
                if schedulers[i] is not None:
                    schedulers[i].step()
            
            # Calculate average loss for this epoch
            avg_losses = [epoch_losses[i] / len(dataloaders[i]) for i in range(self.n_models)]
            avg_loss = np.mean(avg_losses)
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
            
            # Evaluate on test set if requested
            if test_loader is not None and epoch % eval_freq == 0:
                results = self.evaluate(test_loader)
                print(f"\nEpoch {epoch+1}/{n_epochs} - Test RMSE: {results['rmse']:.4f}, Test NLL: {results['nll']:.4f}, ECE: {results['ece']:.4f}, EUC: {results['euc']:.4f}, CRPS: {results['crps']:.4f}")

                if len(batch_X.shape) == 4:
                    n_samples = 10
                    _, axs = plt.subplots(n_samples, 3, figsize=(20, 5 * n_samples))
                    batch_X, _ = next(iter(test_loader))
                    for i in range(n_samples):
                        axs[i, 0].imshow(results['all_inputs'][i].permute(1, 2, 0).cpu().numpy())
                        axs[i, 0].axis("off")
                        axs[i, 1].imshow(results['all_targets'][i].permute(1, 2, 0).cpu().numpy())
                        axs[i, 1].axis("off")
                        axs[i, 2].imshow(results['all_means'][i].permute(1, 2, 0).cpu().numpy())
                        axs[i, 2].axis("off")

                    plt.tight_layout()
                    plt.savefig(f"{results_dir}/base_ensemble_model_{epoch + 1}.png")
                    plt.close()

                if results['rmse'] < self.min_rmse:
                    self.min_rmse = results['rmse']
                    self.save(f"{model_dir}/base_ensemble_best.pth")
                    self.overfit_counter = 0
                else:
                    self.overfit_counter += 1

                # if self.overfit_counter > 8:
                #     print(f"Overfitting detected at epoch {epoch + 1}, loading best model")
                #     break
        
        pbar.close()
        self.load(f"{model_dir}/base_ensemble_best.pth")
    
    def evaluate(self, test_loader):
        """
        Evaluate the ensemble on a test set
        
        Args:
            test_loader: DataLoader containing test data
            
        Returns:
            rmse: Root Mean Squared Error
            nll: Negative Log-Likelihood (if applicable)
        """
        all_means = []
        all_al_sigmas = []
        all_ep_sigmas = []
        all_inputs = []
        all_targets = []
        
        # Collect predictions from all ensemble models
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                batch_preds = self.predict(batch_X)
                
                # Mean prediction across ensemble
                mean_mean = batch_preds['mean']
                al_sigma = batch_preds['al_sigma']
                ep_sigma = batch_preds['ep_sigma']

                # Store predictions and targets
                all_means.append(mean_mean)
                all_al_sigmas.append(al_sigma)
                all_ep_sigmas.append(ep_sigma)
                all_inputs.append(batch_X)
                all_targets.append(batch_y)
                
        # Combine predictions and targets
        all_means = torch.vstack(all_means)
        all_al_sigmas = torch.vstack(all_al_sigmas)
        all_ep_sigmas = torch.vstack(all_ep_sigmas)
        all_inputs = torch.vstack(all_inputs)
        all_targets = torch.vstack(all_targets)
        
        all_mean_total_sigmas = torch.log(all_al_sigmas + all_ep_sigmas + 1e-8)
        
        # Calculate RMSE
        mse = torch.mean((all_means - all_targets) ** 2)
        rmse = torch.sqrt(mse).item()

        nll = gaussian_nll(y_pred=torch.cat([all_means, all_mean_total_sigmas], dim=1), y_true=all_targets, reduce=True)

        residuals = torch.abs(all_means - all_targets)
        ece, empirical_confidence_levels = compute_ece(residuals=residuals, sigma=torch.exp(all_mean_total_sigmas))
        euc, p_value = compute_euc(predictions=all_means, uncertainties=torch.exp(all_mean_total_sigmas), targets=all_targets)
        crps = compute_crps(predictions=all_means, uncertainties=torch.exp(all_mean_total_sigmas), targets=all_targets)

        results = { 
            'rmse': rmse,
            'nll': nll.item(),
            'ece': ece.item(),
            'euc': euc,
            'crps': crps.mean().detach().cpu(),
            'all_means': all_means,
            'all_targets': all_targets,
            'all_inputs': all_inputs
        }
        return results
    
    def predict(self, X, return_individual=False):
        """
        Generate predictions from all models in the ensemble
        
        Args:
            X: Input data
            return_individual: Whether to return individual model predictions
            
        Returns:
            mean_prediction: Mean prediction across models
            std_prediction: Standard deviation of predictions
            all_predictions: (Optional) Individual model predictions if return_individual=True
        """
        # Convert input to tensor if it's a numpy array
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X).to(self.device)
        
        # Set models to evaluation mode
        for model in self.models:
            model.eval()
        
        # Get predictions from each model
        with torch.no_grad():
            mean_preds, sigma_preds, encoder_activations, decoder_activations = [], [], [], []
            for model in self.models:
                if self.return_activations:
                    pred, enc, dec = model(X)
                    encoder_activations.append(enc)
                    decoder_activations.append(dec)
                else:
                    pred = model(X)
                pred = self.infer(pred)
                mean_preds.append(pred['mean'])
                sigma_preds.append(pred['sigma'])
        
        # Stack predictions
        all_predictions = torch.stack(mean_preds, axis=0)
        
        # Compute mean prediction and uncertainty (std)
        mean_prediction = torch.mean(all_predictions, axis=0)
        ep_sigma = torch.zeros_like(mean_prediction) + 1e-8 if self.n_models == 1 else torch.std(all_predictions, axis=0)

        # Stack predictions
        all_sigmas = torch.stack(sigma_preds, axis=0)

        al_sigma = torch.mean(all_sigmas, axis=0)
        
        return {'mean': mean_prediction, 
                'ep_sigma': ep_sigma,
                'al_sigma': al_sigma,
                'all_means': all_predictions,
                'encoder_activations': encoder_activations,
                'decoder_activations': decoder_activations}
            
    def save(self, path):
        """
        Save the ensemble models to disk
        
        Args:
            path: Path to save the models to
        """
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save all models in the ensemble
        state_dict = {
            'n_models': self.n_models,
            'model_states': [model.state_dict() for model in self.models]
        }
        torch.save(state_dict, path)
        
    def load(self, path):
        """
        Load the ensemble models from disk
        
        Args:
            path: Path to load the models from
        """
        # Load the state dictionary
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        
        # Check if the number of models matches
        if state_dict['n_models'] != self.n_models:
            raise ValueError(f"Number of models in saved state ({state_dict['n_models']}) doesn't match number of models in ensemble ({self.n_models})")
        
        # Load state for each model
        for i, model_state in enumerate(state_dict['model_states']):
            self.models[i].load_state_dict(model_state)
            self.models[i].to(self.device) 

