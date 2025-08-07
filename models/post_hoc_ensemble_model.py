import os
import pickle
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from dataloaders.dataset_utils import create_bootstrapped_dataloaders
from evaluation.metrics import compute_ece, compute_euc, compute_crps
from models.model_utils import create_optimizer, create_scheduler, create_model_instances
from evaluation.eval_depth_utils import get_predictions, visualize_results

from predictors.gaussian import gaussian_nll
from predictors.bayescap import predict_bayescap

class PostHocEnsemble(nn.Module):
    def __init__(self, mean_ensemble, model_class, model_params, infer, n_models=5, device=None):
        """
        Ensemble of variance prediction models for post-hoc uncertainty estimation
        
        Args:
            model_class: The model class to use for the ensemble
            model_params: Dictionary of parameters to pass to the model constructor
            n_models: Number of models in the ensemble
            device: Device to run the models on
        """
        super(PostHocEnsemble, self).__init__()
        self.model_class = model_class
        self.model_params = model_params
        self.n_models = n_models
        self.models = create_model_instances(self.model_class, self.model_params, self.n_models)
        self.mean_ensemble = mean_ensemble
        self.device = device
        
        # Move models to device
        for model in self.models:
            model.to(self.device)

        self.infer = infer
        self.is_bayescap = self.infer == predict_bayescap

        self.train_log = {'nll': [], 'rmse': [], 'avg_var': [], 'ece': [], 'euc': [], 'crps': [], 'p_value': []}
        self.overfit_counter = 0


    def optimize(self, results_dir, model_dir, train_loader, test_loader=None, n_epochs=100,
              optimizer_type='Adam', optimizer_params=None, scheduler_type=None, 
              scheduler_params=None, pair_models=False, criterion=None, eval_freq=100, is_bayescap=False):
        """
        Train the post-hoc ensemble for uncertainty estimation
        
        Args:
            train_loader: DataLoader for training data
            mean_ensemble: Ensemble of models for mean prediction
            test_loader: DataLoader for testing/evaluation data (optional)
            n_epochs: Number of epochs to train for
            optimizer_type: Type of optimizer to use ('Adam', 'AdamW', etc.)
            optimizer_params: Parameters for the optimizer
            scheduler_type: Type of learning rate scheduler to use
            scheduler_params: Parameters for the scheduler
            pair_models: Whether to pair each variance model with a specific mean model (1:1)
            criterion: Loss function to use
            eval_freq: Frequency (in epochs) to evaluate on test set
        """
        # Create optimizers, schedulers and dataloaders for each model
        optimizers = []
        schedulers = []
        dataloaders = []
        paired_mean_models = []
        finished = [False] * len(self.models)

        self.min_nll = float('inf')
        
        # Create a bootstrapped dataloader for each model
        for i in range(len(self.models)):
            # Create optimizer using utility function
            optimizer = create_optimizer(
                optimizer_type, 
                self.models[i].parameters(), 
                optimizer_params
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
            
            # Pair with mean model
            mean_model = self.mean_ensemble.models[i % len(self.mean_ensemble.models)]
            mean_model.eval()
            paired_mean_models.append(mean_model)
        
        dataloaders = create_bootstrapped_dataloaders(train_loader, len(self.models))

        # Train all models concurrently
        dataloader_iters = [iter(dataloaders[i]) for i in range(len(self.models))]
        epoch_losses = [0.0 for _ in range(len(self.models))]
        
        pbar = tqdm(range(n_epochs), desc="UQ")
        for epoch in pbar:
            # Reset epoch metrics
            for i in range(len(self.models)):
                epoch_losses[i] = 0.0
                finished[i] = False
                self.models[i].train()
            
            # Train each model for one epoch
            while not all(finished):
                for i in range(len(self.models)):
                    if finished[i]:
                        continue
                    
                    # Get a batch of data
                    try:
                        batch_X, batch_y_true = next(dataloader_iters[i])
                    except StopIteration:
                        finished[i] = True
                        # If dataloader is exhausted, create a new iterator
                        dataloader_iters[i] = iter(dataloaders[i])
                        continue
                    
                    batch_X = batch_X.to(self.device)
                    batch_y_true = batch_y_true.to(self.device)
                    
                    # Generate mean predictions from the mean ensemble
                    with torch.no_grad():
                        if pair_models:
                            batch_y_pred, _, _ = paired_mean_models[i](batch_X)
                            if batch_y_pred.shape[1] > 1:
                                # If the mean ensemble has multiple output heads,
                                # treat the first one as the mean prediction.
                                batch_y_pred = batch_y_pred[:, :1, ...]
                        else:
                            batch_pred = self.mean_ensemble.predict(batch_X)
                            batch_y_pred = batch_pred['mean']
                    
                    optimizers[i].zero_grad()

                    # Forward pass for variance model
                    if self.is_bayescap:
                        params = self.models[i](batch_y_pred)
                        loss = criterion(y_true=batch_y_true, y_pred=batch_y_pred, params=params, epoch=epoch, n_epochs=n_epochs)
                    else:
                        if self.model_params['in_channels'] == (batch_X.shape[1] + 1):
                            params = self.models[i](torch.concat([batch_X, batch_y_pred], dim=1))
                        else:
                            params = self.models[i](batch_X)
                        
                        loss = criterion(y_true=batch_y_true, y_pred=batch_y_pred, params=params, epoch=epoch, n_epochs=n_epochs)
                    
                    # Backward pass
                    loss.backward()
                    optimizers[i].step()
                    
                    # Track losses
                    epoch_losses[i] += loss.item()

            with torch.no_grad():
                if self.is_bayescap:
                    batch_post_hoc_preds = self.predict(batch_y_pred)
                else:
                    batch_post_hoc_preds = self.predict(batch_X, y_pred=batch_y_pred)
                batch_sigma = batch_post_hoc_preds['sigma']
            
            # Update learning rate schedulers
            for i in range(len(schedulers)):
                if schedulers[i] is not None:
                    schedulers[i].step()
            
            # Calculate average loss for this epoch
            avg_losses = [epoch_losses[i] / len(dataloaders[i]) for i in range(len(self.models))]
            avg_loss = np.mean(avg_losses)
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{avg_loss:.4f}",
                              "sigma": f"{batch_sigma.mean().item():.2f}"})
            
            # Evaluate on test set if requested
            if test_loader is not None and (epoch + 1) % eval_freq == 0:
                results = self.evaluate(test_loader)
                
                print(f"\nEpoch {epoch+1}/{n_epochs} - RMSE: {results['metrics']['rmse']:.4f}, NLL: {results['metrics']['nll']:.4f}, ECE: {results['metrics']['ece']:.4f}, EUC: {results['metrics']['euc']:.4f}, CRPS: {results['metrics']['crps']:.4f}")
                self.update_log(results)
                self.pickle_log(f"{results_dir}/post_hoc_ensemble_model_log.pkl")
                if results['metrics']['nll'] < self.min_nll:
                    self.min_nll = results['metrics']['nll']
                    self.save(f"{model_dir}/post_hoc_ensemble_model_{epoch + 1}.pth")
                    self.save(f"{model_dir}/post_hoc_ensemble_model_best.pt")
                    self.overfit_counter = 0
                else:
                    self.overfit_counter += 1

                # We can only do this specific visualization if the dataset is simple_depth
                if len(batch_X.shape) == 4:
                    visualize_results(results, num_samples=5, metric_name="nll", path=f"{results_dir}", suffix=f"_{epoch}")
                
                if self.overfit_counter > 8:
                    self.load(f"{model_dir}/post_hoc_ensemble_model_best.pt")
                    break

                print()
                    
        pbar.close()
        

    def evaluate(self, test_loader):
        """
        Evaluate the ensemble on a test set
        
        Args:
            test_loader: DataLoader containing test data
            mean_ensemble: The ensemble used for mean predictions
            
        Returns:
            rmse: Root Mean Squared Error
            nll: Negative Log-Likelihood
        """
        all_base_means = []
        all_post_hoc_log_sigmas = []
        all_targets = []
        all_errors = []
        
        # Set base models to evaluation mode
        for model in self.mean_ensemble.models:
            model.eval()
        
        # Get mean predictions from mean ensemble
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_average_indices = tuple(range(1, batch_X.ndim))
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Mean predictions from mean ensemble
                batch_preds = self.mean_ensemble.predict(batch_X)
                base_mean = batch_preds['mean']

                all_base_means.append(base_mean)
                
                # Log variance predictions from variance ensemble
                if self.is_bayescap:
                    batch_post_hoc_preds = self.predict(base_mean)
                else:
                    if self.model_params['in_channels'] == (batch_X.shape[1] + 1):
                        batch_post_hoc_preds = self.predict(batch_X, y_pred=base_mean)
                    else:
                        batch_post_hoc_preds = self.predict(batch_X)

                post_hoc_log_sigma = batch_post_hoc_preds['mean_log_sigma']
                # Store predictions and targets
                all_post_hoc_log_sigmas.append(post_hoc_log_sigma)
                all_targets.append(batch_y)

                error_batch = torch.abs(batch_y - base_mean).square().mean(dim=batch_average_indices)
                all_errors.append(error_batch)
        
        # Combine predictions and targets
        all_base_means = torch.vstack(all_base_means)
        all_post_hoc_log_sigmas = torch.vstack(all_post_hoc_log_sigmas)
        all_targets = torch.vstack(all_targets)
        all_errors = torch.concat(all_errors)
        
        # Calculate RMSE
        mse = all_errors
        rmse = torch.sqrt(mse)

        # Calculate NLL using predicted variances
        nll = gaussian_nll(torch.cat([all_base_means, all_post_hoc_log_sigmas], dim=1), all_targets, reduce=False)
        residuals = torch.abs(all_base_means - all_targets)
        ece, empirical_confidence_levels = compute_ece(residuals=residuals, sigma=torch.exp(all_post_hoc_log_sigmas))
        euc, p_value = compute_euc(predictions=all_base_means, uncertainties=torch.exp(all_post_hoc_log_sigmas), targets=all_targets)
        crps = compute_crps(predictions=all_base_means, uncertainties=torch.exp(all_post_hoc_log_sigmas), targets=all_targets)

        # print(f"{self.is_bayescap}, NLL: {nll.mean().detach().item():.4f}, ECE: {ece:.4f}, EUC: {euc:.4f}")
        metrics = {
            'nll': nll.mean().detach().cpu(),
            'rmse': rmse.mean().detach().cpu(),
            'avg_var': torch.exp(post_hoc_log_sigma).mean().detach().cpu(),
            'empirical_confidence_levels': empirical_confidence_levels,
            'ece': ece,
            'euc': euc,
            'p_value': p_value,
            'crps': crps.mean().detach().cpu()
        }
        
        return {
            'images': batch_X[-5:, ...].detach().cpu(),
            'targets': batch_y[-5:, ...].detach().cpu(),
            'errors': error_batch[-5:, ...].detach().cpu(),
            'mu_batch': base_mean[-5:, ...].detach().cpu(),
            'sigma_batch': torch.exp(post_hoc_log_sigma[-5:, ...]).detach().cpu(),
            'nll_batch': nll[-5:, ...].mean(dim=batch_average_indices).detach().cpu(),
            'rmse_batch': rmse[-5:, ...].detach().cpu(),
            'avg_var_batch': torch.exp(post_hoc_log_sigma)[-5:, ...].mean(dim=batch_average_indices).detach().cpu(),
            
            'all_sigmas': torch.exp(all_post_hoc_log_sigmas),
            'metrics': metrics,
        }
    
    def predict(self, X, y_pred=None):
        """
        Generate variance predictions from all models in the ensemble
        
        Args:
            X: Input features
            y_pred: Ensemble predictions (mean) - not used with new architecture
            return_individual: Whether to return individual model predictions
        
        Returns:
            mean_variance: Mean of variance predictions
            std_variance: Standard deviation of variance predictions
            all_variances: (Optional) Individual model predictions if return_individual=True
        """
        # Convert input to tensor if it's a numpy array
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X).to(self.device)
        
        # Set models to evaluation mode
        for model in self.models:
            model.eval()
        
        # Get predictions from each model
        with torch.no_grad():
            if self.model_params['in_channels'] == X.shape[1] + 1:
                params_preds = [self.infer(model(torch.concat([X, y_pred], dim=1))) for model in self.models]
            else:
                params_preds = [self.infer(model(X)) for model in self.models]
            sigma_preds = [pred['sigma'] for pred in params_preds]
        
        # Stack predictions
        all_sigmas = torch.stack(sigma_preds, dim=0)

        # Compute mean and std of variances
        mean_sigma = torch.mean(all_sigmas, dim=0)
        mean_log_sigma = torch.log(mean_sigma)
        var_of_var = torch.var(all_sigmas.square(), dim=0) if self.n_models > 1 else torch.zeros_like(mean_log_sigma)
        
        return {'sigma': mean_sigma, 
                'mean_log_sigma': mean_log_sigma, 
                'var_of_var': var_of_var
        }
            
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
            'n_models': len(self.models),
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
        if state_dict['n_models'] != len(self.models):
            raise ValueError(f"Number of models in saved state ({state_dict['n_models']}) doesn't match number of models in ensemble ({len(self.models)})")
        
        # Load state for each model
        for i, model_state in enumerate(state_dict['model_states']):
            self.models[i].load_state_dict(model_state)
            self.models[i].to(self.device)

    def update_log(self, results, max_keep=16):
        for key in self.train_log.keys():
            self.train_log[key].append(results['metrics'][key])
            if len(self.train_log[key]) > max_keep:
                self.train_log[key].pop(0)

    def pickle_log(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.train_log, f)
