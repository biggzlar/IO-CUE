import os
import pickle
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from dataloaders.dataset_utils import create_bootstrapped_dataloaders
from evaluation.utils import compute_ece, compute_euc, compute_crps, compute_ause_rmse
from models.model_utils import create_optimizer, create_scheduler, create_model_instances
from evaluation.utils import get_predictions, visualize_results

from predictors.gaussian import gaussian_nll
from predictors.bayescap import bayescap_loss, predict_bayescap
from predictors.generalized_gaussian import gen_gaussian_nll

class PostHocEnsemble(nn.Module):
    def __init__(self, mean_ensemble, model_class, model_params, n_models=5, device=None):
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

        # self.infer = infer
        # self.is_bayescap = self.loss == bayescap_loss
        self.is_bayescap = False

        self.train_log = {'nll': [], 'rmse': [], 'avg_var': [], 'ece': [], 'euc': [], 'crps': [], 'p_value': []}
        self.overfit_counter = 0
        # Persistent training state for step-wise optimization
        self._optimizers = None
        self._schedulers = None
        self._paired_mean_models = None
        self._epoch = 0  # 0-based global epoch counter across calls
        self._total_epochs = None  # optional total epochs target across calls
        self.min_nll = float('inf')

    def optimize(self, results_dir, model_dir, train_loader, test_loader=None, n_epochs=100,
              optimizer_type='Adam', optimizer_params=None, scheduler_type=None, 
              scheduler_params=None, pair_models=False, eval_freq=100, one_epoch=False, load_best_at_end=True):
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
            one_epoch: If True, run exactly one epoch and return, keeping optimizer/scheduler state
            load_best_at_end: If True and not one_epoch, load the best checkpoint at the end if available
        """
        # Initialize or reuse persistent training components
        if self._optimizers is None or self._schedulers is None or self._paired_mean_models is None:
            optimizers, schedulers, paired_mean_models = self.initialize_training_components(
                optimizer_type, optimizer_params, scheduler_type, scheduler_params
            )
            self._optimizers = optimizers
            self._schedulers = schedulers
            self._paired_mean_models = paired_mean_models
        else:
            optimizers = self._optimizers
            schedulers = self._schedulers
            paired_mean_models = self._paired_mean_models

        # Remember intended total epochs if provided (first call wins)
        if self._total_epochs is None:
            self._total_epochs = n_epochs
        
        # Create a bootstrapped dataloader for each model
        dataloaders = create_bootstrapped_dataloaders(train_loader, len(self.models))
        
        finished = [False] * len(self.models)
        if self._epoch == 0:
            self.min_nll = float('inf')
        
        # Train all models concurrently
        dataloader_iters = [iter(dataloaders[i]) for i in range(len(self.models))]
        epoch_losses = [0.0 for _ in range(len(self.models))]
        
        epochs_to_run = 1 if one_epoch else n_epochs
        pbar = tqdm(range(epochs_to_run), desc="UQ")
        for _ in pbar:
            global_epoch_index = self._epoch  # 0-based global epoch index
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
                            batch_y_pred = paired_mean_models[i](batch_X)
                            if batch_y_pred.shape[1] > 1:
                                # If the mean ensemble has multiple output heads,
                                # treat the first one as the mean prediction.
                                batch_y_pred = batch_y_pred[:, :1, ...]
                        else:
                            batch_pred = self.mean_ensemble.predict(batch_X)
                            batch_y_pred = batch_pred['mean']
                    
                    optimizers[i].zero_grad()

                    # Forward pass for variance model
                    params = self._predict(X=batch_X, y_pred=batch_y_pred, idx=i)
                        
                    loss = self.loss(
                        y_true=batch_y_true,
                        y_pred=batch_y_pred,
                        params=params,
                        epoch=global_epoch_index,
                        n_epochs=self._total_epochs if self._total_epochs is not None else n_epochs,
                        reduce=True
                    )

                    # Backward pass
                    loss.backward()
                    optimizers[i].step()
                    
                    # Track losses
                    epoch_losses[i] += loss.item()

            with torch.no_grad():
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
            if test_loader is not None and global_epoch_index % eval_freq == 0:
                results = self.evaluate(test_loader)
                
                total_epochs_display = self._total_epochs if self._total_epochs is not None else n_epochs
                print(f"\nEpoch {global_epoch_index+1}/{total_epochs_display} - RMSE: {results['metrics']['rmse']:.3f}, NLL: {results['metrics']['nll']:.3f}, ECE: {results['metrics']['ece']:.3f}, EUC: {results['metrics']['euc']:.3f}, CRPS: {results['metrics']['crps']:.3f}, AUSE RMSE: {results['metrics']['ause_rmse']:.3f}")
                self.update_log(results)
                self.pickle_log(f"{results_dir}/post_hoc_ensemble_model_log.pkl")
                if results['metrics']['nll'] < self.min_nll:
                    self.min_nll = results['metrics']['nll']
                    self.save(f"{model_dir}/post_hoc_ensemble_model_best.pth")
                    self.overfit_counter = 0
                else:
                    self.overfit_counter += 1
                self.save(f"{model_dir}/post_hoc_ensemble_model_{global_epoch_index + 1}.pth")
                # We can only do this specific visualization if the dataset is simple_depth
                if len(batch_X.shape) == 4:
                    visualize_results(results, num_samples=5, metric_name="nll", path=f"{results_dir}", suffix=f"_{global_epoch_index}")
                
                # if self.overfit_counter > 8:
                #     self.load(f"{model_dir}/post_hoc_ensemble_model_best.pt")
                #     break

                print()
            
            # Advance global epoch counter at end of this epoch
            self._epoch += 1
        self.save(f"{model_dir}/post_hoc_ensemble_model_last.pth")
                    
        pbar.close()
        # if not one_epoch and load_best_at_end:
        #     best_path = f"{model_dir}/post_hoc_ensemble_model_best.pt"
        #     if os.path.exists(best_path):
        #         self.load(best_path)
        
    def reset_training_state(self):
        """
        Reset persistent optimizer/scheduler state and epoch counter.
        Call before starting a new independent training run.
        """
        self._optimizers = None
        self._schedulers = None
        self._paired_mean_models = None
        self._epoch = 0
        self._total_epochs = None
        self.overfit_counter = 0
        self.min_nll = float('inf')

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
        all_post_hoc_sigmas = []
        all_targets = []
        all_errors = []
        all_nlls = []
        
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
                batch_means = batch_preds['mean']

                all_base_means.append(batch_means)
                
                # Log variance predictions from variance ensemble
                batch_post_hoc_preds = self.predict(batch_X, y_pred=batch_means)

                batch_post_hoc_sigma = batch_post_hoc_preds['sigma']
                # Store predictions and targets
                all_post_hoc_sigmas.append(batch_post_hoc_sigma)
                all_targets.append(batch_y)

                # Compute NLL using framework's loss function if available
                batch_params = batch_post_hoc_preds['params']
                # BayesCap's loss is not a proper scoring function, so we use the generalized Gaussian NLL.
                if self.is_bayescap:
                    batch_nll = gen_gaussian_nll(y_pred=batch_means, y_true=batch_y, params=batch_post_hoc_preds['params'], reduce=False, epoch=1, n_epochs=1)
                else:
                    batch_nll = self.loss(y_true=batch_y, y_pred=batch_means, params=batch_params, reduce=False, epoch=1, n_epochs=1)
                    
                # batch_nll = self.loss(y_true=batch_y, y_pred=batch_means, params=batch_params, reduce=False, epoch=1, n_epochs=1)
                all_nlls.append(batch_nll)

                batch_errors = torch.abs(batch_y - batch_means).square().mean(dim=batch_average_indices)
                all_errors.append(batch_errors)
        
        # Combine predictions and targets
        all_base_means = torch.vstack(all_base_means)
        all_post_hoc_sigmas = torch.vstack(all_post_hoc_sigmas)
        all_targets = torch.vstack(all_targets)
        all_errors = torch.concat(all_errors)
        
        # Calculate global RMSE across all samples and elements
        rmse_global = torch.sqrt(torch.mean((all_base_means - all_targets) ** 2))

        # Use the NLLs computed in the loop
        nll = torch.cat(all_nlls, dim=0)
        residuals = torch.abs(all_base_means - all_targets)
        ece, empirical_confidence_levels = compute_ece(residuals=residuals, sigma=all_post_hoc_sigmas)
        euc, p_value = compute_euc(predictions=all_base_means, uncertainties=all_post_hoc_sigmas, targets=all_targets)
        crps = compute_crps(predictions=all_base_means, uncertainties=all_post_hoc_sigmas, targets=all_targets)
        ause_rmse, sparsification_error = compute_ause_rmse(predictions=all_base_means, uncertainties=all_post_hoc_sigmas, targets=all_targets)

        # print(f"{self.is_bayescap}, NLL: {nll.mean().detach().item():.4f}, ECE: {ece:.4f}, EUC: {euc:.4f}")
        metrics = {
            'nll': nll.mean().detach().cpu(),
            'rmse': rmse_global.detach().cpu(),
            'avg_var': all_post_hoc_sigmas.mean().detach().cpu(),
            'empirical_confidence_levels': empirical_confidence_levels,
            'ece': ece,
            'euc': euc,
            'p_value': p_value,
            'crps': crps.mean().detach().cpu(),
            'ause_rmse': ause_rmse,
            'sparsification_error': sparsification_error
        }
        
        return {
            'images': batch_X[-5:, ...].detach().cpu(),
            'targets': batch_y[-5:, ...].detach().cpu(),
            'errors': batch_errors[-5:, ...].detach().cpu(),
            'mu_batch': batch_means[-5:, ...].detach().cpu(),
            'sigma_batch': batch_post_hoc_sigma[-5:, ...].detach().cpu(),
            'nll_batch': nll[-5:, ...].mean(dim=batch_average_indices).detach().cpu(),
            'rmse_batch': torch.sqrt(batch_errors)[-5:, ...].detach().cpu(),
            'avg_var_batch': batch_post_hoc_sigma[-5:, ...].mean(dim=batch_average_indices).detach().cpu(),
            
            'all_sigmas': all_post_hoc_sigmas,
            'metrics': metrics,
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

    def initialize_training_components(self, optimizer_type, optimizer_params, scheduler_type, scheduler_params):
        """
        Initialize optimizers, schedulers, and paired mean models for training
        
        Args:
            optimizer_type: Type of optimizer to use
            optimizer_params: Parameters for the optimizer
            scheduler_type: Type of learning rate scheduler to use
            scheduler_params: Parameters for the scheduler
            
        Returns:
            tuple: (optimizers, schedulers, paired_mean_models)
        """
        optimizers = []
        schedulers = []
        paired_mean_models = []
        
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
            
        return optimizers, schedulers, paired_mean_models
