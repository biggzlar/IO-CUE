import torch
import numpy as np

from predictors.bayescap import predict_bayescap, bayescap_loss
from models.post_hoc_ensemble_model import PostHocEnsemble

class BayesCap(PostHocEnsemble):
    def __init__(self, mean_ensemble, model_class, model_params, n_models=5, device=None, infer=predict_bayescap, loss=bayescap_loss):
        """
        Ensemble of variance prediction models for post-hoc uncertainty estimation
        
        Args:
            mean_ensemble: The mean ensemble model
            model_class: The model class to use for the ensemble
            model_params: Dictionary of parameters to pass to the model constructor
            n_models: Number of models in the ensemble
            device: Device to run the models on
            infer: Inference function for post-processing model outputs
            loss: Loss function for training
        """
        super(BayesCap, self).__init__(mean_ensemble, model_class, model_params, n_models, device)
        self.infer = infer
        self.loss  = loss

    
    def _predict(self, X=None, y_pred=None, idx=None):
        inputs = y_pred
        if idx is not None:
            return self.models[idx](inputs)
        else:
            return torch.stack([model(inputs) for model in self.models], dim=0)
        

    def predict(self, X=None, y_pred=None):
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
        if isinstance(y_pred, np.ndarray):
            y_pred = torch.FloatTensor(y_pred).to(self.device)
        
        # Set models to evaluation mode
        for model in self.models:
            model.eval()
        
        with torch.no_grad():
            params_preds = [self.infer(model(y_pred)) for model in self.models]
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
