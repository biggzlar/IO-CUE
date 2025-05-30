import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path to access modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predictors.gaussian import gaussian_nll_detached, post_hoc_predict_gaussian, gaussian_nll, predict_gaussian
from predictors.generalized_gaussian import gen_gaussian_nll, predict_gen_gaussian
from predictors.bayescap import bayescap_loss, predict_bayescap, predict_bayescap
from predictors.mse import mse, predict_mse

class PredictorWrapper:
    """A wrapper class for different predictors"""
    def __init__(self, predictor_type="gaussian"):
        """
        Initialize a predictor wrapper
        
        Args:
            predictor_type: Type of predictor to use
                - "base": Base ensemble predictor
                - "gaussian": Gaussian post-hoc predictor
                - "gen_gaussian": Generalized Gaussian predictor
                - "bayescap": BayesCap predictor
        """
        self.predictor_type = predictor_type
        
        # Map predictor type to predictor function
        self.predictor_map = {
            "gaussian": post_hoc_predict_gaussian,
            "gen_gaussian": predict_gen_gaussian,
            "bayescap": predict_bayescap,
        }
        
        # Map for predictors that work directly with outputs
        self.output_predictor_map = {
            "mse": predict_mse,
            "gaussian": post_hoc_predict_gaussian,
            "post_hoc_gaussian": post_hoc_predict_gaussian,
            "bayescap": predict_bayescap,
        }
        
        if predictor_type not in self.predictor_map:
            raise ValueError(f"Unknown predictor type: {predictor_type}")
        
        self.predictor = self.predictor_map[predictor_type]
    
    def predict(self, preds):
        """Apply the predictor function to the model outputs"""
        return self.predictor(preds)
        
    def predict_from_outputs(self, outputs):
        """Apply the appropriate predictor to model outputs directly"""
        if self.predictor_type not in self.output_predictor_map:
            raise ValueError(f"Direct output prediction not supported for: {self.predictor_type}")
        
        return self.output_predictor_map[self.predictor_type](outputs)


class LossWrapper:
    """A wrapper class for different loss functions"""
    def __init__(self, loss_type="mse", post_hoc=False):
        """ Initialize a loss wrapper
        """
        self.loss_type = loss_type
        self.post_hoc = post_hoc
        
        # Map loss type to loss function and output dimension
        if self.post_hoc:
            self.loss_map = {
                "mse": (mse, 1),
                "gaussian": (gaussian_nll_detached, 1),
                "gen_gaussian": (gen_gaussian_nll, 2),
                "bayescap": (bayescap_loss, 3),
            }
        else:
            self.loss_map = {
                "mse": (mse, 1),
                "gaussian": (gaussian_nll, 2),
            }
        
        if loss_type not in self.loss_map:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        self.loss_fn, self.output_dims = self.loss_map[loss_type]
    
    def calculate_loss(self, y_true, y_pred, params=None, **kwargs):
        """Calculate loss between true values and predictions"""
        if self.post_hoc:
            return self.loss_fn(y_true=y_true, y_pred=y_pred, params=params, **kwargs)
        else:
            return self.loss_fn(y_true=y_true, y_pred=y_pred)


# Direct access to common loss functions and predictors
def get_loss_function(loss_type="mse"):
    """Get a loss function by name"""
    return LossWrapper(loss_type).loss_fn

def get_predictor(predictor_type="gaussian"):
    """Get a predictor function by name"""
    return PredictorWrapper(predictor_type).predictor 