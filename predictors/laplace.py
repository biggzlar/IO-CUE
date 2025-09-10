import torch
import numpy as np
from .registry import register_predictor, register_criterion

@register_criterion("crit_laplace")
def laplace_nll(y_pred, y_true, reduce=True):
    """
    Negative log-likelihood for Laplace distribution.
    
    Args:
        y_pred: Predicted values [mean, log_scale] concatenated
        y_true: True values
        reduce: Whether to return mean loss or per-sample loss
        
    Returns:
        NLL loss
    """
    mean, log_scale = torch.split(y_pred, 1, dim=1)
    scale = torch.exp(log_scale)
    
    # Laplace NLL: log(2*scale) + |y - mean| / scale
    nll = torch.log(2 * scale) + torch.abs(y_true - mean) / scale
    return nll.mean() if reduce else nll

@register_criterion("crit_laplace_detached")
def laplace_nll_detached(y_pred, y_true, params, reduce=True, **kwargs):
    """
    Negative log-likelihood for Laplace distribution with detached predictions.
    
    Args:
        y_pred: Predicted mean values (detached)
        y_true: True values
        params: Log scale parameters
        reduce: Whether to return mean loss or per-sample loss
        
    Returns:
        NLL loss
    """
    log_scale = params
    scale = torch.exp(log_scale)
    
    # Laplace NLL: log(2*scale) + |y - mean| / scale
    nll = torch.log(2 * scale) + torch.abs(y_true - y_pred.detach()) / scale
    return nll.mean() if reduce else nll

@register_predictor("pred_laplace")
def predict_laplace(preds):
    """
    Predict Laplace distribution parameters from model output.
    
    Args:
        preds: Model output [mean, log_scale] concatenated
        
    Returns:
        Dictionary with mean, scale, and log_scale
    """
    mean, log_scale = torch.split(preds, 1, dim=1)
    scale = torch.exp(log_scale)
    return {"mean": mean, "sigma": scale, "log_sigma": log_scale, "params": log_scale}

@register_predictor("pred_laplace_posthoc")
def post_hoc_predict_laplace(preds):
    """
    Post-hoc prediction of Laplace scale parameter.
    
    Args:
        preds: Log scale parameters
        
    Returns:
        Dictionary with scale and log_scale
    """
    log_scale = preds
    scale = torch.exp(log_scale)
    return {'sigma': scale, 'log_sigma': log_scale, 'params': log_scale}
