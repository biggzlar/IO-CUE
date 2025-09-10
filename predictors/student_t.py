import torch
import numpy as np
from .registry import register_predictor, register_criterion

@register_criterion("crit_student_t")
def student_t_nll(y_pred, y_true, reduce=True):
    """
    Negative log-likelihood for Student's t distribution.
    
    Args:
        y_pred: Predicted values [mean, log_scale, log_df] concatenated
        y_true: True values
        reduce: Whether to return mean loss or per-sample loss
        
    Returns:
        NLL loss
    """
    mean, log_scale, log_df = torch.split(y_pred, 1, dim=1)
    scale = torch.exp(log_scale)
    df = torch.exp(log_df) + 2.0  # Ensure df > 2 for finite variance
    
    # Student's t NLL
    # log p(y) = log Γ((ν+1)/2) - log Γ(ν/2) - 0.5*log(π*ν*σ²) - (ν+1)/2 * log(1 + (y-μ)²/(ν*σ²))
    residual = (y_true - mean) / scale
    t_term = (df + 1) / 2 * torch.log(1 + residual**2 / df)
    
    # Log-gamma terms
    log_gamma_df_plus_1 = torch.lgamma((df + 1) / 2)
    log_gamma_df = torch.lgamma(df / 2)
    
    nll = -log_gamma_df_plus_1 + log_gamma_df + 0.5 * torch.log(np.pi * df * scale**2) + t_term
    return nll.mean() if reduce else nll

@register_criterion("crit_student_t_detached")
def student_t_nll_detached(y_pred, y_true, params, reduce=True, **kwargs):
    """
    Negative log-likelihood for Student's t distribution with detached predictions.
    
    Args:
        y_pred: Predicted mean values (detached)
        y_true: True values
        params: [log_scale, log_df] parameters
        reduce: Whether to return mean loss or per-sample loss
        
    Returns:
        NLL loss
    """
    log_scale, log_df = torch.split(params, 1, dim=1)
    scale = torch.exp(log_scale)
    df = torch.exp(log_df) + 2.0  # Ensure df > 2 for finite variance
    
    # Student's t NLL
    residual = (y_true - y_pred.detach()) / scale
    t_term = (df + 1) / 2 * torch.log(1 + residual**2 / df)
    
    # Log-gamma terms
    log_gamma_df_plus_1 = torch.lgamma((df + 1) / 2)
    log_gamma_df = torch.lgamma(df / 2)
    
    nll = -log_gamma_df_plus_1 + log_gamma_df + 0.5 * torch.log(np.pi * df * scale**2) + t_term
    return nll.mean() if reduce else nll

@register_predictor("pred_student_t")
def predict_student_t(preds):
    """
    Predict Student's t distribution parameters from model output.
    
    Args:
        preds: Model output [mean, log_scale, log_df] concatenated
        
    Returns:
        Dictionary with mean, scale, df, and log parameters
    """
    mean, log_scale, log_df = torch.split(preds, 1, dim=1)
    scale = torch.exp(log_scale)
    df = torch.exp(log_df) + 2.0  # Ensure df > 2 for finite variance
    
    return {
        "mean": mean, 
        "sigma": scale, 
        "log_sigma": log_scale,
        "df": df,
        "log_df": log_df,
        "params": torch.cat([log_scale, log_df], dim=1)
    }

@register_predictor("pred_student_t_posthoc")
def post_hoc_predict_student_t(preds):
    """
    Post-hoc prediction of Student's t distribution parameters.
    
    Args:
        preds: [log_scale, log_df] parameters
        
    Returns:
        Dictionary with scale, df, and log parameters
    """
    log_scale, log_df = torch.split(preds, 1, dim=1)
    scale = torch.exp(log_scale)
    df = torch.exp(log_df) + 2.0  # Ensure df > 2 for finite variance
    
    return {
        'sigma': scale, 
        'log_sigma': log_scale,
        'df': df,
        'log_df': log_df,
        'params': torch.cat([log_scale, log_df], dim=1)
    }
