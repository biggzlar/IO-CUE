import torch
import numpy as np
from .registry import register_predictor, register_criterion

@register_criterion("crit_gaussian")
def gaussian_nll(y_pred, y_true, reduce=True):
    mean, log_sigma = torch.split(y_pred, 1, dim=1)
    residual = (y_true - mean)**2
    nll = 0.5 * torch.exp(-2 * log_sigma) * residual + log_sigma
    nll = nll + 0.5 * torch.log(torch.tensor(2 * np.pi))
    return nll.mean() if reduce else nll

@register_criterion("crit_gaussian_detached")
def gaussian_nll_detached(y_pred, y_true, params, reduce=True, **kwargs):
    log_sigma = params
    residual = (y_true - y_pred.detach()) ** 2
    nll = 0.5 * torch.exp(-2 * log_sigma) * residual + log_sigma
    nll = nll + 0.5 * torch.log(torch.tensor(2 * np.pi))
    return nll.mean() if reduce else nll

@register_predictor("pred_gaussian")
def predict_gaussian(preds):
    mu, log_sigma = torch.split(preds, 1, dim=1)
    sigma = torch.exp(log_sigma)
    return {"mean": mu, "sigma": sigma, "log_sigma": log_sigma, "params": log_sigma}

@register_predictor("pred_gaussian_posthoc")
def post_hoc_predict_gaussian(preds):
    log_sigma = preds
    sigma = torch.exp(log_sigma)
    return {'sigma': sigma, 'log_sigma': log_sigma, 'params': log_sigma}
