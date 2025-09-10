import torch
import torch.nn.functional as F
from .registry import register_predictor, register_criterion

# Adapter function to work directly with outputs
@register_predictor("pred_bayescap")
def predict_bayescap(params):
    # Extract parameters from outputs
    mu_tilde, one_over_alpha, beta = torch.split(params, 1, dim=1)
    one_over_alpha = F.softplus(one_over_alpha) + 1e-8
    beta = F.softplus(beta) + 1e-8
    
    # Calculate alpha
    alpha = torch.pow(one_over_alpha, -1)
    
    # Calculate sigma
    variance = torch.square(alpha) * torch.exp(torch.lgamma(3 / beta) - torch.lgamma(1 / beta))
    sigma = variance.sqrt()
    log_sigma = torch.log(sigma)

    return {"mu_tilde": mu_tilde, 
            "sigma": sigma, 
            "log_sigma": log_sigma,
            "alpha": alpha, 
            "beta": beta}


@register_criterion("crit_bayescap")
def bayescap_loss(y_true, y_pred, params, **kwargs):
    mu_tilde, one_over_alpha, beta = torch.split(params, 1, dim=1)
    one_over_alpha = F.softplus(one_over_alpha) + 1e-8
    beta = F.softplus(beta) + 1e-8
    
    # residual = (beta * torch.abs(mu_tilde - y_true).clamp(1e-4, 1e3) * one_over_alpha)
    residual = torch.pow(torch.abs(mu_tilde - y_true).clamp(1e-4, 1e3) * one_over_alpha, beta)
    nll = residual - torch.log(one_over_alpha) - torch.log(beta) + torch.lgamma(torch.pow(beta, -1))

    identity_loss = torch.square(y_pred - mu_tilde)

    rate = kwargs["epoch"] / kwargs["n_epochs"]
    loss = max(1 - rate, 1e-1) * identity_loss + max(rate, 5e-2) * nll

    return loss.mean()
