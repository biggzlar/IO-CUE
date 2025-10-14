import torch
import numpy as np
from scipy.stats import spearmanr
from torch.distributions.normal import Normal

from predictors.gaussian import gaussian_nll_detached


def get_predicted_cdf(residuals: torch.Tensor, sigma: torch.Tensor, num_bins: int = 10):
    """Computes empirical CDF values for predicted uncertainties, given absolute error.
    
    Args:
        residuals: Prediction errors abs(y_pred - y_true)
        sigma: Predicted standard deviations
        num_bins: Number of confidence levels to evaluate
    
    Returns:
        Empirical CDF values for each prediction
    """
    alpha = torch.linspace(0, 1, num_bins, device=residuals.device)
    std_quantiles = torch.special.ndtri((1 + alpha) / 2)
    normalized_residuals = (residuals / sigma).flatten()
    coverages = []
    for q in std_quantiles:
        coverage = torch.less_equal(normalized_residuals, q).float().mean()
        coverages.append(coverage.detach().cpu())
    return coverages


def compute_ece(residuals: torch.Tensor, sigma: torch.Tensor, num_bins: int = 10) -> float:
    pcdf = get_predicted_cdf(residuals, sigma, num_bins)
    confidence_levels = np.arange(0.1, 1.1, 0.1)
    empirical_fractions = np.array([
        np.mean(pcdf <= p) for p in confidence_levels
    ])
    abs_differences = np.abs(empirical_fractions - confidence_levels)
    ece = np.mean(abs_differences)
    return ece, empirical_fractions


def compute_nll(predictions, uncertainties, targets):
    log_sigma = 0.5 * torch.log(uncertainties)
    nll = gaussian_nll_detached(y_pred=predictions, y_true=targets, params=log_sigma, reduce=False)
    return nll


def compute_euc(predictions, uncertainties, targets):
    errors = torch.abs(targets - predictions)
    errors_np = errors.detach().cpu().numpy().flatten()
    uncertainties_np = uncertainties.detach().cpu().numpy().flatten()
    if np.allclose(uncertainties_np, uncertainties_np[0], rtol=1e-5, atol=1e-8):
        return 0.0, 1.0
    uncertainties_np += np.random.normal(0, 1e-10, size=uncertainties_np.shape)
    try:
        correlation, p_value = spearmanr(errors_np, uncertainties_np)
        if np.isnan(correlation):
            correlation, p_value = 0.0, 1.0
        return correlation.item(), p_value
    except:
        return 0.0, 1.0


def compute_crps(predictions, uncertainties, targets):
    uncertainties = torch.clamp(uncertainties, min=1e-6)
    x = (targets - predictions) / uncertainties
    normal = Normal(torch.zeros_like(x), torch.ones_like(x))
    pdf = torch.exp(normal.log_prob(x))
    cdf = normal.cdf(x)
    crps = uncertainties * (x * (2 * cdf - 1) + 2 * pdf - 1. / torch.sqrt(torch.tensor(torch.pi)))
    return crps

__all__ = [
    'get_predicted_cdf',
    'compute_ece',
    'compute_nll',
    'compute_euc',
    'compute_crps',
]


