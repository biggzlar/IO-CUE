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
    # Generate confidence levels from 0 to 1
    alpha = torch.linspace(0, 1, num_bins, device=residuals.device)
    
    # Generate standard normal quantiles
    # equivalent to scipy.stats.norm.ppf(alpha) or torch.distributions.Normal(0, 1).icdf(alpha)
    std_quantiles = torch.special.ndtri((1 + alpha) / 2)
    
    # Normalize residuals by predicted standard deviations
    normalized_residuals = (residuals / sigma).flatten()
    # print(normalized_residuals)
    
    coverages = []
    for q in std_quantiles:
        coverage = torch.less_equal(normalized_residuals, q).float().mean()
        coverages.append(coverage.detach().cpu())
    
    return coverages


def compute_ece(residuals: torch.Tensor, sigma: torch.Tensor, num_bins: int = 10) -> float:
    """Compute Expected Calibration Error for regression uncertainty.
    
    Args:
        pcdf: Predicted CDF values for each sample
        num_bins: Number of confidence levels to evaluate
    
    Returns:
        Expected Calibration Error score
    """
    pcdf = get_predicted_cdf(residuals, sigma, num_bins)

    confidence_levels = np.arange(0.1, 1.1, 0.1)  # [0.1, 0.2, ..., 1.0]
    
    # For each confidence level, compute fraction of samples below that level
    empirical_fractions = np.array([
        np.mean(pcdf <= p) for p in confidence_levels
    ])
    
    # Compute absolute differences from diagonal
    abs_differences = np.abs(empirical_fractions - confidence_levels)
    
    # Equal weighting for bins (could be modified to use actual bin counts)
    ece = np.mean(abs_differences)
    
    return ece, empirical_fractions


def compute_nll(predictions, uncertainties, targets):
    """
    Compute Gaussian Negative Log-Likelihood
    
    Args:
        predictions: Predicted values (torch.Tensor)
        uncertainties: Uncertainty estimates (variance) (torch.Tensor)
        targets: Ground truth values (torch.Tensor)
    
    Returns:
        nll: Negative log-likelihood (torch.Tensor)
    """    
    # Convert uncertainties to log_sigma
    log_sigma = 0.5 * torch.log(uncertainties)
    
    nll = gaussian_nll_detached(y_pred=predictions, y_true=targets, params=log_sigma, reduce=False)

    return nll


def compute_euc(predictions, uncertainties, targets):
    """
    Compute Spearman's correlation between error and uncertainty
    
    Args:
        predictions: Predicted values (torch.Tensor)
        uncertainties: Uncertainty estimates (torch.Tensor)
        targets: Ground truth values (torch.Tensor)
    
    Returns:
        correlation: Spearman's correlation coefficient (float)
        p_value: p-value for the correlation (float)
    """
    # Compute errors
    errors = torch.abs(targets - predictions)
    
    # Convert to numpy for spearmanr
    errors_np = errors.detach().cpu().numpy().flatten()
    uncertainties_np = uncertainties.detach().cpu().numpy().flatten()
    
    # Check if uncertainties are constant
    if np.allclose(uncertainties_np, uncertainties_np[0], rtol=1e-5, atol=1e-8):
        return 0.0, 1.0  # Return zero correlation with p-value of 1.0 for constant uncertainty
    
    # Add tiny noise to break ties (helps with constant/tied values)
    uncertainties_np += np.random.normal(0, 1e-10, size=uncertainties_np.shape)
    
    try:
        # Compute Spearman's correlation
        correlation, p_value = spearmanr(errors_np, uncertainties_np)
        
        # Replace NaN with 0 (happens with constant values)
        if np.isnan(correlation):
            correlation, p_value = 0.0, 1.0
        
        return correlation, p_value
    except:
        # Fallback in case of errors
        return 0.0, 1.0
    

def compute_crps(predictions, uncertainties, targets):
    # Ensure positive sigma
    uncertainties = torch.clamp(uncertainties, min=1e-6)
    
    # Standardized error
    x = (targets - predictions) / uncertainties
    
    # Standard normal
    normal = Normal(torch.zeros_like(x), torch.ones_like(x))
    
    # Compute components
    pdf = torch.exp(normal.log_prob(x))
    cdf = normal.cdf(x)
    
    crps = uncertainties * (x * (2 * cdf - 1) + 2 * pdf - 1. / torch.sqrt(torch.tensor(torch.pi)))
    return crps
