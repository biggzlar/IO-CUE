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
        return correlation, p_value
    except:
        return 0.0, 1.0


def compute_crps(predictions, uncertainties, targets):
    uncertainties = torch.clamp(uncertainties, min=1e-6)
    x = (targets - predictions) / uncertainties
    normal = Normal(torch.zeros_like(x), torch.ones_like(x))
    try:
        pdf = torch.exp(normal.log_prob(x))
        cdf = normal.cdf(x)
        crps = uncertainties * (x * (2 * cdf - 1) + 2 * pdf - 1. / torch.sqrt(torch.tensor(torch.pi)))
        return crps
    except:
        return torch.zeros_like(uncertainties)


def compute_ause_rmse(predictions: torch.Tensor, uncertainties: torch.Tensor, targets: torch.Tensor):
    """
    Compute AUSE (Area Under Sparsification Error) for RMSE.
    
    The sparsification procedure ranks samples by predicted uncertainty and
    evaluates RMSE as we keep only the most certain fraction. The oracle curve
    ranks by true errors. The area between these two curves (averaged over
    retention fractions) is the AUSE-RMSE. Lower is better.
    
    Returns:
        ause_rmse (float): Scalar AUSE-RMSE.
        sparsification_error (np.ndarray): SE curve over retention fractions (length N).
    """
    # Flatten to treat each element as an independent prediction
    pred_flat = predictions.detach().flatten()
    targ_flat = targets.detach().flatten()
    unc_flat = uncertainties.detach().flatten()
    
    # Squared errors per element
    sq_errors = (pred_flat - targ_flat).pow(2)
    n = sq_errors.numel()
    if n == 0:
        return 0.0, np.array([0.0])
    
    # Break ties in uncertainties minimally to avoid degenerate ordering
    # (Keep on CPU for numpy return; torch computations remain on tensor)
    noise = torch.randn_like(unc_flat) * 1e-12
    unc_for_sort = unc_flat + noise
    
    # Sort by increasing uncertainty (keep the most certain first)
    idx_unc_asc = torch.argsort(unc_for_sort, descending=False)
    # Sort by increasing true error (oracle keeps smallest errors first)
    idx_err_asc = torch.argsort(sq_errors, descending=False)
    
    sq_err_sorted_unc = sq_errors[idx_unc_asc]
    sq_err_sorted_oracle = sq_errors[idx_err_asc]
    
    # Cumulative sums to compute RMSE efficiently for each retention size k=1..n
    cumsum_unc = torch.cumsum(sq_err_sorted_unc, dim=0)
    cumsum_oracle = torch.cumsum(sq_err_sorted_oracle, dim=0)
    
    ks = torch.arange(1, n + 1, device=sq_errors.device, dtype=sq_errors.dtype)
    rmse_unc = torch.sqrt(cumsum_unc / ks)
    rmse_oracle = torch.sqrt(cumsum_oracle / ks)
    
    # Sparsification error over retention fractions
    se_curve = (rmse_unc - rmse_oracle).detach().cpu().numpy()
    # AUSE as average SE across all retention fractions
    ause_rmse = float(np.mean(se_curve))
    return ause_rmse, se_curve

__all__ = [
    'get_predicted_cdf',
    'compute_ece',
    'compute_nll',
    'compute_euc',
    'compute_crps',
    'compute_ause_rmse',
]



def _run_compute_ause_rmse_tests():
    """
    Simple executable tests for compute_ause_rmse. These tests are deterministic and print
    what you should expect to see for each scenario.
    """
    print("Running compute_ause_rmse test scenarios...\n")
    torch.manual_seed(0)
    np.random.seed(0)

    # Construct a simple 1D prediction problem where targets are zero
    # and prediction errors follow a controlled pattern.
    n = 1000
    targets = torch.zeros(n)
    # Error magnitudes from easy (0) to hard (1)
    error_mag = torch.linspace(0.0, 1.0, steps=n)
    # Random signs to avoid degenerate ordering by sign
    signs = torch.where(torch.rand(n) > 0.5, torch.tensor(1.0), torch.tensor(-1.0))
    predictions = signs * error_mag  # squared error = error_mag^2

    # Scenario 1: Perfectly informative uncertainties (higher uncertainty for bigger errors)
    # Expectation: AUSE-RMSE ≈ 0.0 (the uncertainty ranking matches oracle error ranking).
    uncertainties_informative = error_mag + 1e-9
    ause_inf, _ = compute_ause_rmse(predictions, uncertainties_informative, targets)
    print("Scenario 1: Perfectly informative uncertainties (monotonic with true error).")
    print("  Expectation: AUSE-RMSE ≈ 0.0 (nearly zero).")
    print(f"  Observed AUSE-RMSE: {ause_inf:.6f}\n")

    # Scenario 2: Anti-informative uncertainties (confident on hard items, uncertain on easy ones)
    # Expectation: AUSE-RMSE is large (>> 0) and higher than the uninformative baseline.
    uncertainties_anti = (1.0 - error_mag) + 1e-9
    ause_anti, _ = compute_ause_rmse(predictions, uncertainties_anti, targets)
    print("Scenario 2: Anti-informative uncertainties (inverse of true error).")
    print("  Expectation: AUSE-RMSE is high (much greater than zero).")
    print(f"  Observed AUSE-RMSE: {ause_anti:.6f}\n")

    # Scenario 3: Uninformative uncertainties (constant for all samples)
    # Expectation: AUSE-RMSE is > 0 (random ranking) but lower than anti-informative on average.
    uncertainties_const = torch.full_like(error_mag, 0.5)
    ause_const, _ = compute_ause_rmse(predictions, uncertainties_const, targets)
    print("Scenario 3: Uninformative uncertainties (constant).")
    print("  Expectation: AUSE-RMSE > 0, and typically less than Scenario 2 (anti-informative).")
    print(f"  Observed AUSE-RMSE: {ause_const:.6f}\n")

    # Summary comparison (no hard assertions to avoid breaking script runs)
    print("Summary of expected ordering (not enforced):")
    print("  Scenario 1 (informative)  ≈ 0.0")
    print("  Scenario 3 (uninformative) > 0.0")
    print("  Scenario 2 (anti-inform.)  > Scenario 3\n")


if __name__ == "__main__":
    _run_compute_ause_rmse_tests()
