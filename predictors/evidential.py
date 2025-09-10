import torch
import torch.nn.functional as F
from .registry import register_predictor, register_criterion

@register_criterion("crit_nig")
def NIG_NLL(y_true, gamma, nu, alpha, beta, reduce=False):
    two_b_lambda = 2 * beta * (1 + nu)
    nll = 0.5 * torch.log(torch.pi / (nu + 1e-8)) \
        - alpha * torch.log(two_b_lambda) \
        + (alpha + 0.5) * torch.log(nu * torch.square(y_true - gamma) + two_b_lambda) \
        + torch.lgamma(alpha) \
        - torch.lgamma(alpha + 0.5)
    
    # nll = 0.5 * torch.log(torch.pi / (nu + 1e-8)) \
    #     + 0.5 * torch.log(two_b_lambda) \
    #     + (alpha + 0.5) * torch.log(1 + (nu * torch.square(y_true - gamma)) / two_b_lambda) \
    #     + torch.lgamma(alpha) \
    #     - torch.lgamma(alpha + 0.5)
        
    return torch.mean(nll) if reduce else nll


def NIG_REG(y_true, gamma, nu, alpha, beta, reduce=False):
    """ The regularizer as proposed by Amini et al. is adapted to maximize
        calibration and accurate recovery of the aleatoric component. Excluding 
        $\nu$ from the total evidence term improves aleatoric uncertainty estimates.

        Dividing the error term by the aleatoric component may increase 
        calibration scores in some instances.
    """
    error = torch.square(y_true - gamma).detach() # / (beta * torch.reciprocal(alpha - 1.0))
    # improved regularizer?
    # evi = torch.reciprocal(beta * torch.reciprocal((alpha - 1.0))) * 2 * alpha
    # evi = torch.reciprocal(beta * torch.reciprocal((alpha - 1.0))) + 2 * alpha + nu
    # evi = torch.reciprocal(beta * torch.reciprocal((alpha - 1.0))) + 2 * alpha

    # Meinert-style regularizer
    # evi = 2 * alpha + nu
    # Amini-style regularizer
    # evi = alpha + 2 * nu
    
    evi = 0.
    reg = error * evi
    
    return torch.mean(reg) if reduce else reg

@register_criterion("crit_nig_detached")
def NIG_NLL_DETACHED(y_true, y_pred, params, reduce=False, coeff=1.0, epoch=None, n_epochs=None):
    nu, alpha, beta = transform_NIG_params(params)
    gamma = y_pred.detach()

    loss_nll = NIG_NLL(y_true, gamma, nu, alpha, beta)
    loss_reg = NIG_REG(y_true, gamma, nu, alpha, beta)
    loss = torch.mean(loss_nll + coeff * loss_reg)
    return loss

@register_predictor("pred_nig_posthoc")
def post_hoc_predict_NIG(params):
    nu, alpha, beta = transform_NIG_params(params)

    sigma = beta * torch.reciprocal((alpha - 1.0))

    return {'sigma': sigma, 'log_sigma': torch.log(sigma), 'nu': nu, 'alpha': alpha, 'beta': beta}


def transform_NIG_params(params):
    nu, alpha, beta = torch.split(params, 1, dim=1)

    alpha = F.softplus(alpha) + 1.
    beta = F.softplus(beta) + 1e-8
    nu = F.softplus(nu) + 1e-8

    return nu, alpha, beta
