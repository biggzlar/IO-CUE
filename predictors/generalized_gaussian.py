import torch
import torch.nn.functional as F

def predict_gen_gaussian(x, net):
    outs = [out.detach() for out in net(x)]
    alpha_map = torch.pow(outs[0] + 1e-8, -1)
    beta_map = outs[1]

    variance = alpha_map**2 * torch.exp(torch.lgamma(3 / beta_map + 1e-8) - torch.lgamma(1 / beta_map + 1e-8))
    sigma = variance.sqrt()

    return {"alpha": alpha_map, "beta": beta_map, "variance": variance, "sigma": sigma}

def gen_gaussian_nll(y_true, y_pred, params, **kwargs):
    one_over_alpha, beta = torch.split(params, 1, dim=1)
    one_over_alpha = F.softplus(one_over_alpha) + 1e-8
    beta = F.softplus(beta) + 1e-8

    residual = torch.pow(torch.abs(y_pred - y_true).clamp(1e-4, 1e3) * one_over_alpha, beta)
    nll = residual - torch.log(one_over_alpha) - torch.log(beta) + torch.lgamma(torch.pow(beta, -1))
    return nll.mean()

def post_hoc_predict_gen_gaussian(preds):
    one_over_alpha, beta = torch.split(preds, 1, dim=1)
    one_over_alpha = F.softplus(one_over_alpha) + 1e-8
    alpha = torch.pow(one_over_alpha, -1)
    beta = F.softplus(beta) + 1e-8

    variance = alpha**2 * torch.exp(torch.lgamma(3 / beta + 1e-8) - torch.lgamma(1 / beta + 1e-8))
    sigma = variance.sqrt()
    log_sigma = torch.log(sigma)

    return {'sigma': sigma, 'log_sigma': log_sigma, 'alpha': alpha, 'beta': beta}