import torch
import numpy as np
import torch.nn.functional as F
from .registry import register_criterion

def _compute_gradient(x):
    """Compute gradients in x and y directions."""
    # Ensure input has batch and channel dimensions
    if len(x.shape) == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif len(x.shape) == 3:
        x = x.unsqueeze(1)

    # Compute gradients using Sobel filters
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                           dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                           dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    
    grad_x = F.conv2d(x, sobel_x, padding=1)
    grad_y = F.conv2d(x, sobel_y, padding=1)
    
    return grad_x, grad_y

@register_criterion("crit_edgy_gaussian_detached")
def edgy_gaussian_nll_detached(y_pred, y_true, params, reduce=True, **kwargs):
    log_sigma = params
    residual = (y_true - y_pred.detach()) ** 2
    nll = 0.5 * torch.exp(-2 * log_sigma) * residual + log_sigma
    nll = nll + 0.5 * torch.log(torch.tensor(2 * np.pi))

    # Compute gradients
    pred_grad_x, pred_grad_y = _compute_gradient(y_pred)
    true_grad_x, true_grad_y = _compute_gradient(y_true)
    
    # Edge loss - emphasize differences in gradients
    # Using L1 loss for gradient differences to preserve edge sharpness
    edge_loss_x = torch.abs(pred_grad_x - true_grad_x)
    edge_loss_y = torch.abs(pred_grad_y - true_grad_y)
    edge_loss = edge_loss_x + 1. * edge_loss_y
    
    total_loss = nll + edge_loss
        
    return nll.mean() if reduce else nll
