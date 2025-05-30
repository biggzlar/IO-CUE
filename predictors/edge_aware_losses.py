import torch
import torch.nn.functional as F
import numpy as np

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

def edge_aware_mse_loss(y_pred, y_true, edge_weight=1.0, reduce=True):
    """
    MSE loss with an additional term to encourage sharper edges.
    
    Args:
        y_pred: Predicted depth map
        y_true: Ground truth depth map
        edge_weight: Weight for the edge loss term
        reduce: Whether to return a scalar loss or per-pixel loss
    """
    # Standard MSE loss
    mse = (y_pred - y_true) ** 2
    
    # Compute gradients
    pred_grad_x, pred_grad_y = _compute_gradient(y_pred)
    true_grad_x, true_grad_y = _compute_gradient(y_true)
    
    # Edge loss - emphasize differences in gradients
    # Using L1 loss for gradient differences to preserve edge sharpness
    edge_loss_x = torch.abs(pred_grad_x - true_grad_x)
    edge_loss_y = torch.abs(pred_grad_y - true_grad_y)
    edge_loss = edge_loss_x + edge_loss_y
    
    # Combine losses
    # Squeeze messes up shape but leads to better results
    total_loss = mse + edge_weight * edge_loss.squeeze(1)
    
    return total_loss.mean() if reduce else total_loss

def edge_aware_gaussian_nll_loss(y_pred, y_true, edge_weight=1.0, reduce=True):
    """
    Gaussian NLL loss with an additional term to encourage sharper edges.
    
    Args:
        y_pred: Predicted depth map and uncertainty (concatenated)
        y_true: Ground truth depth map
        edge_weight: Weight for the edge loss term
        reduce: Whether to return a scalar loss or per-pixel loss
    """
    # Standard Gaussian NLL
    mean, log_sigma = torch.split(y_pred, 1, dim=1)
    residual = (y_true - mean)**2
    nll = 0.5 * torch.exp(-2 * log_sigma) * residual + log_sigma
    
    # Compute gradients
    pred_grad_x, pred_grad_y = _compute_gradient(torch.exp(log_sigma))
    true_grad_x, true_grad_y = _compute_gradient(y_true)
    
    # Edge loss - weighted by uncertainty (exp(-log_sigma))
    # Areas with higher confidence should have more accurate edges
    edge_loss_x = torch.abs(pred_grad_x - true_grad_x)
    edge_loss_y = torch.abs(pred_grad_y - true_grad_y)
    edge_loss = (edge_loss_x + edge_loss_y)
    
    # Combine losses
    total_loss = nll + edge_weight * edge_loss
    
    return total_loss.mean() if reduce else total_loss


def edge_aware_gaussian_nll_loss_detached(y_pred, y_true, params, edge_weight=1e-1, reduce=True, **kwargs):
    """
    Gaussian NLL loss with an additional term to encourage sharper edges.
    
    Args:
        y_pred: Predicted depth map and uncertainty (concatenated)
        y_true: Ground truth depth map
        edge_weight: Weight for the edge loss term
        reduce: Whether to return a scalar loss or per-pixel loss
    """
    # Standard Gaussian NLL
    log_sigma = params
    residual = (y_true - y_pred)**2
    nll = 0.5 * torch.exp(-2 * log_sigma) * residual + log_sigma
    
    # Compute gradients
    pred_grad_x, pred_grad_y = _compute_gradient(torch.exp(log_sigma))
    true_grad_x, true_grad_y = _compute_gradient(residual)
    # # Usually, we be interested in the gradient of the ground truth
    # # depth map, but we want to capture our error accurately, so we use
    # # the gradient of the predicted depth map
    # true_grad_x, true_grad_y = _compute_gradient(y_pred.detach())
    
    edge_loss_x = torch.abs(pred_grad_x - true_grad_x)
    edge_loss_y = torch.abs(pred_grad_y - true_grad_y)
    edge_loss = (edge_loss_x + edge_loss_y)
    total_loss = nll + edge_weight * edge_loss
    # total_loss = nll

    # if kwargs['epoch'] < 10:
    # total_loss += max(1 - kwargs['epoch'] / kwargs['n_epochs'], 1e-1) * 10. \
    #             * torch.square(torch.exp(log_sigma) * torch.sqrt(torch.ones(1) * 2 * np.pi).to(y_pred.device) - 3. * torch.abs(y_pred - y_true))
    
    return total_loss.mean() if reduce else total_loss
