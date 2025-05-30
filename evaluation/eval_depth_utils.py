import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

from predictors.gaussian import gaussian_nll
from predictors.mse import rmse

from networks.unet_model import UNet

plt.rcParams.update(
    {
        "font.size": 12,
        "text.usetex": False,
        "font.family": "stixgeneral",
        "mathtext.fontset": "stix",
    }
)



def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate depth model and plot sample predictions")
    parser.add_argument("--metric", default="rmse", choices=["rmse", "nll", "vloss"],
                        help="Select the metric for best model checkpoint (rmse, nll, vloss)")
    parser.add_argument("--save-dir", type=str, required=True,
                        help="Directory containing saved model checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Computation device (cuda or cpu)")
    parser.add_argument("--num-samples", type=int, default=4,
                        help="Number of samples to visualize")
    parser.add_argument("--post-hoc", action="store_true",
                        help="Use post-hoc model")
    return parser.parse_args()


def get_predictions(model, images, depths, post_hoc_model=None, device=None, is_bayescap=False):
    """
    Run inference on the given images and return predictions.
    
    Args:
        model: The base Gaussian depth model
        images: Batch of input images
        depths: Ground truth depth maps
        post_hoc_model: Optional post-hoc uncertainty model
        device: Computation device
        
    Returns:
        Dictionary containing predictions and input data
    """ 
    # Run forward pass for the batch
    with torch.no_grad():
        outputs = model.predict(images)
        mu_batch, log_sigma_batch = outputs['mean'], torch.log(outputs['log_ep_sigma'])
    
    # Use post-hoc model if provided
    if post_hoc_model is not None:
        with torch.no_grad():
            if is_bayescap:
                outputs = post_hoc_model.predict(mu_batch)
            else:
                outputs = post_hoc_model.predict(X=images, y_pred=mu_batch)
            log_sigma_batch = outputs['mean_log_sigma']
    
    # Calculate metrics for each sample
    preds = torch.concat([mu_batch, log_sigma_batch], dim=1)
    batch_nll = gaussian_nll(y_pred=preds, y_true=depths, reduce=False)
    batch_rmse = rmse(y_pred=mu_batch, y_true=depths, reduce=False)
    batch_sigma = torch.exp(log_sigma_batch)

    batch_average_indices = tuple(range(1, batch_nll.ndim))
    metrics = {
        'rmse': batch_rmse.mean(dim=batch_average_indices),
        'nll': batch_nll.mean(dim=batch_average_indices),
        'avg_var': batch_sigma.mean(dim=batch_average_indices)
    }

    return {
        'images': images.detach(),
        'depths': depths.detach(),
        'mu_batch': mu_batch.detach(),
        'sigma_batch': batch_sigma.detach(),
        'metrics': metrics,
        'error_batch': batch_rmse.detach()
    }


def visualize_results(results, num_samples, metric_name, path=None, suffix=""):
    """
    Visualize the prediction results as a grid of plots.
    
    Args:
        results: Dictionary containing prediction results
        num_samples: Number of samples to visualize
        metric_name: Name of the metric used for evaluation
    """
    images = results['images']
    depths = results['targets']
    mu_batch = results['mu_batch']
    sigma_batch = results['sigma_batch']
    
    # Create results folder
    path = "results" if path is None else path
    os.makedirs(path, exist_ok=True)

    # Plot input RGB and predicted depth side by side for each sample
    fig, axs = plt.subplots(num_samples, 5, figsize=(20, num_samples * 4))
    for i in range(num_samples):
        # Input RGB image
        img = images[i].cpu().permute(1, 2, 0).numpy()
        ax_in = axs[i, 0]
        im0 = ax_in.imshow(img)
        im0.set_clim(0, 1)
        ax_in.set_title(f"Sample {i+1}")
        ax_in.axis("off")

        # Predicted depth map
        mu = mu_batch[i]
        sigma = sigma_batch[i]
        y = depths[i]
        rmse_i = results['rmse_batch'][i]
        nll_i = results['nll_batch'][i]
        avg_var_i = results['avg_var_batch'][i]

        ax_pred = axs[i, 1]
        im1 = ax_pred.imshow(mu.squeeze().cpu().numpy(), cmap="plasma")
        im1.set_clim(0, 1)
        ax_pred.set_title(f"RMSE: {rmse_i:.4f}\nNLL: {nll_i:.4f}\nVar: {avg_var_i:.4f}")
        ax_pred.axis("off")

        ax_gt = axs[i, 2]
        im2 = ax_gt.imshow(y.squeeze().cpu().numpy(), cmap="plasma")
        im2.set_clim(0, 1)
        ax_gt.set_title(f"Ground Truth")
        ax_gt.axis("off")

        ax_error = axs[i, 3]
        im3 = ax_error.imshow(torch.abs(mu - y).squeeze().cpu().numpy(), cmap="plasma")
        im3.set_clim(0, 0.5)
        ax_error.set_title(f"Error")
        ax_error.axis("off")

        ax_var = axs[i, 4]
        im4 = ax_var.imshow(sigma.squeeze().cpu().numpy(), cmap="plasma")
        im4.set_clim(0, 0.5)
        ax_var.set_title(f"Standard Deviation")
        ax_var.axis("off")

    plt.tight_layout()
    out_path = os.path.join(path, f"eval_{metric_name}{suffix}.png")
    plt.savefig(out_path, dpi=300)
    print(f"Saved visualization to {out_path}")
    plt.close()

def load_model(model_type, model_path, inference_fn, model_params, n_models, device, model_class=UNet):
    model = model_type(
        model_class=model_class,
        model_params=model_params,
        n_models=n_models,
        device=device,
        infer=inference_fn
    )
    model.load(model_path)
    return model.eval()