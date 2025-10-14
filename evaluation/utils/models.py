import torch
from predictors.gaussian import gaussian_nll
from predictors.mse import rmse
from networks.unet_model import UNet


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
    with torch.no_grad():
        outputs = model.predict(images)
        mu_batch, log_sigma_batch = outputs['mean'], torch.log(outputs['log_ep_sigma'])

    if post_hoc_model is not None:
        with torch.no_grad():
            if is_bayescap:
                outputs = post_hoc_model.predict(mu_batch)
            else:
                outputs = post_hoc_model.predict(X=images, y_pred=mu_batch)
            log_sigma_batch = outputs['mean_log_sigma']

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


def load_mean_model(model_type, model_path, model_params, n_models, device, model_class=UNet, inference_fn=None):
    model = model_type(
            model_class=model_class,
            model_params=model_params,
            n_models=n_models,
            device=device,
            infer=inference_fn
        )
    model.load(model_path)
    return model


def load_variance_model(mean_ensemble, model_type, model_path, model_params, n_models, device, model_class=UNet):
    model = model_type(
        mean_ensemble=mean_ensemble,
        model_class=model_class,
        model_params=model_params,
        n_models=n_models,
        device=device,
    )
    model.load(model_path)
    return model


