import torch
from .registry import register_predictor, register_criterion

@register_criterion("crit_mse")
def mse(y_pred, y_true, reduce=True):
    mse = (y_pred - y_true) ** 2
    return mse.mean() if reduce else mse

@register_criterion("crit_rmse")
def rmse(y_pred, y_true, reduce=True):
    mse = (y_pred - y_true) ** 2
    rmse = torch.sqrt(mse)
    return rmse.mean() if reduce else rmse

@register_predictor("pred_mse")
def predict_mse(preds):
    return {"mean": preds, "sigma": torch.zeros_like(preds), "log_sigma": torch.zeros_like(preds)}