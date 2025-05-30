import torch

def mse(y_pred, y_true, reduce=True):
    mse = (y_pred - y_true) ** 2
    return mse.mean() if reduce else mse

def rmse(y_pred, y_true, reduce=True):
    mse = (y_pred - y_true) ** 2
    rmse = torch.sqrt(mse)
    return rmse.mean() if reduce else rmse

def predict_mse(preds):
    return {"mean": preds, "sigma": torch.zeros_like(preds), "log_sigma": torch.zeros_like(preds)}