import argparse
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloaders.simple_depth import DepthDataset
from evaluation.utils import load_mean_model
from networks.unet_model import UNet
from predictors.mse import rmse, predict_mse
from predictors.gaussian import predict_gaussian, gaussian_nll
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd


def set_seed(seed: int = 0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_device(batch: Tuple[torch.Tensor, torch.Tensor], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    x, y = batch
    return x.to(device), y.to(device)


def apply_flip(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.flip(x, dims=[-1]), torch.flip(y, dims=[-1])


def apply_color_jitter(x: torch.Tensor, strength: float, rng: np.random.RandomState) -> torch.Tensor:
    # Apply per-image to keep code simple and deterministic under given rng
    out = []
    jitter = transforms.ColorJitter(
        brightness=0.3 * strength,
        contrast=0.3 * strength,
        saturation=0.3 * strength,
        hue=0.05 * strength,
    )
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    for i in range(x.shape[0]):
        img = x[i].detach().cpu()
        pil = to_pil(img)
        # Seed torch RNG via numpy draw to keep variety yet reproducible
        _ = rng.rand()
        pil_aug = jitter(pil)
        out.append(to_tensor(pil_aug))
    return torch.stack(out, dim=0).to(x.device)


def apply_gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return x
    blur = transforms.GaussianBlur(kernel_size=3, sigma=(sigma, sigma))
    out = []
    for i in range(x.shape[0]):
        out.append(blur(x[i]))
    return torch.stack(out, dim=0)


def apply_grayscale(x: torch.Tensor) -> torch.Tensor:
    # Convert to grayscale then back to 3 channels
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    gray = transforms.Grayscale(num_output_channels=3)
    out = []
    for i in range(x.shape[0]):
        pil = to_pil(x[i].detach().cpu())
        out.append(to_tensor(gray(pil)))
    return torch.stack(out, dim=0).to(x.device)


def apply_gaussian_noise(x: torch.Tensor, sigma: float, rng: np.random.RandomState) -> torch.Tensor:
    if sigma <= 0:
        return x
    noise = torch.from_numpy(rng.randn(*x.shape)).to(x.device, dtype=x.dtype) * sigma
    return torch.clamp(x + noise, 0.0, 1.0)


def _compute_nll_from_preds(preds: Dict[str, torch.Tensor], targets: torch.Tensor) -> float:
    try:
        if 'al_sigma' in preds and 'ep_sigma' in preds:
            total_log_sigma = torch.log(preds['al_sigma'] + preds['ep_sigma'] + 1e-8)
        elif 'log_ep_sigma' in preds:
            total_log_sigma = preds['log_ep_sigma']
        else:
            return float('nan')
        y_pred = torch.cat([preds['mean'], total_log_sigma], dim=1)
        return float(gaussian_nll(y_pred=y_pred, y_true=targets, reduce=True).item())
    except Exception:
        return float('nan')


def eval_model_on_loader(model, loader: DataLoader, device: torch.device, num_batches: int = None) -> Dict[str, float]:
    model.eval()
    rmses = []
    with torch.no_grad():
        for idx, (xb, yb) in enumerate(loader):
            xb, yb = xb.to(device), yb.to(device)
            preds = model.predict(xb)
            mu = preds['mean']
            rmses.append(rmse(mu, yb).item())
            if num_batches is not None and (idx + 1) >= num_batches:
                break
    return {'rmse': float(np.mean(rmses))}


def consistency_metric(model, x_clean: torch.Tensor, x_aug: torch.Tensor, inverse_aug=None) -> float:
    # Measures average L2 difference between clean prediction and inverse(aug prediction)
    with torch.no_grad():
        pred_clean = model.predict(x_clean)['mean']
        pred_aug = model.predict(x_aug)['mean']
        if inverse_aug is not None:
            pred_aug = inverse_aug(pred_aug)
        diff = (pred_clean - pred_aug) ** 2
        return float(torch.sqrt(diff.mean()).item())


def load_base_model_auto(path: str, device: torch.device, n_models: int = 5):
    # Try MSE first
    try:
        model = load_mean_model(
            model_type=torch.nn.Module.__subclasses__()[0].__mro__[0].__class__,  # placeholder; ignored by loader
            model_path=path,
            model_params={"in_channels": 3, "out_channels": [1], "drop_prob": 0.2},
            n_models=n_models,
            device=device,
            model_class=UNet,
            inference_fn=predict_mse,
        )
        return model, 'mse'
    except Exception:
        pass
    # Then try Gaussian
    model = load_mean_model(
        model_type=torch.nn.Module.__subclasses__()[0].__mro__[0].__class__,  # placeholder; ignored by loader
        model_path=path,
        model_params={"in_channels": 3, "out_channels": [1, 1], "drop_prob": 0.2},
        n_models=n_models,
        device=device,
        model_class=UNet,
        inference_fn=predict_gaussian,
    )
    return model, 'gaussian'


def build_results_dir(base_dir: str = "results/compare_base_models") -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(base_dir, ts)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def write_csv(path: str, header: List[str], rows: List[List]):
    import csv
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Compare base models under dataset augmentations to infer training scheme")
    parser.add_argument('--models', nargs='+', required=True, help='Paths to base ensemble checkpoints to compare')
    parser.add_argument('-d', '--device', type=str, default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-batches', type=int, default=20, help='Limit number of test batches (None for all)')
    parser.add_argument('--img-size', type=int, nargs=2, default=(128, 160))
    parser.add_argument('--train-split', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    # Data: clean evaluation loader from simple_depth
    dataset = DepthDataset(path=None, img_size=tuple(args.img_size), augment=False, train_split=args.train_split)
    _, test_loader = dataset.get_dataloaders(batch_size=args.batch_size, shuffle=False)

    # Results dir
    out_dir = build_results_dir()

    # Augmentation grids
    noise_sigmas = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4]
    blur_sigmas = [0.0, 0.5, 1.0, 1.5]
    jitter_strengths = [0.0, 0.3, 0.6, 1.0]
    flips = [False, True]
    grays = [False, True]

    header = [
        'model_name', 'inference', 'augment', 'level', 'rmse', 'consistency_rmse', 'nll'
    ]
    rows: List[List] = []

    # Per-model evaluation
    for model_path in args.models:
        model_name = os.path.basename(model_path)
        try:
            # Explicitly construct BaseEnsemble using helper
            from models.base_ensemble_model import BaseEnsemble
            # Try both in order
            try:
                model = BaseEnsemble(model_class=UNet, model_params={"in_channels": 3, "out_channels": [1], "drop_prob": 0.2}, n_models=5, device=device, infer=predict_mse)
                model.load(model_path)
                inference_kind = 'mse'
            except Exception:
                model = BaseEnsemble(model_class=UNet, model_params={"in_channels": 3, "out_channels": [1, 1], "drop_prob": 0.2}, n_models=5, device=device, infer=predict_gaussian)
                model.load(model_path)
                inference_kind = 'gaussian'
        except Exception as e:
            print(f"Failed to load model {model_path}: {e}")
            continue

        model.eval()

        # Prepare RNG for deterministic augmentation sampling
        rng = np.random.RandomState(args.seed)

        # Iterate limited number of batches once and reuse for all aug evaluations for fairness
        cached_batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
        with torch.no_grad():
            for idx, batch in enumerate(test_loader):
                xb, yb = to_device(batch, device)
                cached_batches.append((xb, yb))
                if args.num_batches is not None and (idx + 1) >= args.num_batches:
                    break

        def measure_over_batches(apply_x, apply_xy=None, inverse_on_pred=None) -> Tuple[float, float, float]:
            # Returns (rmse_to_gt, consistency_rmse, nll)
            rmse_vals = []
            cons_vals = []
            nll_vals = []
            for xb, yb in cached_batches:
                if apply_xy is not None:
                    x_aug, y_aug = apply_xy(xb, yb)
                else:
                    x_aug = apply_x(xb)
                    y_aug = yb

                with torch.no_grad():
                    preds = model.predict(x_aug)
                    mu = preds['mean']
                    rmse_vals.append(rmse(mu, y_aug).item())
                    nll_vals.append(_compute_nll_from_preds(preds, y_aug))

                cons_vals.append(consistency_metric(model, xb, x_aug, inverse_on_pred))
            return float(np.mean(rmse_vals)), float(np.mean(cons_vals)), float(np.nanmean(nll_vals))

        # Clean (identity)
        rm, cons, nllv = measure_over_batches(apply_x=lambda x: x)
        rows.append([model_name, inference_kind, 'clean', 0.0, rm, cons, nllv])

        # Flip
        for do_flip in flips:
            if not do_flip:
                continue
            rm, cons, nllv = measure_over_batches(apply_x=lambda x: torch.flip(x, dims=[-1]), apply_xy=apply_flip, inverse_on_pred=lambda t: torch.flip(t, dims=[-1]))
            rows.append([model_name, inference_kind, 'flip', 1.0, rm, cons, nllv])

        # Color Jitter
        for s in jitter_strengths:
            if s <= 0:
                continue
            rm, cons, nllv = measure_over_batches(apply_x=lambda x, s=s: apply_color_jitter(x, s, rng))
            rows.append([model_name, inference_kind, 'colorjitter', s, rm, cons, nllv])

        # Gaussian Blur
        for s in blur_sigmas:
            if s <= 0:
                continue
            rm, cons, nllv = measure_over_batches(apply_x=lambda x, s=s: apply_gaussian_blur(x, s))
            rows.append([model_name, inference_kind, 'gaussianblur', s, rm, cons, nllv])

        # Grayscale
        for g in grays:
            if not g:
                continue
            rm, cons, nllv = measure_over_batches(apply_x=lambda x: apply_grayscale(x))
            rows.append([model_name, inference_kind, 'grayscale', 1.0, rm, cons, nllv])

        # Gaussian Noise
        for s in noise_sigmas:
            if s <= 0:
                continue
            rm, cons, nllv = measure_over_batches(apply_x=lambda x, s=s: apply_gaussian_noise(x, s, rng))
            rows.append([model_name, inference_kind, 'gaussian_noise', s, rm, cons, nllv])

        # Resize sensitivity: scale then immediately rescale back to original size to keep model input shape
        resize_scales = [0.75, 1.25]
        for scale in resize_scales:
            def _resize_to_orig(x: torch.Tensor, s: float = scale) -> torch.Tensor:
                b, c, h, w = x.shape
                nh, nw = int(round(h * s)), int(round(w * s))
                x_res = F.interpolate(x, size=(nh, nw), mode='bilinear', align_corners=False)
                x_res_back = F.interpolate(x_res, size=(h, w), mode='bilinear', align_corners=False)
                return x_res_back

            rm, cons, nllv = measure_over_batches(
                apply_x=lambda x, s=scale: _resize_to_orig(x, s=s),
                apply_xy=None,
                inverse_on_pred=None
            )
            rows.append([model_name, inference_kind, 'resize', scale, rm, cons, nllv])

    # Save CSV
    csv_path = os.path.join(out_dir, 'augmentation_comparison.csv')
    write_csv(csv_path, header, rows)
    print(f"Saved results to {csv_path}")

    # Build single comparison figure with subplots
    df = pd.DataFrame(rows, columns=header)
    augment_order = ['clean', 'flip', 'colorjitter', 'gaussianblur', 'grayscale', 'gaussian_noise', 'resize']
    metrics = ['rmse', 'nll']

    # Determine available metrics (finite)
    available_metrics = []
    for m in metrics:
        vals = pd.to_numeric(df[m], errors='coerce')
        if np.isfinite(vals).any():
            available_metrics.append(m)
    if not available_metrics:
        available_metrics = ['rmse']

    n_rows = len(available_metrics)
    n_cols = len(augment_order)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.2 * n_rows), squeeze=False)

    model_names = sorted(df['model_name'].unique().tolist())
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', None)

    def plot_panel(ax, metric, augment):
        base = df[df['augment'] == 'clean'][['model_name', metric]].copy()
        base[metric] = pd.to_numeric(base[metric], errors='coerce')
        aug_df = df[df['augment'] == augment][['model_name', 'level', metric]].copy()
        aug_df[metric] = pd.to_numeric(aug_df[metric], errors='coerce')

        for i, name in enumerate(model_names):
            base_val = base[base['model_name'] == name][metric].values
            if len(base_val) == 0 or not np.isfinite(base_val[0]):
                continue
            x_levels = [0.0]
            y_vals = [float(base_val[0])]
            sub = aug_df[aug_df['model_name'] == name].dropna(subset=[metric])
            if not sub.empty:
                xs = sub['level'].astype(float).values.tolist()
                ys = sub[metric].astype(float).values.tolist()
                paired = sorted(zip(xs, ys), key=lambda t: t[0])
                x_levels.extend([p[0] for p in paired])
                y_vals.extend([p[1] for p in paired])
            c = None if color_cycle is None else color_cycle[i % len(color_cycle)]
            ax.plot(x_levels, y_vals, marker='o', label=name, color=c)

        ax.set_xlabel('level')
        ax.set_ylabel(metric.upper())
        ax.set_title(augment)
        if augment in ['flip', 'grayscale']:
            ax.set_xticks([0.0, 1.0])
            ax.set_xticklabels(['clean', 'on'])
        elif augment == 'clean':
            ax.set_xticks([0.0])
            ax.set_xticklabels(['clean'])
        ax.grid(True, alpha=0.3)

    for r, metric in enumerate(available_metrics):
        for c, augment in enumerate(augment_order):
            plot_panel(axs[r, c], metric, augment)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=min(len(model_names), 6))
        plt.subplots_adjust(bottom=0.12)
    else:
        plt.tight_layout()

    fig_path = os.path.join(out_dir, 'comparison_grid.png')
    plt.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"Saved comparison figure to {fig_path}")


if __name__ == '__main__':
    main()


