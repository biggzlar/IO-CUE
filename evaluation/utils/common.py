import os
import numpy as np
import matplotlib.pyplot as plt
import torch


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def timestamped_dir(base_dir: str) -> str:
    import time
    out_dir = os.path.join(base_dir, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def write_csv(path: str, header, rows) -> None:
    import csv
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def visualize_results(results, num_samples, metric_name, path=None, suffix=""):
    images = results['images']
    depths = results['targets']
    mu_batch = results['mu_batch']
    sigma_batch = results['sigma_batch']
    empirical_confidence_levels = results['metrics']['empirical_confidence_levels']

    calibration_path = os.path.join(path, "calibration_plots")
    os.makedirs(calibration_path, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 5))
    confidence_levels = np.arange(0., 1.1, 0.1)
    empirical_confidence_levels = np.concatenate([np.zeros(1), empirical_confidence_levels])
    ax.plot(confidence_levels, confidence_levels, linestyle='--', color='black', zorder=-1)
    ax.plot(confidence_levels, empirical_confidence_levels, marker='o', zorder=1)
    ax.set_xlabel('Expected Confidence Level')
    ax.set_ylabel('Empirical Confidence Level')
    plt.locator_params(axis='both', nbins=3)
    plt.savefig(os.path.join(calibration_path, f"calibration_plot{suffix}.png"), dpi=300)
    plt.close()

    examples_path = os.path.join(path, "examples_plots")
    os.makedirs(examples_path, exist_ok=True)

    fig, axs = plt.subplots(num_samples, 5, figsize=(20, num_samples * 4))
    for i in range(num_samples):
        img = images[i].cpu().permute(1, 2, 0).numpy()
        ax_in = axs[i, 0]
        im0 = ax_in.imshow(img)
        im0.set_clim(0, 1)
        ax_in.set_title(f"Sample {i+1}")
        ax_in.axis("off")

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
        ax_error.set_title(f"Error")
        ax_error.axis("off")

        ax_var = axs[i, 4]
        im4 = ax_var.imshow(sigma.squeeze().cpu().numpy(), cmap="plasma")
        ax_var.set_title(f"Standard Deviation")
        ax_var.axis("off")

    plt.tight_layout()
    out_path = os.path.join(examples_path, f"eval_{metric_name}{suffix}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()


