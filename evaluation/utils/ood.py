import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_curve, auc, average_precision_score


def plot_ood_analysis(id_uncertainties, ood_uncertainties, id_errors, ood_errors, method_name, save_dir="results/ood_analysis"):
    os.makedirs(save_dir, exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    axs[0].hist(id_uncertainties.cpu().numpy(), bins=30, alpha=0.5, label='ID')
    axs[0].hist(ood_uncertainties.cpu().numpy(), bins=30, alpha=0.5, label='OOD')
    axs[0].set_title(f"{method_name} - Uncertainty Distribution")
    axs[0].set_xlabel("Uncertainty")
    axs[0].set_ylabel("Frequency")
    axs[0].legend()

    labels = torch.cat([torch.zeros(len(id_uncertainties)), torch.ones(len(ood_uncertainties))])
    scores = torch.cat([id_uncertainties.cpu(), ood_uncertainties.cpu()])

    fpr, tpr, _ = roc_curve(labels.numpy(), scores.numpy())
    roc_auc = auc(fpr, tpr)

    # Compute AUPR and FPR@95 with bootstrap 95% CIs (printed to terminal)
    def compute_fpr_at_tpr(fpr_arr, tpr_arr, target_tpr=0.95):
        # Ensure arrays are numpy and sorted by TPR increasing for interpolation
        order = np.argsort(tpr_arr)
        tpr_sorted = tpr_arr[order]
        fpr_sorted = fpr_arr[order]
        # Clip target into achievable range to avoid NaN
        target = np.clip(target_tpr, tpr_sorted[0], tpr_sorted[-1])
        return float(np.interp(target, tpr_sorted, fpr_sorted))

    y_true = labels.numpy()
    y_score = scores.numpy()
    aupr = float(average_precision_score(y_true, y_score))
    fpr95 = compute_fpr_at_tpr(fpr, tpr, target_tpr=0.95)

    # Bootstrap CIs
    rng = np.random.RandomState(42)
    n_bootstraps = 200
    aupr_boot = []
    fpr95_boot = []
    n_samples = len(y_true)
    indices = np.arange(n_samples)
    for _ in range(n_bootstraps):
        sample_idx = rng.choice(indices, size=n_samples, replace=True)
        y_b = y_true[sample_idx]
        s_b = y_score[sample_idx]
        try:
            fpr_b, tpr_b, _ = roc_curve(y_b, s_b)
            aupr_b = average_precision_score(y_b, s_b)
            fpr95_b = compute_fpr_at_tpr(fpr_b, tpr_b, target_tpr=0.95)
            if not np.isnan(aupr_b) and not np.isnan(fpr95_b):
                aupr_boot.append(aupr_b)
                fpr95_boot.append(fpr95_b)
        except Exception:
            continue
    if len(aupr_boot) > 0:
        aupr_ci_low, aupr_ci_high = np.percentile(aupr_boot, [2.5, 97.5])
    else:
        aupr_ci_low, aupr_ci_high = np.nan, np.nan
    if len(fpr95_boot) > 0:
        fpr95_ci_low, fpr95_ci_high = np.percentile(fpr95_boot, [2.5, 97.5])
    else:
        fpr95_ci_low, fpr95_ci_high = np.nan, np.nan

    print(f"{method_name} - AUPR: {aupr:.3f} (95% CI: [{aupr_ci_low:.3f}, {aupr_ci_high:.3f}])")
    print(f"{method_name} - FPR@95: {fpr95:.3f} (95% CI: [{fpr95_ci_low:.3f}, {fpr95_ci_high:.3f}])")

    axs[1].plot(fpr, tpr, lw=2)
    axs[1].plot([0, 1], [0, 1], 'k--', lw=2)
    axs[1].set_xlim([0.0, 1.0])
    axs[1].set_ylim([0.0, 1.05])
    axs[1].set_xlabel('False Positive Rate')
    axs[1].set_ylabel('True Positive Rate')
    axs[1].set_title(f"{method_name} - ROC Curve (AUROC = {roc_auc:.3f})")

    boxplot_data = [
        id_uncertainties.cpu().numpy(),
        id_errors.cpu().numpy(),
        ood_uncertainties.cpu().numpy(),
        ood_errors.cpu().numpy()
    ]
    axs[2].boxplot(boxplot_data, tick_labels=['ID Unc', 'ID Err', 'OOD Unc', 'OOD Err'])
    axs[2].set_title(f"{method_name} - Uncertainty and Error Box Plot")
    axs[2].set_ylabel("Value")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{method_name.lower().replace(' ', '_')}_ood.png"), dpi=300)
    plt.close(fig)

    return id_uncertainties, ood_uncertainties, fpr, tpr, roc_auc

__all__ = ['plot_ood_analysis']


