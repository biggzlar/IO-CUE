import os
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_curve, auc

def plot_ood_analysis(id_uncertainties, ood_uncertainties, id_errors, ood_errors, method_name, save_dir="results/ood_analysis"):
    """Create plots for OOD analysis of a single uncertainty method."""
    os.makedirs(save_dir, exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Histogram plot
    axs[0].hist(id_uncertainties.cpu().numpy(), bins=30, alpha=0.5, label='ID')
    axs[0].hist(ood_uncertainties.cpu().numpy(), bins=30, alpha=0.5, label='OOD')
    axs[0].set_title(f"{method_name} - Uncertainty Distribution")
    axs[0].set_xlabel("Uncertainty")
    axs[0].set_ylabel("Frequency")
    axs[0].legend()
    
    # 2. ROC curve
    # Create binary labels (0 for ID, 1 for OOD)
    labels = torch.cat([torch.zeros(len(id_uncertainties)), torch.ones(len(ood_uncertainties))])
    scores = torch.cat([id_uncertainties.cpu(), ood_uncertainties.cpu()])
    
    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(labels.numpy(), scores.numpy())
    roc_auc = auc(fpr, tpr)
    
    axs[1].plot(fpr, tpr, lw=2)
    axs[1].plot([0, 1], [0, 1], 'k--', lw=2)
    axs[1].set_xlim([0.0, 1.0])
    axs[1].set_ylim([0.0, 1.05])
    axs[1].set_xlabel('False Positive Rate')
    axs[1].set_ylabel('True Positive Rate')
    axs[1].set_title(f"{method_name} - ROC Curve (AUROC = {roc_auc:.3f})")
    
    # 3. Box plot
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