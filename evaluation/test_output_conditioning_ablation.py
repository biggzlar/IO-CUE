import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
import torchvision.transforms as transforms

from dataloaders.simple_depth import DepthDataset as NYUDEPTH_dataset
from dataloaders.simple_depth import _DepthDataset, load_depth
from dataloaders.apolloscape_depth import ApolloscapeDepthDataset

from models.base_ensemble_model import BaseEnsemble
from models.post_hoc_frameworks import IOCUE, ICUE, InputFisher
from networks.unet_model import UNet

from predictors.gaussian import predict_gaussian
from predictors.mse import predict_mse

from evaluation.utils import load_mean_model, load_variance_model


from evaluation.utils import ensure_dir


def load_models(device):
    # Gaussian MLE single model (predicts mean + log_sigma)
    gaussian_model = load_mean_model(
        BaseEnsemble,
        model_path="results/base_unet_cosine_annealing_256_flip_grayscale_gaussian_single/checkpoints/base_ensemble_best.pth",
        inference_fn=predict_gaussian,
        model_params={"in_channels": 3, "out_channels": [2], "drop_prob": 0.2},
        n_models=1,
        device=device,
        model_class=UNet,
    )

    # Base ensemble (MSE mean predictions, epistemic from spread)
    ensemble_model = load_mean_model(
        BaseEnsemble,
        model_path="results/base_unet_cosine_annealing_256_flip_grayscale/checkpoints/base_ensemble_best.pth",
        # model_path="results/base_unet_cosine_annealing_256_noaug/checkpoints/base_ensemble_best.pth",
        inference_fn=predict_mse,
        model_params={"in_channels": 3, "out_channels": [1], "drop_prob": 0.2},
        n_models=5,
        device=device,
        model_class=UNet,
    )

    iocue_model = load_variance_model(
        mean_ensemble=ensemble_model,
        model_type=IOCUE,
        model_path="results/edgy_depth_gaussian_io_cue/checkpoints/post_hoc_ensemble_model_best.pt",
        model_params={"in_channels": 4, "out_channels": [1], "drop_prob": 0.2},
        n_models=1,
        device=device,
        model_class=UNet,
    )

    icue_model = load_variance_model(
        mean_ensemble=ensemble_model,
        model_type=ICUE,
        model_path="results/edgy_depth_gaussian_icue/checkpoints/post_hoc_ensemble_model_best.pt",
        model_params={"in_channels": 3, "out_channels": [1], "drop_prob": 0.2},
        n_models=1,
        device=device,
        model_class=UNet,
    )

    # Input Fisher (gradient-based, no training)
    input_fisher_model = InputFisher(mean_ensemble=ensemble_model, gaussian_model=gaussian_model, device=device, num_probe_vectors=8)

    return gaussian_model, ensemble_model, iocue_model, icue_model, input_fisher_model


from evaluation.utils.data import get_augmented_nyu_test_loader


def compute_results(loader, device, models, desc="Inference"):
    """Compute errors and uncertainties for a given loader."""
    gaussian_model, ensemble_model, iocue_model, icue_model, input_fisher_model = models
    
    errors = {"gaussian": [], "ensemble": []}
    uncs = {"gaussian": [], "ensemble": [], "iocue": [], "icue": [], "inpfisher": []}
    sample_reduce = (1, 2, 3)
    
    for batch_X, batch_y in tqdm(loader, desc=desc):
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        
        gaussian_preds = gaussian_model.predict(batch_X)
        ensemble_preds = ensemble_model.predict(batch_X)
        
        errors["gaussian"].append(torch.abs(gaussian_preds["mean"] - batch_y))
        errors["ensemble"].append(torch.abs(ensemble_preds["mean"] - batch_y))
        
        uncs["gaussian"].append(gaussian_preds["al_sigma"])  # aleatoric from Gaussian
        uncs["ensemble"].append(ensemble_preds["ep_sigma"])  # epistemic from ensemble spread
        uncs["iocue"].append(iocue_model.predict(batch_X, y_pred=ensemble_preds["mean"])['sigma'])
        uncs["icue"].append(icue_model.predict(batch_X)['sigma'])
        uncs["inpfisher"].append(input_fisher_model.predict(batch_X, y_pred=ensemble_preds["mean"])['sigma'])
    
    errors = {k: torch.cat(v).mean(dim=sample_reduce) for k, v in errors.items()}
    uncs = {k: torch.cat(v).mean(dim=sample_reduce) for k, v in uncs.items()}
    return errors, uncs


def compute_roc_auc(id_uncs, ood_uncs):
    """Compute ROC AUC for OOD detection."""
    roc_results = {}
    for key in ["gaussian", "ensemble", "iocue", "icue", "inpfisher"]:
        labels = torch.cat([torch.zeros(len(id_uncs[key])), torch.ones(len(ood_uncs[key]))])
        scores = torch.cat([id_uncs[key].cpu(), ood_uncs[key].cpu()])
        fpr, tpr, _ = roc_curve(labels.numpy(), scores.numpy())
        roc_auc = auc(fpr, tpr)
        roc_results[key] = {"fpr": fpr, "tpr": tpr, "auc": roc_auc}
    return roc_results


def eval_basic_ood(device, save_root):
    """Evaluate basic OOD scenarios (NYU flip, Apolloscape)."""
    models = load_models(device)
    
    # Load ID dataset (un-augmented NYU)
    id_dataset_nyu = NYUDEPTH_dataset(img_size=(128, 160), augment=False)
    _, id_loader = id_dataset_nyu.get_dataloaders(batch_size=128, shuffle=False)
    id_errors, id_uncs = compute_results(id_loader, device, models, "ID Inference")
    
    # NYU flipped OOD
    ood_loader_nyu = get_augmented_nyu_test_loader(
        img_size=(128, 160), batch_size=128, 
        augment_kwargs={"flip": True}, 
        attr_overrides={"flip_prob": 1.0}, 
        shuffle=False
    )
    ood_errors_nyu, ood_uncs_nyu = compute_results(ood_loader_nyu, device, models, "OOD Inference (NYU_flip)")
    roc_results_nyu = compute_roc_auc(id_uncs, ood_uncs_nyu)
    
    # Apolloscape OOD
    apollo_dataset = ApolloscapeDepthDataset()
    _, ood_loader_apollo = apollo_dataset.get_dataloaders(batch_size=128, shuffle=False)
    ood_errors_apollo, ood_uncs_apollo = compute_results(ood_loader_apollo, device, models, "OOD Inference (Apolloscape)")
    roc_results_apollo = compute_roc_auc(id_uncs, ood_uncs_apollo)
    
    return {
        "NYU_flip": {
            "id_unc": id_uncs, "ood_unc": ood_uncs_nyu, 
            "id_err": id_errors, "ood_err": ood_errors_nyu, 
            "roc": roc_results_nyu
        },
        "Apolloscape": {
            "id_unc": id_uncs, "ood_unc": ood_uncs_apollo, 
            "id_err": id_errors, "ood_err": ood_errors_apollo, 
            "roc": roc_results_apollo
        }
    }


def eval_nyu_augmentations(device, save_root):
    """Evaluate NYU augmentations as OOD."""
    models = load_models(device)
    
    # Load ID dataset (un-augmented NYU) - compute fresh for consistency
    id_dataset_nyu = NYUDEPTH_dataset(img_size=(128, 160), augment=False)
    _, id_loader = id_dataset_nyu.get_dataloaders(batch_size=128, shuffle=False)
    id_errors, id_uncs = compute_results(id_loader, device, models, "ID Inference")
    
    augment_settings = [
        ("grayscale", dict(grayscale=True), {}),
        ("colorjitter", dict(colorjitter=True), {}),
        ("gaussianblur", dict(gaussianblur=True), {}),
    ]
    
    results = {}
    for name, aug_kwargs, attr_overrides in augment_settings:
        ood_loader = get_augmented_nyu_test_loader(
            img_size=(128, 160), batch_size=128, 
            augment_kwargs=aug_kwargs, 
            attr_overrides=attr_overrides, 
            shuffle=False
        )
        ood_errors, ood_uncs = compute_results(ood_loader, device, models, f"OOD Inference (NYU_{name})")
        roc_results = compute_roc_auc(id_uncs, ood_uncs)
        results[f"NYU_{name}"] = {
            "id_unc": id_uncs, "ood_unc": ood_uncs, 
            "id_err": id_errors, "ood_err": ood_errors, 
            "roc": roc_results
        }
    
    return results


def create_roc_plots(all_results, save_root):
    """Create ROC curve plots."""
    methods = ["gaussian", "ensemble", "iocue", "icue", "inpfisher"]
    method_display = {
        "gaussian": "Gaussian MLE", 
        "ensemble": "Ensemble", 
        "iocue": "IO-CUE", 
        "icue": "I-CUE", 
        "inpfisher": "Input-Fisher"
    }
    method_colors = {
        "gaussian": "tab:orange", 
        "ensemble": "cornflowerblue", 
        "iocue": "darkorchid", 
        "icue": "darkgreen", 
        "inpfisher": "crimson"
    }
    
    datasets = list(all_results.keys())
    n_cols = len(datasets)
    n_rows = len(methods) + 1  # +1 for combined plot
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows), squeeze=False)
    
    # Individual method plots
    for r, method in enumerate(methods):
        for c, ds in enumerate(datasets):
            ax = axes[r][c]
            roc = all_results[ds]['roc'].get(method)
            if roc is not None:
                ax.plot(roc["fpr"], roc["tpr"], lw=2, color=method_colors[method])
                ax.plot([0, 1], [0, 1], 'k--', lw=1)
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                if c == 0:
                    ax.set_ylabel('TPR')
                if r == 0:
                    ax.set_title(ds)
                ax.text(0.6, 0.1, f"AUC={roc['auc']:.3f}", transform=ax.transAxes)
            else:
                ax.axis('off')
        
        # Add method label
        axes[r][0].annotate(method_display[method], xy=(0, 0.5), 
                           xytext=(-axes[r][0].yaxis.labelpad - 20, 0),
                           xycoords=axes[r][0].yaxis.label, textcoords='offset points',
                           size='medium', ha='right', va='center', rotation=90)
    
    # Combined plot (all methods together)
    for c, ds in enumerate(datasets):
        ax = axes[-1][c]
        for method in methods:
            roc = all_results[ds]['roc'].get(method)
            if roc is not None:
                ax.plot(roc["fpr"], roc["tpr"], lw=2, color=method_colors[method], 
                       label=method_display[method])
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('FPR')
        if c == 0:
            ax.set_ylabel('TPR')
        ax.legend(fontsize='x-small')
    
    axes[-1][0].annotate("All methods", xy=(0, 0.5), 
                        xytext=(-axes[-1][0].yaxis.labelpad - 20, 0),
                        xycoords=axes[-1][0].yaxis.label, textcoords='offset points',
                        size='medium', ha='right', va='center', rotation=90)
    
    plt.suptitle("ROC Curves - OOD Detection", fontsize=16)
    plt.tight_layout()
    ensure_dir(save_root)
    plt.savefig(os.path.join(save_root, "roc_curves.png"), dpi=300, bbox_inches='tight')
    plt.close()


def create_boxplots(all_results, save_root):
    """Create boxplot comparison."""
    methods = ["gaussian", "ensemble", "iocue", "icue"]  # Exclude input fisher
    method_display = {
        "gaussian": "Gaussian MLE", 
        "ensemble": "Ensemble", 
        "iocue": "IO-CUE", 
        "icue": "I-CUE"
    }
    method_colors = {
        "gaussian": "tab:orange", 
        "ensemble": "cornflowerblue", 
        "iocue": "darkorchid", 
        "icue": "darkgreen"
    }
    
    datasets = list(all_results.keys())
    n_cols = len(datasets)
    fig, axes = plt.subplots(1, n_cols, figsize=(4*n_cols, 4), squeeze=False)
    
    def remove_outliers(data, factor=1.5):
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr
        return data[(data >= lower_bound) & (data <= upper_bound)]
    
    for c, ds in enumerate(datasets):
        ax = axes[0][c]
        data = []
        colors = []
        tick_positions = []
        tick_labels = []
        
        # Get base errors (ensemble) for ID and OOD
        id_err_base = all_results[ds]['id_err']['ensemble'].cpu().numpy()
        ood_err_base = all_results[ds]['ood_err']['ensemble'].cpu().numpy()
        
        pos = 1
        for method in methods:
            id_unc = all_results[ds]['id_unc'][method].cpu().numpy()
            ood_unc = all_results[ds]['ood_unc'][method].cpu().numpy()
            
            # Clean data
            id_unc_clean = remove_outliers(id_unc)
            ood_unc_clean = remove_outliers(ood_unc)
            id_err_clean = remove_outliers(id_err_base)
            ood_err_clean = remove_outliers(ood_err_base)
            
            # Append: ID Unc, ID Err, OOD Unc, OOD Err
            data.extend([id_unc_clean, id_err_clean, ood_unc_clean, ood_err_clean])
            colors.extend([method_colors[method]] * 4)
            tick_positions.append(pos + 1.5)
            tick_labels.append(method_display[method])
            pos += 4
        
        # Create boxplot
        bp = ax.boxplot(data, positions=list(range(1, len(data)+1)), patch_artist=True, widths=0.8)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        
        # Add separators between methods
        for sep in tick_positions:
            ax.axvline(x=sep+2, color='k', alpha=0.05)
        
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=90)
        ax.set_title(ds)
        ax.set_ylabel('Value')
        ax.grid(True, axis='y', alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_root, "boxplots.png"), dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_root = "results/ood_output_conditioning_ablation"
    
    print("Running basic OOD evaluation (NYU flipped, Apolloscape)...")
    basic_results = eval_basic_ood(device, save_root)
    
    print("Running NYU OOD evaluation across augmentations...")
    aug_results = eval_nyu_augmentations(device, save_root)
    
    # Combine results
    all_results = {**basic_results, **aug_results}
    
    print("Creating ROC plots...")
    create_roc_plots(all_results, save_root)
    
    print("Creating boxplots...")
    create_boxplots(all_results, save_root)
    
    print("Done. Results saved to", save_root)