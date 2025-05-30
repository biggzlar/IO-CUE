import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataloaders.simple_depth import DepthDataset as NYUDEPTH_dataset
from dataloaders.hypersim_depth import HyperSimDepthDataset
from dataloaders.apolloscape_depth import ApolloscapeDepthDataset

from models.base_ensemble_model import BaseEnsemble
from models.post_hoc_ensemble_model import PostHocEnsemble
from networks.unet_model import UNet, BabyUNet, MediumUNet
from predictors.gaussian import post_hoc_predict_gaussian, predict_gaussian
from predictors.bayescap import predict_bayescap
from predictors.mse import predict_mse
from evaluation.utils_ood import plot_ood_analysis
from evaluation.eval_depth_utils import load_model


def run_evaluation(id_dataset, ood_dataset, device, dataset_name):
    """Run evaluation with given ID and OOD datasets and return results."""
    _, id_test_loader = id_dataset.get_dataloaders(batch_size=128, shuffle=False)
    _, ood_test_loader = ood_dataset.get_dataloaders(batch_size=128, shuffle=False)

    base_model = load_model(
        BaseEnsemble, 
        model_path="results/pretrained/base_gaussian_ensemble.pth", 
        inference_fn=predict_gaussian, 
        model_params={"in_channels": 3, "out_channels": [1, 1], "drop_prob": 0.2}, 
        n_models=5, 
        device=device)
    
    edgy_model = load_model(
        BaseEnsemble, 
        # model_path="results/base_unet_depth_model_augmented/checkpoints/base_ensemble_model_45.pth",
        model_path="results/pretrained/base_mse_ensemble.pth",
        # model_path="results/base_unet_depth_model_very_augmented/checkpoints/base_ensemble_model_95.pth", 
        inference_fn=predict_mse, 
        model_params={"in_channels": 3, "out_channels": [1], "drop_prob": 0.2}, 
        n_models=5, 
        device=device)
    
    bayescap_model = load_model(
        PostHocEnsemble, 
        model_path="results/pretrained/bayescap.pth", 
        # model_path="results/edgy_depth_bayescap_aug/checkpoints/post_hoc_ensemble_model_best.pt", 
        inference_fn=predict_bayescap, 
        model_params={"in_channels": 1, "out_channels": [1, 1, 1], "drop_prob": 0.3}, 
        n_models=1, 
        device=device)
    
    post_hoc_gaussian_model = load_model(
        PostHocEnsemble, 
        # model_path="results/pretrained/post_hoc_gaussian_aug.pth", 
        model_path="results/edgy_depth/checkpoints/post_hoc_ensemble_model_best.pt", 
        inference_fn=post_hoc_predict_gaussian, 
        model_params={"in_channels": 4, "out_channels": [1], "drop_prob": 0.3}, 
        n_models=1, 
        device=device,
        model_class=UNet)

    id_errors_base = []
    id_errors_edgy = []

    id_sigmas_base = []
    id_sigmas_edgy = []
    id_sigmas_bayescap = []
    id_sigmas_post_hoc_gaussian = []

    ood_errors_base = []
    ood_errors_edgy = []

    ood_sigmas_base = []
    ood_sigmas_edgy = []
    ood_sigmas_bayescap = []
    ood_sigmas_post_hoc_gaussian = []

    sample_wise_reduce_indices = (1, 2, 3)

    for batch_X, batch_y in tqdm(id_test_loader, desc=f"{dataset_name} ID Inference"):
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        base_preds = base_model.predict(batch_X)
        edgy_preds = edgy_model.predict(batch_X)

        id_errors_base.append(torch.abs(base_preds['mean'] - batch_y))
        id_errors_edgy.append(torch.abs(edgy_preds['mean'] - batch_y))

        id_sigmas_base.append(base_preds['ep_sigma'])
        id_sigmas_edgy.append(edgy_preds['ep_sigma'])
        id_sigmas_bayescap.append(bayescap_model.predict(edgy_preds['mean'])['sigma'])
        id_sigmas_post_hoc_gaussian.append(post_hoc_gaussian_model.predict(batch_X, y_pred=edgy_preds['mean'])['sigma'])

    for batch_X, batch_y in tqdm(ood_test_loader, desc=f"{dataset_name} OOD Inference"):
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        base_preds = base_model.predict(batch_X)
        edgy_preds = edgy_model.predict(batch_X)

        ood_errors_base.append(torch.abs(base_preds['mean'] - batch_y))
        ood_errors_edgy.append(torch.abs(edgy_preds['mean'] - batch_y))

        ood_sigmas_base.append(base_preds['ep_sigma'])
        ood_sigmas_edgy.append(edgy_preds['ep_sigma'])
        ood_sigmas_bayescap.append(bayescap_model.predict(edgy_preds['mean'])['sigma'])
        ood_sigmas_post_hoc_gaussian.append(post_hoc_gaussian_model.predict(batch_X, y_pred=edgy_preds['mean'])['sigma'])

    all_id_results = {
        'base': torch.cat(id_sigmas_base).mean(dim=sample_wise_reduce_indices), 
        'edgy': torch.cat(id_sigmas_edgy).mean(dim=sample_wise_reduce_indices),
        'bayescap': torch.cat(id_sigmas_bayescap).mean(dim=sample_wise_reduce_indices),
        'post_hoc_gaussian': torch.cat(id_sigmas_post_hoc_gaussian).mean(dim=sample_wise_reduce_indices),
    }

    all_ood_results = {
        'base': torch.cat(ood_sigmas_base).mean(dim=sample_wise_reduce_indices), 
        'edgy': torch.cat(ood_sigmas_edgy).mean(dim=sample_wise_reduce_indices),
        'bayescap': torch.cat(ood_sigmas_bayescap).mean(dim=sample_wise_reduce_indices),
        'post_hoc_gaussian': torch.cat(ood_sigmas_post_hoc_gaussian).mean(dim=sample_wise_reduce_indices),
    }

    all_id_errors = {
        'base': torch.cat(id_errors_base).mean(dim=sample_wise_reduce_indices),
        'edgy': torch.cat(id_errors_edgy).mean(dim=sample_wise_reduce_indices),
    }

    all_ood_errors = {
        'base': torch.cat(ood_errors_base).mean(dim=sample_wise_reduce_indices),
        'edgy': torch.cat(ood_errors_edgy).mean(dim=sample_wise_reduce_indices),
    }

    # Store results
    results = {
        'id_uncertainties': all_id_results,
        'ood_uncertainties': all_ood_results,
        'id_errors': all_id_errors,
        'ood_errors': all_ood_errors,
    }
    
    # Dictionary for nicer method names in plots
    method_display_names = {
        'base': 'Ensemble',
        'bayescap': 'BayesCap',
        'post_hoc_gaussian': 'IO-CUE',
        'edgy': 'edgy'
    }
    
    # Create individual plots for each method
    roc_results = {}
    boxplot_data = {}
    
    for key in all_id_results.keys():
        if key == 'post_hoc_gaussian' or key == 'bayescap':
            _, _, fpr, tpr, roc_auc = plot_ood_analysis(
                id_uncertainties=all_id_results[key], 
                ood_uncertainties=all_ood_results[key],
                id_errors=all_id_errors['edgy'],
                ood_errors=all_ood_errors['edgy'],
                method_name=f"{method_display_names[key]} ({dataset_name})", 
                save_dir=f"results/ood_basic_analysis/{dataset_name}")
            boxplot_data[key] = {
                'id_unc': all_id_results[key].cpu().numpy(),
                'id_err': all_id_errors['edgy'].cpu().numpy(),
                'ood_unc': all_ood_results[key].cpu().numpy(),
                'ood_err': all_ood_errors['edgy'].cpu().numpy()
            }
        else:
            _, _, fpr, tpr, roc_auc = plot_ood_analysis(
                id_uncertainties=all_id_results[key], 
                ood_uncertainties=all_ood_results[key],
                id_errors=all_id_errors[key],
                ood_errors=all_ood_errors[key],
                method_name=f"{method_display_names[key]} ({dataset_name})", 
                save_dir=f"results/ood_basic_analysis/{dataset_name}")
            boxplot_data[key] = {
                'id_unc': all_id_results[key].cpu().numpy(),
                'id_err': all_id_errors[key].cpu().numpy(),
                'ood_unc': all_ood_results[key].cpu().numpy(),
                'ood_err': all_ood_errors[key].cpu().numpy()
            }
        
        # Only add non-edgy models to the combined plots
        if key != 'edgy':
            roc_results[key] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
    
    return {
        'data': results,
        'roc': roc_results,
        'boxplot_data': boxplot_data,
        'method_display_names': method_display_names
    }


if __name__ == "__main__":
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    
    # Run evaluation with NYU dataset (flipped)
    print("Running evaluation with NYU dataset (flipped)...")
    id_dataset_nyu = NYUDEPTH_dataset(img_size=(128, 160), augment=False)
    ood_dataset_nyu = NYUDEPTH_dataset(img_size=(128, 160), augment=True, flip=True)
    nyu_results = run_evaluation(id_dataset_nyu, ood_dataset_nyu, device, "NYU_flipped")
    
    # Run evaluation with Apolloscape dataset
    print("Running evaluation with Apolloscape dataset...")
    id_dataset_apollo = NYUDEPTH_dataset(img_size=(128, 160), augment=False)
    ood_dataset_apollo = ApolloscapeDepthDataset()
    apollo_results = run_evaluation(id_dataset_apollo, ood_dataset_apollo, device, "Apolloscape")
    
    # Define consistent colors for each method
    method_colors = {
        'base': 'cornflowerblue',
        'bayescap': 'mediumseagreen',
        'post_hoc_gaussian': 'darkorchid'
    }
    
    # Define ID and OOD color tints
    id_alpha = 0.9
    ood_alpha = 0.9
    
    # Create a 1x4 figure with histograms and ROC curve
    fig, axes = plt.subplots(1, 4, figsize=(20, 3))
    
    # Plot histograms for each method in the first three subplots
    methods = ['base', 'bayescap', 'post_hoc_gaussian']
    titles = ['Ensemble', 'BayesCap', 'IO-CUE']
    
    for i, (method, title) in enumerate(zip(methods, titles)):
        ax = axes[i]
        
        # Get data for both datasets
        id_data = nyu_results['boxplot_data'][method]['id_unc']  # ID data is the same for both
        nyu_ood_data = nyu_results['boxplot_data'][method]['ood_unc']
        apollo_ood_data = apollo_results['boxplot_data'][method]['ood_unc']
        
        # Plot histograms
        bins = 30
        
        # ID histogram (plot only once since it's the same for both evaluations)
        ax.hist(id_data, bins=bins, alpha=id_alpha, color=method_colors[method], 
                label='In-Distribution', density=True)
        
        # OOD histograms for both datasets
        ax.hist(nyu_ood_data, bins=bins, alpha=ood_alpha, color='darkorange', 
                label='NYU OOD', density=True, linestyle='--', linewidth=2, histtype='step')
        ax.hist(apollo_ood_data, bins=bins, alpha=ood_alpha, color='firebrick', 
                label='Apolloscape OOD', density=True, linestyle='--', linewidth=2, histtype='step')
        
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Uncertainty')
        ax.set_ylabel('Density')
    
    # Plot ROC curves in the fourth subplot
    ax = axes[3]
    
    # Plot ROC curves for both datasets
    for method in methods:
        nyu_data = nyu_results['roc'][method]
        apollo_data = apollo_results['roc'][method]
        
        # Plot both curves with the same color but different line styles
        ax.plot(nyu_data['fpr'], nyu_data['tpr'], lw=2, color=method_colors[method], 
                linestyle='--')
        ax.plot(apollo_data['fpr'], apollo_data['tpr'], lw=2, color=method_colors[method],
                label=f'{nyu_results["method_display_names"][method]} (NYU: {nyu_data["auc"]:.3f}, Apollo: {apollo_data["auc"]:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend(loc='lower right', fontsize='small')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("results/ood_basic_analysis/combined_datasets_figure.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create individual ROC plots for each dataset
    plt.figure(figsize=(10, 8))
    for method in methods:
        data = nyu_results['roc'][method]
        plt.plot(data['fpr'], data['tpr'], lw=2, color=method_colors[method],
                label=f'{nyu_results["method_display_names"][method]} (AUC = {data["auc"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for NYU Dataset')
    plt.legend()
    plt.grid(True)
    plt.savefig("results/ood_basic_analysis/nyu_roc.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 8))
    for method in methods:
        data = apollo_results['roc'][method]
        plt.plot(data['fpr'], data['tpr'], lw=2, color=method_colors[method],
                label=f'{apollo_results["method_display_names"][method]} (AUC = {data["auc"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Apolloscape Dataset')
    plt.legend()
    plt.grid(True)
    plt.savefig("results/ood_basic_analysis/apollo_roc.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Done did it!")
    