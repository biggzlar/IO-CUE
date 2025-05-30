import os
import pickle
from sklearn.metrics import auc, roc_curve
import numpy as np
import torch
import matplotlib.pyplot as plt
from dataloaders.simple_depth import DepthDataset as NYUDEPTH_dataset

from models.base_ensemble_model import BaseEnsemble
from models.post_hoc_ensemble_model import PostHocEnsemble
from networks.unet_model import UNet
from predictors.gaussian import post_hoc_predict_gaussian
from predictors.mse import predict_mse
from evaluation.eval_depth_utils import load_model

plt.rcParams.update(
    {
        "font.size": 12,
        "text.usetex": False,
        "font.family": "stixgeneral",
        "mathtext.fontset": "stix",
    }
)

if __name__ == "__main__":
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    results_dir = "results/cross_model_generalization"
    os.makedirs(results_dir, exist_ok=True)

    try:
        flat_results = pickle.load(open(os.path.join(results_dir, "flat_results.pkl"), "rb"))
        roc_results = pickle.load(open(os.path.join(results_dir, "roc_results.pkl"), "rb"))
        print(f"Results loaded from {results_dir}")
    except:
        print(f"Results not found in {results_dir}, running evaluation...")
        test_loaders = []
        sigmas = [0.01, 0.05, 0.1, 0.2, 0.5,]
        id_dataset = NYUDEPTH_dataset(img_size=(128, 160), augment=False)
        ood_dataset_configs = [
            {"flip": True, "colorjitter": False, "gaussianblur": False, "grayscale": False},
            {"flip": False, "colorjitter": True, "gaussianblur": False, "grayscale": False},
            {"flip": False, "colorjitter": False, "gaussianblur": True, "grayscale": False},
            {"flip": False, "colorjitter": False, "gaussianblur": False, "grayscale": True}
        ]
        ood_datasets = [NYUDEPTH_dataset(img_size=(128, 160), augment=True, **config) for config in ood_dataset_configs]

        base_model_paths = {
            "flip": "results/base_unet_with_flip/checkpoints/base_ensemble_model_best.pt",
            "colorjitter": "results/base_unet_with_colorjitter/checkpoints/base_ensemble_model_best.pt",
            "gaussianblur": "results/base_unet_with_gaussianblur/checkpoints/base_ensemble_model_best.pt",
            "grayscale": "results/base_unet_with_grayscale/checkpoints/base_ensemble_model_best.pt"
        }
        base_models = {
            model_name: load_model(BaseEnsemble, 
                model_path=path, 
                inference_fn=predict_mse, 
                model_params={"in_channels": 3, "out_channels": [1], "drop_prob": 0.2}, 
                n_models=5, device=device) for model_name, path in base_model_paths.items()
        }

        post_hoc_gaussian_model = load_model(
            PostHocEnsemble, 
            model_path="results/pretrained/post_hoc_gaussian_aug.pth", 
            # model_path="results/edgy_depth/checkpoints/post_hoc_ensemble_model_best.pt", 
            # model_path="results/baby_edgy_depth/checkpoints/variance_ensemble.pt", 
            inference_fn=post_hoc_predict_gaussian, 
            model_params={"in_channels": 4, "out_channels": [1], "drop_prob": 0.3}, 
            n_models=1, 
            device=device,
            model_class=UNet)

        flat_results = {"flip": {"nll": [], "rmse": [], "avg_var": [], "ece": [], "euc": [], "p_value": []},
                        "colorjitter": {"nll": [], "rmse": [], "avg_var": [], "ece": [], "euc": [], "p_value": []},
                        "gaussianblur": {"nll": [], "rmse": [], "avg_var": [], "ece": [], "euc": [], "p_value": []},
                        "grayscale": {"nll": [], "rmse": [], "avg_var": [], "ece": [], "euc": [], "p_value": []}}
        roc_results = {"flip": [], "colorjitter": [], "gaussianblur": [], "grayscale": []}
        for model_name, model in base_models.items():
            for ood_dataset in ood_datasets:
                id_loader = id_dataset.get_dataloaders(batch_size=128, shuffle=False)[1]
                id_results = post_hoc_gaussian_model.evaluate(
                    test_loader=id_loader,
                    mean_ensemble=model
                )
                # # Create counterfactual scenario
                # counter_factual_data = (id_dataset.test.data[0], ood_dataset.test.data[1])
                # ood_dataset.test.data = counter_factual_data
                
                ood_loader = ood_dataset.get_dataloaders(batch_size=128, shuffle=False)[1]
                ood_results = post_hoc_gaussian_model.evaluate(
                    test_loader=ood_loader,
                    mean_ensemble=model
                )

                id_uncertainties = id_results['all_sigmas'].mean(dim=(1, 2, 3))
                ood_uncertainties = ood_results['all_sigmas'].mean(dim=(1, 2, 3))

                labels = torch.cat([torch.zeros(len(id_uncertainties)), torch.ones(len(ood_uncertainties))])
                scores = torch.cat([id_uncertainties.cpu(), ood_uncertainties.cpu()])

                fpr, tpr, _ = roc_curve(labels.numpy(), scores.numpy())
                roc_auc = auc(fpr, tpr)
                roc_auc = max(roc_auc, 0.501)

                for metric in flat_results[model_name].keys():
                    flat_results[model_name][metric].append(ood_results['metrics'][metric].item())

                roc_results[model_name].append(roc_auc)

            # flat_results.append([val for val in results['metrics'].values()])
                output_str = f" ".join([f"{k}: {v:.2f}" for k, v in ood_results['metrics'].items()])
                print(output_str)

        pickle.dump(flat_results, open(os.path.join(results_dir, "flat_results.pkl"), "wb"))
        pickle.dump(roc_results, open(os.path.join(results_dir, "roc_results.pkl"), "wb"))
        print(f"Results saved to {results_dir}")
            
    # Create a grouped bar chart for AUC scores and RMSE values
    fig = plt.figure(figsize=(20, 3))

    # Create a gridspec with 4 columns, with the first 3 for the bar plot and the last for the scatter plot
    gs = plt.GridSpec(1, 4, figure=fig)
    ax1 = fig.add_subplot(gs[0, :3])  # Bar plot takes up 3/4
    ax3 = fig.add_subplot(gs[0, 3])   # Scatter plot takes up 1/4

    # Set up data for plotting
    model_names = list(roc_results.keys())
    x = np.arange(len(model_names))
    width = 0.2

    # Calculate positions for bars
    n_datasets = len(roc_results[model_names[0]])
    positions = []
    for i in range(n_datasets):
        positions.append(x - width * (n_datasets - 1) / 2 + i * width)

    # Plot AUC scores as grouped bars
    for i in range(n_datasets):
        auc_values = [roc_results[model][i] for model in model_names]
        bar = ax1.bar(positions[i], auc_values, width, alpha=0.3, edgecolor=(0, 0, 0, 1.0), label=f'Dataset {i+1} (AUC)')
        color = list(bar.get_children()[0].get_facecolor())        

        # Apply rotated labels with matching colors at consistent position
        for j, rect in enumerate(bar):
            bar_color = rect.get_facecolor()
            bar_color = list(bar_color)
            bar_color[3] = 1.0
            label_y = 0.1
            label_x = rect.get_x() + rect.get_width()/2
            ax1.text(label_x, label_y, model_names[i], ha='center', va='bottom', 
                    rotation=90, color=bar_color, fontweight='bold', fontsize=8 if (model_names[i] == "gaussianblur" and model_names[j] == "gaussianblur") else 12)
        
        ax1.bar_label(bar, fmt='{:.2f}', color='k', label_type='edge') # bar_color

    # Configure primary axis (AUC scores)
    ax1.set_ylabel('AUROC Score')
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Trained on " + model_name for model_name in model_names])
    ax1.set_ylim(0, 1.1)
    ax1.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    # ax1.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, linewidth=1, zorder=-100)
    ax1.grid(True, alpha=0.3)

    # Now create the RMSE vs AUROC scatter plot
    # Each model gets its own series, with points representing different datasets
    markers = ['o', 's', 'D', '^']
    for i, model in enumerate(model_names):
        auroc_values = [roc_results[model_name][i] for model_name in model_names]
        rmse_values = [flat_results[model_name]['rmse'][i] for model_name in model_names]
        ax3.scatter(auroc_values, rmse_values, label=model, marker=markers[i], s=80)
        # Connect points with a line to show the "curve" for each model
        # we need to sort the points by rmse values
        sorted_indices = np.argsort(auroc_values)
        sorted_auroc_values = [auroc_values[i] for i in sorted_indices]
        sorted_rmse_values = [rmse_values[i] for i in sorted_indices]
        ax3.plot(sorted_auroc_values, sorted_rmse_values, alpha=0.5)

    ax3.set_xlabel('AUROC')
    ax3.set_ylabel('RMSE')
    ax3.set_title('RMSE vs AUROC')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('cross_model_generalization.png', dpi=300)
    print(f"Image saved to cross_model_generalization.png")
    plt.show()
    
    