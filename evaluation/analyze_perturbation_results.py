import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams.update(
    {
        "font.size": 12,
        "text.usetex": False,
        "font.family": "stixgeneral",
        "mathtext.fontset": "stix",
    }
)

# Load results
with open('results/input_perturbation/perturbation_results.pkl', 'rb') as f:
    data = pickle.load(f)

sample_data = data.pop('data')
print("Available models:", list(data.keys()))
print("Data structure:", {k: {sk: len(v[sk]) for sk in v} for k, v in data.items()})
print("Noise levels:", sorted(set(data['base_model']['noise_levels'])))

# Group results by noise level
noise_levels = sorted(set(data['base_model']['noise_levels']))
results_by_noise = {}

for sigma in noise_levels:
    results_by_noise[sigma] = {}
    for model in data:
        results_by_noise[sigma][model] = {
            'rmses': [],
            'nlls': [],
            'avg_vars': []
        }
        
        # Get indices for this noise level
        for i, nl in enumerate(data[model]['noise_levels']):
            if nl == sigma:
                results_by_noise[sigma][model]['rmses'].append(data[model]['rmses'][i])
                results_by_noise[sigma][model]['nlls'].append(data[model]['nlls'][i])
                results_by_noise[sigma][model]['avg_vars'].append(data[model]['avg_vars'][i])
# Print average metrics per noise level
print("\nAverage RMSE per noise level:")
for sigma in noise_levels:
    print(f"Sigma {sigma}:")
    for model in data:
        avg_rmse = np.mean(results_by_noise[sigma][model]['rmses'])
        print(f"  {model}: {avg_rmse:.4f}")
    print()

print("\nAverage NLL per noise level:")
for sigma in noise_levels:
    print(f"Sigma {sigma}:")
    for model in data:
        if model == 'edgy_model':
            # Skip edgy model since it doesn't have meaningful NLL values
            continue
        avg_nll = np.mean(results_by_noise[sigma][model]['nlls'])
        print(f"  {model}: {avg_nll:.4f}")
    print()

# Create results directory
os.makedirs('results/input_perturbation/plots', exist_ok=True)
model_labels = {
    'base_model': 'Gaussian Ensemble',
    'edgy_model': 'Ensemble',
    'post_hoc_model': r'IO-CUE $g(x_{\sigma}, f(x_{\sigma}))$',
    'post_hoc_model_clean': r'IO-CUE Counterfactual $g(x_{\sigma}, f(x))$'
}

data.pop('base_model')

### PLOTTING ###
# Plot RMSE vs noise level
plt.figure(figsize=(6, 6))
for model in data:
    avg_rmses = [np.mean(results_by_noise[sigma][model]['rmses']) for sigma in noise_levels]
    plt.plot(noise_levels, avg_rmses, marker='o', label=model_labels[model])
    plt.fill_between(noise_levels, 
        avg_rmses - (np.std(results_by_noise[sigma][model]['rmses']) * np.array(noise_levels) * 10.), 
        avg_rmses + (np.std(results_by_noise[sigma][model]['rmses']) * np.array(noise_levels) * 10.), alpha=0.2)

plt.locator_params(nbins=5)
plt.xlabel('Noise Level (σ)')
plt.ylabel('RMSE')
# plt.grid(True)
plt.legend()
plt.savefig('results/input_perturbation/plots/rmse_vs_noise.png', dpi=300)

plt.figure(figsize=(6, 6))
for model in data:
    avg_nlls = [np.mean(results_by_noise[sigma][model]['nlls']) for sigma in noise_levels]
    plt.plot(noise_levels, avg_nlls, marker='o', label=model_labels[model])
    plt.fill_between(noise_levels, avg_nlls - np.std(results_by_noise[sigma][model]['nlls']), avg_nlls + np.std(results_by_noise[sigma][model]['nlls']), alpha=0.2)

plt.locator_params(nbins=5)
plt.xlabel('Noise Level (σ)')
plt.ylabel('NLL')
# plt.grid(True)
plt.legend()
plt.savefig('results/input_perturbation/plots/nll_vs_noise.png', dpi=300)

# Plot avg_vars vs noise level
plt.figure(figsize=(6, 6))
for model in data:
    avg_vars = [np.mean(results_by_noise[sigma][model]['avg_vars']) for sigma in noise_levels]
    plt.plot(noise_levels, avg_vars, marker='o', label=model_labels[model])
    plt.fill_between(noise_levels, 
        avg_vars - (np.std(results_by_noise[sigma][model]['avg_vars']) * np.array(noise_levels) * 10.), 
        avg_vars + (np.std(results_by_noise[sigma][model]['avg_vars']) * np.array(noise_levels) * 10.), alpha=0.2)

plt.locator_params(nbins=5)
plt.xlabel('Noise Level (σ)')
plt.ylabel(r'Average $\sigma$')
# plt.grid(True)
plt.legend()
plt.savefig('results/input_perturbation/plots/avg_var_vs_noise.png', dpi=300)

# Plot rmses vs avg_var
plt.figure(figsize=(6, 6))
for model in data:
    avg_rmses = [np.mean(results_by_noise[sigma][model]['rmses']) for sigma in noise_levels]
    avg_vars = [np.mean(results_by_noise[sigma][model]['avg_vars']) for sigma in noise_levels]
    plt.plot(avg_rmses, avg_vars, marker='o', label=model_labels[model])
    plt.fill_between(avg_rmses, 
        avg_vars - (np.std(results_by_noise[sigma][model]['avg_vars']) * np.array(noise_levels) * 10.), 
        avg_vars + (np.std(results_by_noise[sigma][model]['avg_vars']) * np.array(noise_levels) * 10.), alpha=0.2)

plt.locator_params(nbins=5)
plt.xlabel('RMSE')
plt.ylabel(r'Average $\sigma$')
plt.xlim(0, 0.19)
plt.ylim(0, 0.19)
# plt.grid(True)
plt.legend()
plt.savefig('results/input_perturbation/plots/rmse_vs_avg_var.png', dpi=300)

print("\nPlots saved to results/input_perturbation/plots/") 

fig, axs = plt.subplots(5, len(noise_levels), figsize=(4 * len(noise_levels), 16))

for i, sigma in enumerate(noise_levels):
    axs[0, i].imshow(sample_data['samples'][sigma][0])
    axs[1, i].imshow(data['edgy_model']['preds'][sigma][0], cmap="viridis")
    axs[2, i].imshow(np.abs(sample_data['clean_targets'][0] - data['edgy_model']['preds'][sigma][0]), cmap="rainbow")
    # axs[3, i].imshow(data['edgy_model']['pred_stds'][sigma][0])
    axs[3, i].imshow(data['post_hoc_model']['pred_stds'][sigma][0], cmap="rainbow")
    axs[4, i].imshow(data['post_hoc_model_clean']['pred_stds'][sigma][0], cmap="rainbow")

    # Add titles but as y-labels
    if i == 0:
        axs[0, i].set_ylabel('Noisy Input', fontsize=32)
        axs[1, i].set_ylabel('Prediction', fontsize=32)
        axs[2, i].set_ylabel('Error Map', fontsize=32)
        # axs[3, i].set_ylabel('Ensemble Std', fontsize=18)
        axs[3, i].set_ylabel('IO-CUE', fontsize=32)
        axs[4, i].set_ylabel('IO-CUE\nCounterfactual', fontsize=32)

# Turn off all axes
for row_idx, row in enumerate(axs):
    for col_idx, ax in enumerate(row):
        # Hide all spines, ticks and tick labels
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
            
        # Only show y-labels for the first column
        if col_idx == 0:
            ax.yaxis.label.set_visible(True)
        else:
            ax.yaxis.label.set_visible(False)

plt.tight_layout()

# Add a colorbar for uncertainty that spans the full height
cbar_ax = fig.add_axes([1., 0.03, 0.02, 0.91])  # [left, bottom, width, height]
norm = plt.Normalize(0, 0.3)  # Match the uncertainty plot range
sm = plt.cm.ScalarMappable(cmap="rainbow", norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label(r"Error / Uncertainty ($\sigma$)")


plt.savefig('results/input_perturbation/plots/perturbation_examples.png', dpi=300)

