import os
import pickle
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from configs.config_utils import setup_result_directories
from evaluation.utils import compute_ece, compute_euc, compute_crps
from evaluation.utils import get_predictions, visualize_results

from predictors.gaussian import gaussian_nll

from models.basic_la import _precision_to_scale_tril_robust, get_hessian, _nn_predictive_samples

class PostHocLaplace(nn.Module):
    def __init__(self, mean_ensemble, filter=['upconv8'], device=None):
        """
        Ensemble of variance prediction models for post-hoc uncertainty estimation
        
        Args:
            model_class: The model class to use for the ensemble
            model_params: Dictionary of parameters to pass to the model constructor
            n_models: Number of models in the ensemble
            device: Device to run the models on
        """
        super(PostHocLaplace, self).__init__()
        self.mean_ensemble = mean_ensemble.models[0]
        self.device = device

        self.train_log = {'nll': [], 'rmse': [], 'avg_var': [], 'ece': [], 'euc': [], 'crps': [], 'p_value': []}
        self.filter = filter


    def optimize(self, results_dir, model_dir, train_loader, test_loader=None):
        clean_train_loader =  train_loader # [(sample["input"].squeeze(), sample["target"].squeeze()) for sample in train_loader]

        sigma_path = os.path.join(model_dir, 'laplace_covariance.pt')

        print("Computing Hessian and covariance matrix...")
        hessian = get_hessian(self.mean_ensemble, self.filter, clean_train_loader, device=self.device)
        sigma_L = _precision_to_scale_tril_robust(hessian)
        sigma = sigma_L @ sigma_L.T
        torch.save(sigma, sigma_path)
        print(f"Covariance matrix saved to {sigma_path}")

        self.sigma = sigma
        
        if test_loader is not None:
            results = self.evaluate(test_loader)
            
            print(f"\n LLLA - RMSE: {results['metrics']['rmse']:.4f}, NLL: {results['metrics']['nll']:.4f}, ECE: {results['metrics']['ece']:.4f}, EUC: {results['metrics']['euc']:.4f}, CRPS: {results['metrics']['crps']:.4f}")
            self.update_log(results)
            self.pickle_log(f"{results_dir}/post_hoc_ensemble_model_log.pkl")

            X, _ = next(iter(test_loader))
            if len(X.shape) == 4:
                # We can only do this specific visualization if the dataset is simple_depth
                visualize_results(results, num_samples=5, metric_name="nll", path=f"{results_dir}", suffix=f"")

            print()
        

    def evaluate(self, test_loader):
        """
        Evaluate the ensemble on a test set
        
        Args:
            test_loader: DataLoader containing test data
            mean_ensemble: The ensemble used for mean predictions
            
        Returns:
            rmse: Root Mean Squared Error
            nll: Negative Log-Likelihood
        """
        all_images = []
        all_base_means = []
        all_post_hoc_log_sigmas = []
        all_targets = []
        all_errors = []
        
        # Set base models to evaluation mode
        self.mean_ensemble.eval()
        
        # Get mean predictions from mean ensemble
        with torch.no_grad():
            for batch_X, batch_y in tqdm(test_loader):
                batch_average_indices = tuple(range(1, batch_X.ndim))
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Mean predictions from mean ensemble
                base_mean = self.mean_ensemble(batch_X)
                # base_mean = batch_preds['mean']
                all_base_means.append(base_mean)
                
                batch_post_hoc_preds = self.predict(batch_X)

                post_hoc_log_sigma = batch_post_hoc_preds['mean_log_sigma']
                
                # Store predictions and targets
                all_post_hoc_log_sigmas.append(post_hoc_log_sigma)
                all_images.append(batch_X)
                all_targets.append(batch_y)

                error_batch = torch.abs(batch_y - base_mean).square().mean(dim=batch_average_indices)
                all_errors.append(error_batch)
        
        # Combine predictions and targets
        all_base_means = torch.vstack(all_base_means)
        all_post_hoc_log_sigmas = torch.vstack(all_post_hoc_log_sigmas)
        all_images = torch.vstack(all_images)
        all_targets = torch.vstack(all_targets)
        all_errors = torch.concat(all_errors)
        
        # Calculate RMSE
        mse = all_errors
        rmse = torch.sqrt(mse)

        # Calculate NLL using predicted variances
        nll = gaussian_nll(torch.cat([all_base_means, all_post_hoc_log_sigmas], dim=1), all_targets, reduce=False)
        residuals = torch.abs(all_base_means - all_targets)
        ece, empirical_confidence_levels = compute_ece(residuals=residuals, sigma=torch.exp(all_post_hoc_log_sigmas))
        euc, p_value = compute_euc(predictions=all_base_means, uncertainties=torch.exp(all_post_hoc_log_sigmas), targets=all_targets)
        crps = compute_crps(predictions=all_base_means, uncertainties=torch.exp(all_post_hoc_log_sigmas), targets=all_targets)

        # print(f"{self.is_bayescap}, NLL: {nll.mean().detach().item():.4f}, ECE: {ece:.4f}, EUC: {euc:.4f}")
        metrics = {
            'nll': nll.mean().detach().cpu(),
            'rmse': rmse.mean().detach().cpu(),
            'avg_var': torch.exp(post_hoc_log_sigma).mean().detach().cpu(),
            'empirical_confidence_levels': empirical_confidence_levels,
            'ece': ece,
            'euc': euc,
            'p_value': p_value,
            'crps': crps.mean().detach().cpu()
        }
        
        return {
            'images': all_images[-5:, ...].detach().cpu(),
            'targets': all_targets[-5:, ...].detach().cpu(),
            'errors': all_errors[-5:, ...].detach().cpu(),
            'mu_batch': all_base_means[-5:, ...].detach().cpu(),
            'sigma_batch': torch.exp(all_post_hoc_log_sigmas[-5:, ...]).detach().cpu(),
            'nll_batch': nll[-5:, ...].mean(dim=batch_average_indices).detach().cpu(),
            'rmse_batch': rmse[-5:, ...].detach().cpu(),
            'avg_var_batch': torch.exp(all_post_hoc_log_sigmas)[-5:, ...].mean(dim=batch_average_indices).detach().cpu(),
            
            'all_sigmas': torch.exp(all_post_hoc_log_sigmas),
            'metrics': metrics,
        }
    
    def predict(self, X, y_pred=None):
        fs = _nn_predictive_samples(self.mean_ensemble, X=X, posterior_covariance=self.sigma, filter=self.filter)
        sigma = fs.var(dim=0).sqrt()
        mean_log_sigma = sigma.log()
        return {'sigma': sigma, 
                'mean_log_sigma': mean_log_sigma, 
                'var_of_var': 0 # var_of_var
        }

    def update_log(self, results, max_keep=16):
        for key in self.train_log.keys():
            self.train_log[key].append(results['metrics'][key])
            if len(self.train_log[key]) > max_keep:
                self.train_log[key].pop(0)

    def pickle_log(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.train_log, f)


if __name__=="__main__":
    import os
    import torch
    import numpy as np
    import pickle
    from dataloaders.simple_depth import DepthDataset as NYUDepthDataset
    from models import BaseEnsemble
    from networks.unet_model import UNet
    from predictors.mse import mse, predict_mse
    from predictors.edge_aware_losses import edge_aware_mse_loss

    # Set random seeds for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Model configuration
    AUGMENTS = {
        "flip": False,
        "colorjitter": False,
        "gaussianblur": False,
        "grayscale": False
    }
    N_MODELS = 5  # Number of models in the ensemble
    N_EPOCHS = 64
    EVAL_FREQ = 5

    # Dataset configuration
    IMG_SIZE = (128, 160)
    TRAIN_SPLIT = 1.0
    BATCH_SIZE = 32

    # Model parameters
    MODEL_CLASS = UNet
    MODEL_PARAMS = {
        "in_channels": 3,
        "out_channels": [1],
        "drop_prob": 0.2
    }

    # Optimizer configuration
    OPTIMIZER_TYPE = "Adam"
    OPTIMIZER_PARAMS = {
        "lr": 5.0e-5,
        "weight_decay": 1.0e-5
    }

    # Scheduler configuration
    SCHEDULER_TYPE = "CosineAnnealingLR"
    SCHEDULER_PARAMS = {
        "T_max": N_EPOCHS
    }

    # Loss function
    CRITERION = edge_aware_mse_loss
    # Create result directories
    model_dir, results_dir = setup_result_directories("laplace")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Save configuration for reproducibility
    config = locals().copy()
    # with open(os.path.join(results_dir, "configuration.pkl"), "wb") as f:
    #     pickle.dump(config, f)
    
    # Load dataset
    dataset = NYUDepthDataset(img_size=IMG_SIZE, augment=True, **AUGMENTS)
    train_loader, test_loader = dataset.get_dataloaders(batch_size=32)
    print(f"Dataset loaded: {len(train_loader)} training batches, {len(test_loader)} test batches")
    
    # Create model
    print(f"Creating base ensemble with {N_MODELS} models...")
    mean_ensemble = BaseEnsemble(
        model_class=MODEL_CLASS,
        model_params=MODEL_PARAMS,
        n_models=N_MODELS,
        device=DEVICE,
        infer=predict_mse
    )
    mean_ensemble_path = "results/pretrained/base_ensemble_model_best.pth"
    mean_ensemble.load(mean_ensemble_path)
    print(f"Loaded mean ensemble from {mean_ensemble_path}")

    LA = PostHocLaplace(
        mean_ensemble=mean_ensemble,
        device=DEVICE,
    )

    LA.optimize(results_dir=results_dir, model_dir=model_dir, train_loader=train_loader, test_loader=test_loader)
