"""
Utility functions for managing experiment configurations.
"""
import os
import string
import random
import torch
import torch.nn as nn
import yaml
from networks.simple_regression_model import SimpleRegressionModel
from networks.unet_model import UNet
from dataloaders.simple_depth import DepthDataset
from predictors.gaussian import post_hoc_predict_gaussian, gaussian_nll, gaussian_nll_detached, predict_gaussian
from predictors.edge_aware_losses import edge_aware_gaussian_nll_loss_detached, edge_aware_mse_loss, edge_aware_gaussian_nll_loss
from predictors.mse import predict_mse, rmse
from predictors.bayescap import predict_bayescap, bayescap_loss
from predictors.generalized_gaussian import post_hoc_predict_gen_gaussian, gen_gaussian_nll

from models import BaseEnsemble
from models.post_hoc_frameworks import IOCUE, BayesCap

def load_yaml_config(file_path, device):
    """
    Load configuration from a YAML file.
    
    Args:
        file_path (str): Path to the YAML file
        
    Returns:
        dict: Configuration dictionary or None if loading fails
    """
    # If path doesn't include the full path, assume it's in configs/yaml_configs/
    if '/' not in file_path:
        file_path = f"configs/yaml_configs/{file_path}"
    
    if not file_path.endswith('.yaml') and not file_path.endswith('.yml'):
        file_path += '.yaml'
    
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert the loaded YAML to a fully-functioning config
    config = resolve_config_references(config, device)
    return config
    # try:
    #     with open(file_path, 'r') as f:
    #         config = yaml.safe_load(f)
        
    #     # Convert the loaded YAML to a fully-functioning config
    #     config = resolve_config_references(config)
    #     return config
    # except Exception as e:
    #     print(f"Error loading YAML configuration: {str(e)}")
    #     return None

def resolve_config_references(config, device):
    """
    Resolve string references to Python objects in the configuration.
    
    Args:
        config (dict): Configuration dictionary with string references
        
    Returns:
        dict: Configuration dictionary with resolved Python objects
    """
    # Create a copy of the config to avoid modifying the original
    resolved_config = config.copy()
    
    # Resolve model classes
    # For backward compatibility, check both model_class and mean_model_class
    if 'model_class' in config:
        model_class_name = config['model_class']
        resolved_config['mean_model_class'] = resolve_model_class(model_class_name)
        resolved_config['variance_model_class'] = resolve_model_class(model_class_name)
    
    # If separate model classes are specified, they take precedence
    if 'mean_model_class' in config:
        model_class_name = config['mean_model_class']
        resolved_config['mean_model_class'] = resolve_model_class(model_class_name)
    
    if 'variance_model_class' in config:
        model_class_name = config['variance_model_class']
        resolved_config['variance_model_class'] = resolve_model_class(model_class_name)
    
    # Resolve predictor
    if 'mean_predictor' in config:
        predictor_name = config['mean_predictor']
        resolved_config['mean_predictor'] = resolve_predictor(predictor_name)

    # Resolve criterion for mean model
    if 'mean_criterion' in config:
        criterion_name = config['mean_criterion']
        resolved_config['mean_criterion'] = resolve_criterion(criterion_name)
    
    # Generate dataloaders from dataset attributes if present
    if 'dataset_attrs' in config:
        dataset_name = config.get('dataset_name')
        resolved_config.update(resolve_dataset(dataset_name, config['dataset_attrs'], config['batch_size']))

    # Add device
    if 'device' not in resolved_config:
        resolved_config['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Handle scheduler configurations
    # For mean scheduler
    if 'mean_scheduler_type' in config:
        if config['mean_scheduler_type'] is None:
            resolved_config['mean_scheduler_type'] = None
            resolved_config['mean_scheduler_params'] = None
        elif 'mean_scheduler_params' in config:
            resolved_config['mean_scheduler_params']['T_max'] = config['n_epochs']
    
    # For variance scheduler
    if 'variance_scheduler_type' in config:
        if config['variance_scheduler_type'] is None:
            resolved_config['variance_scheduler_type'] = None
            resolved_config['variance_scheduler_params'] = None
        elif 'variance_scheduler_params' in config:
            resolved_config['variance_scheduler_params']['T_max'] = config['n_epochs']

    # Resolve mean model
    resolved_config['mean_model'] = BaseEnsemble(
        model_class=resolved_config['mean_model_class'],
        model_params=resolved_config['mean_model_params'],
        n_models=resolved_config['n_ensemble_models'], 
        device=device,
        infer=resolved_config['mean_predictor']
    )

    # Resolve variance model
    variance_framework = resolve_framework(config.get('variance_framework'))
    resolved_config['variance_model'] = variance_framework(
        mean_ensemble=resolved_config['mean_model'],
        model_class=resolved_config['variance_model_class'],
        model_params=resolved_config['variance_model_params'],
        n_models=resolved_config['n_variance_models'], 
        device=device,
    )
    
    return resolved_config

def resolve_framework(variance_framework_name):
    if variance_framework_name == "iocue" or variance_framework_name == "io-cue":
        return IOCUE
    elif variance_framework_name == "bayescap" or variance_framework_name == "bayes-cap":
        return BayesCap
    else:
        raise ValueError(f"Unknown framework: {variance_framework_name}")
    

def resolve_predictor(predictor_name):
    """
    Resolve predictor from string name.
    
    Args:
        predictor_name (str): Name of the predictor
        
    Returns:
        function: Predictor function
    """
    if predictor_name == "predict_mse":
        return predict_mse
    elif predictor_name == "predict_gaussian":
        return predict_gaussian
    elif predictor_name == "post_hoc_predict_gaussian":
        return post_hoc_predict_gaussian
    elif predictor_name == "predict_bayescap":
        return predict_bayescap
    elif predictor_name == "post_hoc_predict_gen_gaussian":
        return post_hoc_predict_gen_gaussian
    else:
        raise ValueError(f"Unknown predictor: {predictor_name}")

def resolve_dataset(dataset_name, dataset_attrs, batch_size):
    """
    Resolve dataset based on name and attributes.
    
    Args:
        dataset_name (str): Name of the dataset
        dataset_attrs (dict): Dataset-specific attributes
        batch_size (int): Batch size for dataloaders
        
    Returns:
        dict: Dictionary containing train_loader, test_loader, and other dataset info
    """
    result = {}
    
    if dataset_name == "simple_depth":
        # Load depth dataset
        dataset_path = dataset_attrs.get('dataset_path', None)
        dataset = DepthDataset(path=dataset_path, 
            img_size=dataset_attrs.get('img_size', (128, 160)),
            augment=dataset_attrs.get('augment', False),
            train_split=dataset_attrs.get('train_split', 1.0),
            **dataset_attrs.get('augmentations', {}))
        
        # Get train and test loaders
        train_loader, test_loader = dataset.get_dataloaders(
            batch_size=batch_size,
            shuffle=dataset_attrs.get('shuffle', True)
        )
        
        print("Dataset augment: ", dataset.augment)
        print(dataset.colorjitter, dataset.gaussianblur, dataset.augment)
        print("Dataset train split: ", dataset.train_split_idx)
        print("Dataset train length: ", len(train_loader))
        result['train_loader'] = train_loader
        result['test_loader'] = test_loader
        # Use a default noise level for depth data
        result['noise_level'] = dataset_attrs.get('noise_level', 0.1)
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return result

def resolve_model_class(model_class_name):
    """
    Resolve model class from string name.
    
    Args:
        model_class_name (str): Name of the model class
        
    Returns:
        class: Model class
    """
    if model_class_name == "SimpleRegressionModel":
        return SimpleRegressionModel
    elif model_class_name == "UNet":
        return UNet
    else:
        raise ValueError(f"Unknown model class: {model_class_name}")

def resolve_criterion(criterion_name):
    """
    Resolve loss function from string name.
    
    Args:
        criterion_name (str): Name of the criterion
        
    Returns:
        function: Loss function
    """
    if criterion_name == "MSELoss":
        return nn.MSELoss()
    elif criterion_name == "gaussian_nll_loss":
        return gaussian_nll
    elif criterion_name == "L1Loss":
        return nn.L1Loss()
    elif criterion_name == "gaussian_nll_loss_detached":
        return gaussian_nll_detached
    elif criterion_name == "rmse_loss":
        return rmse
    elif criterion_name == "edge_aware_mse_loss":
        return edge_aware_mse_loss
    elif criterion_name == "edge_aware_gaussian_nll_loss":
        return edge_aware_gaussian_nll_loss
    elif criterion_name == "edge_aware_gaussian_nll_loss_detached":
        return edge_aware_gaussian_nll_loss_detached
    elif criterion_name == "bayescap_loss":
        return bayescap_loss
    elif criterion_name == "gen_gaussian_nll":
        return gen_gaussian_nll
    else:
        raise ValueError(f"Unknown criterion: {criterion_name}")

def get_random_id(length=6):
    """
    Generate a random ID for experiment tracking.
    
    Args:
        length (int): Length of the ID
        
    Returns:
        str: Random ID
    """
    chars = string.ascii_uppercase + string.digits
    return ''.join(random.choice(chars) for _ in range(length))

def setup_result_directories(config_name):
    """
    Set up the directory structure for experiment results.
    
    Args:
        config_name (str): Name of the configuration
        
    Returns:
        tuple: Paths to model and results directories
    """
    model_dir = f"results/{config_name}/checkpoints"
    results_dir = f"results/{config_name}/figs"
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    return model_dir, results_dir

def process_config_from_args(args, device):
    """
    Process configuration from command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        tuple: (config, config_name)
    """
    config = load_yaml_config(args.yaml_config, device)
    config_name = os.path.basename(args.yaml_config).replace('.yaml', '').replace('.yml', '')

    return config, config_name 