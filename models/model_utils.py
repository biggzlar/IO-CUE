import torch.optim as optim

def create_optimizer(optimizer_type, model_parameters, optimizer_params):
    """
    Create an optimizer based on the specified type and parameters
    
    Args:
        optimizer_type: Type of optimizer ('Adam', 'AdamW', 'SGD', etc.)
        model_parameters: Model parameters to optimize
        optimizer_params: Dictionary of optimizer parameters
    
    Returns:
        Configured optimizer instance
    """
    if optimizer_type == 'Adam':
        return optim.Adam(model_parameters, **optimizer_params)
    elif optimizer_type == 'AdamW':
        return optim.AdamW(model_parameters, **optimizer_params)
    elif optimizer_type == 'SGD':
        return optim.SGD(model_parameters, **optimizer_params)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

def create_scheduler(scheduler_type, optimizer, scheduler_params):
    """
    Create a scheduler based on the specified type and parameters
    
    Args:
        scheduler_type: Type of scheduler ('CosineAnnealingLR', 'ReduceLROnPlateau', etc.)
        optimizer: Optimizer to schedule
        scheduler_params: Dictionary of scheduler parameters
    
    Returns:
        Configured scheduler instance or None if scheduler_type is None
    """
    if scheduler_type is None:
        return None
    
    if scheduler_type == 'CosineAnnealingLR':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_params)
    elif scheduler_type == 'ReduceLROnPlateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_params)
    elif scheduler_type == 'CosineAnnealingWarmRestarts':
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **scheduler_params)
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

def create_model_instances(model_class, model_params, n_instances, return_activations=False):
    """
    Create multiple instances of a model class with different random initializations
    
    Args:
        model_class: The model class to instantiate
        model_params: Dictionary of parameters to pass to the model constructor
        n_instances: Number of model instances to create
    
    Returns:
        List of initialized model instances
    """
    models = []
    for _ in range(n_instances):
        # Create a new instance with its own initialization
        model = model_class(**model_params, return_activations=return_activations)
        models.append(model)
    
    return models 