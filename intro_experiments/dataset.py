import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def ground_truth_function(x):
    """
    Ground truth function with a strong periodic component using cos(x)
    """
    # return (np.cos(6*x) * np.abs(x)) / 4.
    return np.cos(x * 6.)

def ground_truth_noise(x):
    """
    Ground truth noise function
    """
    return abs(x - 1)

def generate_data(n_samples=200, x_range=(-2, 2), noise_level=0.2, batch_size=32):
    """
    Generate data from a 1D periodic function
    
    Args:
        n_samples: Number of samples to generate
        x_range: Range of x values
        noise_level: Amount of noise to add
        batch_size: Batch size for returned dataloader
    
    Returns:
        data_loader: DataLoader with data
        X: Raw data features
        y: Raw data targets
        true_std: True noise standard deviation
    """
    # Generate random x values within the specified range
    X = np.random.uniform(x_range[0], x_range[1], n_samples)
    
    # Calculate clean target values
    y_clean = np.array([ground_truth_function(x) for x in X])
    
    # Calculate heteroscedastic noise levels (varying based on x position)
    # Higher noise in the middle, lower at the edges
    # true_std = noise_level * (1 + 0.5 * np.sin(X)**2) 
    true_std = noise_level * ground_truth_noise(X)
    
    # Generate noise with varying standard deviation
    noise = np.random.normal(0, 1, size=len(y_clean)) * true_std
    
    # Add noise to targets
    y = y_clean + noise
    
    # Reshape X for the model
    X_reshaped = X.reshape(-1, 1)
    
    # Create dataset and dataloader
    dataset = CustomDataset(X_reshaped, y.reshape(-1, 1))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return data_loader, X_reshaped, y.reshape(-1, 1), true_std.reshape(-1, 1)

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_bootstrapped_loader(X, y, n_samples=None, batch_size=32):
    """Create a bootstrapped dataloader by sampling with replacement"""
    if n_samples is None:
        n_samples = len(X)
    
    indices = np.random.choice(len(X), size=n_samples, replace=True)
    
    X_boot = X[indices]
    y_boot = y[indices]
    
    dataset = CustomDataset(X_boot, y_boot)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return loader 