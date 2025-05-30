import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader

class UCIDataset(Dataset):
    """PyTorch Dataset for UCI data."""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class UCIDatasets:
    """Class to handle various UCI datasets"""
    
    AVAILABLE_DATASETS = {
        "concrete": "concrete",
        "energy": "energy-efficiency",
        "kin8nm": "kin8nm",
        "naval_propulsion": "naval",
        "power_plant": "power-plant", 
        "protein": "protein",
        "wine": "wine",
        "yacht": "yacht",
        "boston": "boston"
    }
    
    def __init__(self, dataset_name, batch_size=32, test_size=0.1, random_state=42):
        """
        Initialize UCI dataset handler
        
        Args:
            dataset_name: Name of the UCI dataset to use
            batch_size: Batch size for dataloaders
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
        """
        if dataset_name not in self.AVAILABLE_DATASETS:
            raise ValueError(f"Dataset {dataset_name} not available. Choose from {list(self.AVAILABLE_DATASETS.keys())}")
        
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.test_size = test_size
        self.random_state = random_state
        
        # Load and preprocess the dataset
        self._load_dataset()
        
    def _load_dataset(self):
        """Load and preprocess the specified dataset"""
        try:
            # Map internal dataset name to loader dataset name
            repo_dataset_name = self.AVAILABLE_DATASETS[self.dataset_name]
            
            # Load the dataset
            X, y = self._load_uci_data(repo_dataset_name)
            X = X.astype(np.float32)
            y = y.astype(np.float32)
            
            # Standardize features
            def standardize(data):
                mu = data.mean(axis=0, keepdims=1)
                scale = data.std(axis=0, keepdims=1)
                scale[scale < 1e-10] = 1.0
                
                data = (data - mu) / scale
                return data, mu, scale
            
            # Create train/test split
            if self.random_state == -1:  # Do not shuffle
                permutation = range(X.shape[0])
            else:
                rs = np.random.RandomState(self.random_state)
                permutation = rs.permutation(X.shape[0])
                
            # Some datasets require different test fractions
            test_fraction = self.test_size
            if self.dataset_name == "boston" or self.dataset_name == "wine":
                test_fraction = 0.2
                
            size_train = int(np.round(X.shape[0] * (1 - test_fraction)))
            index_train = permutation[0:size_train]
            index_test = permutation[size_train:]
            
            X_train = X[index_train, :]
            X_test = X[index_test, :]
            
            # Reshape y to have a second dimension if needed
            if len(y.shape) == 1:
                y = y.reshape(-1, 1)
                
            y_train = y[index_train]
            y_test = y[index_test]
            
            # Standardize the data
            X_train, x_train_mu, x_train_scale = standardize(X_train)
            X_test = (X_test - x_train_mu) / x_train_scale
            
            y_train, y_train_mu, y_train_scale = standardize(y_train)
            y_test = (y_test - y_train_mu) / y_train_scale
            
            # Save scaling info for later use
            self.y_train_mu = y_train_mu
            self.y_train_scale = y_train_scale
            
            # Store the split data
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test
            
            # Create PyTorch datasets and dataloaders
            self.train_dataset = UCIDataset(self.X_train, self.y_train)
            self.test_dataset = UCIDataset(self.X_test, self.y_test)
            
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True
            )
            
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset {self.dataset_name}: {str(e)}")
    
    def _load_uci_data(self, dataset_name):
        """Load UCI dataset based on name"""
        data_dir = "/mnt/data/uci"  # Use the specified data directory
        
        if dataset_name == "concrete":
            data_file = os.path.join(data_dir, 'concrete/Concrete_Data.xls')
            data = pd.read_excel(data_file)
            X = data.values[:, :-1]
            y = data.values[:, -1]
            
        elif dataset_name == "energy-efficiency":
            data_file = os.path.join(data_dir, 'energy-efficiency/ENB2012_data.xlsx')
            data = pd.read_excel(data_file)
            X = data.values[:, :-2]
            y = data.values[:, -1]  # We use cooling load by default
            
        elif dataset_name == "kin8nm":
            data_file = os.path.join(data_dir, 'kin8nm/dataset_2175_kin8nm.csv')
            data = pd.read_csv(data_file, sep=',')
            X = data.values[:, :-1]
            y = data.values[:, -1]
            
        elif dataset_name == "naval":
            data = np.loadtxt(os.path.join(data_dir, 'naval/data.txt'))
            X = data[:, :-2]
            y = data[:, -1]  # Use turbine decay state coefficient
            
        elif dataset_name == "power-plant":
            data_file = os.path.join(data_dir, 'power-plant/Folds5x2_pp.xlsx')
            data = pd.read_excel(data_file)
            X = data.values[:, :-1]
            y = data.values[:, -1]
            
        elif dataset_name == "protein":
            data_file = os.path.join(data_dir, 'protein/CASP.csv')
            data = pd.read_csv(data_file, sep=',')
            X = data.values[:, 1:]
            y = data.values[:, 0]
            
        elif dataset_name == "wine":
            data_file = os.path.join(data_dir, 'wine-quality/wine_data_new.txt')
            data = pd.read_csv(data_file, sep=' ', header=None)
            X = data.values[:, :-1]
            y = data.values[:, -1]
            
        elif dataset_name == "yacht":
            data_file = os.path.join(data_dir, 'yacht/yacht_hydrodynamics.data')
            data = pd.read_csv(data_file, sep='\s+')
            X = data.values[:, :-1]
            y = data.values[:, -1]
            
        elif dataset_name == "boston":
            data = np.loadtxt(os.path.join(data_dir, 'boston-housing/boston_housing.txt'))
            X = data[:, :-1]
            y = data[:, -1]
            
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")
            
        return X, y
    
    def get_train_loader(self):
        """Get training dataloader"""
        return self.train_loader
    
    def get_test_loader(self):
        """Get test dataloader"""
        return self.test_loader
    
    def get_bootstrapped_loader(self, n_samples=None, batch_size=None):
        """
        Create a bootstrapped version of the training dataloader
        
        Args:
            n_samples: Number of samples to generate (uses train set size if None)
            batch_size: Batch size for the new dataloader (uses original if None)
            
        Returns:
            A new dataloader with bootstrapped samples
        """
        if n_samples is None:
            n_samples = len(self.X_train)
            
        if batch_size is None:
            batch_size = self.batch_size
            
        indices = np.random.choice(len(self.X_train), size=n_samples, replace=True)
        
        X_boot = self.X_train[indices]
        y_boot = self.y_train[indices]
        
        dataset = UCIDataset(X_boot, y_boot)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def verify_dataset_availability():
    """Check that all datasets are available and can be loaded."""
    results = {}
    data_dir = "/mnt/data/uci"
    
    print(f"Checking datasets in {data_dir}")
    
    for internal_name, repo_name in UCIDatasets.AVAILABLE_DATASETS.items():
        print(f"Checking dataset {internal_name} (mapped to {repo_name})...")
        
        if repo_name == "concrete":
            file_path = os.path.join(data_dir, 'concrete/Concrete_Data.xls')
        elif repo_name == "energy-efficiency":
            file_path = os.path.join(data_dir, 'energy-efficiency/ENB2012_data.xlsx')
        elif repo_name == "kin8nm":
            file_path = os.path.join(data_dir, 'kin8nm/dataset_2175_kin8nm.csv')
        elif repo_name == "naval":
            file_path = os.path.join(data_dir, 'naval/data.txt')
        elif repo_name == "power-plant":
            file_path = os.path.join(data_dir, 'power-plant/Folds5x2_pp.xlsx')
        elif repo_name == "protein":
            file_path = os.path.join(data_dir, 'protein/CASP.csv')
        elif repo_name == "wine":
            file_path = os.path.join(data_dir, 'wine-quality/wine_data_new.txt')
        elif repo_name == "yacht":
            file_path = os.path.join(data_dir, 'yacht/yacht_hydrodynamics.data')
        elif repo_name == "boston":
            file_path = os.path.join(data_dir, 'boston-housing/boston_housing.txt')
        else:
            file_path = None
            
        if file_path and os.path.exists(file_path):
            results[internal_name] = {
                "status": "available",
                "file_path": file_path
            }
            print(f"  ✓ Dataset file found: {file_path}")
        else:
            results[internal_name] = {
                "status": "unavailable",
                "error": f"File not found: {file_path}"
            }
            print(f"  ✗ Dataset file not found: {file_path}")
    
    # Print summary
    available_count = sum(1 for info in results.values() if info["status"] == "available")
    print(f"\nFound {available_count} out of {len(results)} datasets.")
    
    return results


if __name__ == '__main__':
    # First verify datasets are available
    print("Checking dataset availability...")
    verify_dataset_availability()
    
    # Test the implementation with different datasets
    for dataset_name in UCIDatasets.AVAILABLE_DATASETS.keys():
        print(f"\nTesting {dataset_name} dataset:")
        try:
            # Initialize dataset
            dataset = UCIDatasets(dataset_name, batch_size=32)
            
            # Get loaders
            train_loader = dataset.get_train_loader()
            test_loader = dataset.get_test_loader()
            
            # Test a batch
            X_batch, y_batch = next(iter(train_loader))
            print(f"Train batch shape - X: {X_batch.shape}, y: {y_batch.shape}")
            
            # Test bootstrapped loader
            boot_loader = dataset.get_bootstrapped_loader()
            X_boot, y_boot = next(iter(boot_loader))
            print(f"Bootstrapped batch shape - X: {X_boot.shape}, y: {y_boot.shape}")
            
            # Print dataset statistics
            print(f"Total samples: {len(dataset.X_train) + len(dataset.X_test)}")
            print(f"Feature dimensions: {dataset.X_train.shape[1]}")
            print(f"Train samples: {len(dataset.X_train)}")
            print(f"Test samples: {len(dataset.X_test)}")
            
        except Exception as e:
            print(f"Error with {dataset_name}: {str(e)}") 