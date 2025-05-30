import torch
import random

def create_bootstrapped_dataloaders(dataloader, n_splits):
    dataset = dataloader.dataset
    dataset_size = len(dataset)
    
    bootstrapped_dataloaders = []
    
    for _ in range(n_splits):
        # Create bootstrapped indices (random sampling with replacement)
        indices = [random.randint(0, dataset_size - 1) for _ in range(dataset_size)]
        
        # Create a new dataloader with bootstrapped indices
        bootstrapped_dataloader = torch.utils.data.DataLoader(
            dataset=torch.utils.data.Subset(dataset, indices),
            batch_size=dataloader.batch_size,
            shuffle=True,  # Already shuffled by bootstrap sampling
            num_workers=dataloader.num_workers,
            pin_memory=dataloader.pin_memory,
            drop_last=dataloader.drop_last
        )
        
        bootstrapped_dataloaders.append(bootstrapped_dataloader)
    
    return bootstrapped_dataloaders
