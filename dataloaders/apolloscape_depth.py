import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import math

def load_apolloscape_depth():
    """Load the Apolloscape dataset"""
    path = "/mnt/data/simple_depth_data/apolloscape_test.h5"
    dataset = h5py.File(path, "r")
    return (dataset["image"], dataset["depth"])

class ApolloscapeDepthDataset:
    def __init__(self, img_size=None):
        """Initialize the Apolloscape depth dataset
        
        Args:
            img_size: Optional (H, W) to resize samples to. If None, keep original size.
        """
        data = load_apolloscape_depth()
        self.test = _ApolloscapeDepthDataset(data, img_size=img_size)

    def get_dataloaders(self, batch_size, shuffle=False):
        """Return dataloader for the test set"""
        test_loader = DataLoader(self.test, batch_size=batch_size, shuffle=shuffle, num_workers=4)
        return None, test_loader


class _ApolloscapeDepthDataset(Dataset):
    def __init__(self, data, img_size=None):
        """Initialize the internal dataset class"""
        self.data = data
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.data[0])
    
    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        if self.data[0][idx].dtype == np.uint8:
            image = self.data[0][idx].astype(np.float32) / 255.0
            depth = self.data[1][idx].astype(np.float32) / 255.0
        else:
            image = self.data[0][idx]
            depth = self.data[1][idx]
        # Sanitize inputs to remove NaNs/Infs and clamp to valid ranges
        image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
        depth = np.nan_to_num(depth, nan=0.0, posinf=1.0, neginf=0.0)
        # Clamp images to [0,1], depth to [0,1] since it's normalized above
        image = np.clip(image, 0.0, 1.0)
        depth = np.clip(depth, 0.0, 1.0)
        img_t = self.transform(image)
        depth_t = self.transform(depth)
        # Resize if requested to ensure compatibility with model input sizes
        if self.img_size is not None:
            # img_t: [C,H,W], depth_t: [1,H,W]
            img_t = F.interpolate(img_t.unsqueeze(0), size=self.img_size, mode='bilinear', align_corners=False).squeeze(0)
            # Use nearest for depth to preserve label values
            depth_t = F.interpolate(depth_t.unsqueeze(0), size=self.img_size, mode='nearest').squeeze(0)
        # Final tensor-level sanitation to be safe
        img_t = torch.nan_to_num(img_t, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        depth_t = torch.nan_to_num(depth_t, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        return img_t, depth_t


if __name__ == "__main__":
    """Test the dataset"""
    import matplotlib.pyplot as plt
    
    dataset = ApolloscapeDepthDataset()
    test_loader = dataset.get_dataloaders(batch_size=4)[1]
    
    # Show a few samples
    for i, (images, depths) in enumerate(test_loader):
        print(f"Batch {i}: images {images.shape}, depths {depths.shape}")
        
        # Plot the first sample in the batch
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(images[0].permute(1, 2, 0).numpy())
        axs[0].set_title("RGB Image")
        axs[0].axis('off')
        
        axs[1].imshow(depths[0].permute(1, 2, 0).numpy(), cmap='plasma')
        axs[1].set_title("Depth Map")
        axs[1].axis('off')
        
        plt.savefig(f"apolloscape_sample_{i}.png")
        plt.close()
        
        if i >= 2:  # Show just a few samples
            break 