import os
import glob
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import decode_image, ImageReadMode, read_file

def load_hypersim_depth(data_root="/mnt/data/hypersim", train_split=1.0):
    """
    Load HyperSim dataset
    
    Args:
        data_root: Root directory of HyperSim dataset
        train_split: Fraction of data to use for training (0.0-1.0)
    
    Returns:
        Tuple of train and test data
    """
    # Find all scene folders (ai_*)
    scene_folders = glob.glob(os.path.join(data_root, "ai_*"))
    
    rgb_files = []
    depth_files = []
    
    for scene in scene_folders:
        # Find all final_preview folders for RGB
        rgb_folders = sorted(glob.glob(os.path.join(scene, "**/images/*final_preview"), recursive=True))
        # Find all geometry_preview folders for depth
        depth_folders = sorted(glob.glob(os.path.join(scene, "**/images/*geometry_preview"), recursive=True))
        
        # Match RGB and depth files
        for rgb_folder, depth_folder in zip(rgb_folders, depth_folders):
            rgb_images = sorted(glob.glob(os.path.join(rgb_folder, "*color.jpg")))
            depth_images = sorted(glob.glob(os.path.join(depth_folder, "*depth_meters.png")))
            
            # Ensure we only add pairs that exist in both folders
            for rgb_path, depth_path in zip(rgb_images, depth_images):
                rgb_files.append(rgb_path)
                depth_files.append(depth_path)

    # Create dataset with paired data up to train_fraction
    n_pairs = len(rgb_files)
    n_train = int(n_pairs * 0.95 * train_split)  # 95% for training
    n_test = int(n_pairs * 0.05)  # 5% for testing
    
    # Split the data
    train_rgb = rgb_files[:n_train]
    train_depth = depth_files[:n_train]
    test_rgb = rgb_files[-n_test:]
    test_depth = depth_files[-n_test:]
    
    return (train_rgb, train_depth), (test_rgb, test_depth)


class HyperSimDepthDataset:
    def __init__(self, data_root="/mnt/data/hypersim", train_split=1.0, img_size=(224, 224)):
        self.img_size = img_size
        data = load_hypersim_depth(data_root, train_split)
        self.train = _HyperSimDepthDataset(data[0], img_size=self.img_size, is_train=True)
        self.test = _HyperSimDepthDataset(data[1], img_size=self.img_size, is_train=False)

        self.train.__getitem__(0)

    def get_dataloaders(self, batch_size, shuffle=True):
        train_loader = DataLoader(self.train, batch_size=batch_size, shuffle=shuffle, num_workers=8, persistent_workers=True)
        test_loader = DataLoader(self.test, batch_size=batch_size, shuffle=False, num_workers=8, persistent_workers=True)
        return train_loader, test_loader


class _HyperSimDepthDataset(Dataset):
    def __init__(self, data, img_size=(224, 224), is_train=False):
        self.rgb_files, self.depth_files = data
        self.img_size = img_size
        self.is_train = is_train
        
        # Basic transforms for all images
        self.basic_rgb_transform = transforms.Compose([
            transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            # transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.basic_depth_transform = transforms.Compose([
            transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.NEAREST),
            # transforms.ToTensor()
        ])
        
        # Add augmentations for training
        if self.is_train:
            # Spatial augmentations - applied to both RGB and depth images
            self.spatial_transforms = transforms.Compose([
                # Horizontal flip
                transforms.RandomHorizontalFlip(p=0.5),
                # Random crop that maintains spatial alignment
                transforms.RandomResizedCrop(self.img_size, scale=(0.8, 1.0), 
                                           ratio=(0.9, 1.1), 
                                           interpolation=transforms.InterpolationMode.BICUBIC)
            ])
            
            # RGB-specific augmentations
            self.rgb_augmentations = transforms.Compose([
                # Color jitter
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                # Occasional grayscale conversion
                transforms.RandomGrayscale(p=0.1),
                # Gaussian blur
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))
            ])

    def __len__(self):
        return len(self.rgb_files)
    
    def load_image(self, path, mode='RGB'):
        """Load image using PIL."""
        # return Image.open(path).convert(mode)
        return decode_image(read_file(path), mode=mode).to(torch.float32)
    
    def _add_depth_dropout(self, depth_tensor, max_dropout_pct=0.05, p=0.2):
        """Randomly drop pixels in depth maps (simulating sensor failures)"""
        if not self.is_train or np.random.random() > p:
            return depth_tensor
            
        # Create random dropout mask
        mask_size = int(np.prod(depth_tensor.shape) * np.random.uniform(0, max_dropout_pct))
        if mask_size > 0:
            # Create a flat index list of randomly selected positions
            flat_idx = torch.randperm(np.prod(depth_tensor.shape))[:mask_size]
            
            # Convert flat indices to coordinates
            coords = []
            for dim_size in depth_tensor.shape:
                coords.append(flat_idx % dim_size)
                flat_idx = flat_idx // dim_size
            
            # Set values to zero (missing depth)
            for i in range(mask_size):
                idx = tuple(coords[d][i] for d in range(len(coords)))
                depth_tensor[idx] = 0.0
        
        return depth_tensor
    
    def get_depth_meters(self, depth_image):
        intWidth = 1024
        intHeight = 768
        fltFocal = 886.81
        npyImageplaneX = torch.linspace((-0.5 * intWidth) + 0.5, (0.5 * intWidth) - 0.5, intWidth).reshape(1, intWidth).repeat_interleave(intHeight, 0).to(torch.float32)[:, :, None]
        npyImageplaneY = torch.linspace((-0.5 * intHeight) + 0.5, (0.5 * intHeight) - 0.5, intHeight).reshape(intHeight, 1).repeat_interleave(intWidth, 1).to(torch.float32)[:, :, None]
        npyImageplaneZ = torch.full(size=(intHeight, intWidth, 1), fill_value=fltFocal, dtype=torch.float32)
        npyImageplane = torch.concatenate([npyImageplaneX, npyImageplaneY, npyImageplaneZ], 2)

        
        npyDepth = depth_image.squeeze() / torch.linalg.norm(npyImageplane, 2, 2) * fltFocal
        # import ipdb; ipdb.set_trace()
        return npyDepth.unsqueeze(0)
    
    def __getitem__(self, idx):
        # Load images
        rgb_image = self.load_image(self.rgb_files[idx], mode=ImageReadMode.RGB)
        # depth_image = self.load_image(self.depth_files[idx], mode='L')
        depth_image = self.load_image(self.depth_files[idx], mode=ImageReadMode.GRAY)
        
        # Apply transforms with synchronized randomness for spatial augmentations
        # if self.is_train:
        #     # Set the same random seed for both transforms to ensure spatial alignment
        #     seed = np.random.randint(2147483647)
        #     torch.manual_seed(seed)
            
        #     # Apply spatial transforms
        #     rgb_image = self.spatial_transforms(rgb_image)
            
        #     # Reset seed for depth to match RGB transformation
        #     torch.manual_seed(seed)
        #     depth_image = self.spatial_transforms(depth_image)
            
        #     # Apply RGB-specific augmentations
        #     rgb_image = self.rgb_augmentations(rgb_image)
        
        # Apply basic transforms
        rgb_tensor = self.basic_rgb_transform(rgb_image) / 255.
        rgb_tensor = rgb_tensor.clamp(0, 1)
        depth_ = self.get_depth_meters(depth_image)
        depth_ = transforms.functional.crop(depth_, top=0, left=0, height=int(768 * 0.75), width=int(1024 * 0.75))
        depth_tensor = self.basic_depth_transform(depth_) / 255.

        # max_depth = 20.0
        # min_depth = 1 / 20.
        # disparity_tensor = 1 / (depth_tensor + min_depth)
        # disparity_tensor = (disparity_tensor - 0.0025) / (20. - 0.0025)

        # import ipdb; ipdb.set_trace()
        disparity_tensor = 1 - (torch.clamp(depth_tensor * 20., min=1 / 20., max=10.) / 10.)

        # print(disparity_tensor.min(), disparity_tensor.max())
        # d_min, d_max = 1 / max_depth, 1 / min_depth
        # disparity_tensor = (disparity_tensor - d_min) / (d_max - d_min)
        # Apply depth-specific augmentations
        # if self.is_train:
        #     disparity_tensor = self._add_depth_dropout(disparity_tensor)

        # fuck_tensor = self.basic_depth_transform(depth_image)
        # fuck_tensor = 1 / (fuck_tensor + min_depth)
        # fuck_tensor = (fuck_tensor - 0.0025) / (20. - 0.0025)
        
        return rgb_tensor, disparity_tensor  # , fuck_tensor


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib import cm
    
    # Set the path to your HyperSim dataset
    data_root = "/mnt/data/hypersim"  # Update with your path
    
    # Load dataset and create dataloaders
    dataset = HyperSimDepthDataset(data_root=data_root, img_size=(128, 160))
    train_loader, test_loader = dataset.get_dataloaders(batch_size=4)
    
    print(f"Train set size: {len(dataset.train)}, Test set size: {len(dataset.test)}")
    
    # Visualize a few samples
    for i, (images, depths) in enumerate(test_loader):
        print(f"Batch {i+1}: Images shape: {images.shape}, Depths shape: {depths.shape}")
        print(f"Images range: [{images.min():.2f}, {images.max():.2f}], Depths range: [{depths.min():.2f}, {depths.max():.2f}]")
        
        # Plot images and depths
        fig, axes = plt.subplots(2, 4, figsize=(16, 12))
        for j in range(min(4, images.shape[0])):
            # Denormalize the image for visualization
            img = images[j].numpy().transpose(1, 2, 0)
            # img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            
            # Display image
            axes[0, j].imshow(img)
            axes[0, j].set_title(f"RGB {j+1}")
            axes[0, j].axis("off")
            
            # Display depth with viridis colormap
            depth_map = depths[j].squeeze().numpy()
            axes[1, j].imshow(depth_map, cmap='plasma')
            axes[1, j].set_title(f"Depth {j+1}")
            axes[1, j].axis("off")

            # import ipdb; ipdb.set_trace()
        
            # fuck_map = fucks[j].squeeze().numpy()
            # axes[2, j].imshow(fuck_map, cmap='plasma')
            # axes[2, j].set_title(f"Fuck {j+1}")
            # axes[2, j].axis("off")
        
        plt.tight_layout()
        plt.savefig(f"hypersim_sample_{i}.png")
        plt.close()
        
        if i >= 2:
            break
    
    print("Test completed - check the output images to verify the data loading.") 