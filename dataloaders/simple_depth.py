import h5py
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

def load_depth_path(path):
    dataset = h5py.File(path, "r")
    return (dataset["image"], dataset["depth"])

def load_depth():
    train = h5py.File("/mnt/data/simple_depth_data/depth_train.h5", "r")
    test = h5py.File("/mnt/data/simple_depth_data/depth_test.h5", "r")
    return (train["image"], train["depth"]), (test["image"], test["depth"])

def split_data(data, train_split):
    X, y = data
    X_ = X[:train_split]
    y_ = y[:train_split]
    return (X_, y_)


class DepthDataset:
    def __init__(self, path=None, img_size=None, augment=True, augment_test_data=True, train_split=1.0, 
                 flip=False, colorjitter=False, gaussianblur=False, grayscale=False, gaussian_noise=False):
        self.augment = augment
        self.augment_test_data = augment_test_data
        self.flip = flip
        self.colorjitter = colorjitter
        self.gaussianblur = gaussianblur
        self.grayscale = grayscale
        self.gaussian_noise = gaussian_noise

        self.flip_prob = 0.5 if flip else 0

        self.path = path
        self.img_size = img_size
        self.train_split = train_split

    def get_dataloaders(self, batch_size, shuffle=True):
        if self.path is None:
            data = load_depth()
            self.train_split_idx = int(len(data[0][0]) * self.train_split)
            train_data = split_data(data[0], self.train_split_idx)
            test_data = data[1]

            self.train = _DepthDataset(train_data, img_size=self.img_size, augment=self.augment, 
                flip=self.flip, colorjitter=self.colorjitter, 
                gaussianblur=self.gaussianblur, grayscale=self.grayscale,
                gaussian_noise=self.gaussian_noise, flip_prob=self.flip_prob
            )
            self.test = _DepthDataset(test_data, img_size=self.img_size, augment=self.augment_test_data, 
                flip=self.flip, colorjitter=self.colorjitter, 
                gaussianblur=self.gaussianblur, grayscale=self.grayscale,
                gaussian_noise=self.gaussian_noise, flip_prob=self.flip_prob)
        else:
            data = load_depth_path(self.path)
            self.train = None
            self.test = _DepthDataset(data, img_size=self.img_size)

        train_loader = None
        test_loader = None

        if self.train is not None:
            train_loader = DataLoader(self.train, batch_size=batch_size, shuffle=shuffle, num_workers=8, prefetch_factor=4)
        if self.test is not None:
            test_loader = DataLoader(self.test, batch_size=batch_size, shuffle=shuffle, num_workers=8, prefetch_factor=4)

        return train_loader, test_loader
    

class _DepthDataset(Dataset):
    def __init__(self, data, img_size=None, augment=False, 
                 flip=False, colorjitter=False, gaussianblur=False, grayscale=False, gaussian_noise=False, flip_prob=0):
        self.data = data
        self.augment = augment
        self.flip = flip
        self.flip_prob = flip_prob
        self.gaussian_noise = gaussian_noise
        
        # Create input transform without flip
        input_transforms = [transforms.ToPILImage()]
        
        # Add other transforms (not flip)
        if augment:
            if colorjitter and np.random.random() < 0.8:
                input_transforms.append(transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05))
            if gaussianblur and np.random.random():
                input_transforms.append(transforms.GaussianBlur(kernel_size=3, sigma=(1.0)))
            if grayscale:
                input_transforms.append(transforms.RandomGrayscale(p=0.7))

        # Add final transforms
        input_transforms.extend([
            transforms.ToTensor(),
            transforms.Resize(img_size) if img_size is not None else transforms.Lambda(lambda x: x),
            # transforms.Resize(img_size, interpolation=InterpolationMode.BILINEAR, antialias=True) if img_size is not None else transforms.Lambda(lambda x: x),
        ])
        
        self.input_transform = transforms.Compose(input_transforms)
        
        # Output transform only includes resize
        self.output_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(img_size) if img_size is not None else transforms.Lambda(lambda x: x),
            # transforms.Resize(img_size, interpolation=InterpolationMode.NEAREST) if img_size is not None else transforms.Lambda(lambda x: x),
        ])

    def __len__(self):
        return len(self.data[0])
    
    def __getitem__(self, idx):
        if self.data[0][idx].dtype == np.uint8:
            image = self.data[0][idx].astype(np.float32) / 255.0
            depth = self.data[1][idx].astype(np.float32) / 255.0
        else:
            image = self.data[0][idx]
            depth = self.data[1][idx]

        # Convert to tensor with other transforms
        image_tensor = self.input_transform(image)
        depth_tensor = self.output_transform(depth)
        
        # Apply horizontal flip consistently to both input and output
        if self.augment and self.flip and np.random.random() < self.flip_prob:
            image_tensor = torch.flip(image_tensor, dims=[-1])
            depth_tensor = torch.flip(depth_tensor, dims=[-1])

        if self.augment and self.gaussian_noise and np.random.random() < 0.5:
            image_tensor = torch.clamp(image_tensor + torch.randn_like(image_tensor) * 0.1, 0, 1)
            
        return image_tensor, depth_tensor
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    dataset = DepthDataset(augment=True, flip=True, train_split=0.1)
    train_loader, test_loader = dataset.get_dataloaders(batch_size=16)
    for i, (images, depths) in enumerate(train_loader):
        print(images.shape, depths.shape)
        print(images[i].min(), images[i].max(), depths[i].min(), depths[i].max())
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(images[0].numpy().transpose(1, 2, 0))
        axs[1].imshow(depths[0].numpy().transpose(1, 2, 0))
        plt.savefig(f"results/train_sample_{i}.png")
        plt.close()
        if i > 3:
            break

