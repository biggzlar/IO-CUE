import numpy as np
import torch
import torchvision.transforms as transforms

from dataloaders.simple_depth import _DepthDataset, load_depth


def get_augmented_nyu_test_loader(img_size, batch_size, augment_kwargs, attr_overrides=None, shuffle=False):
    """Create an augmented NYU test loader for OOD evaluation/analysis.

    Arguments mirror existing script helpers to avoid caller changes.
    """
    _, test_data = load_depth()
    flip_prob = 1.0 if augment_kwargs.get("flip", False) else 0
    if attr_overrides and "flip_prob" in attr_overrides:
        flip_prob = attr_overrides["flip_prob"]

    class DeterministicDepthDataset(_DepthDataset):
        def __init__(self, data, img_size=None, augment=False,
                     flip=False, colorjitter=False, gaussianblur=False, grayscale=False, gaussian_noise=False, flip_prob=0):
            super().__init__(data, img_size, augment, flip, colorjitter, gaussianblur, grayscale, gaussian_noise, flip_prob)
            input_transforms = [transforms.ToPILImage()]
            if augment:
                if colorjitter:
                    input_transforms.append(transforms.ColorJitter(brightness=1, contrast=1, saturation=1, hue=0.05))
                if gaussianblur:
                    input_transforms.append(transforms.GaussianBlur(kernel_size=5, sigma=(1.0)))
                if grayscale:
                    input_transforms.append(transforms.Grayscale(num_output_channels=3))
            input_transforms.extend([
                transforms.ToTensor(),
                transforms.Resize(img_size) if img_size is not None else transforms.Lambda(lambda x: x),
            ])
            self.input_transform = transforms.Compose(input_transforms)

        def __getitem__(self, idx):
            if len(self.data[0][idx].shape) == 3:
                image = self.data[0][idx].astype(np.float32) / 255.0
                depth = self.data[1][idx].astype(np.float32) / 255.0
            else:
                image = self.data[0][idx]
                depth = self.data[1][idx]
            image_tensor = self.input_transform(image)
            depth_tensor = self.output_transform(depth)
            if self.augment and self.flip:
                image_tensor = torch.flip(image_tensor, dims=[-1])
                depth_tensor = torch.flip(depth_tensor, dims=[-1])
            if self.augment and self.gaussian_noise:
                image_tensor = torch.clamp(image_tensor + torch.randn_like(image_tensor) * 0.1, 0, 1)
            return image_tensor, depth_tensor

    ds = DeterministicDepthDataset(
        test_data,
        img_size=img_size,
        augment=True,
        flip=augment_kwargs.get("flip", False),
        colorjitter=augment_kwargs.get("colorjitter", False),
        gaussianblur=augment_kwargs.get("gaussianblur", False),
        grayscale=augment_kwargs.get("grayscale", False),
        gaussian_noise=False,
        flip_prob=flip_prob,
    )
    from torch.utils.data import DataLoader
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=8, prefetch_factor=4)


__all__ = ["get_augmented_nyu_test_loader"]


