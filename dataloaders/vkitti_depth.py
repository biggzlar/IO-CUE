import os
import glob
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import decode_image, ImageReadMode, read_file
from PIL import Image


def _pair_vkitti_rgb_depth(root: str, split: str = "train") -> List[Tuple[str, str]]:
    """Return paired (rgb, depth) paths for VKITTI.

    Expected layout:
        <root>/<split>/rgb/*.jpg
        <root>/<split>/depth/*.png  (uint16 depth, millimeters)

    Pairing is done by deterministic mapping from depth path -> rgb path by
    replacing directory and token `_depth_` -> `_rgb_`, and extension png->jpg.
    """
    rgb_dir = os.path.join(root, split, "rgb")
    depth_dir = os.path.join(root, split, "depth")
    if not (os.path.isdir(rgb_dir) and os.path.isdir(depth_dir)):
        return []

    depth_paths = sorted(glob.glob(os.path.join(depth_dir, "*.png")))
    pairs: List[Tuple[str, str]] = []
    for dpth in depth_paths:
        basename = os.path.basename(dpth)
        rgb_basename = basename.replace("_depth_", "_rgb_").replace(".png", ".jpg")
        rgb_path = os.path.join(rgb_dir, rgb_basename)
        if os.path.isfile(rgb_path):
            pairs.append((rgb_path, dpth))
    return pairs


class VKITTIDepthDataset:
    def __init__(
        self,
        data_root: str = "/mnt/data/vkitti",
        split: str = "train",
        img_size: Optional[Tuple[int, int]] = (256, 512),
        max_depth_meters: float = 80.0,
        train_split: float = 1.0,
        augment: bool = False,
        flip: bool = False,
        colorjitter: bool = False,
        gaussianblur: bool = False,
        grayscale: bool = False,
        gaussian_noise: bool = False,
    ):
        """VKITTI Depth dataset wrapper exposing train and test dataloaders.

        We construct a 95%/5% split from the available `split` set and then
        apply `train_split` to the 95% train portion.
        """
        self.data_root = data_root
        self.split = split
        self.img_size = img_size
        self.max_depth_meters = max_depth_meters
        self.train_split = train_split

        all_pairs = _pair_vkitti_rgb_depth(self.data_root, self.split)
        if len(all_pairs) == 0:
            self.train = None
            self.test = None
            return

        rng = np.random.RandomState(0)
        indices = np.arange(len(all_pairs))
        rng.shuffle(indices)
        all_pairs = [all_pairs[i] for i in indices]

        n_total = len(all_pairs)
        n_test = max(1, int(0.05 * n_total))
        base_train = all_pairs[:-n_test]
        test_pairs = all_pairs[-n_test:]

        if 0.0 < self.train_split <= 1.0:
            n_keep = max(1, int(len(base_train) * self.train_split))
            train_pairs = base_train[:n_keep]
        else:
            train_pairs = base_train

        self.train = _VKITTIDepthDataset(
            train_pairs,
            img_size=self.img_size,
            is_train=True,
            max_depth_meters=self.max_depth_meters,
            augment=augment,
            flip=flip,
            colorjitter=colorjitter,
            gaussianblur=gaussianblur,
            grayscale=grayscale,
            gaussian_noise=gaussian_noise,
        ) if len(train_pairs) > 0 else None

        self.test = _VKITTIDepthDataset(
            test_pairs,
            img_size=self.img_size,
            is_train=False,
            max_depth_meters=self.max_depth_meters,
            augment=False,
        ) if len(test_pairs) > 0 else None

    def get_dataloaders(self, batch_size: int, shuffle: bool = True) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
        train_loader = None
        test_loader = None
        if self.train is not None:
            train_loader = DataLoader(self.train, batch_size=batch_size, shuffle=shuffle, num_workers=8, prefetch_factor=4)
        if self.test is not None:
            test_loader = DataLoader(self.test, batch_size=batch_size, shuffle=False, num_workers=8, prefetch_factor=4)
        return train_loader, test_loader


class _VKITTIDepthDataset(Dataset):
    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        img_size: Optional[Tuple[int, int]] = (256, 512),
        is_train: bool = False,
        max_depth_meters: float = 80.0,
        augment: bool = False,
        flip: bool = False,
        colorjitter: bool = False,
        gaussianblur: bool = False,
        grayscale: bool = False,
        gaussian_noise: bool = False,
    ):
        self.pairs = pairs
        self.img_size = img_size
        self.is_train = is_train
        self.max_depth_meters = max_depth_meters
        self.augment = augment
        self.flip = flip
        self.colorjitter = colorjitter
        self.gaussianblur = gaussianblur
        self.grayscale = grayscale
        self.gaussian_noise = gaussian_noise

        rgb_transforms = []
        depth_transforms = []
        if self.img_size is not None:
            rgb_transforms.append(transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.BICUBIC))
            depth_transforms.append(transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.NEAREST))

        if self.is_train and self.augment:
            if self.colorjitter:
                rgb_transforms.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05))
            if self.gaussianblur:
                rgb_transforms.append(transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)))
            if self.grayscale:
                rgb_transforms.append(transforms.RandomGrayscale(p=0.1))

        self.rgb_transform = transforms.Compose(rgb_transforms)
        self.depth_transform = transforms.Compose(depth_transforms)

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_rgb(self, path: str) -> torch.Tensor:
        img = decode_image(read_file(path), mode=ImageReadMode.RGB).to(torch.float32) / 255.0
        img = self.rgb_transform(img)
        if self.is_train and self.augment and self.gaussian_noise:
            if np.random.random() < 0.5:
                img = torch.clamp(img + torch.randn_like(img) * 0.03, 0.0, 1.0)
        return img

    def _load_depth(self, path: str) -> torch.Tensor:
        # VKITTI depth is uint16 PNG storing millimeters. Convert to meters.
        with Image.open(path) as im:
            depth_arr = np.array(im)
        if depth_arr.dtype != np.uint16 and depth_arr.max() <= 255:
            # Unexpected, but handle as uint8 normalized to 0..1 meters up to max_depth_meters
            depth_lin01 = depth_arr.astype(np.float32) / 255.0
            depth_m = depth_lin01 * self.max_depth_meters
        else:
            depth_m = depth_arr.astype(np.float32) / 1000.0  # millimeters -> meters
        depth_t = torch.from_numpy(depth_m).unsqueeze(0)  # [1, H, W]
        if self.depth_transform is not None:
            depth_t = self.depth_transform(depth_t)
        depth_norm = torch.clamp(depth_t / self.max_depth_meters, 0.0, 1.0)
        return depth_norm

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rgb_path, depth_path = self.pairs[idx]
        rgb = self._load_rgb(rgb_path)
        depth = self._load_depth(depth_path)

        if self.is_train and self.augment and self.flip and np.random.random() < 0.5:
            rgb = torch.flip(rgb, dims=[-1])
            depth = torch.flip(depth, dims=[-1])

        return rgb, depth






