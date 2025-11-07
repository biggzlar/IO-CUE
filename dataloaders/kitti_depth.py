import os
import glob
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import decode_image, ImageReadMode, read_file
from PIL import Image
from pathlib import Path

# from tqdm import tqdm

def _list_sequence_dirs(root: str) -> List[str]:
    """Return sorted list of sequence directories for a split (train/val)."""
    if not os.path.isdir(root):
        return []
    seqs = [os.path.join(root, d) for d in sorted(os.listdir(root))]
    return [d for d in seqs if os.path.isdir(d)]


def _candidate_dirs(sequence_dir: str, camera_id: str, data_root: Optional[str] = None) -> Tuple[List[str], List[str]]:
    """Return candidate (rgb_dirs, depth_dirs) for a KITTI sequence.

    We try multiple known layouts:
    - RGB: <seq>/<camera_id>/data or <seq>/<camera_id>
    - Depth GT: <seq>/proj_depth/groundtruth/<camera_id>
    - Depth raw proj: <seq>/proj_depth/velodyne_raw/<camera_id>
    - Fallback: <seq>/proj_depth/<camera_id>
    """
    rgb_dirs = [
        os.path.join(sequence_dir, camera_id, "data"),  # rarely populated under split root
        os.path.join(sequence_dir, camera_id),
    ]
    # Add date-based raw RGB location: <root>/<date>/<sequence>/<camera_id>/data
    if data_root is not None:
        seq_name = os.path.basename(sequence_dir.rstrip(os.sep))
        date_prefix = seq_name.split("_drive")[0]  # e.g., 2011_09_26
        rgb_dirs.insert(0, os.path.join(data_root, date_prefix, seq_name, camera_id, "data"))
    depth_dirs = [
        os.path.join(sequence_dir, "proj_depth", "groundtruth", camera_id),
        os.path.join(sequence_dir, "proj_depth", "velodyne_raw", camera_id),
        os.path.join(sequence_dir, "proj_depth", camera_id),
    ]
    return rgb_dirs, depth_dirs


def _first_existing_dir(dirs: List[str]) -> Optional[str]:
    for d in dirs:
        if os.path.isdir(d):
            return d
    return None


def _pair_rgb_depth(rgb_dir: str, depth_dir: str) -> List[Tuple[str, str]]:
    """Pair RGB and depth files by common filename.

    KITTI uses 6-10 digit zero-padded filenames with .png extension for both.
    We match by basename to be robust to subdir differences.
    """
    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png")) + glob.glob(os.path.join(rgb_dir, "*.jpg")))
    depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.png")))

    if not rgb_files or not depth_files:
        return []

    # Match by stem (without extension) to handle .jpg vs .png
    def _stem(p: str) -> str:
        return os.path.splitext(os.path.basename(p))[0]

    depth_map = {_stem(p): p for p in depth_files}
    pairs = []
    for rgb_path in rgb_files:
        key = _stem(rgb_path)
        if key in depth_map:
            pairs.append((rgb_path, depth_map[key]))
    return pairs


def read_pairs(pairs_file: str, use_dense_output: bool = True, dense_dirs: Optional[List[str]] = None) -> List[Tuple[str, str]]:
    """Read (rgb, depth) pairs from an eigen-style pairs file.

    Lines contain two whitespace-separated relative paths. Paths are resolved
    relative to the parent directory of the pairs file's parent, matching the
    logic in KITTI_Dense_Depth's run_depth_extraction.py.
    """
    pf = Path(pairs_file)
    root = pf.parent.parent
    pairs: List[Tuple[str, str]] = []
    dense_dir_candidates = dense_dirs or ["dense_map_py", "dense_depth_py"]
    chosen_dense_dir: Optional[Path] = None
    if use_dense_output:
        for cand in dense_dir_candidates:
            dd = root / cand
            if dd.exists() and dd.is_dir():
                chosen_dense_dir = dd
                break

    with pf.open('r') as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            img_rel, dep_rel = line.split()
            img_path = str(root / img_rel)
            if use_dense_output and chosen_dense_dir is not None:
                dense_path = chosen_dense_dir / f"{i:05d}.png"
                pairs.append((img_path, str(dense_path)))
            else:
                pairs.append((img_path, str(root / dep_rel)))
    return pairs


def scan_kitti_pairs(data_root: str, split: str, camera_id: str) -> List[Tuple[str, str]]:
    """Scan KITTI data_root for (rgb, depth) pairs for the given split and camera.

    Args:
        data_root: Root of KITTI data, e.g., /mnt/data/kitti_data
        split: 'train' or 'val'
        camera_id: e.g., 'image_02' (left) or 'image_03' (right)
    """
    split_root = os.path.join(data_root, split)
    sequence_dirs = _list_sequence_dirs(split_root)

    all_pairs: List[Tuple[str, str]] = []
    for seq_dir in sequence_dirs:
        rgb_dirs, depth_dirs = _candidate_dirs(seq_dir, camera_id, data_root=data_root)
        rgb_dir = _first_existing_dir(rgb_dirs)
        depth_dir = _first_existing_dir(depth_dirs)
        if rgb_dir is None or depth_dir is None:
            continue
        pairs = _pair_rgb_depth(rgb_dir, depth_dir)
        if pairs:
            all_pairs.extend(pairs)

    return all_pairs


class KITTIDepthDataset:
    def __init__(
        self,
        # Common/simple-depth-style args
        path: Optional[str] = None,
        img_size: Optional[Tuple[int, int]] = (128, 256),
        augment: bool = False,
        augment_test_data: bool = False,
        train_split: float = 1.0,
        flip: bool = False,
        colorjitter: bool = False,
        gaussianblur: bool = False,
        grayscale: bool = False,
        gaussian_noise: bool = False,
        # KITTI-specific optional args (kept for flexibility/back-compat)
        data_root: str = "/mnt/data/kitti_data",
        camera_id: str = "image_02",
        pairs_file: Optional[str] = None,
        use_dense_output: bool = True,
        dense_dirs: Optional[List[str]] = None,
        max_depth_meters: float = 80.0,
        dense_assumed_max_meters: float = 80.0,
    ):
        """KITTI Depth dataset wrapper that exposes train and val dataloaders.

        Args:
            data_root: Root folder containing 'train/' and 'val/' sequences
            camera_id: 'image_02' (left) or 'image_03' (right)
            img_size: Output size (H, W)
            max_depth_meters: Depth normalization upper bound
            train_split: Fraction of training frames to keep (0-1]
            augment: Apply augmentations to train split
        """
        # Store config
        self.path = path
        self.data_root = data_root
        self.camera_id = camera_id
        self.img_size = img_size
        self.max_depth_meters = max_depth_meters
        self.train_split = train_split
        self.dense_assumed_max_meters = dense_assumed_max_meters
        self.augment = augment
        self.augment_test_data = augment_test_data
        self.flip = flip
        self.colorjitter = colorjitter
        self.gaussianblur = gaussianblur
        self.grayscale = grayscale
        self.gaussian_noise = gaussian_noise

        # Default: use constant eigen pairs path like other loaders with fixed sources
        default_pairs = "/mnt/data/KITTI_Dense_Depth/utils/eigen_train_pairs.txt"

        # Resolve source based on provided path/pairs_file/data_root
        use_pairs_path: Optional[str] = None
        scan_from_root: bool = False

        if self.path is not None:
            if os.path.isfile(self.path):
                use_pairs_path = self.path
            elif os.path.isdir(self.path):
                # Treat provided path as data root
                self.data_root = self.path
                scan_from_root = True
        elif pairs_file is not None and os.path.isfile(pairs_file):
            use_pairs_path = pairs_file
        else:
            # Fallback to default eigen pairs if available, else scan data_root
            if os.path.isfile(default_pairs):
                use_pairs_path = default_pairs
            else:
                scan_from_root = True

        if use_pairs_path is not None and os.path.isfile(use_pairs_path):
            all_pairs = read_pairs(use_pairs_path, use_dense_output=use_dense_output, dense_dirs=dense_dirs)
            # Create a 95%/5% split like HyperSim; apply train_split to the 95% portion
            if len(all_pairs) == 0:
                self.train = None
                self.test = None
            else:
                # Deterministic shuffle
                rng = np.random.RandomState(0)
                indices = np.arange(len(all_pairs))
                rng.shuffle(indices)
                all_pairs = [all_pairs[i] for i in indices]

                n_total = len(all_pairs)
                n_test = max(1, int(0.05 * n_total))
                base_train = all_pairs[:-n_test]
                test_pairs = all_pairs[-n_test:]

                # Apply train_split to the base_train portion
                if 0.0 < self.train_split <= 1.0:
                    n_keep = max(1, int(len(base_train) * self.train_split))
                    train_pairs = base_train[:n_keep]
                else:
                    train_pairs = base_train

                self.train = _KITTIDepthDataset(
                    train_pairs,
                    img_size=self.img_size,
                    is_train=True,
                    max_depth_meters=self.max_depth_meters,
                    dense_assumed_max_meters=self.dense_assumed_max_meters,
                    augment=self.augment,
                    flip=self.flip,
                    colorjitter=self.colorjitter,
                    gaussianblur=self.gaussianblur,
                    grayscale=self.grayscale,
                    gaussian_noise=self.gaussian_noise,
                ) if len(train_pairs) > 0 else None

                self.test = _KITTIDepthDataset(
                    test_pairs,
                    img_size=self.img_size,
                    is_train=False,
                    max_depth_meters=self.max_depth_meters,
                    dense_assumed_max_meters=self.dense_assumed_max_meters,
                    augment=self.augment_test_data,
                    flip=self.flip,
                    colorjitter=self.colorjitter,
                    gaussianblur=self.gaussianblur,
                    grayscale=self.grayscale,
                    gaussian_noise=self.gaussian_noise,
                ) if len(test_pairs) > 0 else None
        elif scan_from_root:
            # Fallback: try scanning a conventional KITTI directory layout
            train_pairs = scan_kitti_pairs(self.data_root, "train", self.camera_id)
            val_pairs = scan_kitti_pairs(self.data_root, "val", self.camera_id)

            if 0.0 < self.train_split < 1.0 and len(train_pairs) > 0:
                cutoff = int(len(train_pairs) * self.train_split)
                train_pairs = train_pairs[:cutoff]

            self.train = _KITTIDepthDataset(
                train_pairs,
                img_size=self.img_size,
                is_train=True,
                max_depth_meters=self.max_depth_meters,
                dense_assumed_max_meters=self.dense_assumed_max_meters,
                augment=self.augment,
                flip=self.flip,
                colorjitter=self.colorjitter,
                gaussianblur=self.gaussianblur,
                grayscale=self.grayscale,
                gaussian_noise=self.gaussian_noise,
            ) if train_pairs else None

            self.test = _KITTIDepthDataset(
                val_pairs,
                img_size=self.img_size,
                is_train=False,
                max_depth_meters=self.max_depth_meters,
                dense_assumed_max_meters=self.dense_assumed_max_meters,
                augment=self.augment_test_data,
                flip=self.flip,
                colorjitter=self.colorjitter,
                gaussianblur=self.gaussianblur,
                grayscale=self.grayscale,
                gaussian_noise=self.gaussian_noise,
            ) if val_pairs else None

    def get_dataloaders(self, batch_size: int, shuffle: bool = True) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
        train_loader = None
        test_loader = None
        if self.train is not None:
            train_loader = DataLoader(self.train, batch_size=batch_size, shuffle=shuffle, num_workers=8, prefetch_factor=4)
        if self.test is not None:
            # Match simple depth behavior: allow shuffle control on test loader too
            test_loader = DataLoader(self.test, batch_size=batch_size, shuffle=shuffle, num_workers=8, prefetch_factor=4)
        return train_loader, test_loader


class _KITTIDepthDataset(Dataset):
    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        img_size: Tuple[int, int] = (128, 256),
        is_train: bool = False,
        max_depth_meters: float = 80.0,
        dense_assumed_max_meters: float = 80.0,
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
        self.dense_assumed_max_meters = dense_assumed_max_meters
        self.augment = augment
        self.flip = flip
        self.colorjitter = colorjitter
        self.gaussianblur = gaussianblur
        self.grayscale = grayscale
        self.gaussian_noise = gaussian_noise

        # Build transforms
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
        # Clamp after interpolation/augs to avoid minor overshoots (e.g., bicubic) causing values > 1
        img = torch.clamp(img, 0.0, 1.0)
        if self.is_train and self.augment and self.gaussian_noise:
            if np.random.random() < 0.5:
                img = torch.clamp(img + torch.randn_like(img) * 0.03, 0.0, 1.0)
        return img

    def _load_depth(self, path: str) -> torch.Tensor:
        # KITTI groundtruth depth is stored as 16-bit PNG with scaling factor 256
        # Use PIL to preserve 16-bit, then convert to tensor
        with Image.open(path) as im:
            depth_arr = np.array(im)

        # Branch on bit depth: uint16 is GT meters scaled by 256; uint8 is dense output normalized 0..255
        if depth_arr.dtype == np.uint16 or depth_arr.max() > 255:
            depth_m = torch.from_numpy(depth_arr.astype(np.float32)).unsqueeze(0) / 256.0  # [1, H, W]
            if self.depth_transform is not None:
                depth_m = self.depth_transform(depth_m)
            depth_norm = torch.clamp(depth_m / self.max_depth_meters, 0.0, 1.0)
            return depth_norm
        else:
            # Treat dense uint8 as linear depth up to dense_assumed_max_meters, then cap by max_depth_meters
            depth_lin01 = torch.from_numpy(depth_arr.astype(np.float32)).unsqueeze(0) / 255.0
            depth_m = depth_lin01 * self.dense_assumed_max_meters
            if self.depth_transform is not None:
                depth_m = self.depth_transform(depth_m)
            depth_norm = torch.clamp(depth_m / self.max_depth_meters, 0.0, 1.0)
            return depth_norm

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rgb_path, depth_path = self.pairs[idx]
        rgb = self._load_rgb(rgb_path)
        depth = self._load_depth(depth_path)

        # Optional paired horizontal flip
        if self.is_train and self.augment and self.flip and np.random.random() < 0.5:
            rgb = torch.flip(rgb, dims=[-1])
            depth = torch.flip(depth, dims=[-1])

        return rgb, depth


if __name__ == "__main__":
    # Plot 5 random samples (RGB left, depth right) in a single figure without resizing
    import random
    import matplotlib.pyplot as plt

    pairs_path = "/mnt/data/KITTI_Dense_Depth/utils/eigen_train_pairs.txt"

    ds = KITTIDepthDataset(
        data_root="/mnt/data/kitti_data",
        camera_id="image_02",
        img_size=None,  # keep original sizes in dataset transforms
        augment=False,
        pairs_file=pairs_path,
        use_dense_output=True,
        dense_assumed_max_meters=80.0,
    )

    dataset_obj = ds.train if ds.train is not None else ds.test
    if dataset_obj is None or len(dataset_obj) == 0:
        raise RuntimeError("No KITTI samples found to visualize.")

    num_rows = min(5, len(dataset_obj.pairs))
    idxs = random.sample(range(len(dataset_obj.pairs)), k=num_rows)

    fig, axes = plt.subplots(num_rows, 2, figsize=(12, 3 * num_rows))
    if num_rows == 1:
        axes = np.array([axes])

    for r, idx in enumerate(idxs):
        # rgb_path, depth_path = dataset_obj.pairs[idx]
        # import ipdb; ipdb.set_trace()

        # with Image.open(rgb_path) as im_rgb:
        #     rgb_np = np.array(im_rgb.convert("RGB"))

        # with Image.open(depth_path) as im_d:
        #     depth_arr = np.array(im_d)
        # if depth_arr.dtype == np.uint16 or depth_arr.max() > 255:
        #     depth_m = depth_arr.astype(np.float32) / 256.0
        #     depth_vis = np.clip(depth_m / ds.max_depth_meters, 0.0, 1.0)
        # else:
        #     depth_lin01 = depth_arr.astype(np.float32) / 255.0
        #     depth_m = depth_lin01 * ds.dense_assumed_max_meters
        #     depth_vis = np.clip(depth_m / ds.max_depth_meters, 0.0, 1.0)

        rgb, depth = dataset_obj.__getitem__(idx)
        axes[r, 0].imshow(rgb.permute(1, 2, 0).numpy())
        axes[r, 0].set_title("RGB")
        axes[r, 0].axis("off")
        axes[r, 1].imshow(depth.squeeze().numpy(), cmap="plasma")
        axes[r, 1].set_title("Depth (normalized)")
        axes[r, 1].axis("off")
        # import ipdb; ipdb.set_trace()

    # for x, y in tqdm(dataset_obj):
    #     if x.min() < 0.0 or x.max() > 1.0 or y.min() < 0.0 or y.max() > 1.0:
    #         import ipdb; ipdb.set_trace()
    #         pass        

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "kitti_random5_pairs.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


