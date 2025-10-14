#!/usr/bin/env python
"""
Script to evaluate trained Gaussian depth model and visualize sample predictions.
"""
import os
import argparse
import torch
import matplotlib.pyplot as plt
from dataloaders.simple_depth import DepthDataset
from models.depth import gaussian as gaussian_model
import torch.nn.functional as F
import re
from evaluation.utils import get_predictions, visualize_results

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate depth model and plot sample predictions")
    parser.add_argument("--metric", default="rmse", choices=["rmse", "nll", "vloss"],
                        help="Select the metric for best model checkpoint (rmse, nll, vloss)")
    parser.add_argument("--save-dir", type=str, required=True,
                        help="Directory containing saved model checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Computation device (cuda or cpu)")
    parser.add_argument("--num-samples", type=int, default=4,
                        help="Number of samples to visualize")
    parser.add_argument("--post-hoc", action="store_true",
                        help="Use post-hoc model")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Load dataset
    dataset = DepthDataset()
    _, test_loader = dataset.get_dataloaders(batch_size=args.num_samples)

    # Grab a batch to infer input shape
    images, depths = next(iter(test_loader))
    images = images.to(device)
    depths = depths.to(device)

    # Build model architecture
    model, opts = gaussian_model.create(input_shape=images.shape[1:], activation=F.relu, num_class=1)
    model.to(device)
    model.eval()

    # Locate checkpoint file for the selected metric
    ckpt_files = [f for f in os.listdir(args.save_dir)
                  if f.startswith(f"unet_{args.metric}_") and f.endswith(".pth")]
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoints found for metric {args.metric} in {args.save_dir}")
    # sort by iteration number in filename
    ckpt_files.sort(key=lambda x: int(re.search(r"(\d+)\.pth$", x).group(1)))
    ckpt_path = os.path.join(args.save_dir, ckpt_files[-1])
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))

    # Set up post-hoc model if needed
    post_hoc_model = None
    if args.post_hoc:
        post_hoc_model, _ = gaussian_model.create_post_hoc(input_shape=images.shape[1:], activation=F.relu, num_class=1)
        post_hoc_model.to(device)
        post_hoc_model.eval()
        post_hoc_model.load_state_dict(torch.load("save/post_hoc_model.pth", map_location=device, weights_only=True))

    # Get predictions
    results = get_predictions(model, images, depths, post_hoc_model, device)
    
    # Visualize results
    visualize_results(results, args.num_samples, args.metric)


if __name__ == "__main__":
    main() 