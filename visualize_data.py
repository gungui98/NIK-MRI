import argparse
import random

import torch
import numpy as np

from utils.basic import parse_config
from datasets.brain import BrainDataset

from utils.vis import (visualize_reconstruction, visualize_all_echoes, find_best_slice)

def main():
    # parse args and get config
    config = parse_config('configs/config_brain.yml')

    # create dataset
    dataset = BrainDataset(config)

    # if requested, visualize the "perfect fit" by using ground-truth k-space targets
    ktarget = torch.from_numpy(dataset.kspace).to(torch.complex64) # 12, 31, 92, 224
    kpred = ktarget

        # New: Reconstruct images using legacy approach and visualize
    img_recon = dataset.reconstruct_images(k_space=kpred)  # (slices, echoes, kx, ky)
    print(f"Reconstructed image shape: {img_recon.shape}")

    # Choose a representative slice to visualize
    best_slice = find_best_slice(img_recon)

    # Save single echo visualization
    visualize_reconstruction(img_recon, slice_idx=best_slice, echo_idx=0,
                             output_prefix=f"{'selected' if config['is_selected'] else 'all'}_best")

    # Save all echoes visualization for that slice
    visualize_all_echoes(img_recon, slice_idx=best_slice,
                         output_prefix=f"{'selected' if config['is_selected'] else 'all'}_best")

if __name__ == '__main__':
    main()