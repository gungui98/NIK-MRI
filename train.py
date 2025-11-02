import argparse
import random

import torch
import numpy as np
import wandb
from torch.utils.data import DataLoader


from models.siren import NIKSiren
# from models.insngp_tcnn import NIKHashSiren
from utils.basic import parse_config
from datasets.cardiac import RadialDataset
from datasets.brain import BrainDataset

from utils.vis import (k2img,visualize_reconstruction, visualize_all_echoes, find_best_slice)

def main():
    # parse args and get config
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/config_brain.yml')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('-s', '--slice_name', type=str, default='CINE_S1_rad_AA')
    parser.add_argument('-seed', '--seed', type=int, default=0)
    args = parser.parse_args()

    # enable Double precision
    torch.set_default_dtype(torch.float32)

    # set gpu and random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # parse config
    slice_name = args.slice_name
    config = parse_config(args.config)
    config['slice_name'] = slice_name
    config['gpu'] = args.gpu

    # create dataset
    if config['type'] == 'brain':
        dataset = BrainDataset(config)
    elif config['type'] == 'cardiac':
        dataset = RadialDataset(config)
    else:
        raise ValueError(f"Unknown dataset_type: {config['type']}. Must be 'brain' or 'cardiac'")
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    # config['eps'] = dataset.eps
    # create model
    if config['model'] == 'siren':
        NIKmodel = NIKSiren(config)
    # elif config['model'] == 'hashsiren':
    #     NIKmodel = NIKHashSiren(config)

    NIKmodel.init_train()

    for epoch in range(config['num_steps']):
        loss_epoch = 0
        # for i, sample in enumerate(dataloader):
        #     # kcoord, kv = sample['coords'], sample['target']
        #     loss = NIKmodel.train_batch(sample)
        #     print(f"Epoch: {epoch}, Iter: {i}, Loss: {loss}")
        #     loss_epoch += loss

        # if requested, visualize the "perfect fit" by using ground-truth k-space targets
        ktarget = torch.from_numpy(dataset.kspace_data_original).to(torch.complex64) # 12, 31, 92, 224
        # kpred = NIKmodel.test_batch(kspace_data_original=dataset.kspace_data_original)
        
        kpred = ktarget

        # New: Reconstruct images using legacy approach and visualize
        if config['type'] == 'brain' and epoch % 10 == 0:
            print(f"\nGenerating legacy-style reconstruction visualizations for epoch {epoch}...")
            img_recon = dataset.reconstruct_images(k_space=kpred)  # (slices, echoes, kx, ky)
            print(f"Reconstructed image shape: {img_recon.shape}")

            # Choose a representative slice to visualize
            best_slice = find_best_slice(img_recon)

            # Save single echo visualization
            visualize_reconstruction(img_recon, slice_idx=best_slice, echo_idx=0,
                                     output_prefix=f"train_recon_epoch{epoch}")

            # Save all echoes visualization for that slice
            visualize_all_echoes(img_recon, slice_idx=best_slice,
                                 output_prefix=f"train_recon_all_echoes_epoch{epoch}")
        

if __name__ == '__main__':
    main()