import torch
import argparse
import numpy as np
import random

import wandb

from models.siren import NIKSiren
# from models.insngp_tcnn import NIKHashSiren
from utils.basic import parse_config
from torch.utils.data import DataLoader
from datasets.cardiac import RadialDataset
from datasets.brain import BrainDataset

from utils.vis import (k2img, 
                       visualize_reconstruction, visualize_all_echoes)

def main():
    # parse args and get config
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/config_brain.yml')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('-s', '--slice_name', type=str, default='CINE_S1_rad_AA')
    parser.add_argument('-seed', '--seed', type=int, default=0)
    # parser.add_argument('--use_targets', action='store_true', help='Use full training targets instead of model output')
    # parser.add_argument('-s', '--seed', type=int, default=0)
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
        for i, sample in enumerate(dataloader):
            # kcoord, kv = sample['coords'], sample['target']
            loss = NIKmodel.train_batch(sample)
            print(f"Epoch: {epoch}, Iter: {i}, Loss: {loss}")
            loss_epoch += loss

        # if requested, visualize the "perfect fit" by using ground-truth k-space targets
        ktarget = torch.from_numpy(dataset.kspace_data_original).to(torch.complex64) # 12, 31, 92, 224
        kpred = NIKmodel.test_batch(kspace_data_original=dataset.kspace_data_original)
        
        # kpred = ktarget

        # kpred[kpred != 0] = (dataset.eps / torch.abs(kpred[kpred != 0]) - dataset.eps) * (kpred[kpred != 0] / torch.abs(kpred[kpred != 0]))

        # Existing k2img visualization
        # vis_img = k2img(kpred, dataset.csm)
        # vis_img_all_slices = k2img_all_slices(kpred, dataset.csm)
        # write_to_video(vis_img['k_mag'], vis_img['combined_mag'], vis_img['combined_phase'], vis_img['combined_img'], kpred)
        # write_slices_to_video(vis_img_all_slices, fps=5, output_prefix=f"train_recon_all_slices_epoch{epoch}")
        # write to nii.gz files
        # write_slices_to_nii(vis_img_all_slices, output_prefix=f"train_recon_all_slices_epoch{epoch}")
        
        # New: Reconstruct images using legacy approach and visualize
        if config['type'] == 'brain' and epoch % 10 == 0:
            print(f"\nGenerating legacy-style reconstruction visualizations for epoch {epoch}...")
            img_recon = dataset.reconstruct_images(k_space=kpred)  # (echoes, kx, ky)
            print(f"Reconstructed image shape: {img_recon.shape}")
            
            # Add batch dimension for consistency with multi-slice format
            img_for_vis = img_recon[None]  # (1, echoes, kx, ky)
            
            # Save single echo visualization
            visualize_reconstruction(img_for_vis, slice_idx=0, echo_idx=0, 
                                    output_prefix=f"train_recon_epoch{epoch}")
            
            # Save all echoes visualization
            visualize_all_echoes(img_for_vis, slice_idx=0,
                               output_prefix=f"train_recon_all_echoes_epoch{epoch}")
        
        # log_dict = {
        #     'loss': loss_epoch/len(dataloader),
        #     'k': wandb.Video(vis_img['k_mag'].transpose(0,2,3,1), fps=10, format="gif"),
        #     'img': wandb.Video(vis_img['combined_mag'].transpose(0,2,3,1), fps=10, format="gif"), 
        #     'img_phase': wandb.Video(vis_img['combined_phase'].transpose(0,2,3,1), fps=10, format="gif"),
        #     'img_combined': wandb.Video(vis_img['combined_img'].transpose(0,2,3,1), fps=10, format="gif"),
        #     'khist': wandb.Histogram(torch.view_as_real(kpred).detach().cpu().numpy().flatten()),
        # }
        
        # NIKmodel.exp_summary_log(log_dict)



if __name__ == '__main__':
    main()