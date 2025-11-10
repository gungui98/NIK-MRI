import argparse
import random
import os

import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from models.siren import NIKSiren
# from models.insngp_tcnn import NIKHashSiren
from utils.basic import parse_config
from datasets.cardiac import RadialDataset
from datasets.brain import BrainDataset

from utils.vis import visualize_reconstruction, visualize_all_echoes, visualize_kspace_error

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
    if config['model'] == 'siren':
        NIKmodel = NIKSiren(config)
    else:
        raise ValueError(f"Unsupported model: {config['model']}. Only 'siren' is currently supported.")

    NIKmodel.init_train()
    
    # Create output directory for images (separate from wandb run directory)
    output_dir = os.path.join('outputs', NIKmodel.exp_id)
    os.makedirs(output_dir, exist_ok=True)
    
    ktarget = torch.from_numpy(dataset.kspace_data_original).to(torch.complex64) # 12, 31, 92, 224
    img_recon_target = dataset.reconstruct_images(k_space=ktarget)
    

    for epoch in range(config['num_steps']):
        loss_epoch = 0
        for i, sample in enumerate(dataloader):
            # kcoord, kv = sample['coords'], sample['target']
            loss = NIKmodel.train_batch(sample)
            print(f"Epoch: {epoch}, Iter: {i}, Loss: {loss}")
            loss_epoch += loss

        # Log average loss per epoch to wandb
        avg_loss = loss_epoch / len(dataloader) if len(dataloader) > 0 else loss_epoch
        
        # Update learning rate scheduler
        if NIKmodel.lr_scheduler is not None:
            if isinstance(NIKmodel.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                NIKmodel.lr_scheduler.step(avg_loss)
            else:
                NIKmodel.lr_scheduler.step()
            current_lr = NIKmodel.optimizer.param_groups[0]['lr']
        else:
            current_lr = config['lr']
        
        # Prepare wandb log dictionary
        log_dict = {}
        if config.get('exp_summary') == 'wandb' and NIKmodel.exp_summary is not None:
            log_dict['loss'] = float(avg_loss)
            log_dict['epoch'] = epoch
            log_dict['learning_rate'] = current_lr
        
        # New: Reconstruct images using legacy approach and visualize
        if config['type'] == 'brain' and epoch % 10 == 0:
            print(f"\nGenerating legacy-style reconstruction visualizations for epoch {epoch}...")
            kpred = NIKmodel.test_batch(dataset=dataset)
            # De-normalize prediction back to original k-space scale for fair comparison
            if hasattr(dataset, 'norm_factor') and dataset.norm_factor is not None:
                kpred = kpred * torch.as_tensor(dataset.norm_factor, dtype=kpred.dtype, device=kpred.device)
            
            # Debug: Check predicted k-space statistics
            kpred_abs = torch.abs(kpred)
            print(f"  Predicted k-space stats: min={kpred_abs.min().item():.2e}, max={kpred_abs.max().item():.2e}, mean={kpred_abs.mean().item():.2e}")
            
            img_recon_pred = dataset.reconstruct_images(k_space=kpred)
            print(f"Reconstructed image shape: {img_recon_pred.shape}")
            
            # Debug: Check reconstructed image statistics
            img_pred_abs = np.abs(img_recon_pred)
            print(f"  Reconstructed image stats: min={img_pred_abs.min():.2e}, max={img_pred_abs.max():.2e}, mean={img_pred_abs.mean():.2e}")

            # Save single echo visualization (with comparison) to output directory
            result = visualize_reconstruction(img_recon_pred, slice_idx=0, echo_idx=0,
                                     output_prefix=f"train_recon_epoch{epoch}",
                                     img_target=img_recon_target, compare=True,
                                     output_dir=output_dir, return_fig=True)

            # Save all echoes visualization for that slice (with comparison)
            result_all_echoes = visualize_all_echoes(img_recon_pred, slice_idx=0,
                                 output_prefix=f"train_recon_all_echoes_epoch{epoch}",
                                 img_target=img_recon_target, compare=True,
                                 output_dir=output_dir, return_fig=True)

            # Visualize k-space learning errors
            print(f"\nGenerating k-space error visualizations for epoch {epoch}...")
            kspace_error_result = visualize_kspace_error(
                kpred=kpred, 
                ktarget=ktarget, 
                slice_idx=0, 
                echo_idx=0,
                output_prefix=f"train_kspace_error_epoch{epoch}",
                output_dir=output_dir, 
                return_fig=True
            )
            
            # Add images to log dict for wandb
            if config.get('exp_summary') == 'wandb' and NIKmodel.exp_summary is not None:
                import wandb
                # Add images to log dict
                if len(result) == 3:  # return_fig=True case
                    _magnitude, _phase, fig = result
                    log_dict['reconstruction_slice0_echo0'] = wandb.Image(fig)
                    fig.clf()  # Clear figure after logging
                    plt.close(fig)
                
                if isinstance(result_all_echoes, tuple):  # return_fig=True case
                    _output_file, fig_all = result_all_echoes
                    log_dict['reconstruction_all_echoes_slice0'] = wandb.Image(fig_all)
                    fig_all.clf()  # Clear figure after logging
                    plt.close(fig_all)
                
                # Add k-space error visualization to wandb
                if isinstance(kspace_error_result, tuple):  # return_fig=True case
                    _error_stats, fig_kspace_error = kspace_error_result
                    log_dict['kspace_error_analysis_slice0_echo0'] = wandb.Image(fig_kspace_error)
                    # Also log error metrics as scalars
                    log_dict['kspace_mean_complex_error'] = _error_stats['mean_complex_error']
                    log_dict['kspace_mean_complex_error_norm'] = _error_stats['mean_complex_error_norm']
                    log_dict['kspace_mean_phase_error'] = _error_stats['mean_phase_error']
                    log_dict['kspace_rmse'] = _error_stats['rmse']
                    log_dict['kspace_good_learning_ratio'] = _error_stats['good_learning_ratio']
                    log_dict['kspace_poor_learning_ratio'] = _error_stats['poor_learning_ratio']
                    fig_kspace_error.clf()  # Clear figure after logging
                    plt.close(fig_kspace_error)
        
        # Log everything to wandb at once with consistent step to ensure monotonic increase
        if config.get('exp_summary') == 'wandb' and NIKmodel.exp_summary is not None and log_dict:
            NIKmodel.exp_summary.log(log_dict, step=epoch)
        

if __name__ == '__main__':
    main()