import logging
import os

import matplotlib.pyplot as plt
import torch
import numpy as np
import wandb

from utils.vis import visualize_reconstruction, visualize_all_echoes, visualize_kspace_error

class Logger:
    def __init__(self, config, exp_summary=None):
        """
        Initialize Logger with config and optional wandb run object.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary
        exp_summary : wandb.run, optional
            Wandb run object for logging. If None, wandb logging will be disabled.
        """
        self.config = config
        self.exp_summary = exp_summary
        self.use_wandb = config.get('exp_summary') == 'wandb' and exp_summary is not None
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        self.logger.handlers.clear()
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        self.logger.addHandler(console_handler)
        
        logs_dir = 'logs'
        if os.path.exists(logs_dir) or os.path.exists(os.path.dirname(logs_dir)):
            os.makedirs(logs_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(logs_dir, 'train.log'))
            file_handler.setLevel(logging.INFO)
            self.logger.addHandler(file_handler)
    
    def log_batch_loss(self, epoch, iteration, loss):
        """Log individual batch loss to console."""
        if isinstance(loss, torch.Tensor):
            loss = loss.item()
        print(f"Epoch: {epoch}, Iter: {iteration}, Loss: {loss}")
    
    def log_epoch(self, epoch, avg_loss, current_lr, kpred=None, img_recon_pred=None):
        """
        Log training metrics and k-space/image stats for each epoch.
        
        Parameters
        ----------
        epoch : int
            Current epoch number
        avg_loss : float or torch.Tensor
            Average loss for the epoch
        current_lr : float
            Current learning rate
        kpred : torch.Tensor, optional
            Predicted k-space data for stats logging
        img_recon_pred : np.ndarray, optional
            Reconstructed image for stats logging
        
        Returns
        -------
        dict
            Dictionary containing metrics to log (for wandb)
        """
        # Convert loss to float if tensor
        if isinstance(avg_loss, torch.Tensor):
            avg_loss = float(avg_loss.item())
        
        # Log k-space stats if provided
        if kpred is not None:
            kpred_abs = torch.abs(kpred)
            print(f"  Predicted k-space stats: min={kpred_abs.min().item():.2e}, max={kpred_abs.max().item():.2e}, mean={kpred_abs.mean().item():.2e}")
        
        # Log image stats if provided
        if img_recon_pred is not None:
            img_pred_abs = np.abs(img_recon_pred)
            print(f"  Reconstructed image stats: min={img_pred_abs.min():.2e}, max={img_pred_abs.max():.2e}, mean={img_pred_abs.mean():.2e}")
            print(f"Reconstructed image shape: {img_recon_pred.shape}")
        
        # Create wandb log dictionary
        log_dict = {}
        if self.use_wandb:
            log_dict['loss'] = float(avg_loss)
            log_dict['epoch'] = epoch
            log_dict['learning_rate'] = current_lr
        
        return log_dict
    
    def visualize_kspace(self, epoch, kpred, ktarget, output_dir, slice_idx=0, echo_idx=0):
        """
        Visualize k-space errors and learning quality.
        
        Parameters
        ----------
        epoch : int
            Current epoch number
        kpred : torch.Tensor
            Predicted k-space data
        ktarget : torch.Tensor
            Target k-space data
        output_dir : str
            Directory to save visualizations
        slice_idx : int
            Slice index to visualize
        echo_idx : int
            Echo index to visualize
        
        Returns
        -------
        dict
            Dictionary containing k-space error metrics and visualization for wandb
        """
        print(f"\nGenerating k-space error visualizations for epoch {epoch}...")
        
        # Generate k-space error visualization
        kspace_error_result = visualize_kspace_error(
            kpred=kpred, 
            ktarget=ktarget, 
            slice_idx=slice_idx, 
            echo_idx=echo_idx,
            output_prefix=f"train_kspace_error_epoch{epoch}",
            output_dir=output_dir, 
            return_fig=True
        )
        
        log_dict = {}
        if self.use_wandb and isinstance(kspace_error_result, tuple):
            error_stats, fig_kspace_error = kspace_error_result
            log_dict['kspace_error_analysis_slice0_echo0'] = wandb.Image(fig_kspace_error)
            # Also log error metrics as scalars
            log_dict['kspace_mean_complex_error'] = error_stats['mean_complex_error']
            log_dict['kspace_mean_complex_error_norm'] = error_stats['mean_complex_error_norm']
            log_dict['kspace_mean_phase_error'] = error_stats['mean_phase_error']
            log_dict['kspace_rmse'] = error_stats['rmse']
            log_dict['kspace_good_learning_ratio'] = error_stats['good_learning_ratio']
            log_dict['kspace_poor_learning_ratio'] = error_stats['poor_learning_ratio']
            fig_kspace_error.clf()
            plt.close(fig_kspace_error)
        
        return log_dict
    
    def visualize_images(self, epoch, img_recon_pred, img_recon_target, output_dir, slice_idx=0):
        """
        Visualize reconstructed images (single echo and all echoes).
        
        Parameters
        ----------
        epoch : int
            Current epoch number
        img_recon_pred : np.ndarray
            Predicted reconstructed image
        img_recon_target : np.ndarray
            Target reconstructed image
        output_dir : str
            Directory to save visualizations
        slice_idx : int
            Slice index to visualize
        
        Returns
        -------
        dict
            Dictionary containing image visualizations for wandb
        """
        print(f"\nGenerating legacy-style reconstruction visualizations for epoch {epoch}...")
        
        # Generate single echo visualization
        result = visualize_reconstruction(
            img_recon_pred, 
            slice_idx=slice_idx, 
            echo_idx=0,
            output_prefix=f"train_recon_epoch{epoch}",
            img_target=img_recon_target, 
            compare=True,
            output_dir=output_dir, 
            return_fig=True
        )
        
        # Generate all echoes visualization
        result_all_echoes = visualize_all_echoes(
            img_recon_pred, 
            slice_idx=slice_idx,
            output_prefix=f"train_recon_all_echoes_epoch{epoch}",
            img_target=img_recon_target, 
            compare=True,
            output_dir=output_dir, 
            return_fig=True
        )
        
        log_dict = {}
        if self.use_wandb:
            # Add reconstruction image (single echo)
            if len(result) == 3:  # return_fig=True case
                _magnitude, _phase, fig = result
                log_dict['reconstruction_slice0_echo0'] = wandb.Image(fig)
                fig.clf()
                plt.close(fig)
            
            # Add all echoes visualization
            if isinstance(result_all_echoes, tuple):  # return_fig=True case
                _output_file, fig_all = result_all_echoes
                log_dict['reconstruction_all_echoes_slice0'] = wandb.Image(fig_all)
                fig_all.clf()
                plt.close(fig_all)
        
        return log_dict
    
    def log_to_wandb(self, log_dict, step):
        """Log dictionary to wandb."""
        if self.use_wandb and log_dict:
            self.exp_summary.log(log_dict, step=step)
