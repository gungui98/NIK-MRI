import argparse
import random
import os

import torch
import numpy as np
from torch.utils.data import DataLoader

from models.siren import NIKSiren
from models.nerf import NIKNeRF
# from models.insngp_tcnn import NIKHashSiren
from utils.basic import parse_config
from datasets.brain import BrainDataset
from utils.logger import Logger

class Trainer:
    def __init__(self, config):
        self.config = config
        self.dataset = BrainDataset(config)
        self.dataloader = DataLoader(self.dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
        
        model_type = config.get('model', 'nerf').lower()
        if model_type == 'nerf':
            self.NIKmodel = NIKNeRF(config)
        else:
            self.NIKmodel = NIKSiren(config)
            
        self.NIKmodel.init_train()
        self.output_dir = os.path.join('outputs', self.NIKmodel.exp_id)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize logger with config and wandb run
        self.logger = Logger(config, self.NIKmodel.exp_summary)
        
        # Prepare target k-space and reconstructed images for visualization
        self.ktarget = torch.from_numpy(self.dataset.kspace_data_original).to(torch.complex64)
        self.img_recon_target = self.dataset.reconstruct_images(k_space=self.ktarget)
        
    def train(self):
        for epoch in range(self.config['num_epochs']):
            loss_epoch = 0
            for i, sample in enumerate(self.dataloader):
                loss = self.NIKmodel.train_batch(sample)
                self.logger.log_batch_loss(epoch, i, loss)
                loss_epoch += loss

            # Calculate average loss and update learning rate scheduler
            avg_loss = loss_epoch / len(self.dataloader) if len(self.dataloader) > 0 else loss_epoch
            
            # Update learning rate scheduler
            if self.NIKmodel.lr_scheduler is not None:
                if isinstance(self.NIKmodel.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.NIKmodel.lr_scheduler.step(avg_loss)
                else:
                    self.NIKmodel.lr_scheduler.step()
                current_lr = self.NIKmodel.optimizer.param_groups[0]['lr']
            else:
                current_lr = self.config['lr']
            
            # Generate predictions and visualizations every 10 epochs
            if epoch % 10 == 0:
                kpred = self.NIKmodel.test_batch(dataset=self.dataset)
                # De-normalize prediction back to original k-space scale for fair comparison
                kpred = self.dataset.denormalize_kspace(kpred)
                img_recon_pred = self.dataset.reconstruct_images(k_space=kpred)
                
                # Log training metrics + k-space/image stats
                log_dict = self.logger.log_epoch(epoch, avg_loss, current_lr, kpred=kpred, img_recon_pred=img_recon_pred)
                
                # Visualize images
                log_dict.update(self.logger.visualize_images(epoch, img_recon_pred, self.img_recon_target, self.output_dir, slice_idx=0))
                
                # Visualize k-space errors
                log_dict.update(self.logger.visualize_kspace(epoch, kpred, self.ktarget, self.output_dir, slice_idx=0, echo_idx=0))
            else:
                # Log epoch metrics only (no visualizations)
                log_dict = self.logger.log_epoch(epoch, avg_loss, current_lr)
            
            # Log everything to wandb
            self.logger.log_to_wandb(log_dict, epoch)
    
    def test(self):
        for epoch in range(self.config['num_epochs']):
            loss_epoch = 0
            for i, sample in enumerate(self.dataloader):
                loss = self.NIKmodel.test_batch(sample)
                self.logger.log_batch_loss(epoch, i, loss)
                loss_epoch += loss
            avg_loss = loss_epoch / len(self.dataloader) if len(self.dataloader) > 0 else loss_epoch
            if self.NIKmodel.lr_scheduler is not None:
                if isinstance(self.NIKmodel.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.NIKmodel.lr_scheduler.step(avg_loss)
                else:
                    self.NIKmodel.lr_scheduler.step()
            
def main():
    # parse args and get config
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/config_brain.yml')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('-seed', '--seed', type=int, default=0)
    args = parser.parse_args()

    # set gpu and random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # parse config
    config = parse_config(args.config)
    config['gpu'] = args.gpu

    # create trainer and train
    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
