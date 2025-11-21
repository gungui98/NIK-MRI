import argparse
import random
import os
os.environ['WANDB_DISABLED'] = 'true'   

import torch
import numpy as np
from torch.utils.data import DataLoader

from models.siren import NIKSiren
# from models.insngp_tcnn import NIKHashSiren
from utils.basic import parse_config
from datasets.brain import BrainDataset
from utils.logger import Logger

if __name__ == '__main__':
    config = parse_config('configs/config_brain.yml')
    dataset = BrainDataset(config)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    # For complex tensors, compute min/max on real and imaginary parts separately
    min_real = float('inf')
    max_real = float('-inf')
    min_imag = float('inf')
    max_imag = float('-inf')
    for sample in dataloader:
        targets = sample['targets']
        real_part = targets.real
        imag_part = targets.imag
        min_real = min(min_real, real_part.min().item())
        max_real = max(max_real, real_part.max().item())
        min_imag = min(min_imag, imag_part.min().item())
        max_imag = max(max_imag, imag_part.max().item())
    print(f'Min real value: {min_real}, Max real value: {max_real}')
    print(f'Min imag value: {min_imag}, Max imag value: {max_imag}')
    NIKmodel = NIKSiren(config)
    NIKmodel.init_train()
    # create a sample with value range [min_value, max_value]
    sample = {'coords': torch.randn(config['batch_size'], 4), 
              'targets': (torch.randn(config['batch_size'], 1) * (max_real - min_real) + min_real) + 1j * (torch.randn(config['batch_size'], 1) * (max_imag - min_imag) + min_imag)}
    for epoch in range(10000):
        loss_epoch = 0
        loss = NIKmodel.train_batch(sample)
        loss_epoch += loss
        print(f'Epoch {epoch}, Loss: {loss}')