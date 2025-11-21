import os
import torch
import torch.nn as nn
import numpy as np
from abc import abstractmethod, ABC
from utils.mri import ifft2c_mri, coilcombine
from utils.loss import HDRLoss_FF, AdaptiveHDRLoss, MixedLoss, WeightedHDRLoss_FF, ComplexSpectralLoss
from datetime import datetime


"""
This is the base class for all the NIK models.
The NIK model classes handle the all details of the training and testing 
process except the network structure. The network structure is defined in
the mlp model classes.
Therefore, if you want to create a new model with existing mlp models, 
you need to inherit this class and implement the following methods:
TODO
If you want to create a new model with new mlp models, you also need to
define a nn.Module class for the mlp model, and implement the following 
methods:
TODO
"""

class NIKBase(nn.Module, ABC):
    """
    This is the base class for all the NIK models.
    If you want to create a new model, you need to inherit this class and implement the following methods:
        1. create_mlp: Create the MLP network.
        2. pre_process(optional): Pre process the coordinates. Default is to return the coordinates as it is.
        3. post_process(optional): Post process the output of the network. Default is to return the output as it is.
    Then the forward function will be implemented automatically by:
        post_process(mlp(pre_process(coords)))
    To keep the code clean, we recommend you to create a new file for each model, and seperate the pre_process 
    and post_process out from the forward of mlp.
    """
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        # needed for both training and testing
        # will be set in corresponding functions
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network = None
        self.output = None

        # needed for training
        self.model_save_path = None
        self.criterion = None
        self.optimizer = None
        self.lr_scheduler = None
        self.exp_id = None
        self.exp_summary = None
        self.global_step = 0

        # needed for testing
        self.weight_path = None
        self.result_save_path = None


    @abstractmethod
    def create_network(self) -> nn.Module:
        """Create the MLP network.
        Should be reimplemented to create an MLP model for self.network.
        """
        pass

    def load_network(self):
        """Load the network parameters from the path."""
        path = self.config['weight_path']
        self.network.load_state_dict(torch.load(path, map_location=self.device))

    def save_network(self, name):
        """Save the network parameters to the path."""
        path = os.path.join(self.model_save_path, name)
        torch.save(self.network.state_dict(), path)
    
    def init_expsummary(self):
        """
        Initialize the visualization tools.
        Should be called in init_train after the initialization of self.exp_id.
        """
        if self.config['exp_summary'] == 'wandb':
            import wandb
            # Set wandb directory to outputs/wandb_runs to avoid writing to src
            wandb_dir = os.path.join('outputs', 'wandb_runs')
            os.makedirs(wandb_dir, exist_ok=True)
            self.exp_summary = wandb.init(
                project=self.config['wandb_project'], 
                name=self.exp_id,
                config=self.config,
                dir=wandb_dir,  # Set directory for wandb run files
            )

    def exp_summary_log(self, log_dict):
        """Log the summary to the visualization tools."""
        if self.config['exp_summary'] == 'wandb':
            self.exp_summary.log(log_dict)

    def init_train(self):
        """Initialize the network for training.
        Should be called before training.
        It does the following things:
            1. set the network to train mode
            2. create the optimizer to self.optimizer
            3. create the model save directory
            4. initialize the visualization tools
        If you want to add more things, you can override this function.
        """
        self.network.train()

        self.create_criterion()
        self.create_optimizer()
        self.create_lr_scheduler()
        self.global_step = 0

        exp_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.exp_id = exp_id
        self.model_save_path = os.path.join('model_checkpoints', exp_id)
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        self.init_expsummary()
        

    def init_test(self):
        """Initialize the network for testing.
        Should be called before testing.
        It does the following things:
            1. set the network to eval mode
            2. load the network parameters from the weight file path
        If you want to add more things, you can override this function.
        """
        self.weightself.self.config['weight_path']
        self.load_network()
        self.network.eval()

        exp_id = self.weight_path.split('/')[-2]
        epoch_id = self.weight_path.split('/')[-1].split('.')[0]
        # TODO: add exp and epoch id to the result save path when needed

        # setup model save dir
        results_save_dir = os.path.join('results', f'{self.config["nt"]}f', 
                                        f'{self.config["num_cardiac_cycles"]}hb', 
                                        'nik', f'{self.config["hdr_ff_factor"]}FF')
        if not os.path.exists(results_save_dir):
            os.makedirs(results_save_dir)


    def create_optimizer(self):
        """Create the optimizer."""
        # self.optimizer = torch.optim.Adam([self.parameters(), self.network.parameters()], lr=self.config['lr'])
        # TODO: NEED TO CHECK IF THIS IS CORRECT
        self.optimizer = torch.optim.Adam(self.parameters(), lr=float(self.config['lr']))
        # for param in self.named_parameters():
        #     print(param[0])

    def create_lr_scheduler(self):
        """Create the learning rate scheduler."""
        scheduler_type = self.config.get('lr_scheduler', 'cosine').lower()
        num_epochs = self.config.get('num_epochs', 500)
        
        if scheduler_type == 'cosine':
            # Cosine annealing with warm restarts
            T_max = num_epochs
            eta_min = float(self.config.get('lr_min', 1e-6))
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=T_max, eta_min=eta_min
            )
        elif scheduler_type == 'step':
            step_size = self.config.get('lr_step_size', num_epochs // 3)
            gamma = float(self.config.get('lr_gamma', 0.5))
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=gamma
            )
        elif scheduler_type == 'exponential':
            gamma = float(self.config.get('lr_gamma', 0.995))
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=gamma
            )
        elif scheduler_type == 'plateau':
            # Reduce LR when loss plateaus
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=50, verbose=True
            )
        else:
            self.lr_scheduler = None

    def create_criterion(self):
        """Create the loss function."""
        loss_type = self.config.get('loss_type', 'hdr_ff').lower()
        
        if loss_type == 'mixed':
            self.criterion = MixedLoss(self.config)
        elif loss_type == 'weighted_hdr':
            self.criterion = WeightedHDRLoss_FF(self.config)
        elif loss_type == 'hdr_ff':
            self.criterion = HDRLoss_FF(self.config)
        elif loss_type == 'adaptive_hdr':
            self.criterion = AdaptiveHDRLoss(self.config)
        elif loss_type == 'complex_spectral':
            self.criterion = ComplexSpectralLoss(self.config)
        elif loss_type == 'mse':
            self.criterion = torch.nn.MSELoss()
        else:
            # Default to HDR
            self.criterion = HDRLoss_FF(self.config)

    def pre_process(self, inputs):
        """
        Encodings the coordinates
        If not implemented, it will return the coordinates as it is.
        """
        return inputs

    def post_process(self, output):
        """
        Post process the output of the network.
        If not implemented, it will return the output as it is.
        """
        return output

    def forward(self, inputs) -> torch.Tensor:
        """Get the coordinates and return the output of the network with required shape."""
        out = self.post_process(self.network(self.pre_process(inputs)))
        return out

    def train_batch(self, sample):
        """
        Train the network with a batch of points.
        Args:
            sample: A batch of data formed as a dict. Must contain the following keys:
                coords: The coordinates of the data.
                target: The target of the data.
        """
        self.optimizer.zero_grad()
        output = self.forward(sample['coords'])
        if hasattr(self.criterion, 'set_global_step'):
            self.criterion.set_global_step(self.global_step)
        loss = self.criterion(output, sample['targets'], sample['coords'])
        loss = loss[0] if isinstance(loss, tuple) else loss  # Handle tuple returns
        loss.backward()
        
        # Gradient clipping for stability
        max_grad_norm = self.config.get('max_grad_norm', None)
        if max_grad_norm is not None and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
        
        self.optimizer.step()
        self.global_step += 1
        return loss

    def test_batch(self, sample=None):
        """
        Test the network with a cartesian grid.
        if sample is not None, it will return image combined with coil sensitivity.
        """
        with torch.no_grad():
            nt = self.config['nt']
            nx = self.config['nx']
            ny = self.config['ny']

            ts = torch.linspace(-1+1/nt, 1-1/nt, nt)
            kxs = torch.linspace(-1, 1-2/nx, nx)
            kys = torch.linspace(-1, 1-2/ny, ny)
            nc = len(self.config['coil_select'])
            kc = torch.linspace(-1, 1, nc)

            # TODO: disgard the outside coordinates before prediction
            grid_coords = torch.stack(torch.meshgrid(ts, kxs, kys, kc, indexing='ij'), -1).to(self.device) # nt, nx, ny, nc, 4
            dist_to_center = torch.sqrt(grid_coords[:,:,:,:,1]**2 + grid_coords[:,:,:,:,2]**2)

            # split t for memory saving
            t_split = 3
            t_split_num = np.ceil(nt / t_split).astype(int)

            kpred_list = []
            for t_batch in range(t_split_num):
                grid_coords_batch = grid_coords[t_batch*t_split:(t_batch+1)*t_split]

                grid_coords_batch = grid_coords_batch.reshape(-1, 4).requires_grad_(False)
                # get prediction
                kpred_batch = self.forward(grid_coords_batch)
                kpred.append(kpred_batch)
            kpred = torch.concat(kpred, 0)
            
            kpred_list.append(kpred)
            kpred = torch.mean(torch.stack(kpred_list, 0), 0) #* filter_value.reshape(-1, 1)


            # TODO: clearning this part of code
            if sample is not None:
                sensitivity_map = sample['sensitivity_map']

                kpred = kpred.reshape(nt, nx, ny, -1)
                k_outer = 1
                kpred[dist_to_center>=k_outer] = 0
                kpred = kpred.permute(0, 3, 1, 2)
                coil_img = ifft2c_mri(kpred)

                k_img = kpred[:,0,:,:].abs().unsqueeze(1).detach().cpu().numpy()        # nt, nx, ny   
                # combined_img_motion = coil_img_motion.abs()
                combined_img = coilcombine(coil_img, coil_dim=1, csm=sensitivity_map)
                combined_phase = torch.angle(combined_img).detach().cpu().numpy()
                combined_mag = combined_img.abs()
                k_img = np.log(np.abs(k_img) + 1e-4)


            return kpred, combined_img
    


