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
            
            # Watch the model to log gradients and parameters during training
            if self.network is not None:
                watch_log = self.config.get('wandb_watch_log', 'gradients')  # 'gradients', 'parameters', 'all', or None
                watch_freq = self.config.get('wandb_watch_freq', 100)  # Log every N steps
                log_graph = self.config.get('wandb_watch_log_graph', False)  # Log computation graph
                
                self.exp_summary.watch(
                    self.network,
                    log=watch_log,
                    log_freq=watch_freq,
                    log_graph=log_graph
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
        self.weight_path = self.config['weight_path']
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
        self.optimizer = torch.optim.Adam(self.parameters(), lr=float(self.config['lr']))

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
        Preprocess the input coordinates.
        Base implementation moves coordinates to device.
        Subclasses should override to add model-specific encoding.
        """
        inputs['coords'] = inputs['coords'].to(self.device)
        if 'targets' in inputs:
            inputs['targets'] = inputs['targets'].to(self.device)
        return inputs

    def post_process(self, output):
        """
        Post process the output of the network.
        If not implemented, it will return the output as it is.
        """
        return output

    def forward(self, inputs) -> torch.Tensor:
        """
        Forward pass through the network.
        Expects inputs to be a dict with 'coords' key (or 'features' if pre_processed).
        """
        # If inputs is already pre-processed and has 'features', use it directly
        if isinstance(inputs, dict) and 'features' in inputs:
            return self.network(inputs['features'])
        # Otherwise, pre-process first
        processed = self.pre_process(inputs)
        if 'features' in processed:
            return self.network(processed['features'])
        # Fallback: assume processed['coords'] can be passed directly to network
        return self.network(processed['coords'])

    def train_batch(self, sample):
        """
        Train the network with a batch of points.
        Args:
            sample: A batch of data formed as a dict. Must contain the following keys:
                coords: The coordinates of the data.
                targets: The target of the data.
        """
        self.optimizer.zero_grad()
        sample = self.pre_process(sample)
        output = self.forward(sample)
        output = self.post_process(output)
        if hasattr(self.criterion, 'set_global_step'):
            self.criterion.set_global_step(self.global_step)
        loss_result = self.criterion(output, sample['targets'], sample['coords'])
        
        # Handle different loss return formats
        if isinstance(loss_result, tuple):
            loss = loss_result[0]
        else:
            loss = loss_result
        
        loss.backward()
        
        # Gradient clipping for stability
        max_grad_norm = self.config.get('max_grad_norm', None)
        if max_grad_norm is not None and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
        
        self.optimizer.step()
        self.global_step += 1
        return loss

    def test_batch(self, kspace_data_original=None, dataset=None):
        """
        Test the network with a cartesian grid.
        Uses grid coordinates from dataset if provided, otherwise constructs grid from config.
        
        Parameters
        ----------
        kspace_data_original : torch.Tensor or np.ndarray, optional
            Original k-space data for shape inference
        dataset : BrainDataset, optional
            Dataset object to get grid coordinates from. If provided, uses its coordinate system.
            
        Returns
        -------
        torch.Tensor
            Predicted k-space with shape:
            - (num_slices, num_echoes, kx_dim, ky_dim) if dataset is provided with 4D coords
            - (1, num_echoes, num_coils, kx_dim, ky_dim) if dataset is provided with legacy 3D coords
            - (num_echoes, num_coils, kx_dim, ky_dim) if dataset is None (legacy behavior)
        """
        with torch.no_grad():
            # Initialize format flags
            new_format = False
            coil_less = False
            ns = None
            ne = None
            nt = None
            nc = None
            
            # Use grid from dataset if provided
            if dataset is not None:
                if hasattr(dataset, 'get_grid_coordinates'):
                    grid_coords = dataset.get_grid_coordinates().to(self.device)
                    # grid_coords shape (brain, new format): (num_slices, kx_dim, ky_dim, num_echoes, 4)
                    if grid_coords.shape[-1] == 4 and len(grid_coords.shape) == 5:
                        ns, nx, ny, ne = grid_coords.shape[:4]
                        nc = 1
                        coil_less = True
                        new_format = True
                    elif grid_coords.shape[-1] == 3:
                        # Legacy 3D format: (num_echoes, kx_dim, ky_dim, 3)
                        nt, nx, ny = grid_coords.shape[:3]
                        nc = 1
                        coil_less = True
                        new_format = False
                    else:
                        # legacy path with coils: (num_echoes, num_coils, kx_dim, ky_dim, 4)
                        nt, nc, nx, ny = grid_coords.shape[:4]
                        coil_less = False
                        new_format = False
                else:
                    raise ValueError("Dataset does not have get_grid_coordinates method. Use BrainDataset or provide grid manually.")
            else:
                # Fallback to original implementation using config
                nt = self.config['nt']
                nx = self.config['nx']
                ny = self.config['ny']

                ts = torch.linspace(-1+1/nt, 1-1/nt, nt)
                # Infer coil dimension
                if kspace_data_original is not None:
                    # Support numpy arrays and torch tensors
                    try:
                        nc = int(kspace_data_original.shape[1])
                    except Exception as e:
                        raise ValueError("kspace_data_original must have a shape with coil at dim=1") from e
                elif self.config.get('coil_select'):
                    nc = len(self.config['coil_select'])
                elif 'nc' in self.config:
                    nc = int(self.config['nc'])
                else:
                    raise ValueError("Cannot infer number of coils (nc). Provide kspace_data_original, dataset, or set coil_select/nc in config.")
                kc = torch.linspace(-1, 1, nc)
                kxs = torch.linspace(-1, 1-2/nx, nx)
                kys = torch.linspace(-1, 1-2/ny, ny)
                
                grid_coords = torch.stack(torch.meshgrid(ts, kc, kxs, kys, indexing='ij'), -1).to(self.device) # nt, nc, nx, ny, 4

            # Compute distance to center for masking
            if new_format:
                # New 4D format: [slice, kx, ky, echo] - kx, ky at indices 1, 2
                dist_to_center = torch.sqrt(grid_coords[:,:,:,:,1]**2 + grid_coords[:,:,:,:,2]**2)
            elif coil_less:
                # Legacy 3D format: kx, ky at indices 1,2
                dist_to_center = torch.sqrt(grid_coords[:,:,:,1]**2 + grid_coords[:,:,:,2]**2)
            else:
                # legacy: kx, ky at indices 2,3
                dist_to_center = torch.sqrt(grid_coords[:,:,:,:,2]**2 + grid_coords[:,:,:,:,3]**2)

            # Flatten coordinates for prediction
            if new_format:
                # New 4D format: (num_slices, kx_dim, ky_dim, num_echoes, 4)
                grid_coords_flat = grid_coords.reshape(-1, 4).requires_grad_(False)
                # Process all at once (or split if memory is an issue)
                sample = {'coords': grid_coords_flat}
                sample = self.pre_process(sample)
                kpred = self.forward(sample)
                kpred = self.post_process(kpred)
                # Reshape to (num_slices, kx_dim, ky_dim, num_echoes) then permute to (num_slices, num_echoes, kx_dim, ky_dim)
                kpred = kpred.reshape(ns, nx, ny, ne)
                kpred = kpred.permute(0, 3, 1, 2)  # (num_slices, num_echoes, kx_dim, ky_dim)
            else:
                # Legacy format: split t (echoes) for memory saving
                if coil_less:
                    t_split = 1
                    t_split_num = np.ceil(nt / t_split).astype(int)
                else:
                    t_split = 1
                    t_split_num = np.ceil(nt / t_split).astype(int)

                kpred_list = []
                for t_batch in range(t_split_num):
                    grid_coords_batch = grid_coords[t_batch*t_split:(t_batch+1)*t_split]

                    # Flatten coordinates appropriately
                    if 'coil_less' in locals() and coil_less:
                        grid_coords_batch = grid_coords_batch.reshape(-1, 3).requires_grad_(False)
                    else:
                        grid_coords_batch = grid_coords_batch.reshape(-1, 4).requires_grad_(False)
                    # get prediction
                    sample = {'coords': grid_coords_batch}
                    sample = self.pre_process(sample)
                    kpred_batch = self.forward(sample)
                    kpred_batch = self.post_process(kpred_batch)
                    kpred_list.append(kpred_batch)
                kpred = torch.concat(kpred_list, 0)
                
                # Reshape prediction
                if coil_less:
                    kpred = kpred.reshape(nt, nx, ny)
                else:
                    kpred = kpred.reshape(nt, nc, nx, ny)
            
            # Mask outer k-space points (optional)
            # Get masking configuration from config, with defaults
            k_mask_enabled = self.config.get('k_mask_enabled', False)  # Enable/disable masking (default: False for full k-space)
            k_mask_type = self.config.get('k_mask_type', 'rectangular')  # 'circular', 'rectangular', or 'elliptical'
            k_outer = self.config.get('k_outer', 1.0)  # Outer radius/threshold
            
            if k_mask_enabled:
                if k_mask_type == 'circular':
                    # Circular mask: dist_to_center >= k_outer
                    # Note: This will appear as an oval if nx != ny (rectangular grid)
                    if new_format:
                        kpred = kpred.permute(0, 2, 3, 1)  # (num_slices, kx_dim, ky_dim, num_echoes) for masking
                        kpred[dist_to_center>=k_outer] = 0
                        kpred = kpred.permute(0, 3, 1, 2)  # Back to (num_slices, num_echoes, kx_dim, ky_dim)
                    else:
                        kpred[dist_to_center>=k_outer] = 0
                elif k_mask_type == 'rectangular':
                    # Rectangular mask: max(|kx|, |ky|) >= k_outer
                    if new_format:
                        kx_coords = grid_coords[:,:,:,:,1].abs()
                        ky_coords = grid_coords[:,:,:,:,2].abs()
                        mask = torch.maximum(kx_coords, ky_coords) >= k_outer
                        kpred = kpred.permute(0, 2, 3, 1)  # (num_slices, kx_dim, ky_dim, num_echoes) for masking
                        kpred[mask] = 0
                        kpred = kpred.permute(0, 3, 1, 2)  # Back to (num_slices, num_echoes, kx_dim, ky_dim)
                    elif 'coil_less' in locals() and coil_less:
                        kx_coords = grid_coords[:,:,:,1].abs()
                        ky_coords = grid_coords[:,:,:,2].abs()
                        mask = torch.maximum(kx_coords, ky_coords) >= k_outer
                        kpred[mask] = 0
                    else:
                        kx_coords = grid_coords[:,:,:,:,2].abs()
                        ky_coords = grid_coords[:,:,:,:,3].abs()
                        mask = torch.maximum(kx_coords, ky_coords) >= k_outer
                        kpred[mask] = 0
                elif k_mask_type == 'elliptical':
                    # Elliptical mask accounting for aspect ratio
                    # Normalize by aspect ratio so it appears circular on rectangular grid
                    aspect_ratio = ny / nx if nx > 0 else 1.0
                    if new_format:
                        kx_coords = grid_coords[:,:,:,:,1]
                        ky_coords = grid_coords[:,:,:,:,2]
                        dist_to_center_ellipse = torch.sqrt(kx_coords**2 + (ky_coords * aspect_ratio)**2)
                        kpred = kpred.permute(0, 2, 3, 1)  # (num_slices, kx_dim, ky_dim, num_echoes) for masking
                        kpred[dist_to_center_ellipse>=k_outer] = 0
                        kpred = kpred.permute(0, 3, 1, 2)  # Back to (num_slices, num_echoes, kx_dim, ky_dim)
                    elif coil_less:
                        kx_coords = grid_coords[:,:,:,1]
                        ky_coords = grid_coords[:,:,:,2]
                        dist_to_center_ellipse = torch.sqrt(kx_coords**2 + (ky_coords * aspect_ratio)**2)
                        kpred[dist_to_center_ellipse>=k_outer] = 0
                    else:
                        kx_coords = grid_coords[:,:,:,:,2]
                        ky_coords = grid_coords[:,:,:,:,3]
                        dist_to_center_ellipse = torch.sqrt(kx_coords**2 + (ky_coords * aspect_ratio)**2)
                        kpred[dist_to_center_ellipse>=k_outer] = 0
                else:
                    raise ValueError(f"Unknown k_mask_type: {k_mask_type}. Must be 'circular', 'rectangular', or 'elliptical'")
            
            # For legacy format, add slice dimension if dataset was provided (for compatibility with reconstruct_images)
            # New format already has correct shape (num_slices, num_echoes, kx_dim, ky_dim)
            if dataset is not None and not new_format:
                if coil_less:
                    kpred = kpred.unsqueeze(0)  # (1, nt, nx, ny)
                else:
                    kpred = kpred.unsqueeze(0)  # (1, nt, nc, nx, ny)
            
            return kpred
    


