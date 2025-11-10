import torch
import torch.nn as nn
import numpy as np

from utils.mri import coilcombine, ifft2c_mri
from .base import NIKBase

class NIKSiren(NIKBase):
    def __init__(self, config) -> None:
        super().__init__(config)
        B = torch.randn((self.config['coord_dim'], self.config['feature_dim']//2), dtype=torch.float32)
        self.register_buffer('B', B)
        self.create_network()
        self.to(self.device)
        
        
    def create_network(self):
        feature_dim = self.config["feature_dim"]
        num_layers = self.config["num_layers"]
        out_dim = self.config["out_dim"]
        self.network = Siren(feature_dim, num_layers, out_dim).to(self.device)

    def pre_process(self, inputs):
        """
        Preprocess the input coordinates.
        """
        inputs['coords'] = inputs['coords'].to(self.device)
        if inputs.keys().__contains__('targets'):
            inputs['targets'] = inputs['targets'].to(self.device)
        features = torch.cat([torch.sin(inputs['coords'] @ self.B),
                              torch.cos(inputs['coords'] @ self.B)] , dim=-1)
        inputs['features'] = features
        return inputs
    
    def post_process(self, output):
        """
        Convert the real output to a complex-valued output.
        The first half of the output is the real part, and the second half is the imaginary part.
        """
        output = torch.complex(output[...,0:self.config["out_dim"]], output[...,self.config["out_dim"]:])
        return output

    def train_batch(self, sample):
        self.optimizer.zero_grad()
        sample = self.pre_process(sample)
        output = self.forward(sample)
        output = self.post_process(output)
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
            - (1, num_echoes, num_coils, kx_dim, ky_dim) if dataset is provided
            - (num_echoes, num_coils, kx_dim, ky_dim) if dataset is None (legacy behavior)
        """
        with torch.no_grad():
            # Use grid from dataset if provided
            if dataset is not None:
                if hasattr(dataset, 'get_grid_coordinates'):
                    grid_coords = dataset.get_grid_coordinates().to(self.device)
                    # grid_coords shape (brain, coil-less): (num_echoes, kx_dim, ky_dim, 3)
                    if grid_coords.shape[-1] == 3:
                        nt, nx, ny = grid_coords.shape[:3]
                        nc = 1
                        coil_less = True
                    else:
                        # legacy path with coils: (num_echoes, num_coils, kx_dim, ky_dim, 4)
                        nt, nc, nx, ny = grid_coords.shape[:4]
                        coil_less = False
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
            if 'coil_less' in locals() and coil_less:
                # kx, ky at indices 1,2
                dist_to_center = torch.sqrt(grid_coords[:,:,:,1]**2 + grid_coords[:,:,:,2]**2)
            else:
                # legacy: kx, ky at indices 2,3
                dist_to_center = torch.sqrt(grid_coords[:,:,:,:,2]**2 + grid_coords[:,:,:,:,3]**2)

            # split t (echoes) for memory saving
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
                kpred = self.forward(sample)
                kpred = self.post_process(kpred)
                kpred_list.append(kpred)
            kpred = torch.concat(kpred_list, 0)
            
            # Reshape prediction
            if 'coil_less' in locals() and coil_less:
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
                    kpred[dist_to_center>=k_outer] = 0
                elif k_mask_type == 'rectangular':
                    # Rectangular mask: max(|kx|, |ky|) >= k_outer
                    if 'coil_less' in locals() and coil_less:
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
                    if 'coil_less' in locals() and coil_less:
                        kx_coords = grid_coords[:,:,:,1]
                        ky_coords = grid_coords[:,:,:,2]
                        # Scale ky by aspect ratio to make mask circular on display
                        dist_to_center_ellipse = torch.sqrt(kx_coords**2 + (ky_coords * aspect_ratio)**2)
                        kpred[dist_to_center_ellipse>=k_outer] = 0
                    else:
                        kx_coords = grid_coords[:,:,:,:,2]
                        ky_coords = grid_coords[:,:,:,:,3]
                        dist_to_center_ellipse = torch.sqrt(kx_coords**2 + (ky_coords * aspect_ratio)**2)
                        kpred[dist_to_center_ellipse>=k_outer] = 0
                else:
                    raise ValueError(f"Unknown k_mask_type: {k_mask_type}. Must be 'circular', 'rectangular', or 'elliptical'")
            
            # Add slice dimension if dataset was provided (for compatibility with reconstruct_images)
            # reconstruct_images now expects (slices, echoes, kx, ky) for brain
            if dataset is not None:
                if 'coil_less' in locals() and coil_less:
                    kpred = kpred.unsqueeze(0)  # (1, nt, nx, ny)
                else:
                    kpred = kpred.unsqueeze(0)  # (1, nt, nc, nx, ny)
            
            return kpred
    
    def forward(self, inputs):
        return self.network(inputs['features'])

"""
The following code is a demo of mlp with sine activation function.
We suggest to only use the mlp model class to do the very specific 
mlp task: takes a feature vector and outputs a vector. The encoding 
and post-process of the input coordinates and output should be done 
outside of the mlp model (e.g. in the prepocess and postprocess 
function in your NIK model class).
"""

class Siren(nn.Module):
    def __init__(self, hidden_features, num_layers, out_dim, omega_0=30, exp_out=True) -> None:
        super().__init__()

        self.net = [SineLayer(hidden_features, hidden_features, is_first=True, omega_0=omega_0)]
        for i in range(num_layers-1):
            self.net.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=omega_0))
        final_linear = nn.Linear(hidden_features, out_dim*2)
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / omega_0, 
                                          np.sqrt(6 / hidden_features) / omega_0)
        self.net.append(final_linear)
        self.net = nn.Sequential(*self.net)
    
    def forward(self, features):
        return self.net(features)



class SineLayer(nn.Module):
    """Linear layer with sine activation. Adapted from Siren repo"""
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    