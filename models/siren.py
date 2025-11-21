import torch
import torch.nn as nn
import numpy as np

from utils.mri import coilcombine, ifft2c_mri
from .base import NIKBase

class NIKSiren(NIKBase):
    def __init__(self, config) -> None:
        super().__init__(config)
        # Optionally include polar coordinates [r, theta] in the encoded inputs
        self.use_polar_inputs = bool(self.config.get('use_polar_inputs', False))
        input_coord_dim = int(self.config['coord_dim']) + (2 if self.use_polar_inputs else 0)
        B = torch.randn((input_coord_dim, self.config['feature_dim']//2), dtype=torch.float32)
        self.register_buffer('B', B)
        self.create_network()
        self.to(self.device)
        
        
    def create_network(self):
        feature_dim = self.config["feature_dim"]
        num_layers = self.config["num_layers"]
        out_dim = self.config["out_dim"]
        omega0_first = float(self.config.get("omega0_first", 60.0))
        omega0_hidden = float(self.config.get("omega0_hidden", 1.0))
        self.network = Siren(feature_dim, num_layers, out_dim,
                             omega_0_first=omega0_first,
                             omega_0_hidden=omega0_hidden).to(self.device)

    def pre_process(self, inputs):
        """
        Preprocess the input coordinates with SIREN-specific encoding.
        Applies positional encoding (sin/cos) and optionally adds polar coordinates.
        """
        # Call parent to move coords to device
        inputs = super().pre_process(inputs)
        
        coords = inputs['coords']
        
        # SIREN-specific: Add polar coordinates if enabled
        if self.use_polar_inputs:
            # kx, ky are at indices 1, 2 in [slice, kx, ky, echo] coordinate system
            kx = coords[..., 1]
            ky = coords[..., 2]
            r = torch.sqrt(kx * kx + ky * ky)
            theta = torch.atan2(ky, kx)
            aug_coords = torch.stack([r, theta], dim=-1)
            coords = torch.cat([coords, aug_coords], dim=-1)
        
        # SIREN-specific: Positional encoding with sin/cos
        features = torch.cat([torch.sin(coords @ self.B),
                              torch.cos(coords @ self.B)] , dim=-1)
        inputs['features'] = features
        return inputs
    
    def post_process(self, output):
        """
        SIREN-specific: Convert the real output to a complex-valued output.
        The first half of the output is the real part, and the second half is the imaginary part.
        """
        output = torch.complex(output[...,0:self.config["out_dim"]], output[...,self.config["out_dim"]:])
        return output
    
    def forward(self, inputs):
        """
        Forward pass through the SIREN network.
        Expects inputs to have 'features' key from pre_process.
        """
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
    def __init__(self, hidden_features, num_layers, out_dim,
                 omega_0_first=30, omega_0_hidden=30) -> None:
        super().__init__()

        self.net = [SineLayer(hidden_features, hidden_features, is_first=True, omega_0=omega_0_first)]
        for i in range(num_layers-1):
            self.net.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=omega_0_hidden))
        final_linear = nn.Linear(hidden_features, out_dim*2)
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / omega_0_hidden, 
                                          np.sqrt(6 / hidden_features) / omega_0_hidden)
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
    