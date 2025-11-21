import torch
import torch.nn as nn
import numpy as np
from .base import NIKBase

class NIKNeRF(NIKBase):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.num_frequencies = self.config.get('num_frequencies', 10)
        self.include_input = self.config.get('include_input', True)
        
        # Optionally include polar coordinates [r, theta] before encoding
        self.use_polar_inputs = bool(self.config.get('use_polar_inputs', False))
        
        # Calculate input dimension after positional encoding
        coord_dim = int(self.config['coord_dim']) + (2 if self.use_polar_inputs else 0)
        
        self.embed_fns = []
        self.out_dim = 0
        
        if self.include_input:
            self.embed_fns.append(lambda x: x)
            self.out_dim += coord_dim
            
        log_sampling = self.config.get('log_sampling', True)
        if log_sampling:
            freq_bands = 2.**torch.linspace(0., self.num_frequencies - 1, steps=self.num_frequencies)
        else:
            freq_bands = torch.linspace(2.**0., 2.**(self.num_frequencies - 1), steps=self.num_frequencies)
            
        for freq in freq_bands:
            self.embed_fns.append(lambda x, freq=freq: torch.sin(x * freq * np.pi))
            self.embed_fns.append(lambda x, freq=freq: torch.cos(x * freq * np.pi))
            self.out_dim += coord_dim * 2
            
        self.create_network()
        self.to(self.device)

    def create_network(self):
        feature_dim = self.out_dim # Input dimension to MLP is the encoded dimension
        hidden_dim = self.config["feature_dim"] # Reuse feature_dim as hidden_dim
        num_layers = self.config["num_layers"]
        out_dim = self.config["out_dim"] * 2 # Real + Imaginary parts
        skips = self.config.get("skips", [4])
        
        self.network = NeRFMLP(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            skips=skips
        ).to(self.device)

    def pre_process(self, inputs):
        """
        Preprocess the input coordinates with NeRF positional encoding.
        """
        # Call parent to move coords to device
        inputs = super().pre_process(inputs)
        
        coords = inputs['coords']
        
        # Add polar coordinates if enabled
        if self.use_polar_inputs:
            # kx, ky are at indices 1, 2 in [slice, kx, ky, echo] coordinate system
            # Note: Adjust indices if your coordinate system differs
            if coords.shape[-1] >= 3:
                kx = coords[..., 1]
                ky = coords[..., 2]
            else: 
                # Fallback for simpler coords (e.g. 2D or 3D legacy)
                # Assuming last two dims are spatial if not 4D
                kx = coords[..., -2]
                ky = coords[..., -1]

            r = torch.sqrt(kx * kx + ky * ky)
            theta = torch.atan2(ky, kx)
            aug_coords = torch.stack([r, theta], dim=-1)
            coords = torch.cat([coords, aug_coords], dim=-1)
            
        # Positional encoding
        # Note: embed_fns captures freq by value, so this works
        # We need to ensure freq bands are on the same device if they were tensors, 
        # but here we bake them into the lambda or use scalar mult.
        # Ideally, pre-compute freq bands on device if used directly, but simple lambda works for scalar freq.
        
        encoded = []
        for fn in self.embed_fns:
            encoded.append(fn(coords))
            
        features = torch.cat(encoded, dim=-1)
        inputs['features'] = features
        return inputs

    def post_process(self, output):
        """
        Convert the real output to a complex-valued output.
        The first half of the output is the real part, and the second half is the imaginary part.
        """
        output = torch.complex(output[..., 0:self.config["out_dim"]], output[..., self.config["out_dim"]:])
        return output

    def forward(self, inputs):
        return self.network(inputs['features'])


class NeRFMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, num_layers=8, skips=[4]):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.skips = skips
        
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        
        # Hidden layers
        for i in range(num_layers - 1):
            if i + 1 in skips:
                self.layers.append(nn.Linear(hidden_dim + input_dim, hidden_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                
        # Output layer
        self.output_linear = nn.Linear(hidden_dim, out_dim)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        input_x = x
        h = x
        
        for i, layer in enumerate(self.layers):
            # Apply skip connection if needed (before the layer operation that expects concatenated input)
            # Note: The layer at index `i` corresponds to transition from layer `i` to `i+1`.
            # If `i` (0-indexed) matches a skip index, it means the layer we are about to apply 
            # expects concatenated input.
            # Wait, let's align with standard NeRF:
            # Layer 0: input -> hidden
            # Layer 1: hidden -> hidden
            # ...
            # Layer 4 (if skip=4): [hidden, input] -> hidden
            
            # In __init__:
            # i=0: Linear(input, hidden) -> creates layer 0
            # i=1..num_layers-1:
            #   if i in skips: Linear(hidden+input, hidden)
            #   else: Linear(hidden, hidden)
            
            # In forward:
            # loop i=0..num_layers-1
            #   if i in skips: concat
            #   h = layer(h)
            #   h = relu(h)
            
            # Logic check:
            # init loop runs for i = 0 to num_layers-2.
            # layer 0 is created before loop.
            # loop index j (0 to num_layers-2):
            #   creates layer j+1.
            #   if j+1 in skips: layer j+1 takes concat.
            
            # forward loop:
            # i = 0 (layer 0): takes input_x (handled by `h=x` init logic? No, layer 0 takes `input_x`).
            # Wait, logic in forward needs to match init.
            
            if i == 0:
                h = layer(input_x)
            elif i in self.skips:
                h = torch.cat([h, input_x], dim=-1)
                h = layer(h)
            else:
                h = layer(h)
            
            h = self.relu(h)
            
        out = self.output_linear(h)
        return out

