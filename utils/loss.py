import torch

class HDRLoss_FF(torch.nn.Module):
    """
    HDR loss function with frequency filtering (v4)
    """
    def __init__(self, config):
        super().__init__()
        self.sigma = float(config['hdr_ff_sigma'])
        self.eps = float(config['hdr_eps'])
        self.factor = float(config['hdr_ff_factor'])

    def forward(self, input, target, kcoords, weights=None, reduce=True):
        # target_max = target.abs().max()
        # target /= target_max
        # input = input / target_max
        # input_nograd = input.clone()
        # input_nograd = input_nograd.detach()
        dist_to_center2 = kcoords[...,1]**2 + kcoords[...,2]**2
        filter_value = torch.exp(-dist_to_center2/(2*self.sigma**2)).unsqueeze(-1)

        if input.dtype == torch.float:
            input = torch.view_as_complex(input) #* filter_value
        if target.dtype == torch.float:
            target = torch.view_as_complex(target)
        
        assert input.shape == target.shape
        error = input - target
        # error = error * filter_value

        loss = (error.abs()/(input.detach().abs()+self.eps))**2
        if weights is not None:
            loss = loss * weights.unsqueeze(-1)

        reg_error = (input - input * filter_value)
        reg = self.factor * (reg_error.abs()/(input.detach().abs()+self.eps))**2
        # reg = torch.matmul(torch.conj(reg).t(), reg)
        # reg = reg.abs() * self.factor
        # reg = torch.zeros([1]).mean()

        if reduce:
            return loss.mean() + reg.mean(), reg.mean()
        else:
            return loss, reg

class MixedLoss(torch.nn.Module):
    """
    Mixed loss combining HDR loss with MSE/L1 for better reconstruction quality.
    This helps when HDR loss alone doesn't converge well. The MSE/L1 components
    provide absolute scale constraints that prevent the model from predicting
    unrealistically small values.
    """
    def __init__(self, config):
        super().__init__()
        self.hdr_loss = HDRLoss_FF(config)
        self.mse_weight = float(config.get('mse_weight', 0.5))  # Increased default to help with scale
        self.l1_weight = float(config.get('l1_weight', 0.3))    # Increased default
        self.hdr_weight = float(config.get('hdr_weight', 0.2))  # Reduced default to prevent scale collapse
        
    def forward(self, input, target, kcoords, weights=None, reduce=True):
        # Convert to complex if needed
        if input.dtype == torch.float:
            input = torch.view_as_complex(input)
        if target.dtype == torch.float:
            target = torch.view_as_complex(target)
        
        error = input - target
        
        # MSE loss component (on complex values) - provides absolute scale constraint
        mse_loss = torch.abs(error) ** 2
        
        # L1 loss component (on complex values) - provides absolute scale constraint
        l1_loss = torch.abs(error)
        
        # HDR loss component (normalized, helps with high dynamic range)
        hdr_loss, reg = self.hdr_loss(input, target, kcoords, weights, reduce)
        
        if weights is not None:
            mse_loss = mse_loss * weights.unsqueeze(-1)
            l1_loss = l1_loss * weights.unsqueeze(-1)
        
        if reduce:
            mse_loss = mse_loss.mean()
            l1_loss = l1_loss.mean()
            total_loss = self.hdr_weight * hdr_loss + self.mse_weight * mse_loss + self.l1_weight * l1_loss
            return total_loss, reg
        else:
            mse_mean = mse_loss.mean() if weights is None else mse_loss
            l1_mean = l1_loss.mean() if weights is None else l1_loss
            return (self.hdr_weight * hdr_loss + self.mse_weight * mse_mean + self.l1_weight * l1_mean, 
                    mse_mean, l1_mean, reg)

class WeightedHDRLoss_FF(torch.nn.Module):
    """
    HDR loss with improved weighting that emphasizes low-frequency k-space regions.
    This is important for MRI reconstruction quality.
    """
    def __init__(self, config):
        super().__init__()
        self.sigma = float(config['hdr_ff_sigma'])
        self.eps = float(config['hdr_eps'])
        self.factor = float(config['hdr_ff_factor'])
        self.low_freq_weight = float(config.get('low_freq_weight', 2.0))  # Weight for low-freq regions
        
    def forward(self, input, target, kcoords, weights=None, reduce=True):
        dist_to_center2 = kcoords[...,1]**2 + kcoords[...,2]**2
        filter_value = torch.exp(-dist_to_center2/(2*self.sigma**2)).unsqueeze(-1)
        
        # Create frequency weighting: higher weight for low frequencies
        freq_weight = 1.0 + (self.low_freq_weight - 1.0) * filter_value

        if input.dtype == torch.float:
            input = torch.view_as_complex(input)
        if target.dtype == torch.float:
            target = torch.view_as_complex(target)
        
        assert input.shape == target.shape
        error = input - target

        # Normalize by target magnitude instead of input for more stable training
        # This helps when predictions are far from target
        target_mag = target.abs()
        loss = (error.abs() / (target_mag + self.eps)) ** 2
        
        # Apply frequency weighting
        loss = loss * freq_weight
        
        if weights is not None:
            loss = loss * weights.unsqueeze(-1)

        reg_error = (input - input * filter_value)
        reg = self.factor * (reg_error.abs() / (input.detach().abs() + self.eps)) ** 2

        if reduce:
            return loss.mean() + reg.mean(), reg.mean()
        else:
            return loss, reg

class AdaptiveHDRLoss(torch.nn.Module):
    """
    HDR loss function with frequency filtering (v4)
    """
    def __init__(self, config):
        super().__init__()
        self.sigma = float(config['hdr_ff_sigma'])
        self.eps = float(config['eps'])
        self.factor = float(config['hdr_ff_factor'])

    def forward(self, input, target, reduce=True):
        # target_max = target.abs().max()
        # target /= target_max
        # input = input / target_max
        # input_nograd = input.clone()
        # input_nograd = input_nograd.detach()

        if input.dtype == torch.float:
            input = torch.view_as_complex(input) #* filter_value
        if target.dtype == torch.float:
            target = torch.view_as_complex(target)
        
        assert input.shape == target.shape
        error = input - target
        # error = error * filter_value

        loss = (-error.abs()/((input.detach().abs()+self.eps)**2))**2
        # if weights is not None:
        #     loss = loss * weights.unsqueeze(-1)

        # reg_error = (input - input * filter_value)
        # reg = self.factor * (reg_error.abs()/(input.detach().abs()+self.eps))**2
        # reg = torch.matmul(torch.conj(reg).t(), reg)
        # reg = reg.abs() * self.factor
        # reg = torch.zeros([1]).mean()

        if reduce:
            return loss.mean()