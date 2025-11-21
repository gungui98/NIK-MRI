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

    def forward(self, pred, target, kcoords, weights=None, reduce=True):
        # target_max = target.abs().max()
        # target /= target_max
        # input = input / target_max
        # input_nograd = input.clone()
        # input_nograd = input_nograd.detach()
        dist_to_center2 = kcoords[...,1]**2 + kcoords[...,2]**2
        filter_value = torch.exp(-dist_to_center2/(2*self.sigma**2)).unsqueeze(-1)

        if pred.dtype == torch.float:
            pred = torch.view_as_complex(pred) #* filter_value
        if target.dtype == torch.float:
            target = torch.view_as_complex(target)
        
        assert pred.shape == target.shape
        error = pred - target
        # error = error * filter_value

        loss = (error.abs()/(pred.detach().abs()+self.eps))**2
        if weights is not None:
            loss = loss * weights.unsqueeze(-1)

        reg_error = (pred - pred * filter_value)
        reg = self.factor * (reg_error.abs()/(pred.detach().abs()+self.eps))**2
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
        
    def forward(self, pred, target, kcoords, weights=None, reduce=True):
        # Convert to complex if needed
        if pred.dtype == torch.float:
            pred = torch.view_as_complex(pred)
        if target.dtype == torch.float:
            target = torch.view_as_complex(target)
        
        error = pred - target
        
        # MSE loss component (on complex values) - provides absolute scale constraint
        mse_loss = torch.abs(error) ** 2
        
        # L1 loss component (on complex values) - provides absolute scale constraint
        l1_loss = torch.abs(error)
        
        # HDR loss component (normalized, helps with high dynamic range)
        hdr_loss, reg = self.hdr_loss(pred, target, kcoords, weights, reduce)
        
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
        
    def forward(self, pred, target, kcoords, weights=None, reduce=True):
        dist_to_center2 = kcoords[...,1]**2 + kcoords[...,2]**2
        filter_value = torch.exp(-dist_to_center2/(2*self.sigma**2)).unsqueeze(-1)
        
        # Create frequency weighting: higher weight for low frequencies
        freq_weight = 1.0 + (self.low_freq_weight - 1.0) * filter_value

        if pred.dtype == torch.float:
            pred = torch.view_as_complex(pred)
        if target.dtype == torch.float:
            target = torch.view_as_complex(target)
        
        assert pred.shape == target.shape
        error = pred - target

        # Normalize by target magnitude instead of input for more stable training
        # This helps when predictions are far from target
        target_mag = target.abs()
        loss = (error.abs() / (target_mag + self.eps)) ** 2
        
        # Apply frequency weighting
        loss = loss * freq_weight
        
        if weights is not None:
            loss = loss * weights.unsqueeze(-1)

        reg_error = (pred - pred * filter_value)
        reg = self.factor * (reg_error.abs() / (pred.detach().abs() + self.eps)) ** 2

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

    def forward(self, pred, target, reduce=True):
        # target_max = target.abs().max()
        # target /= target_max
        # input = input / target_max
        # input_nograd = input.clone()
        # input_nograd = input_nograd.detach()

        if pred.dtype == torch.float:
            pred = torch.view_as_complex(pred) #* filter_value
        if target.dtype == torch.float:
            target = torch.view_as_complex(target)
        
        assert pred.shape == target.shape
        error = pred - target
        # error = error * filter_value

        loss = (-error.abs()/((pred.detach().abs()+self.eps)**2))**2
        # if weights is not None:
        #     loss = loss * weights.unsqueeze(-1)

        # reg_error = (input - input * filter_value)
        # reg = self.factor * (reg_error.abs()/(input.detach().abs()+self.eps))**2
        # reg = torch.matmul(torch.conj(reg).t(), reg)
        # reg = reg.abs() * self.factor
        # reg = torch.zeros([1]).mean()

        if reduce:
            return loss.mean()

class ComplexSpectralLoss(torch.nn.Module):
    """
    Complex k-space loss with phase-aware alignment and log-magnitude stabilization,
    with optional radial up-weighting to emphasize high-|k|.
    L = lambda_k * |Khat - K|^2
        + lambda_angle * (1 - Re(Khat * conj(K)) / (|Khat||K| + eps))
        + lambda_log * (log(|Khat| + eps) - log(|K| + eps))^2
    Weighted by w(r) = 1 + alpha * r^2 where r = sqrt(kx^2 + ky^2).
    """
    def __init__(self, config):
        super().__init__()
        self.eps = float(config.get('spectral_eps', 1e-6))
        self.lambda_k = float(config.get('lambda_k', 1.0))

        # Warmup-aware weights for phase and log-magnitude terms
        self.lambda_angle_final = float(config.get('lambda_angle', 0.2))
        self.lambda_log_final = float(config.get('lambda_log', 0.1))
        self.lambda_angle_initial = float(config.get('lambda_angle_start', 0.05))
        self.lambda_log_initial = float(config.get('lambda_log_start', 0.02))
        self.spectral_warmup_steps = int(config.get('spectral_warmup_steps', 100))
        if self.spectral_warmup_steps <= 0:
            self._current_lambda_angle = self.lambda_angle_final
            self._current_lambda_log = self.lambda_log_final
        else:
            self._current_lambda_angle = self.lambda_angle_initial
            self._current_lambda_log = self.lambda_log_initial

        self.radial_alpha = float(config.get('radial_alpha', 3.0))
        self.use_radial_weight = bool(config.get('use_radial_weight', True))
    
    def set_global_step(self, step: int):
        """Update warmup weights based on training step."""
        if self.spectral_warmup_steps <= 0:
            return
        t = min(max(step, 0) / self.spectral_warmup_steps, 1.0)
        self._current_lambda_angle = self.lambda_angle_initial + t * (self.lambda_angle_final - self.lambda_angle_initial)
        self._current_lambda_log = self.lambda_log_initial + t * (self.lambda_log_final - self.lambda_log_initial)
    
    def forward(self, pred, target, kcoords, weights=None, reduce=True):
        if pred.dtype == torch.float:
            pred = torch.view_as_complex(pred)
        if target.dtype == torch.float:
            target = torch.view_as_complex(target)
        
        # radius from coordinates: assume kx, ky at indices 1 and 2
        kx = kcoords[..., 1]
        ky = kcoords[..., 2]
        r = torch.sqrt(kx * kx + ky * ky)
        weight_r = (1.0 + self.radial_alpha * (r * r)).unsqueeze(-1) if self.use_radial_weight else 1.0
        
        error = pred - target
        mag_input = pred.abs()
        mag_target = target.abs()
        
        # Complex MSE term
        lk = (error.abs() ** 2)
        
        # Phase-aware cosine similarity (unwrap-free)
        # 1 - Re(<input, target_conj>) / (|input||target| + eps)
        dot_real = (pred.real * target.real + pred.imag * target.imag)
        denom = (mag_input * mag_target + self.eps)
        langle = 1.0 - (dot_real / denom)
        
        # Log-magnitude term
        log_input = torch.log(mag_input + self.eps)
        log_target = torch.log(mag_target + self.eps)
        llog = (log_input - log_target) ** 2
        
        loss = self.lambda_k * lk + self._current_lambda_angle * langle + self._current_lambda_log * llog
        
        # Apply radial weights and any external sample weights
        loss = loss * weight_r
        if weights is not None:
            loss = loss * weights.unsqueeze(-1)
        
        if reduce:
            return loss.mean()
        else:
            return loss