"""
Compatibility module for medutils functions
Implements the required functions locally to avoid dependency issues
"""
import torch
import numpy as np

def center_crop(data, crop_shape):
    """
    Center crop data to specified shape
    
    Args:
        data: Input data (torch.Tensor or np.ndarray)
        crop_shape: Target shape (tuple)
    
    Returns:
        Cropped data
    """
    if isinstance(data, np.ndarray):
        return _center_crop_numpy(data, crop_shape)
    elif isinstance(data, torch.Tensor):
        return _center_crop_torch(data, crop_shape)
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

def _center_crop_numpy(data, crop_shape):
    """Center crop for numpy arrays"""
    if len(crop_shape) == 2:
        h, w = crop_shape
        if len(data.shape) == 2:
            return _crop_2d_numpy(data, h, w)
        elif len(data.shape) == 3:
            c, h_in, w_in = data.shape
            cropped = np.zeros((c, h, w), dtype=data.dtype)
            for i in range(c):
                cropped[i] = _crop_2d_numpy(data[i], h, w)
            return cropped
        else:
            raise ValueError(f"Unsupported numpy array shape: {data.shape}")
    else:
        raise ValueError(f"Unsupported crop shape: {crop_shape}")

def _crop_2d_numpy(data, h, w):
    """Crop 2D numpy array"""
    h_in, w_in = data.shape
    start_h = (h_in - h) // 2
    start_w = (w_in - w) // 2
    return data[start_h:start_h+h, start_w:start_w+w]

def _center_crop_torch(data, crop_shape):
    """Center crop for torch tensors"""
    if len(crop_shape) == 2:
        h, w = crop_shape
        if len(data.shape) == 2:
            return _crop_2d_torch(data, h, w)
        elif len(data.shape) == 3:
            c, h_in, w_in = data.shape
            cropped = torch.zeros((c, h, w), dtype=data.dtype, device=data.device)
            for i in range(c):
                cropped[i] = _crop_2d_torch(data[i], h, w)
            return cropped
        elif len(data.shape) == 4:
            b, c, h_in, w_in = data.shape
            cropped = torch.zeros((b, c, h, w), dtype=data.dtype, device=data.device)
            for i in range(b):
                for j in range(c):
                    cropped[i, j] = _crop_2d_torch(data[i, j], h, w)
            return cropped
        else:
            raise ValueError(f"Unsupported torch tensor shape: {data.shape}")
    else:
        raise ValueError(f"Unsupported crop shape: {crop_shape}")

def _crop_2d_torch(data, h, w):
    """Crop 2D torch tensor"""
    h_in, w_in = data.shape
    start_h = (h_in - h) // 2
    start_w = (w_in - w) // 2
    return data[start_h:start_h+h, start_w:start_w+w]

def rss(data, coil_axis=0):
    """
    Root sum of squares combination
    
    Args:
        data: Input data (torch.Tensor or np.ndarray)
        coil_axis: Axis along which coils are stacked
    
    Returns:
        RSS combined data
    """
    if isinstance(data, np.ndarray):
        return np.sqrt(np.sum(np.abs(data)**2, axis=coil_axis))
    elif isinstance(data, torch.Tensor):
        return torch.sqrt(torch.sum(torch.abs(data)**2, dim=coil_axis))
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")
