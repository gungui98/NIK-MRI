import torch
from utils.medutils_compat import center_crop
import numpy as np

def ifft2c(k):
    x = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k, (-2,-1)), norm='ortho'), (-2,-1))
    return x

def fft2c(x):
    k = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x, (-2,-1)), norm='ortho'), (-2,-1))
    return k

def ifft2c_mri(k):
    x = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(k, (-2,-1)), norm='ortho'), (-2,-1))
    return x

def fft2c_mri(img):
    k = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(img, (-2,-1)), norm='ortho'), (-2,-1))
    return k

def coilcombine(img, im_shape=None, coil_dim=-1, mode='csm', csm=None):
    if mode == 'rss':
        return torch.sqrt(torch.sum(img**2, dim=coil_dim, keepdim=True))
    elif mode == 'csm':
        # csm = csm.unsqueeze(0)
        csm = torch.from_numpy(csm).to(img.device)
        img = center_crop(img, im_shape)
        csm = center_crop(csm, im_shape)
        return torch.sum(img*torch.conj(csm), dim=coil_dim, keepdim=True)
    else:
        raise NotImplementedError

def rss(img, coil_axis=-1):
    """ Compute root-sum-of-squares reconstruction
    :param img: input image (torch.Tensor)
    :param coil_axis: coil dimension
    :return: root-sum-of-squares reconstruction
    """
    return torch.sqrt(torch.sum(torch.abs(img)**2, coil_axis))