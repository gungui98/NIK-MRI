import h5py
import matplotlib
import numpy as np
import torch
from utils.medutils_compat import center_crop
from utils.mri import coilcombine, ifft2c_mri
from utils.medutils_compat import rss

import os
import json
import nibabel as nib
import matplotlib.colors

def get_yshift(hf_file):
    """Get the y_shift to be applied on reconstructed raw images."""

    tmp = hf_file['mrecon_header']['Parameter']['YRange'][()]
    if len(np.unique(tmp[0])) > 1 or len(np.unique(tmp[1])) > 1:
        print('Error: different y shifts for different echoes!')
    return -int((tmp[0, 0] + tmp[1, 0]) / 2)


def load_raw_data(file_path):
    """Load raw data from the T2*-MOVE dataset."""

    with h5py.File(file_path, "r") as hf_file:
        raw_data = hf_file['kspace'][()]
        sens_maps = hf_file['sens_maps'][()]
        y_shift = get_yshift(hf_file)

    return raw_data, sens_maps, y_shift


def pad_sensitivity_maps(sens_maps, kspace_shape):
    """Pad coil sensitivity maps to have same shape as images."""

    pad_width = ((0, 0), (0, 0), (0, 0), (0, 0),
                 (int((kspace_shape[-1] - sens_maps.shape[-1]) / 2),
                  int((kspace_shape[-1] - sens_maps.shape[-1]) / 2))
                 )
    sens_maps = np.pad(sens_maps, pad_width, mode='constant')
    return np.nan_to_num(sens_maps / rss(sens_maps, 2)[:, None])


def remove_readout_oversampling(data, nr_lines):
    """Remove readout oversampling."""

    return data[..., nr_lines:-nr_lines]


def compute_coil_combined_reconstructions(kspace, sens_maps,
                                          y_shift, remove_oversampling=True):
    """Compute coil combined reconstructions."""

    coil_imgs = ifft2c_mri(kspace)
    coil_imgs = np.roll(coil_imgs, shift=y_shift, axis=-2)
    sens_maps = pad_sensitivity_maps(sens_maps, kspace.shape)
    img_cc = np.sum(coil_imgs * np.conj(sens_maps), axis=2)
    if remove_oversampling:
        img_cc = remove_readout_oversampling(img_cc,
                                             int(img_cc.shape[-1] / 4))
    return img_cc


def load_coil_combined_reconstruction(file_path):
    """Load the coil-combined reconstruction from the T2*-MOVE dataset."""

    with h5py.File(file_path, "r") as hf_file:
        img_cc = hf_file['reconstruction'][()]
        nii_header = {}
        for key in hf_file['nifti_header'].keys():
            nii_header[key] = hf_file['nifti_header'][key][()]

    return img_cc, nii_header


def load_reference_mask(file_path):
    """Load the reference exclusion mask for the motion-corrupted acquisition."""

    if os.path.exists(file_path):
        tmp = np.loadtxt(file_path, unpack=True).T
        # shift to match the correct timing:
        tmp = np.roll(tmp, 3, axis=1)
        tmp[:, 0:3] = 1
        # mask_timing = np.take(tmp, idx, axis=0)
        return tmp

    else:
        print(f"Reference mask file {file_path} does not exist.")
        return None


def load_segmentation(file_path, binary=True):
    """Load mask from nii file."""

    mask = nib.load(file_path).get_fdata()[10:-10][::-1, ::-1, :]
    mask = np.rollaxis(mask, 2, 0)
    if binary:
        mask = np.where(mask < 0.5, 0, 1)

    return mask


def load_motion_data(file_path):
    """Load motion data from a JSON file."""

    with open(os.path.join(file_path), 'r') as f:
        data = json.load(f)

    data.pop("RMS_displacement")
    data.pop("max_displacement")
    data.pop("motion_free")

    return data

def k2img(k, csm=None, im_size=None, norm_factor=1):
    """
    Convert k-space to image space
    :param k: k-space data on a Cartesian grid
    :param csm: coil sensitivity maps
    :return: image
    """

    coil_img = ifft2c_mri(k)
    if im_size is not None:
        coil_img = center_crop(coil_img, im_size)
        if csm is not None:
            csm = center_crop(csm, im_size)

    k_mag = k[:,4,:,:].abs().unsqueeze(1).detach().cpu().numpy()        # nt, nx, ny   
    # combined_img_motion = coil_img_motion.abs()
    if csm is not None:
        if len(csm.shape) == len(coil_img.shape):
            im_shape = csm.shape[2:]        # (nx, ny)
        else:
            im_shape = csm.shape[1:]        # (nx, ny)
        combined_img = coilcombine(coil_img, im_shape, coil_dim=1, csm=csm)
    else:
        combined_img = coilcombine(coil_img, coil_dim=1, mode='rss')
    combined_phase = torch.angle(combined_img).detach().cpu().numpy()
    combined_mag = combined_img.abs().detach().cpu().numpy()
    k_mag = np.log(np.abs(k_mag) + 1e-4)
    
    k_min = np.min(k_mag)
    k_max = np.max(k_mag)
    max_int = 255

    # combined_mag_nocenter = combined_mag
    # combined_mag_nocenter[:,:,combined_img.shape[-2]//2-10:combined_img.shape[-2]//2+10,combined_img.shape[-1]//2-10:combined_img.shape[-1]//2+10] = 0
    combined_mag_max = combined_mag.max() / norm_factor

    k_mag = (k_mag - k_min)*(max_int)/(k_max - k_min)
    k_mag = np.minimum(max_int, np.maximum(0.0, k_mag))
    k_mag = k_mag.astype(np.uint8)
    combined_mag = (combined_mag / combined_mag_max * 255)#.astype(np.uint8)
    combined_phase = angle2color(combined_phase, cmap='viridis', vmin=-np.pi, vmax=np.pi)
    k_mag = np.clip(k_mag, 0, 255).astype(np.uint8)
    combined_mag = np.clip(combined_mag, 0, 255).astype(np.uint8)
    combined_phase = np.clip(combined_phase, 0, 255).astype(np.uint8)

    combined_img = combined_img.detach().cpu().numpy()
    vis_dic = {
        'k_mag': k_mag, 
        'combined_mag': combined_mag, 
        'combined_phase': combined_phase, 
        'combined_img': combined_img
    }
    return vis_dic

def angle2color(value_arr, cmap='viridis', vmin=None, vmax=None):
    """
    Convert a value to a color using a colormap
    :param value: the value to convert
    :param cmap: the colormap to use
    :return: the color
    """
    if vmin is None:
        vmin = value_arr.min()
    if vmax is None:
        vmax = value_arr.max()
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    try:
        value_arr = value_arr.squeeze(0)
    except:
        value_arr = value_arr.squeeze()
    if len(value_arr.shape) == 3:
        color_arr = np.zeros((*value_arr.shape, 4))
        for i in range(value_arr.shape[0]):
            color_arr[i] = mapper.to_rgba(value_arr[i], bytes=True)
        color_arr = color_arr.transpose(0, 3, 1, 2)
    elif len(value_arr.shape) == 2:
        color_arr = mapper.to_rgba(value_arr, bytes=True)
    return color_arr
