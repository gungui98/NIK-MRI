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
import matplotlib.pyplot as plt

def get_yshift(hf_file):
    """Get the y_shift to be applied on reconstructed raw images.

    Falls back to 0 if header fields are missing.
    """

    try:
        tmp = hf_file['mrecon_header']['Parameter']['YRange'][()]
        if len(np.unique(tmp[0])) > 1 or len(np.unique(tmp[1])) > 1:
            print('Error: different y shifts for different echoes!')
        return -int((tmp[0, 0] + tmp[1, 0]) / 2)
    except Exception:
        return 0


def inspect_h5(file_path):
    """Inspect an HDF5 file: list top-level keys and dataset shapes."""

    with h5py.File(file_path, "r") as f:
        info = {}
        def _gather(group, prefix=""):
            for k in group.keys():
                obj = group[k]
                name = f"{prefix}{k}"
                if isinstance(obj, h5py.Dataset):
                    info[name] = obj.shape
                elif isinstance(obj, h5py.Group):
                    _gather(obj, prefix=name + "/")
        _gather(f)
    return info


def _first_existing_key(h5file, candidates):
    for key in candidates:
        if key in h5file:
            return key
    return None


def load_brain_h5(file_path):
    """Load brain data from an HDF5 file with flexible key detection.

    Returns
    -------
    kspace : np.ndarray
        Expected shape [slices, echoes, coils, kx, ky] or [echoes, coils, kx, ky]
    sens_maps : np.ndarray
        Sensitivity maps. If readout-oversampled, ky is half-size.
    y_shift : int
        Shift along phase-encode to align reconstructions.
    """

    with h5py.File(file_path, "r") as f:
        k_key = _first_existing_key(f, [
            'kspace', 'kdata', 'k', 'raw/kspace', 'raw/kdata'
        ])
        s_key = _first_existing_key(f, [
            'sens_maps', 'smap', 'csm', 'sensitivity_maps', 'sensmaps'
        ])

        if k_key is None:
            raise KeyError("Could not find a k-space dataset. Tried: 'kspace','kdata','k', ...")
        if s_key is None:
            raise KeyError("Could not find a sensitivity maps dataset. Tried: 'sens_maps','smap','csm', ...")

        kspace = f[k_key][()]
        sens_maps = f[s_key][()]
        try:
            y_shift = get_yshift(f)
        except Exception:
            y_shift = 0

    return kspace, sens_maps, y_shift


def load_raw_data(file_path):
    """Load raw data (legacy alias) using flexible loader."""

    return load_brain_h5(file_path)


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
    """Compute coil combined reconstructions.

    Accepts numpy arrays, performs reconstruction in torch, returns numpy.
    """

    # Convert inputs to torch tensors
    if isinstance(kspace, np.ndarray):
        kspace_t = torch.from_numpy(kspace)
    else:
        kspace_t = kspace

    coil_imgs = ifft2c_mri(kspace_t)
    coil_imgs = torch.roll(coil_imgs, shifts=y_shift, dims=-2)

    # Prepare sensitivity maps
    if isinstance(sens_maps, np.ndarray):
        if sens_maps.ndim == 5:
            # raw shape (S,1,C,Kx,Ky_half) -> pad to (S,1,C,Kx,Ky)
            sens_maps_pad = pad_sensitivity_maps(sens_maps, kspace_t.shape)
            sens_maps_np = sens_maps_pad
        elif sens_maps.ndim == 3:
            # already padded (C,Kx,Ky) -> expand slice/echo dims
            sens_maps_np = sens_maps[None, None]
        else:
            raise ValueError(f"Unsupported sens_maps shape: {sens_maps.shape}")
        sens_maps_t = torch.from_numpy(sens_maps_np)
    else:
        # torch.Tensor
        if sens_maps.ndim == 3:
            sens_maps_t = sens_maps[None, None]
        else:
            sens_maps_t = sens_maps

    img_cc = torch.sum(coil_imgs * torch.conj(sens_maps_t), dim=2)
    if remove_oversampling:
        tmp = int(img_cc.shape[-1] / 4)
        img_cc = img_cc[..., tmp:-tmp]
    return img_cc.detach().cpu().numpy()


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

def write_to_video(k_mag, combined_mag, combined_phase, combined_img, kpred):
    """Write the images to a video gif file using imageio."""
    import imageio
    imageio.mimsave('k_mag.gif', k_mag[:,0], fps=10, loop=0)
    imageio.mimsave('combined_mag.gif', combined_mag[:,0], fps=10, loop=0)
    imageio.mimsave('combined_phase.gif', combined_phase[:,0], fps=10, loop=0)
    # imageio.mimsave('combined_img.gif', combined_img[:,0], fps=10, loop=0)


def find_best_slice(img_cc_fs):
    """Find slice with maximum signal for visualization.
    
    Parameters
    ----------
    img_cc_fs : np.ndarray
        Complex reconstruction image (slices, echoes, kx, ky) or (echoes, kx, ky)
    
    Returns
    -------
    int
        Index of slice with highest mean magnitude, or 0 if single slice
    """
    # Handle both multi-slice and single-slice cases
    if img_cc_fs.ndim == 3:
        # Single slice (echoes, kx, ky)
        return 0
    
    # Multi-slice (slices, echoes, kx, ky)
    slice_magnitudes = []
    for slice_idx in range(img_cc_fs.shape[0]):
        slice_mag = np.abs(img_cc_fs[slice_idx]).mean()
        slice_magnitudes.append(slice_mag)
    
    return int(np.argmax(slice_magnitudes))


def visualize_reconstruction(img_cc_fs, slice_idx=0, echo_idx=0, output_prefix="recon"):
    """Visualize reconstructed images using legacy approach.
    
    Based on legacy code: converts magnitude and phase, normalizes properly,
    and saves with matplotlib for better visualization.
    
    Parameters
    ----------
    img_cc_fs : np.ndarray
        Complex reconstruction image (slices, echoes, kx, ky) or (echoes, kx, ky)
    slice_idx : int
        Which slice to visualize
    echo_idx : int
        Which echo to visualize
    output_prefix : str
        Prefix for output filenames
        
    Returns
    -------
    tuple
        (magnitude_norm, phase) - normalized magnitude and phase arrays
    """
    # Handle both multi-slice and single-slice cases
    if img_cc_fs.ndim == 3:
        # Single slice (echoes, kx, ky)
        img = img_cc_fs[echo_idx]
    else:
        # Multi-slice (slices, echoes, kx, ky)
        img = img_cc_fs[slice_idx, echo_idx]
    
    # Extract magnitude and phase (legacy approach)
    magnitude = np.abs(img)
    phase = np.angle(img)  # Range: -π to π
    
    # Normalize magnitude to [0, 1] for better visualization
    max_mag = np.max(magnitude)
    if max_mag > 0:
        magnitude_norm = magnitude / max_mag
    else:
        magnitude_norm = magnitude
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    
    # Magnitude (transposed as in legacy code)
    im1 = axes[0].imshow(magnitude_norm.T, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(f'Magnitude (Slice {slice_idx}, Echo {echo_idx})', fontsize=10)
    axes[0].axis('off')
    cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.ax.tick_params(labelsize=8)
    
    # Phase (transposed as in legacy code)
    im2 = axes[1].imshow(phase.T, cmap='viridis', vmin=-np.pi, vmax=np.pi)
    axes[1].set_title(f'Phase (Slice {slice_idx}, Echo {echo_idx})', fontsize=10)
    axes[1].axis('off')
    cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.ax.tick_params(labelsize=8)
    
    plt.tight_layout()
    output_file = f"{output_prefix}_slice{slice_idx}_echo{echo_idx}.png"
    plt.savefig(output_file, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"  Saved visualization: {output_file}")
    
    return magnitude_norm, phase


def visualize_all_echoes(img_cc_fs, slice_idx=0, output_prefix="recon_echoes"):
    """Visualize all echoes for a single slice (legacy multi-echo approach).
    
    Parameters
    ----------
    img_cc_fs : np.ndarray
        Complex reconstruction image (slices, echoes, kx, ky) or (echoes, kx, ky)
    slice_idx : int
        Which slice to visualize
    output_prefix : str
        Prefix for output filename
        
    Returns
    -------
    str
        Path to saved visualization file
    """
    # Handle both multi-slice and single-slice cases
    if img_cc_fs.ndim == 3:
        # Single slice (echoes, kx, ky)
        img_slice = img_cc_fs
    else:
        # Multi-slice (slices, echoes, kx, ky)
        img_slice = img_cc_fs[slice_idx]
    
    num_echoes = img_slice.shape[0]
    
    # Create figure with echoes as columns
    fig, axes = plt.subplots(2, num_echoes, figsize=(2*num_echoes, 4), dpi=150)
    
    # Calculate global min/max for consistent scaling (legacy approach)
    magnitude_all = np.abs(img_slice)
    max_mag = np.max(magnitude_all)
    
    for echo_idx in range(num_echoes):
        if max_mag > 0:
            magnitude = np.abs(img_slice[echo_idx]) / max_mag
        else:
            magnitude = np.abs(img_slice[echo_idx])
        phase = np.angle(img_slice[echo_idx])
        
        # Magnitude row
        axes[0, echo_idx].imshow(magnitude.T, cmap='gray', vmin=0, vmax=1)
        axes[0, echo_idx].set_title(f'Echo {echo_idx}', fontsize=8)
        axes[0, echo_idx].axis('off')
        
        # Phase row
        axes[1, echo_idx].imshow(phase.T, cmap='viridis', vmin=-np.pi, vmax=np.pi)
        axes[1, echo_idx].axis('off')
    
    # Add row labels
    axes[0, 0].set_ylabel('Magnitude', fontsize=9)
    axes[1, 0].set_ylabel('Phase', fontsize=9)
    
    plt.tight_layout()
    output_file = f"{output_prefix}_slice{slice_idx}.png"
    plt.savefig(output_file, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"  Saved multi-echo visualization: {output_file}")
    
    return output_file