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
    """Compute coil combined reconstructions."""
    coil_imgs = ifft2c_mri(kspace)
    coil_imgs = torch.roll(coil_imgs, shifts=y_shift, dims=(-2,))
    img_cc = torch.sum(coil_imgs * torch.conj(sens_maps), dim=2)
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
    :param k: k-space data on a Cartesian grid, shape: (echo, slices, kx, ky)
    :param csm: coil sensitivity maps, shape: (slice, kx, ky)
    :return: image with shape (slices, kx, ky)
    """

    coil_img = ifft2c_mri(k)
    if im_size is not None:
        coil_img = center_crop(coil_img, im_size)
        if csm is not None:
            csm = center_crop(csm, im_size)

    # k has shape (echo, slices, kx, ky), select middle echo for k_mag visualization
    k_mag = k[k.shape[0]//2,:,:,:].abs().unsqueeze(1).detach().cpu().numpy()        # slices, 1, kx, ky   
    # combined_img_motion = coil_img_motion.abs()
    if csm is not None:
        if len(csm.shape) == len(coil_img.shape):
            im_shape = csm.shape[1:]        # (kx, ky)
        else:
            im_shape = csm.shape[1:]        # (kx, ky)
        # Combine along echo dimension (dim=0), output will be (slices, kx, ky)
        combined_img = coilcombine(coil_img, im_shape, coil_dim=0, csm=csm)
    else:
        combined_img = coilcombine(coil_img, coil_dim=0, mode='rss')
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


def visualize_reconstruction(img_cc_fs, slice_idx=0, echo_idx=0, output_prefix="recon", 
                            img_target=None, compare=True, output_dir=None, return_fig=False):
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
    img_target : np.ndarray, optional
        Target/ground truth image for comparison (same shape as img_cc_fs)
    compare : bool
        If True and img_target is provided, show comparison plots
    output_dir : str, optional
        Directory to save the image. If None, saves to current directory.
    return_fig : bool
        If True, returns the figure object for wandb logging
        
    Returns
    -------
    tuple
        (magnitude_norm, phase) - normalized magnitude and phase arrays
        If return_fig=True, also returns the figure object
    """
    # Handle both multi-slice and single-slice cases
    if img_cc_fs.ndim == 3:
        # Single slice (echoes, kx, ky)
        img_pred = img_cc_fs[echo_idx]
    else:
        # Multi-slice (slices, echoes, kx, ky)
        img_pred = img_cc_fs[slice_idx, echo_idx]
    
    # Extract magnitude and phase for prediction (legacy approach)
    magnitude_pred = np.abs(img_pred)
    phase_pred = np.angle(img_pred)  # Range: -π to π
    
    # Normalize magnitude to [0, 1] for better visualization
    max_mag_pred = np.max(magnitude_pred)
    min_mag_pred = np.min(magnitude_pred)
    
    # Debug output to diagnose black images
    if compare:
        print(f"  Predicted magnitude range: [{min_mag_pred:.2e}, {max_mag_pred:.2e}]")
    
    if max_mag_pred > 0:
        magnitude_norm_pred = magnitude_pred / max_mag_pred
    else:
        print("  WARNING: Predicted magnitude is all zero or near zero!")
        magnitude_norm_pred = magnitude_pred
    
    # Initialize figure variables for return_fig
    fig_compare = None
    fig_single = None
    
    if compare and img_target is not None:
        # Handle target extraction same way as prediction
        if img_target.ndim == 3:
            img_tgt = img_target[echo_idx]
        else:
            img_tgt = img_target[slice_idx, echo_idx]
        
        magnitude_tgt = np.abs(img_tgt)
        phase_tgt = np.angle(img_tgt)
        
        # Debug output
        max_mag_tgt = np.max(magnitude_tgt)
        min_mag_tgt = np.min(magnitude_tgt)
        print(f"  Target magnitude range: [{min_mag_tgt:.2e}, {max_mag_tgt:.2e}]")
        
        # Normalize each separately to avoid one appearing black when scales are very different
        # This can happen when predicted k-space is normalized but target is not
        max_mag_pred = np.max(magnitude_pred) if np.max(magnitude_pred) > 0 else 1.0
        max_mag_tgt_val = np.max(magnitude_tgt) if np.max(magnitude_tgt) > 0 else 1.0
        
        magnitude_norm_pred = magnitude_pred / max_mag_pred
        magnitude_norm_tgt = magnitude_tgt / max_mag_tgt_val
        
        # Create comparison figure with prediction and target side-by-side
        _fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=150)
        
        # Row 0: Magnitude comparison
        im1 = axes[0, 0].imshow(magnitude_norm_tgt.T, cmap='gray', vmin=0, vmax=1)
        axes[0, 0].set_title('Target Magnitude', fontsize=10)
        axes[0, 0].axis('off')
        cbar1 = plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
        cbar1.ax.tick_params(labelsize=8)
        
        im2 = axes[0, 1].imshow(magnitude_norm_pred.T, cmap='gray', vmin=0, vmax=1)
        axes[0, 1].set_title('Predicted Magnitude', fontsize=10)
        axes[0, 1].axis('off')
        cbar2 = plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
        cbar2.ax.tick_params(labelsize=8)
        
        # Difference image
        diff_mag = np.abs(magnitude_norm_tgt - magnitude_norm_pred)
        im3 = axes[0, 2].imshow(diff_mag.T, cmap='hot', vmin=0, vmax=1)
        axes[0, 2].set_title('Magnitude Difference', fontsize=10)
        axes[0, 2].axis('off')
        cbar3 = plt.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)
        cbar3.ax.tick_params(labelsize=8)
        
        # Row 1: Phase comparison
        im4 = axes[1, 0].imshow(phase_tgt.T, cmap='viridis', vmin=-np.pi, vmax=np.pi)
        axes[1, 0].set_title('Target Phase', fontsize=10)
        axes[1, 0].axis('off')
        cbar4 = plt.colorbar(im4, ax=axes[1, 0], fraction=0.046, pad=0.04)
        cbar4.ax.tick_params(labelsize=8)
        
        im5 = axes[1, 1].imshow(phase_pred.T, cmap='viridis', vmin=-np.pi, vmax=np.pi)
        axes[1, 1].set_title('Predicted Phase', fontsize=10)
        axes[1, 1].axis('off')
        cbar5 = plt.colorbar(im5, ax=axes[1, 1], fraction=0.046, pad=0.04)
        cbar5.ax.tick_params(labelsize=8)
        
        # Phase difference using complex difference (avoids phase wrap issues)
        # Compute phase error via complex product: arg(F_pred * F_tgt*)
        # This properly handles phase wrapping and avoids discontinuities
        complex_diff = img_pred * np.conj(img_tgt)
        phase_err = np.angle(complex_diff)  # Range: -π to π
        phase_err_mag = np.abs(phase_err)  # Magnitude of phase error (0 to π)
        
        im6 = axes[1, 2].imshow(phase_err_mag.T, cmap='hot', vmin=0, vmax=np.pi)
        axes[1, 2].set_title('Phase Difference (Complex)', fontsize=10)
        axes[1, 2].axis('off')
        cbar6 = plt.colorbar(im6, ax=axes[1, 2], fraction=0.046, pad=0.04)
        cbar6.ax.tick_params(labelsize=8)
        
        plt.suptitle(f'Slice {slice_idx}, Echo {echo_idx}', fontsize=12, y=0.98)
        plt.tight_layout()
        
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{output_prefix}_slice{slice_idx}_echo{echo_idx}.png")
        else:
            output_file = f"{output_prefix}_slice{slice_idx}_echo{echo_idx}.png"
        
        plt.savefig(output_file, bbox_inches='tight', dpi=150)
        
        if return_fig:
            fig_compare = plt.gcf()
        else:
            plt.close()
    else:
        # Original single-image visualization
        _fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
        
        # Magnitude (transposed as in legacy code)
        im1 = axes[0].imshow(magnitude_norm_pred.T, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title(f'Magnitude (Slice {slice_idx}, Echo {echo_idx})', fontsize=10)
        axes[0].axis('off')
        cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        cbar1.ax.tick_params(labelsize=8)
        
        # Phase (transposed as in legacy code)
        im2 = axes[1].imshow(phase_pred.T, cmap='viridis', vmin=-np.pi, vmax=np.pi)
        axes[1].set_title(f'Phase (Slice {slice_idx}, Echo {echo_idx})', fontsize=10)
        axes[1].axis('off')
        cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        cbar2.ax.tick_params(labelsize=8)
        
        plt.tight_layout()
        
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{output_prefix}_slice{slice_idx}_echo{echo_idx}.png")
        else:
            output_file = f"{output_prefix}_slice{slice_idx}_echo{echo_idx}.png"
        
        plt.savefig(output_file, bbox_inches='tight', dpi=150)
        
        if return_fig:
            fig_single = plt.gcf()
        else:
            plt.close()
    
    print(f"  Saved visualization: {output_file}")
    
    if return_fig:
        # Determine which figure to return
        if compare and img_target is not None:
            fig = fig_compare
        else:
            fig = fig_single
        if fig is None:
            # Fallback: create a dummy figure (shouldn't happen, but just in case)
            fig = plt.figure()
        return magnitude_norm_pred, phase_pred, fig
    else:
        return magnitude_norm_pred, phase_pred


def visualize_all_echoes(img_cc_fs, slice_idx=0, output_prefix="recon_echoes", 
                         img_target=None, compare=True, output_dir=None, return_fig=False):
    """Visualize all echoes for a single slice (legacy multi-echo approach).
    
    Parameters
    ----------
    img_cc_fs : np.ndarray
        Complex reconstruction image (slices, echoes, kx, ky) or (echoes, kx, ky)
    slice_idx : int
        Which slice to visualize
    output_prefix : str
        Prefix for output filename
    img_target : np.ndarray, optional
        Target/ground truth image for comparison (same shape as img_cc_fs)
    compare : bool
        If True and img_target is provided, show comparison plots
    output_dir : str, optional
        Directory to save the image. If None, saves to current directory.
    return_fig : bool
        If True, returns the figure object for wandb logging
        
    Returns
    -------
    str
        Path to saved visualization file
    If return_fig=True, also returns the figure object
    """
    # Handle both multi-slice and single-slice cases
    if img_cc_fs.ndim == 3:
        # Single slice (echoes, kx, ky)
        img_slice_pred = img_cc_fs
    else:
        # Multi-slice (slices, echoes, kx, ky)
        img_slice_pred = img_cc_fs[slice_idx]
    
    num_echoes = img_slice_pred.shape[0]
    
    if compare and img_target is not None:
        # Handle target extraction same way as prediction
        if img_target.ndim == 3:
            img_slice_tgt = img_target
        else:
            img_slice_tgt = img_target[slice_idx]
        
        # Normalize each separately to avoid one appearing black when scales are very different
        # This can happen when predicted k-space is normalized but target is not
        magnitude_all_pred = np.abs(img_slice_pred)
        magnitude_all_tgt = np.abs(img_slice_tgt)
        max_mag_pred = np.max(magnitude_all_pred) if np.max(magnitude_all_pred) > 0 else 1.0
        max_mag_tgt = np.max(magnitude_all_tgt) if np.max(magnitude_all_tgt) > 0 else 1.0
        
        # Debug output
        print(f"  All echoes - Predicted magnitude range: [0, {max_mag_pred:.2e}]")
        print(f"  All echoes - Target magnitude range: [0, {max_mag_tgt:.2e}]")
        
        # Create figure with 3 rows: Target, Predicted, Difference
        _fig, axes = plt.subplots(3, num_echoes, figsize=(2*num_echoes, 6), dpi=150)
        
        for echo_idx in range(num_echoes):
            # Target magnitude - normalize by target max
            magnitude_tgt = np.abs(img_slice_tgt[echo_idx]) / max_mag_tgt
            _phase_tgt = np.angle(img_slice_tgt[echo_idx])  # Computed but not visualized in this mode
            
            # Predicted magnitude - normalize by prediction max
            magnitude_pred = np.abs(img_slice_pred[echo_idx]) / max_mag_pred
            _phase_pred = np.angle(img_slice_pred[echo_idx])  # Computed but not visualized in this mode
            
            # Row 0: Target Magnitude and Phase
            axes[0, echo_idx].imshow(magnitude_tgt.T, cmap='gray', vmin=0, vmax=1)
            axes[0, echo_idx].set_title(f'Target Echo {echo_idx}', fontsize=8)
            axes[0, echo_idx].axis('off')
            
            # Row 1: Predicted Magnitude and Phase
            axes[1, echo_idx].imshow(magnitude_pred.T, cmap='gray', vmin=0, vmax=1)
            axes[1, echo_idx].set_title(f'Predicted Echo {echo_idx}', fontsize=8)
            axes[1, echo_idx].axis('off')
            
            # Row 2: Difference
            # For difference, use normalized values from separate normalizations
            # This shows where they differ in relative terms
            diff_mag = np.abs(magnitude_tgt - magnitude_pred)
            axes[2, echo_idx].imshow(diff_mag.T, cmap='hot', vmin=0, vmax=1)
            axes[2, echo_idx].set_title(f'Diff Echo {echo_idx}', fontsize=8)
            axes[2, echo_idx].axis('off')
        
        # Add row labels
        axes[0, 0].set_ylabel('Target', fontsize=9)
        axes[1, 0].set_ylabel('Predicted', fontsize=9)
        axes[2, 0].set_ylabel('Difference', fontsize=9)
    else:
        # Create figure with echoes as columns
        _fig, axes = plt.subplots(2, num_echoes, figsize=(2*num_echoes, 4), dpi=150)
        
        # Calculate global min/max for consistent scaling (legacy approach)
        magnitude_all = np.abs(img_slice_pred)
        max_mag = np.max(magnitude_all)
        
        for echo_idx in range(num_echoes):
            if max_mag > 0:
                magnitude = np.abs(img_slice_pred[echo_idx]) / max_mag
            else:
                magnitude = np.abs(img_slice_pred[echo_idx])
            phase = np.angle(img_slice_pred[echo_idx])
            
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
    
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{output_prefix}_slice{slice_idx}.png")
    else:
        output_file = f"{output_prefix}_slice{slice_idx}.png"
    
    plt.savefig(output_file, bbox_inches='tight', dpi=150)
    
    if return_fig:
        fig = plt.gcf()
        print(f"  Saved multi-echo visualization: {output_file}")
        return output_file, fig
    else:
        plt.close()
        print(f"  Saved multi-echo visualization: {output_file}")
        return output_file


def visualize_kspace_error(kpred, ktarget, slice_idx=0, echo_idx=0, output_prefix="kspace_error",
                           output_dir=None, return_fig=False):
    """Visualize k-space learning errors to identify which parts are learned correctly.
    
    This function creates comprehensive visualizations showing:
    - Target and predicted k-space magnitude (log scale)
    - Target and predicted k-space phase
    - Error maps showing where predictions differ from targets
    - Normalized error to identify regions with high/low learning quality
    
    Parameters
    ----------
    kpred : torch.Tensor or np.ndarray
        Predicted k-space data, shape (slices, echoes, kx, ky) or (echoes, kx, ky)
    ktarget : torch.Tensor or np.ndarray
        Target/ground truth k-space data, same shape as kpred
    slice_idx : int
        Which slice to visualize (if multi-slice)
    echo_idx : int
        Which echo to visualize
    output_prefix : str
        Prefix for output filenames
    output_dir : str, optional
        Directory to save the image. If None, saves to current directory.
    return_fig : bool
        If True, returns the figure object for wandb logging
        
    Returns
    -------
    dict or tuple
        Dictionary with error metrics, or tuple with error metrics and figure if return_fig=True
    """
    # Convert to numpy if needed
    if isinstance(kpred, torch.Tensor):
        kpred = kpred.detach().cpu().numpy()
    if isinstance(ktarget, torch.Tensor):
        ktarget = ktarget.detach().cpu().numpy()
    
    # Ensure complex type
    kpred = kpred.astype(np.complex64)
    ktarget = ktarget.astype(np.complex64)
    
    # Handle shape mismatches - common case: kpred has singleton first dim, ktarget has multiple slices
    kpred_slice_idx = slice_idx  # Default to provided slice_idx
    if kpred.shape != ktarget.shape:
        # Case 1: kpred has singleton first dimension, ktarget has multiple slices
        # e.g., kpred: (1, echoes, kx, ky), ktarget: (slices, echoes, kx, ky)
        if (kpred.ndim == ktarget.ndim and 
            kpred.shape[0] == 1 and 
            ktarget.shape[0] > 1 and
            kpred.shape[1:] == ktarget.shape[1:]):
            # Squeeze kpred and use slice_idx=0 for kpred (only one slice)
            kpred = kpred.squeeze(0)
            kpred_slice_idx = 0
        # Case 2: kpred has extra singleton dimension
        elif kpred.ndim == ktarget.ndim + 1 and kpred.shape[0] == 1:
            kpred = kpred.squeeze(0)
        # Case 3: ktarget has extra singleton dimension
        elif ktarget.ndim == kpred.ndim + 1 and ktarget.shape[0] == 1:
            ktarget = ktarget.squeeze(0)
        else:
            # Check if remaining dimensions match (for informative error)
            if kpred.ndim == ktarget.ndim:
                if kpred.shape[1:] == ktarget.shape[1:]:
                    # Dimensions match except first - this is handled above, but if we get here, 
                    # it means the first dimension check failed, so provide helpful error
                    raise ValueError(
                        f"Shape mismatch: kpred shape {kpred.shape} != ktarget shape {ktarget.shape}. "
                        f"kpred has {kpred.shape[0]} slices, ktarget has {ktarget.shape[0]} slices. "
                        f"Remaining dimensions match: {kpred.shape[1:]}. "
                        f"Please ensure kpred and ktarget have compatible shapes."
                    )
            raise ValueError(f"Shape mismatch: kpred shape {kpred.shape} != ktarget shape {ktarget.shape}")
    
    # Handle both multi-slice and single-slice cases
    if kpred.ndim == 3:
        # Single slice (echoes, kx, ky)
        if echo_idx >= kpred.shape[0]:
            raise ValueError(f"echo_idx {echo_idx} out of range for k-space with {kpred.shape[0]} echoes")
        kpred_slice = kpred[echo_idx]
        # For ktarget, check if it's 3D or 4D
        if ktarget.ndim == 3:
            ktarget_slice = ktarget[echo_idx]
        else:
            ktarget_slice = ktarget[slice_idx, echo_idx]
    else:
        # Multi-slice (slices, echoes, kx, ky)
        if kpred_slice_idx >= kpred.shape[0]:
            raise ValueError(f"slice_idx {kpred_slice_idx} out of range for k-space with {kpred.shape[0]} slices")
        if echo_idx >= kpred.shape[1]:
            raise ValueError(f"echo_idx {echo_idx} out of range for k-space with {kpred.shape[1]} echoes")
        kpred_slice = kpred[kpred_slice_idx, echo_idx]
        # For ktarget, use the provided slice_idx
        if slice_idx >= ktarget.shape[0]:
            raise ValueError(f"slice_idx {slice_idx} out of range for ktarget with {ktarget.shape[0]} slices")
        if echo_idx >= ktarget.shape[1]:
            raise ValueError(f"echo_idx {echo_idx} out of range for ktarget with {ktarget.shape[1]} echoes")
        ktarget_slice = ktarget[slice_idx, echo_idx]
    
    # Compute k-space magnitudes (log scale for better visualization)
    kpred_mag = np.abs(kpred_slice)
    ktarget_mag = np.abs(ktarget_slice)
    
    # Use log scale for k-space magnitude visualization (common in MRI)
    kpred_mag_log = np.log(kpred_mag + 1e-8)
    ktarget_mag_log = np.log(ktarget_mag + 1e-8)
    
    # Compute phases
    kpred_phase = np.angle(kpred_slice)
    ktarget_phase = np.angle(ktarget_slice)
    
    # Compute various error metrics
    # 1. Magnitude error (absolute)
    mag_error_abs = np.abs(kpred_mag - ktarget_mag)
    
    # 2. Relative magnitude error (normalized by target magnitude)
    mag_error_rel = np.zeros_like(mag_error_abs)
    nonzero_mask = ktarget_mag > 1e-8
    mag_error_rel[nonzero_mask] = mag_error_abs[nonzero_mask] / ktarget_mag[nonzero_mask]
    
    # 3. Complex error (magnitude of difference)
    complex_error = np.abs(kpred_slice - ktarget_slice)
    
    # 4. Normalized complex error (relative to max target magnitude)
    max_target_mag = np.max(ktarget_mag)
    complex_error_norm = complex_error / (max_target_mag + 1e-8)
    
    # 5. Phase error (using complex product to handle phase wrapping)
    complex_product = kpred_slice * np.conj(ktarget_slice)
    phase_error = np.angle(complex_product)  # Range: -π to π
    phase_error_mag = np.abs(phase_error)  # Magnitude of phase error (0 to π)
    
    # 6. Error ratio: |kpred - ktarget| / |ktarget|
    error_ratio = np.zeros_like(complex_error)
    error_ratio[nonzero_mask] = complex_error[nonzero_mask] / ktarget_mag[nonzero_mask]
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(3, 4, figsize=(20, 15), dpi=150)
    
    # Row 0: Magnitude comparison
    vmin_mag_log = min(np.min(kpred_mag_log), np.min(ktarget_mag_log))
    vmax_mag_log = max(np.max(kpred_mag_log), np.max(ktarget_mag_log))
    
    im1 = axes[0, 0].imshow(ktarget_mag_log.T, cmap='gray', vmin=vmin_mag_log, vmax=vmax_mag_log)
    axes[0, 0].set_title('Target K-space Magnitude (log)', fontsize=10)
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
    
    im2 = axes[0, 1].imshow(kpred_mag_log.T, cmap='gray', vmin=vmin_mag_log, vmax=vmax_mag_log)
    axes[0, 1].set_title('Predicted K-space Magnitude (log)', fontsize=10)
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    im3 = axes[0, 2].imshow(mag_error_abs.T, cmap='hot', vmin=0)
    axes[0, 2].set_title('Magnitude Error (Absolute)', fontsize=10)
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    im4 = axes[0, 3].imshow(mag_error_rel.T, cmap='hot', vmin=0, vmax=np.percentile(mag_error_rel[nonzero_mask], 95))
    axes[0, 3].set_title('Magnitude Error (Relative)', fontsize=10)
    axes[0, 3].axis('off')
    plt.colorbar(im4, ax=axes[0, 3], fraction=0.046, pad=0.04)
    
    # Row 1: Phase comparison
    im5 = axes[1, 0].imshow(ktarget_phase.T, cmap='viridis', vmin=-np.pi, vmax=np.pi)
    axes[1, 0].set_title('Target K-space Phase', fontsize=10)
    axes[1, 0].axis('off')
    plt.colorbar(im5, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    im6 = axes[1, 1].imshow(kpred_phase.T, cmap='viridis', vmin=-np.pi, vmax=np.pi)
    axes[1, 1].set_title('Predicted K-space Phase', fontsize=10)
    axes[1, 1].axis('off')
    plt.colorbar(im6, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    im7 = axes[1, 2].imshow(phase_error.T, cmap='RdBu', vmin=-np.pi, vmax=np.pi)
    axes[1, 2].set_title('Phase Error (Signed)', fontsize=10)
    axes[1, 2].axis('off')
    plt.colorbar(im7, ax=axes[1, 2], fraction=0.046, pad=0.04)
    
    im8 = axes[1, 3].imshow(phase_error_mag.T, cmap='hot', vmin=0, vmax=np.pi)
    axes[1, 3].set_title('Phase Error (Magnitude)', fontsize=10)
    axes[1, 3].axis('off')
    plt.colorbar(im8, ax=axes[1, 3], fraction=0.046, pad=0.04)
    
    # Row 2: Complex error metrics
    im9 = axes[2, 0].imshow(complex_error.T, cmap='hot', vmin=0)
    axes[2, 0].set_title('Complex Error |kpred - ktarget|', fontsize=10)
    axes[2, 0].axis('off')
    plt.colorbar(im9, ax=axes[2, 0], fraction=0.046, pad=0.04)
    
    im10 = axes[2, 1].imshow(complex_error_norm.T, cmap='hot', vmin=0, vmax=1)
    axes[2, 1].set_title('Normalized Complex Error', fontsize=10)
    axes[2, 1].axis('off')
    plt.colorbar(im10, ax=axes[2, 1], fraction=0.046, pad=0.04)
    
    im11 = axes[2, 2].imshow(error_ratio.T, cmap='hot', vmin=0, vmax=np.percentile(error_ratio[nonzero_mask], 95))
    axes[2, 2].set_title('Error Ratio |kpred-ktarget|/|ktarget|', fontsize=10)
    axes[2, 2].axis('off')
    plt.colorbar(im11, ax=axes[2, 2], fraction=0.046, pad=0.04)
    
    # Error map highlighting regions with high/low learning quality
    # Use normalized complex error with threshold
    error_threshold = 0.1  # 10% of max target magnitude
    learning_quality = np.ones_like(complex_error_norm)
    learning_quality[complex_error_norm > error_threshold] = 0  # Poor learning
    learning_quality[(complex_error_norm > error_threshold * 0.5) & 
                     (complex_error_norm <= error_threshold)] = 0.5  # Moderate learning
    
    im12 = axes[2, 3].imshow(learning_quality.T, cmap='RdYlGn', vmin=0, vmax=1)
    axes[2, 3].set_title(f'Learning Quality\n(Green=Good, Red=Poor, threshold={error_threshold:.2f})', fontsize=10)
    axes[2, 3].axis('off')
    plt.colorbar(im12, ax=axes[2, 3], fraction=0.046, pad=0.04)
    
    plt.suptitle(f'K-space Learning Analysis - Slice {slice_idx}, Echo {echo_idx}', 
                 fontsize=14, y=0.995)
    plt.tight_layout()
    
    # Compute summary statistics
    error_stats = {
        'mean_mag_error': np.mean(mag_error_abs),
        'mean_mag_error_rel': np.mean(mag_error_rel[nonzero_mask]) if np.any(nonzero_mask) else 0,
        'mean_complex_error': np.mean(complex_error),
        'mean_complex_error_norm': np.mean(complex_error_norm),
        'mean_phase_error': np.mean(phase_error_mag),
        'max_complex_error': np.max(complex_error),
        'rmse': np.sqrt(np.mean(complex_error**2)),
        'good_learning_ratio': np.mean(learning_quality > 0.5),
        'poor_learning_ratio': np.mean(learning_quality < 0.5),
    }
    
    # Print summary statistics
    print(f"\nK-space Error Statistics (Slice {slice_idx}, Echo {echo_idx}):")
    print(f"  Mean magnitude error: {error_stats['mean_mag_error']:.2e}")
    print(f"  Mean relative magnitude error: {error_stats['mean_mag_error_rel']:.2e}")
    print(f"  Mean complex error: {error_stats['mean_complex_error']:.2e}")
    print(f"  Mean normalized complex error: {error_stats['mean_complex_error_norm']:.2e}")
    print(f"  Mean phase error: {error_stats['mean_phase_error']:.2e} rad ({np.degrees(error_stats['mean_phase_error']):.2f} deg)")
    print(f"  RMSE: {error_stats['rmse']:.2e}")
    print(f"  Good learning ratio: {error_stats['good_learning_ratio']:.2%}")
    print(f"  Poor learning ratio: {error_stats['poor_learning_ratio']:.2%}")
    
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{output_prefix}_slice{slice_idx}_echo{echo_idx}.png")
    else:
        output_file = f"{output_prefix}_slice{slice_idx}_echo{echo_idx}.png"
    
    plt.savefig(output_file, bbox_inches='tight', dpi=150)
    print(f"  Saved k-space error visualization: {output_file}")
    
    if return_fig:
        return error_stats, fig
    else:
        plt.close()
        return error_stats