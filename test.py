"""
Standalone test script with legacy reconstruction code embedded.
No dependencies on local utils - all functions copied from legacy code.
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


def ifft2c(x):
    """Centered 2D inverse FFT (legacy version using numpy)."""
    x = np.fft.ifftshift(x, axes=(-2, -1))
    x = np.fft.ifft2(x, norm='ortho')
    x = np.fft.fftshift(x, axes=(-2, -1))
    return x


def rss(data, axis=0):
    """Root sum of squares along specified axis."""
    return np.sqrt(np.sum(np.abs(data)**2, axis=axis))


def get_yshift(hf_file):
    """Get the y_shift to be applied on reconstructed raw images.
    
    Handles both legacy format (out/Parameter/YRange) and new format (mrecon_header).
    """
    # Try legacy format
    try:
        tmp = hf_file['out']['Parameter']['YRange'][:]
        if len(np.unique(tmp[0])) > 1 or len(np.unique(tmp[1])) > 1:
            print('Warning: different y shifts for different echoes!')
        return -int((tmp[0, 0] + tmp[1, 0]) / 2)
    except:
        pass
    
    # Try new format
    try:
        tmp = hf_file['mrecon_header']['Parameter']['YRange'][()]
        if len(np.unique(tmp[0])) > 1 or len(np.unique(tmp[1])) > 1:
            print('Warning: different y shifts for different echoes!')
        return -int((tmp[0, 0] + tmp[1, 0]) / 2)
    except:
        pass
    
    # Default to 0 if not found
    print("Could not read y_shift from file. Using 0.")
    return 0


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


def process_raw_data(hf_file):
    """Load raw data from h5 file and process to proper complex data.
    
    Handles both legacy format (out/Data) and new format (kspace/kdata).
    """
    # Try legacy format first
    if 'out' in hf_file:
        raw_data = hf_file['out']['Data'][:, :, 0, 0, :, 0]
        sens_maps = hf_file['out']['SENSE']['maps'][:, :, 0, 0, :, 0]

        if (isinstance(raw_data, np.ndarray) and
                raw_data.dtype == [('real', '<f4'), ('imag', '<f4')]):
            return (raw_data.view(np.complex64).astype(np.complex64),
                    sens_maps.view(np.complex64).astype(np.complex64))
        else:
            print('Error: Unexpected data format:', raw_data.dtype)
            return None, None
    
    # Try new format with flexible keys
    else:
        # Try to find k-space data
        kspace_keys = ['kspace', 'kdata', 'k', 'raw/kspace', 'raw/kdata']
        sens_keys = ['sens_maps', 'smap', 'csm', 'sensitivity_maps', 'sensmaps']
        
        kspace = None
        sens_maps = None
        
        for key in kspace_keys:
            if key in hf_file:
                kspace = hf_file[key][()]
                break
        
        for key in sens_keys:
            if key in hf_file:
                sens_maps = hf_file[key][()]
                break
        
        if kspace is None:
            print(f"Error: Could not find k-space data. Available keys: {list(hf_file.keys())}")
            return None, None
        
        if sens_maps is None:
            print(f"Error: Could not find sensitivity maps. Available keys: {list(hf_file.keys())}")
            return None, None
        
        # Ensure complex type
        if not np.iscomplexobj(kspace):
            kspace = kspace.astype(np.complex64)
        if not np.iscomplexobj(sens_maps):
            sens_maps = sens_maps.astype(np.complex64)
        
        return kspace, sens_maps


def compute_coil_combined_reconstructions(kspace, sens_maps,
                                          y_shift, remove_oversampling=True):
    """Compute coil combined reconstructions."""
    coil_imgs = ifft2c(kspace)
    coil_imgs = np.roll(coil_imgs, shift=y_shift, axis=-2)
    img_cc = np.sum(coil_imgs * np.conj(sens_maps), axis=2)
    if remove_oversampling:
        img_cc = remove_readout_oversampling(img_cc,
                                             int(img_cc.shape[-1] / 4))
    return img_cc


def load_raw_data(filename):
    """Load and reconstruct data from legacy h5 file."""
    with h5py.File(filename, "r") as f:
        kspace, sens_maps = process_raw_data(f)
        y_shift = get_yshift(f)
        sens_maps = pad_sensitivity_maps(sens_maps, kspace.shape)

    img_cc_fs = compute_coil_combined_reconstructions(
        kspace, sens_maps, y_shift
    )
   
    return sens_maps, img_cc_fs, kspace


def visualize_reconstruction(img_cc_fs, slice_idx=0, echo_idx=0, output_prefix="recon"):
    """Visualize reconstructed images using legacy approach.
    
    Based on legacy code: converts magnitude and phase, normalizes properly,
    and saves with matplotlib for better visualization.
    
    Parameters
    ----------
    img_cc_fs : np.ndarray
        Complex reconstruction image (slices, echoes, kx, ky)
    slice_idx : int
        Which slice to visualize
    echo_idx : int
        Which echo to visualize
    output_prefix : str
        Prefix for output filenames
    """
    # Select single slice and echo
    img = img_cc_fs[slice_idx, echo_idx]  # (kx, ky)
    
    # Extract magnitude and phase (legacy approach)
    magnitude = np.abs(img)
    phase = np.angle(img)  # Range: -π to π
    
    # Normalize magnitude to [0, 1] for better visualization
    magnitude_norm = magnitude / np.max(magnitude)
    
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
        Complex reconstruction image (slices, echoes, kx, ky)
    slice_idx : int
        Which slice to visualize
    output_prefix : str
        Prefix for output filename
    """
    num_echoes = img_cc_fs.shape[1]
    
    # Create figure with echoes as columns
    fig, axes = plt.subplots(2, num_echoes, figsize=(2*num_echoes, 4), dpi=150)
    
    # Get all echoes for this slice
    img_slice = img_cc_fs[slice_idx]  # (echoes, kx, ky)
    
    # Calculate global min/max for consistent scaling (legacy approach)
    magnitude_all = np.abs(img_slice)
    max_mag = np.max(magnitude_all)
    
    for echo_idx in range(num_echoes):
        magnitude = np.abs(img_slice[echo_idx]) / max_mag
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


if __name__ == "__main__":
    filename = "/u/home/nguyv/Documents/data/helmholtz/val_recon/sub-07/t2s_gre_fr.hf"
    
    print("="*60)
    print("Legacy-style MRI Reconstruction Test")
    print("="*60)
    print(f"\nLoading data from: {filename}")
    
    sens_maps, img_cc_fs, kspace = load_raw_data(filename)
    
    print("\nData shapes:")
    print(f"  K-space:      {kspace.shape}")
    print(f"  Sens maps:    {sens_maps.shape}")
    print(f"  Recon image:  {img_cc_fs.shape}")
    
    print("\nData ranges:")
    print(f"  K-space magnitude: {np.abs(kspace).min():.2e} to {np.abs(kspace).max():.2e}")
    print(f"  Image magnitude:   {np.abs(img_cc_fs).min():.2e} to {np.abs(img_cc_fs).max():.2e}")
    
    # Check signal per slice to find non-empty ones
    print("\n" + "="*60)
    print("Checking signal per slice...")
    print("="*60)
    
    slice_magnitudes = []
    for slice_idx in range(img_cc_fs.shape[0]):
        slice_mag = np.abs(img_cc_fs[slice_idx]).mean()
        slice_magnitudes.append(slice_mag)
        if slice_idx < 5 or slice_mag > 100:  # Show first few and any with good signal
            print(f"  Slice {slice_idx:2d}: mean magnitude = {slice_mag:.2f}")
    
    # Find slice with maximum signal
    best_slice = np.argmax(slice_magnitudes)
    print(f"\n  → Best slice: {best_slice} (mean magnitude = {slice_magnitudes[best_slice]:.2f})")
    
    # Check signal per echo
    print("\nChecking signal per echo (for best slice)...")
    for echo_idx in range(img_cc_fs.shape[1]):
        echo_mag = np.abs(img_cc_fs[best_slice, echo_idx]).mean()
        print(f"  Echo {echo_idx:2d}: mean magnitude = {echo_mag:.2f}")
    
    # Visualize reconstructions (legacy approach)
    print("\n" + "="*60)
    print("Generating visualizations (legacy style)...")
    print("="*60)
    
    # Visualize best slice, first echo
    print(f"\n1. Best slice (slice {best_slice}), echo 0:")
    visualize_reconstruction(img_cc_fs, slice_idx=best_slice, echo_idx=0, output_prefix="recon_best")
    
    # Visualize all echoes for best slice
    print(f"\n2. Multi-echo visualization (slice {best_slice}):")
    visualize_all_echoes(img_cc_fs, slice_idx=best_slice, output_prefix="recon_all_echoes_best")
    
    print("\n" + "="*60)
    print("Done! Check the generated PNG files.")
    print("="*60)