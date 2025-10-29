import os
from torch.utils.data import Dataset
import torch
import numpy as np
import h5py
from utils.vis import load_brain_h5, pad_sensitivity_maps
from utils.medutils_compat import rss


class BrainDataset(Dataset):
    """Dataset for loading and processing multi-echo brain MRI k-space data.
    
    This dataset handles Cartesian k-space brain MRI data with multiple echoes,
    converting 4D coordinates (echo, coil, kx, ky) to flattened format for training.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Construct path to subject data file (flexible)
        subject_id = config.get('subject', 'sub-07')
        subject_dir = f"{config['data_root']}/{subject_id}"
        file_name = config.get('file_name', None)
        if file_name is None:
            # Try to auto-detect .h5/.hf file
            candidates = []
            try:
                for fname in os.listdir(subject_dir):
                    if fname.endswith('.h5') or fname.endswith('.hf'):
                        candidates.append(fname)
            except FileNotFoundError:
                raise FileNotFoundError(f"Subject directory not found: {subject_dir}")
            if not candidates:
                raise FileNotFoundError(f"No .h5/.hf file found under {subject_dir}. Provide config['file_name'].")
            data_file = f"{subject_dir}/{candidates[0]}"
        else:
            data_file = f"{subject_dir}/{file_name}"
        
        # Load brain data with multi-echo acquisition
        self.brain_data = self.get_subject_data(data_file, config['slice'])
        
        # kspace_data dimensions: [num_echoes, num_coils, kx, ky]
        
        # Flatten k-space data for point-wise training: [total_kpoints, 1]
        self.kspace_data_flat = np.reshape(self.brain_data['kdata'].astype(np.complex64), (-1, 1))

        # Keep original k-space data shape for reference
        self.kspace_data_original = self.brain_data['kdata'].astype(np.complex64)
        
        # Extract k-space dimensions
        num_echoes, num_coils, kx_dim, ky_dim = self.brain_data['kdata'].shape
        self.total_kpoints = self.kspace_data_flat.shape[0]
        
        # Create 4D coordinate system for Cartesian k-space
        # Initialize 4D coordinate array: [echo, coil, kx, ky]
        kspace_coordinates = np.zeros((num_echoes, num_coils, kx_dim, ky_dim, 4))
        
        # Echo coordinate: normalize from [0, num_echoes-1] to [-1, 1]
        echo_coordinates = np.linspace(-1, 1, num_echoes)
        kspace_coordinates[:,:,:,:,0] = echo_coordinates.reshape(num_echoes, 1, 1, 1)
        
        # Coil coordinate: linear spacing from -1 to 1 for each coil
        coil_coordinates = np.linspace(-1, 1, num_coils)
        kspace_coordinates[:,:,:,:,1] = coil_coordinates.reshape(1, num_coils, 1, 1)
        
        # kx coordinate: Cartesian grid normalized to [-1, 1]
        kx_coordinates = np.linspace(-1, 1, kx_dim)
        kspace_coordinates[:,:,:,:,2] = kx_coordinates.reshape(1, 1, kx_dim, 1)
        
        # ky coordinate: Cartesian grid normalized to [-1, 1]
        ky_coordinates = np.linspace(-1, 1, ky_dim)
        kspace_coordinates[:,:,:,:,3] = ky_coordinates.reshape(1, 1, 1, ky_dim)

        # Flatten coordinates for point-wise training: [total_kpoints, 4]
        self.kspace_coordinates_flat = np.reshape(kspace_coordinates.astype(np.float32), (-1, 4))

        # Normalize k-space data magnitude for stable training
        # Use same normalization strategy as cardiac dataset
        self.kspace_data_flat = self.kspace_data_flat / (np.max(np.abs(self.kspace_data_flat)) + 1e-9)

        # Convert numpy arrays to PyTorch tensors for training
        # Note: device transfer handled in model, not here
        self.kspace_data_flat = torch.from_numpy(self.kspace_data_flat)      # shape: [total_kpoints, 1]
        self.kspace_coordinates_flat = torch.from_numpy(self.kspace_coordinates_flat)  # shape: [total_kpoints, 4]
        
    
    def get_subject_data(self, data_file, slice_idx):
        """Load and process subject-specific brain MRI data from HDF5 file.
        
        Args:
            data_file: path to HDF5 file containing brain data
            slice_idx: which slice to select from the volume
            
        Returns:
            dict: processed brain data including k-space and coil sensitivity maps
        """
        # Load raw data using flexible loader
        kspace_raw, sens_maps_raw, y_shift = load_brain_h5(data_file)
        
        # kspace_raw shape: (num_slices, num_echoes, num_coils, kx, ky)
        # sens_maps_raw shape: (num_slices, 1, num_coils, kx, ky_half) - half due to readout oversampling
        
        # Support files with or without slice dimension
        if kspace_raw.ndim == 5:
            num_slices, num_echoes, num_coils, kx_dim, ky_dim = kspace_raw.shape
        elif kspace_raw.ndim == 4:
            num_slices = 1
            num_echoes, num_coils, kx_dim, ky_dim = kspace_raw.shape
            kspace_raw = kspace_raw[None]
            # sens_maps may be [slices, 1, coils, kx, ky_half] or [1, coils, kx, ky_half]
            if sens_maps_raw.ndim == 4:
                sens_maps_raw = sens_maps_raw[None]
        else:
            raise ValueError(f"Unexpected kspace shape: {kspace_raw.shape}")
        
        # Select coils if requested
        coil_select = self.config.get('coil_select', [])

        # Select single slice
        kspace_slice = kspace_raw[slice_idx]  # (E, C, Kx, Ky)
        if coil_select:
            kspace_slice = kspace_slice[:, coil_select]
        # sens_maps_raw expected shape: (S, 1, C, Kx, Ky_half) or (S, C, Kx, Ky_half)
        if sens_maps_raw.ndim == 5:
            sens_maps_slice = sens_maps_raw[slice_idx, 0]
        elif sens_maps_raw.ndim == 4:
            sens_maps_slice = sens_maps_raw[slice_idx]
        else:
            raise ValueError(f"Unexpected sens_maps shape: {sens_maps_raw.shape}")

        if coil_select:
            sens_maps_slice = sens_maps_slice[coil_select]
        
        # Pad sensitivity maps to match k-space dimensions
        sens_maps_padded = pad_sensitivity_maps(
            sens_maps_slice[np.newaxis, np.newaxis, :, :, :],
            kspace_slice[0:1].shape
        )[0, 0]
        
        # Normalize coil sensitivity maps to avoid division by zero
        # Compute root-sum-of-squares (RSS) across coils for normalization
        coil_sensitivity_maps_rss = rss(sens_maps_padded, coil_axis=0)
        coil_sensitivity_maps = np.nan_to_num(sens_maps_padded / coil_sensitivity_maps_rss)
        self.csm = coil_sensitivity_maps  # Store for later use in reconstruction
        self.y_shift = y_shift  # Store y-shift for reconstruction
        
        # Prepare processed brain data dictionary
        processed_brain_data = {
            'subject': data_file.split('/')[-2],
            'slice': slice_idx,
            'kdata': kspace_slice,
            'csm': coil_sensitivity_maps,
            'y_shift': y_shift,
        }
        
        return processed_brain_data
    
    def __len__(self):
        """Return total number of k-space points for training."""
        return self.total_kpoints

    def __getitem__(self, index):
        """Get a single k-space point for training.
        
        Args:
            index: index of the k-space point to retrieve
            
        Returns:
            dict: sample containing coordinates and target k-space value
        """
        # Point-wise sampling for neural implicit k-space training
        # TODO: set index = 0 for now to see if the dataset is working
        index = 0
        sample = {
            'coords': self.kspace_coordinates_flat[index],  # 4D coordinates [echo, coil, kx, ky]
            'targets': self.kspace_data_flat[index]         # Target k-space value (complex)
        }
        return sample
    
    def reconstruct_images(self, k_space):
        """Reconstruct coil-combined images from k-space data.
        
        Uses the stored k-space data, coil sensitivity maps, and y-shift to
        perform coil-combined reconstruction following the legacy approach.
        
        Returns:
            np.ndarray: Reconstructed images (echoes, kx, ky) - complex valued
        """
        from utils.vis import compute_coil_combined_reconstructions
        
        # Use stored k-space and CSM (already unnormalized original data)
        kspace = k_space  # (echoes, coils, kx, ky)
        sens_maps = self.csm  # (coils, kx, ky)
        y_shift = self.y_shift
        
        # Add batch dims for reconstruction function
        # Function expects (slices, echoes, coils, kx, ky) and (slices, 1, coils, kx, ky)
        kspace_batch = kspace[None]  # (1, echoes, coils, kx, ky)
        sens_maps_batch = sens_maps[None, None]  # (1, 1, coils, kx, ky)
        
        # Reconstruct
        img_recon = compute_coil_combined_reconstructions(
            kspace_batch, sens_maps_batch, y_shift, remove_oversampling=True
        )
        
        # Remove batch dim -> (echoes, kx, ky)
        return img_recon[0]

