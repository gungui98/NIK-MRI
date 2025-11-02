from typing import Any


import os
from torch.utils.data import Dataset
import torch
import numpy as np
import h5py
from utils.vis import load_brain_h5, pad_sensitivity_maps, compute_coil_combined_reconstructions
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
        csm, y_shift, kspace = self.load_raw_data(data_file)
        
        if config['is_selected']:
            self.selected_slices = config['selected_slices']
            self.selected_echoes = config['selected_echoes']
            self.selected_coils = config['selected_coils']
            
            # k_space_idx = np.ix_(self.selected_slices, self.selected_echoes, self.selected_coils)
            # csm_idx = np.ix_(self.selected_slices, [0], self.selected_coils) # we only need one sensitivity map for reconstruction
            # select all slices, echoes, and coils
            k_space_idx = np.ix_(self.selected_slices, self.selected_echoes, self.selected_coils)
            csm_idx = np.ix_(self.selected_slices, [0], self.selected_coils) # we only need one sensitivity map for reconstruction
            kspace = kspace[k_space_idx]
            csm = csm[csm_idx]
        
        self.csm = csm
        self.y_shift = y_shift
        self.kspace = kspace

        # Flatten k-space data for point-wise training: [total_kpoints, 1]
        self.kspace_data_flat = np.reshape(kspace.astype(np.complex64), (-1, 1))

        # Keep original k-space data shape for reference
        self.kspace_data_original = kspace.astype(np.complex64)
        
        # Extract k-space dimensions
        num_slices, num_echoes, num_coils, kx_dim, ky_dim = kspace.shape
        self.total_kpoints = self.kspace_data_flat.shape[0]
        
        # Create 4D coordinate system [echo, coil, kx, ky] broadcast over slices
        # Normalized to [-1, 1] per dimension
        echo_lin = np.linspace(-1, 1, num_echoes, dtype=np.float32).reshape(1, num_echoes, 1, 1, 1)
        coil_lin = np.linspace(-1, 1, num_coils, dtype=np.float32).reshape(1, 1, num_coils, 1, 1)
        kx_lin = np.linspace(-1, 1, kx_dim, dtype=np.float32).reshape(1, 1, 1, kx_dim, 1)
        ky_lin = np.linspace(-1, 1, ky_dim, dtype=np.float32).reshape(1, 1, 1, 1, ky_dim)

        target_shape = (num_slices, num_echoes, num_coils, kx_dim, ky_dim)
        echo_grid = np.broadcast_to(echo_lin, target_shape)
        coil_grid = np.broadcast_to(coil_lin, target_shape)
        kx_grid = np.broadcast_to(kx_lin, target_shape)
        ky_grid = np.broadcast_to(ky_lin, target_shape)

        # Stack into last dim → (..., 4)
        coords = np.stack([echo_grid, coil_grid, kx_grid, ky_grid], axis=-1)

        # Flatten coordinates for point-wise training: [total_kpoints, 4]
        self.kspace_coordinates_flat = coords.reshape(-1, 4).astype(np.float32)

        # Normalize k-space data magnitude for stable training
        # Use same normalization strategy as cardiac dataset
        self.kspace_data_flat = self.kspace_data_flat / (np.max(np.abs(self.kspace_data_flat)) + 1e-9)

        # Convert numpy arrays to PyTorch tensors for training
        # Note: device transfer handled in model, not here
        self.kspace_data_flat = torch.from_numpy(self.kspace_data_flat)      # shape: [total_kpoints, 1]
        self.kspace_coordinates_flat = torch.from_numpy(self.kspace_coordinates_flat)  # shape: [total_kpoints, 4]
        
    def load_raw_data(self, filename):
        """Load raw data from h5 file."""
        with h5py.File(filename, "r") as f:
            kspace, sens_maps = self.process_raw_data(f)
            y_shift = self.get_yshift(f)
            sens_maps = pad_sensitivity_maps(sens_maps, kspace.shape)
    
        return sens_maps, y_shift, kspace
    
    def process_raw_data(self, hf_file):
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

    
    def get_yshift(self, hf_file):
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
        sample = {
            'coords': self.kspace_coordinates_flat[index],  # 4D coordinates [echo, coil, kx, ky]
            'targets': self.kspace_data_flat[index]         # Target k-space value (complex)
        }
        return sample
    
    def reconstruct_images(self, k_space):
        """Reconstruct coil-combined images for one or multiple slices.

        Parameters
        ----------
        k_space : np.ndarray or torch.Tensor
            K-space with shape (slices, echoes, coils, kx, ky)

        Returns
        -------
        np.ndarray
            Complex images with shape (slices, echoes, kx, ky)
        """

        # Use stored sensitivity maps and y-shift
        kspace = k_space  # (slices, echoes, coils, kx, ky)
        sens_maps = self.csm  # (slices, 1, coils, kx, ky_half or ky), may need padding
        y_shift = self.y_shift

        # Reconstruct; utility handles padding and types
        img_recon = compute_coil_combined_reconstructions(
            kspace, sens_maps, y_shift, remove_oversampling=True
        )

        # (slices, echoes, kx, ky)
        return img_recon

