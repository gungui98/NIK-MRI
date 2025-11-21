import os
from torch.utils.data import Dataset
import torch
import numpy as np
import h5py
from utils.vis import pad_sensitivity_maps, compute_coil_combined_reconstructions
from utils.medutils_compat import rss


class BrainDataset(Dataset):
    """Dataset for loading and processing multi-echo brain MRI k-space data.
    
    This dataset handles Cartesian k-space brain MRI data with multiple echoes,
    converting 4D coordinates (slice, kx, ky, echo) to flattened format for training.
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
            # Coils will be dropped after SENSE combine; ignore any provided selection and warn
            if 'selected_coils' in config and config['selected_coils'] is not None:
                print('Warning: selected_coils provided but will be ignored due to coil combination.')

            # Select slices and echoes; keep all coils for combination
            k_space_idx = np.ix_(self.selected_slices, self.selected_echoes, np.arange(kspace.shape[2]))
            csm_idx = np.ix_(self.selected_slices, [0], np.arange(csm.shape[2]))  # maps typically 1 echo
            kspace = kspace[k_space_idx]
            csm = csm[csm_idx]

            # Renormalize selected coils: RSS across coils should be 1 to avoid spatial bias
            csm_rss_selected = rss(csm, 2)
            csm = np.nan_to_num(csm / csm_rss_selected[:, :, None, :, :])

        # Perform SENSE coil combination to drop coil dimension
        # 1) Reconstruct images per (slice, echo) using sensitivity maps
        # 2) FFT back to k-space to get single-virtual-coil k-space
        # compute_coil_combined_reconstructions returns (slices, echoes, kx, ky) complex image
        kspace_t = torch.from_numpy(kspace)
        csm_t = torch.from_numpy(csm)
        img_combined = compute_coil_combined_reconstructions(kspace_t, csm_t, y_shift, remove_oversampling=True)

        # Use centered numpy FFT to avoid linear phase ramps
        img_np = img_combined.detach().cpu().numpy()
        kspace_combined = np.fft.fftshift(
            np.fft.fft2(
                np.fft.ifftshift(img_np, axes=(-2, -1)),
                axes=(-2, -1),
                norm='ortho',
            ),
            axes=(-2, -1),
        ).astype(np.complex64)
        
        # Store combined representations; csm is no longer needed for dataset usage
        self.csm = None
        self.y_shift = y_shift
        self.kspace = kspace_combined

        # Flatten k-space data for point-wise training: [total_kpoints, 1]
        self.kspace_data_flat = np.reshape(self.kspace.astype(np.complex64), (-1,))

        # Keep original k-space data shape for reference
        self.kspace_data_original = self.kspace.astype(np.complex64)
        
        # Extract k-space dimensions
        num_slices, num_echoes, kx_dim, ky_dim = self.kspace.shape
        self.num_slices = num_slices
        self.num_echoes = num_echoes
        self.num_coils = 1
        self.kx_dim = kx_dim
        self.ky_dim = ky_dim
        self.total_kpoints = self.kspace_data_flat.shape[0]
        
        # Create 4D coordinate system [slice, kx, ky, echo]
        # Normalized to [-1, 1] per dimension
        slice_lin = np.linspace(-1, 1, num_slices, dtype=np.float32).reshape((num_slices, 1, 1, 1))
        kx_lin = np.linspace(-1, 1, kx_dim, dtype=np.float32).reshape((1, kx_dim, 1, 1))
        ky_lin = np.linspace(-1, 1, ky_dim, dtype=np.float32).reshape((1, 1, ky_dim, 1))
        echo_lin = np.linspace(-1, 1, num_echoes, dtype=np.float32).reshape((1, 1, 1, num_echoes))

        target_shape = (num_slices, kx_dim, ky_dim, num_echoes)
        slice_grid = np.broadcast_to(slice_lin, target_shape)
        kx_grid = np.broadcast_to(kx_lin, target_shape)
        ky_grid = np.broadcast_to(ky_lin, target_shape)
        echo_grid = np.broadcast_to(echo_lin, target_shape)

        # Stack into last dim → (..., 4)
        coords = np.stack([slice_grid, kx_grid, ky_grid, echo_grid], axis=-1)

        # Flatten coordinates for point-wise training: [total_kpoints, 4]
        self.kspace_coordinates_flat = coords.reshape(-1, 4).astype(np.float32)

        # Complex standardization: ensure std(Re) = std(Im) = 1
        real = self.kspace_data_flat.real
        imag = self.kspace_data_flat.imag
        self.real_mean = float(real.mean())
        self.imag_mean = float(imag.mean())
        self.real_std = float(real.std() + 1e-9)
        self.imag_std = float(imag.std() + 1e-9)
        real_norm = (real - self.real_mean) / self.real_std
        imag_norm = (imag - self.imag_mean) / self.imag_std
        normalized_complex = real_norm + 1j * imag_norm
        self.kspace_data_flat = normalized_complex.astype(np.complex64)
        # Legacy attribute retained for compatibility (set to None -> skip scalar rescaling)
        self.norm_factor = None

        # Convert numpy arrays to PyTorch tensors for training
        # Note: device transfer handled in model, not here
        self.kspace_data_flat = torch.from_numpy(self.kspace_data_flat).unsqueeze(-1)      # shape: [total_kpoints, 1]
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
            'coords': self.kspace_coordinates_flat[index],  # 4D coordinates [slice, kx, ky, echo]
            'targets': self.kspace_data_flat[index]         # Target k-space value (complex)
        }
        return sample

    def denormalize_kspace(self, k_space):
        """Convert normalized predictions back to the original complex scale."""
        if not hasattr(self, 'real_std'):
            return k_space
        if isinstance(k_space, torch.Tensor):
            real = k_space.real * self.real_std + self.real_mean
            imag = k_space.imag * self.imag_std + self.imag_mean
            return torch.complex(real, imag)
        else:
            real = np.real(k_space) * self.real_std + self.real_mean
            imag = np.imag(k_space) * self.imag_std + self.imag_mean
            return real + 1j * imag
    
    def reconstruct_images(self, k_space):
        """Reconstruct images from combined k-space (no coil dimension).

        Parameters
        ----------
        k_space : np.ndarray or torch.Tensor
            K-space with shape (slices, echoes, kx, ky)

        Returns
        -------
        np.ndarray
            Complex images with shape (slices, echoes, kx, ky)
        """

        if not isinstance(k_space, torch.Tensor):
            k = torch.from_numpy(k_space)
        else:
            k = k_space

        # Use centered numpy IFFT to avoid linear phase ramps
        k_np = k.detach().cpu().numpy() if isinstance(k, torch.Tensor) else k
        img = np.fft.ifft2(
            np.fft.fftshift(k_np, axes=(-2, -1)),
            axes=(-2, -1),
            norm='ortho',
        )
        img = np.fft.ifftshift(img, axes=(-2, -1))
        return img.astype(np.complex64)
    
    def get_grid_coordinates(self):
        """Get grid coordinates in the format expected by test_batch.
        
        The coordinate system matches the dataset's training coordinate system exactly.
            
        Returns
        -------
        torch.Tensor
            Grid coordinates with shape (num_slices, kx_dim, ky_dim, num_echoes, 4)
            in format [slice, kx, ky, echo]
        """
        # Create coordinate grids matching the dataset's coordinate system
        # Use same normalization as dataset: np.linspace(-1, 1, ...)
        slice_lin = torch.linspace(-1, 1, self.num_slices, dtype=torch.float32)
        kx_lin = torch.linspace(-1, 1, self.kx_dim, dtype=torch.float32)
        ky_lin = torch.linspace(-1, 1, self.ky_dim, dtype=torch.float32)
        echo_lin = torch.linspace(-1, 1, self.num_echoes, dtype=torch.float32)

        # Create meshgrid matching the coordinate order [slice, kx, ky, echo]
        grid = torch.stack(torch.meshgrid(slice_lin, kx_lin, ky_lin, echo_lin, indexing='ij'), dim=-1)
        # Shape: (num_slices, kx_dim, ky_dim, num_echoes, 4)
        return grid
