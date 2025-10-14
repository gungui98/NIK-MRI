import os
from torch.utils.data import Dataset
import torch
import numpy as np
import h5py
from utils.medutils_compat import rss


class RadialDataset(Dataset):
    """Dataset for loading and processing radial cardiac MRI k-space data.
    
    This dataset handles radial trajectory cardiac MRI data with ECG-gated binning,
    converting 4D coordinates (time, coil, kx, ky) to flattened format for training.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Construct path to patient-specific data folder
        patient_data_folder = f"{config['data_root']}/{config['slice_name']}"
        
        # Load and process patient data with cardiac cycle alignment
        self.patient_data = self.get_subject_data(patient_data_folder, config['num_cardiac_cycles'])
        
        # kspace_data dimensions: [num_coils, num_spokes, num_frequency_encodes]
        # kspace_trajectory dimensions: [2, num_spokes, num_frequency_encodes] (kx, ky)
        
        # Flatten k-space data for point-wise training: [total_kpoints, 1]
        self.kspace_data_flat = np.reshape(self.patient_data['kdata'].astype(np.complex64), (-1, 1))

        # Keep original k-space data shape for reference
        self.kspace_data_original = self.patient_data['kdata'].astype(np.complex64)
        
        # Extract k-space dimensions
        num_coils, num_spokes, num_frequency_encodes = self.patient_data['kdata'].shape
        self.total_kpoints = self.kspace_data_flat.shape[0]
        
        # Create 4D coordinate system from trajectory and timing data
        # kspace_trajectory: normalized k-space positions [-1, 1] for kx, ky
        kspace_trajectory = self.patient_data['ktraj']    # shape: [2, num_spokes, num_frequency_encodes]
        
        # temporal_spokes: relative time stamps for each spoke within cardiac cycle [0, 1]
        temporal_spokes = self.patient_data['tspokes']    # shape: [num_spokes]

        # Initialize 4D coordinate array: [time, coil, kx, ky]
        kspace_coordinates = np.zeros((num_coils, num_spokes, num_frequency_encodes, 4))
        
        # Time coordinate: normalize temporal_spokes from [0,1] to [-1,1]
        kspace_coordinates[:,:,:,0] = np.reshape(temporal_spokes * 2 - 1, (1, num_spokes, 1))
        
        # Coil coordinate: linear spacing from -1 to 1 for each coil
        coil_coordinates = torch.linspace(-1, 1, num_coils)
        kspace_coordinates[:,:,:,1] = np.reshape(coil_coordinates, [num_coils, 1, 1])
        
        # kx coordinate: k-space trajectory x-component
        kspace_coordinates[:,:,:,2] = kspace_trajectory[0, :][None]  # shape: [1, num_spokes, num_frequency_encodes]
        
        # ky coordinate: k-space trajectory y-component  
        kspace_coordinates[:,:,:,3] = kspace_trajectory[1, :][None]  # shape: [1, num_spokes, num_frequency_encodes]

        # Flatten coordinates for point-wise training: [total_kpoints, 4]
        self.kspace_coordinates_flat = np.reshape(kspace_coordinates.astype(np.float32), (-1, 4))

        # ############################
        # # eps = np.median(np.abs(self.kdata_flat))
        # eps = np.mean(np.abs(self.kdata_flat)) * 2
        # # # self.kdata_flat = eps / (self.kdata_flat + eps)
        # self.kdata_flat = eps / (np.abs(self.kdata_flat) + eps) * (self.kdata_flat / np.abs(self.kdata_flat))
        # self.eps = eps
        # ############################

        # Convert numpy arrays to PyTorch tensors for training
        # Note: device transfer handled in model, not here
        self.kspace_data_flat = torch.from_numpy(self.kspace_data_flat)      # shape: [total_kpoints, 1]
        self.kspace_coordinates_flat = torch.from_numpy(self.kspace_coordinates_flat)  # shape: [total_kpoints, 4]
        

    def align_to_one_cycle(self, ecg_rwave_timings, acquisition_start_time, repetition_time, total_spokes, num_cardiac_cycles):
        r"""Retrospective ECG-gated binning of cardiac data to align spokes to cardiac cycles.
        
        This method aligns radial spokes to specific cardiac phases by binning them
        according to ECG R-wave detections and creating a normalized cardiac cycle.
        
        Args:
            ecg_rwave_timings: array with timings of ECG R-wave detections (ms)
            acquisition_start_time: start time of MRI acquisition (same time axis as ECG)
            repetition_time: time between successive RF pulses (TR in ms)
            total_spokes: total number of spokes acquired
            num_cardiac_cycles: number of cardiac cycles to use for alignment

        Returns:
            aligned_cardiac_data(dict):
                'target_RR': target R-R interval duration (ms)
                'num_cycles': number of cardiac cycles used
                'tspokes': relative time stamps of spokes within normalized cardiac cycle [0,1]
                'idxspokes': original indices of spokes sorted by cardiac phase
        """

        # Remove initial ECG R-waves that occur during transient magnetization period
        # This avoids including data during initial signal stabilization
        while ecg_rwave_timings[0] < self.config['transient_magnetization']:
            ecg_rwave_timings = ecg_rwave_timings[1:]

        # Calculate actual acquisition time stamps for each spoke
        spoke_acquisition_times = acquisition_start_time + np.arange(total_spokes) * repetition_time
        spoke_indices = np.arange(total_spokes)

        # Calculate average R-R interval from ECG data
        target_rr_interval = np.mean(np.diff(ecg_rwave_timings))

        # Bin spokes according to cardiac cycles
        aligned_spoke_indices = []
        aligned_spoke_times = []
        
        # Loop through each cardiac cycle to find spokes within each heartbeat
        for ecg_cycle_idx in range(min(ecg_rwave_timings.size - 1, num_cardiac_cycles)):
            cycle_start_time = ecg_rwave_timings[ecg_cycle_idx]
            cycle_end_time = ecg_rwave_timings[ecg_cycle_idx + 1]
            current_rr_duration = cycle_end_time - cycle_start_time

            # Find spokes acquired during this cardiac cycle
            spokes_in_current_cycle = np.logical_and(
                spoke_acquisition_times >= cycle_start_time, 
                spoke_acquisition_times < cycle_end_time
            )
            
            # Extract spoke indices and times for this cycle
            current_cycle_spoke_indices = spoke_indices[spokes_in_current_cycle]
            current_cycle_spoke_times = spoke_acquisition_times[spokes_in_current_cycle]
            
            # Normalize times to relative cardiac phase [0,1] within this cycle
            relative_cardiac_phase = (current_cycle_spoke_times - cycle_start_time) / current_rr_duration
            
            # Store aligned data
            aligned_spoke_indices.extend(current_cycle_spoke_indices)
            aligned_spoke_times.extend(relative_cardiac_phase)
        
        # Sort spokes by their relative cardiac phase across all cycles
        (aligned_spoke_times, aligned_spoke_indices) = list(zip(*sorted(zip(aligned_spoke_times, aligned_spoke_indices))))
        
        aligned_cardiac_data = {
            'target_RR': target_rr_interval,
            'num_cycles': num_cardiac_cycles,
            'tspokes': aligned_spoke_times,    # Relative cardiac phase [0,1] for each spoke
            'idx_spokes': aligned_spoke_indices  # Original indices of spokes sorted by cardiac phase
        }
        return aligned_cardiac_data
    
    def get_subject_data(self, patient_data_folder, num_cardiac_cycles):
        """Load and process patient-specific cardiac MRI data from HDF5 file.
        
        Args:
            patient_data_folder: path to folder containing patient data
            num_cardiac_cycles: number of cardiac cycles to use for ECG gating
            
        Returns:
            dict: processed patient data including k-space, trajectories, and coil sensitivity maps
        """
        # Extract patient case name from folder path
        patient_case_name = patient_data_folder.split('/')[-1]
        
        # Load patient data from HDF5 file
        patient_h5_file = h5py.File(f'{patient_data_folder}/CINE_testrun.h5', 'r', libver='latest', swmr=True)

        # Load coil sensitivity maps (CSM) for multi-coil reconstruction
        coil_sensitivity_maps = patient_h5_file['csm_real'][self.config['coil_select']] + \
            1j*patient_h5_file['csm_imag'][self.config['coil_select']]

        # Load continuously acquired k-space data from all coils
        full_kspace_data = patient_h5_file['fullkdata_real'][self.config['coil_select']] + \
            1j * patient_h5_file['fullkdata_imag'][self.config['coil_select']]
            
        # Load k-space trajectory (radial spoke positions)
        full_kspace_trajectory = patient_h5_file['fullkpos'][()]

        # frequency padding for better boundary condition
        # if padding:
        #     nFE = full_kdata.shape[-1]
        #     n_fpad = int(0.15 * nFE)
        #     full_kdata = np.pad(full_kdata, ((0, 0), (0, 0), (n_fpad, n_fpad)), 'constant')
        #     # full_kdata = np.pad(full_kdata, ((0, 0), (0, 0), (n_fpad, n_fpad)), 'reflect')
        #     full_kpos_pad = np.pad(full_kpos, ((0, 0), (0, 0), (n_fpad, n_fpad)), 'constant')
        #     full_kpos_pad[:,:,0:n_fpad] = full_kpos[:,:,0:1] + full_kpos[:,:,nFE//2-n_fpad:nFE//2]
        #     full_kpos_pad[:,:,-n_fpad:] = full_kpos[:,:,nFE//2:nFE//2+n_fpad] + full_kpos[:,:,-1:]
        #     full_kpos = full_kpos_pad

        # Normalize coil sensitivity maps to avoid division by zero
        # Compute root-sum-of-squares (RSS) across coils for normalization
        coil_sensitivity_maps_rss = rss(coil_sensitivity_maps, coil_axis=0)
        coil_sensitivity_maps = np.nan_to_num(coil_sensitivity_maps/coil_sensitivity_maps_rss)
        self.csm = coil_sensitivity_maps  # Store for later use in reconstruction

        # Extract k-space data dimensions
        num_coils, total_spokes_acquired, num_frequency_encodes = full_kspace_data.shape

        # Extract ECG and acquisition timing information for cardiac gating
        ecg_rwave_timings = patient_h5_file['ECG'][0]  # ECG R-wave detection times
        acquisition_start_time = patient_h5_file['read_marker'][0][0]  # Start time of acquisition

        # Calculate repetition time (TR) from read markers
        # TR = (total_acquisition_time + single_read_duration) / total_spokes
        repetition_time = (patient_h5_file['read_marker'][0][-1] - patient_h5_file['read_marker'][0][0] 
                + patient_h5_file['read_marker'][0][1] - patient_h5_file['read_marker'][0][0]) / total_spokes_acquired
        patient_h5_file.close()
        
        # Align spokes to normalized cardiac cycles using ECG gating
        aligned_cardiac_data = self.align_to_one_cycle(
            ecg_rwave_timings, acquisition_start_time, repetition_time, 
            total_spokes_acquired, num_cardiac_cycles
        )
        
        # Extract aligned cardiac timing and spoke information
        aligned_spoke_times = aligned_cardiac_data['tspokes']  # Relative cardiac phase [0,1]
        aligned_spoke_indices = aligned_cardiac_data['idx_spokes']  # Spoke indices sorted by cardiac phase
        target_rr_interval = aligned_cardiac_data['target_RR']  # Average R-R interval

        # # padding for temporal dimension
        # if padding:
        #     n_tpad = int(0.15 * len(tshots))
        #     idxshots = list(reversed(idxshots[:n_tpad])) + idxshots + list(reversed(idxshots[-n_tpad:]))
        #     tshots = list(list(reversed(tshots[0] - tshots[:n_tpad])) + tshots[0]) + tshots + list(list(reversed(tshots[-1] - tshots[-n_tpad:])) + tshots[-1])


        # Select spokes according to cardiac-gated alignment
        gated_kspace_data = full_kspace_data[:, aligned_spoke_indices, :]  # [num_coils, num_gated_spokes, num_frequency_encodes]

        # Extract and normalize k-space trajectory to [-1, 1] range
        gated_kspace_trajectory = full_kspace_trajectory[:, aligned_spoke_indices, :] * 2  # Scale to [-1, 1]
        gated_kspace_trajectory = np.roll(gated_kspace_trajectory, shift=1, axis=0)  # Swap kx, ky axes

        # Normalize k-space data magnitude for stable training
        # TODO: Implement more sophisticated normalization strategy
        gated_kspace_data = gated_kspace_data / np.max(np.abs(gated_kspace_data))

        # Prepare processed patient data dictionary
        processed_patient_data = {
            'caseid': patient_case_name,                                    # Patient identifier
            'kdata': gated_kspace_data,                                    # ECG-gated k-space data [num_coils, num_spokes, num_frequency_encodes]
            'ktraj': gated_kspace_trajectory,                             # K-space trajectory [2, num_spokes, num_frequency_encodes]
            'tspokes': np.array(aligned_spoke_times),                     # Relative cardiac phase [0,1] for each spoke
            'RR': target_rr_interval,                                     # Average R-R interval duration (ms)
            'csm': coil_sensitivity_maps,                                 # Normalized coil sensitivity maps
        }
        
        return processed_patient_data
    
    # TODO: add method for traditional binning 
    # def get_binned_data(self, file_folder, num_cycles):
    #    pass

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
            'coords': self.kspace_coordinates_flat[index],  # 4D coordinates [time, coil, kx, ky]
            'targets': self.kspace_data_flat[index]         # Target k-space value (complex)
        }
        return sample
