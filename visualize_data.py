import matplotlib.pyplot as plt
from utils.vis import compute_coil_combined_reconstructions, load_coil_combined_reconstruction,\
    load_reference_mask, load_segmentation, load_motion_data, angle2color, k2img, get_yshift,\
    compute_coil_combined_reconstructions
import h5py

def load_raw_data(file_path):
    """Load raw data from the T2*-MOVE dataset."""

    with h5py.File(file_path, "r") as hf_file:
        raw_data = hf_file['kspace'][()]
        sens_maps = hf_file['sens_maps'][()]
        y_shift = get_yshift(hf_file)

    return raw_data, sens_maps, y_shift


# define the path to the T2*-MOVE dataset:
data_path = "/Users/phinguyen/Documents/qMRI/data/helmholtz/val_recon"


"""Demo code for loading the T2*-MOVE dataset"""
"""1) Loading the MRI data:"""
# define the subset and subject to load:
subject = "sub-07"

# 1a) Load the (motion-free, full-resolution) raw data and perform coil-combined reconstruction:
kspace, sens_maps, y_shift = load_raw_data(
    f"{data_path}/{subject}/t2s_gre_fr.hf"
)

img_coil_combined = compute_coil_combined_reconstructions(
    kspace, sens_maps, y_shift, remove_oversampling=True
)

plt.imshow(abs(img_coil_combined[15, 0].T), cmap="gray")
plt.title(f"{subject} - Coil Combined Reconstruction")
plt.axis("off")
plt.show()


# 1b) Load the coil-combined reconstruction (equivalent to the above):
img_coil_combined, nifti_header = load_coil_combined_reconstruction(
    f"{data_path}/{subject}/t2s_gre_fr_recon.hf"
)
# the nifti_header can be used to save the data as nifti for further processing


# 1c) Load the reference exclusion mask for the motion-corrupted acquisition (if available):
ref_mask = load_reference_mask(
    f"{data_path}/{subject}/motion_mask_t2s_gre_fr_move.txt"
)
# shape [36, 92] for 36 slices and 92 PE lines;
# 0: subject was instructed to move during acquisition of this PE line
# 1: subject was instructed to stay still during acquisition of this PE line


# 1d) Load the segmentations:
brain_mask = load_segmentation(
    f"{data_path}/{subject}/seg_brain_reg-to-t2s.nii"
)
white_matter = load_segmentation(
    f"{data_path}/{subject}/seg_wm_reg-to-t2s.nii"
)
gray_matter = load_segmentation(
    f"{data_path}/{subject}/seg_gm_reg-to-t2s.nii"
)


"""2) Loading the motion data:"""
motion_data = load_motion_data(
    f"{data_path}/{subject}/motion_data.json"
)
# dicitionary with keys: 't_x', 't_y', 't_z', 'r_x', 'r_y', 'r_z', 'time'
# (translational and rotational parameters in mm and °) and time points in seconds
