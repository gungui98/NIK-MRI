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

print(img_coil_combined.shape)