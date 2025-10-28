# Brain MRI Reconstruction and Visualization Guide

This guide explains how to use the integrated legacy-style reconstruction and visualization pipeline.

## Overview

The reconstruction pipeline has been integrated into the main codebase with the following components:

### 1. Utility Functions (`utils/vis.py`)

Three new visualization functions have been added:

- **`find_best_slice(img_cc_fs)`** - Automatically finds the slice with maximum signal
- **`visualize_reconstruction(img_cc_fs, slice_idx, echo_idx, output_prefix)`** - Creates side-by-side magnitude/phase visualization
- **`visualize_all_echoes(img_cc_fs, slice_idx, output_prefix)`** - Creates multi-echo grid visualization

These complement the existing reconstruction functions:
- `load_brain_h5()` - Load k-space and sensitivity maps from HDF5
- `compute_coil_combined_reconstructions()` - Perform coil-combined reconstruction

### 2. BrainDataset Method (`datasets/brain.py`)

A new method has been added to the `BrainDataset` class:

```python
img_recon = dataset.reconstruct_images()
```

This reconstructs coil-combined images from the stored k-space data using the legacy approach.

### 3. Training Integration (`train.py`)

The training script now includes reconstruction visualization when using brain datasets. During training, it will generate:
- Single echo visualizations: `train_recon_epoch{N}_slice0_echo0.png`
- Multi-echo visualizations: `train_recon_all_echoes_epoch{N}_slice0.png`

### 4. Demo Script (`visualize_brain_reconstruction.py`)

A standalone demo script for quick reconstruction and visualization.

## Usage Examples

### Using the Demo Script

```bash
# Basic usage
python visualize_brain_reconstruction.py --data_file /path/to/data.h5

# With custom output prefix
python visualize_brain_reconstruction.py --data_file /path/to/data.h5 --output_prefix my_recon

# Specify slice and echo
python visualize_brain_reconstruction.py --data_file /path/to/data.h5 --slice_idx 17 --echo_idx 5
```

### Using in Your Code

```python
from utils.vis import (load_brain_h5, compute_coil_combined_reconstructions,
                       visualize_reconstruction, find_best_slice)

# Load data
kspace, sens_maps, y_shift = load_brain_h5('data.h5')

# Reconstruct
img_recon = compute_coil_combined_reconstructions(
    kspace, sens_maps, y_shift, remove_oversampling=True
)

# Find best slice and visualize
best_slice = find_best_slice(img_recon)
visualize_reconstruction(img_recon, slice_idx=best_slice, echo_idx=0)
```

### Using with BrainDataset

```python
from datasets.brain import BrainDataset
from utils.basic import parse_config

# Load dataset
config = parse_config('configs/config_brain.yml')
dataset = BrainDataset(config)

# Reconstruct images
img_recon = dataset.reconstruct_images()  # Returns (echoes, kx, ky)

# Visualize
from utils.vis import visualize_all_echoes
visualize_all_echoes(img_recon[None], slice_idx=0)  # Add batch dim
```

## Output Files

The visualization functions create PNG files with the following naming:
- Single echo: `{prefix}_slice{N}_echo{M}.png`
- All echoes: `{prefix}_all_echoes_slice{N}.png`

Each visualization includes:
- **Magnitude**: Grayscale image normalized to [0, 1]
- **Phase**: Viridis colormap ranging from -π to π

## Legacy Compatibility

The reconstruction functions match the behavior of the legacy code in `test.py`:
- Uses torch-based `ifft2c_mri` for inverse FFT
- Applies y-shift correction in image space
- Removes readout oversampling (crops last dimension)
- Pads sensitivity maps and normalizes by RSS
- Transposes images for proper anatomical orientation

## Notes

- The `test.py` file remains unchanged as a standalone reference implementation
- All functions handle both single-slice and multi-slice data
- Images are automatically normalized for optimal visualization
- The best slice is auto-detected based on mean magnitude when not specified

