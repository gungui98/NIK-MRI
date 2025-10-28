# BrainDataset Documentation

## Overview

The `BrainDataset` class loads and processes multi-echo brain MRI data from the T2*-MOVE dataset. It handles Cartesian k-space data and converts it to a flattened format for neural implicit k-space (NIK) training.

## Data Structure

### Input Data Format
- **File format**: HDF5 (`.hf` files)
- **K-space shape**: `(36 slices, 12 echoes, 31 coils, 92, 224)`
- **Sensitivity maps shape**: `(36 slices, 1, 31 coils, 92, 112)`
- **K-space type**: Cartesian (uniform grid)

### Coordinate System
The dataset creates 4D coordinates normalized to `[-1, 1]`:
1. **Echo dimension** (temporal): 12 echoes linearly spaced from -1 to 1
2. **Coil dimension**: 31 coils linearly spaced from -1 to 1
3. **kx dimension** (spatial): 92 points linearly spaced from -1 to 1
4. **ky dimension** (spatial): 224 points linearly spaced from -1 to 1

## Usage

### Basic Example

```python
from datasets.brain import BrainDataset

config = {
    'data_root': '/path/to/helmholtz/val_recon',
    'subject': 'sub-07',  # Options: sub-07, sub-08, sub-09
    'slice': 15,  # 0-35
}

dataset = BrainDataset(config)
print(f"Dataset size: {len(dataset)} k-space points")

# Get a sample
sample = dataset[0]
coords = sample['coords']  # Shape: [4] - [echo, coil, kx, ky]
target = sample['targets']  # Shape: [1] - complex k-space value
```

### Training with BrainDataset

To train a model with the brain dataset, modify `train.py` line 42:

```python
# Replace this line:
dataset = RadialDataset(config)

# With:
dataset = BrainDataset(config)
```

Then run:
```bash
python train.py --config configs/config_brain.yml
```

## Configuration Parameters

### Required Parameters
- `data_root`: Path to the data directory containing subject folders
- `subject`: Subject ID (e.g., 'sub-07', 'sub-08', 'sub-09')
- `slice`: Slice index to process (0-35)

### Recommended Training Parameters
- `batch_size`: 30000 (or adjust based on GPU memory)
- `num_workers`: 8
- `coord_dim`: 4 (echo, coil, kx, ky)
- `nt`: 12 (number of echoes for reconstruction)
- `nx`: 92 (k-space x dimension)
- `ny`: 224 (k-space y dimension with oversampling)

## Key Differences from RadialDataset (Cardiac)

| Aspect | RadialDataset (Cardiac) | BrainDataset |
|--------|------------------------|--------------|
| K-space type | Radial (non-uniform) | Cartesian (uniform) |
| Temporal dimension | Cardiac phases (ECG-gated) | Multi-echo (T2* weighting) |
| Coordinate system | [time, coil, kx, ky] radial | [echo, coil, kx, ky] Cartesian |
| ECG gating | Yes (complex alignment) | No (simple echo ordering) |
| Trajectory | Non-uniform radial spokes | Uniform Cartesian grid |

## Dataset Attributes

After initialization, the `BrainDataset` object provides:

- `kspace_data_flat`: Flattened k-space data (PyTorch tensor)
- `kspace_data_original`: Original 4D k-space data
- `kspace_coordinates_flat`: Flattened 4D coordinates (PyTorch tensor)
- `csm`: Coil sensitivity maps for reconstruction
- `y_shift`: Y-shift value for proper image reconstruction
- `total_kpoints`: Total number of k-space points

## Testing

Run the test script to verify the dataset works correctly:

```bash
python test_brain_dataset.py
```

This will display:
- Dataset dimensions and statistics
- Sample coordinate and target information
- Data normalization ranges
- Metadata about coil sensitivity maps

## Notes

1. **Normalization**: K-space data is normalized by dividing by the maximum absolute value
2. **Memory**: A full dataset with all k-space points (~7.6M points) requires significant GPU memory. Use batch sampling or DataLoader with appropriate batch size.
3. **Coil Sensitivity Maps**: Automatically padded and normalized to match k-space dimensions
4. **Y-shift**: Stored for proper reconstruction alignment (compensates for readout trajectory)

## Example Output

```
Dataset size: 7,666,176 k-space points
Original k-space shape: (12, 31, 92, 224)
  - Echoes: 12
  - Coils: 31  
  - K-space dimensions: 92 x 224
```

