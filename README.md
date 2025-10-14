# NIK_MRI - Neural Implicit K-space Reconstruction

Neural Implicit K-space (NIK) reconstruction for cardiac MRI data using SIREN networks.

## Environment Setup

This project uses `uv` for Python package management. The environment has been pre-configured with all necessary dependencies.

### Quick Start

1. **Activate the environment:**
   ```bash
   source activate_env.sh
   ```
   
   Or manually:
   ```bash
   source .venv/bin/activate
   ```

2. **Verify installation:**
   ```bash
   python -c "from models.siren import NIKSiren; print('✅ Environment ready!')"
   ```

### Dependencies

The environment includes:
- **PyTorch** (2.8.0) - Deep learning framework
- **NumPy** (2.3.3) - Numerical computing
- **Matplotlib** (3.10.7) - Visualization
- **PyYAML** (6.0.3) - Configuration parsing
- **H5Py** (3.14.0) - HDF5 file handling
- **Weights & Biases** (0.22.2) - Experiment tracking
- **SciPy** (1.16.2) - Scientific computing
- **Pandas** (2.3.3) - Data manipulation

### Usage

#### Training
```bash
python train.py -c configs/config.yml -g 0 -s CINE_S1_rad_AA
```

#### Testing
```bash
python test.py
```

### Configuration

Edit `configs/config.yml` to customize:
- Data paths and parameters
- Model architecture (SIREN network)
- Training hyperparameters
- Loss function settings
- Reconstruction parameters

### Project Structure

```
NIK_MRI/
├── configs/          # Configuration files
├── datasets/         # Data loading and preprocessing
├── models/           # Neural network models (base, SIREN)
├── utils/            # Utilities (loss, MRI processing, visualization)
├── train.py          # Training script
├── test.py           # Testing script
├── pyproject.toml    # Project configuration
└── activate_env.sh   # Environment activation script
```

### Notes

- The project implements local compatibility functions for `medutils` to avoid dependency issues
- GPU support is configured for CUDA devices
- Weights & Biases integration is included for experiment tracking

### Troubleshooting

If you encounter import errors, ensure the environment is activated:
```bash
source .venv/bin/activate
```

To reinstall dependencies:
```bash
uv pip install -e .
```
