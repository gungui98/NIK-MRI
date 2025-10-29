#!/bin/bash
# Activation script for NIK_MRI environment
echo "Activating deeplearning conda environment..."
conda activate deeplearning
echo "Environment activated! You can now run:"
echo "  python train.py --help"
echo "  python test.py"
echo ""
echo "To deactivate, run: conda deactivate"
