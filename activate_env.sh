#!/bin/bash
# Activation script for NIK_MRI environment
echo "Activating NIK_MRI environment..."
source .venv/bin/activate
echo "Environment activated! You can now run:"
echo "  python train.py --help"
echo "  python test.py"
echo ""
echo "To deactivate, run: deactivate"
