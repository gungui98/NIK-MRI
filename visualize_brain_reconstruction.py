"""
Demo script showing how to load brain data, reconstruct images,
and visualize using the legacy-style approach.

This script demonstrates the integrated reconstruction pipeline using
functions from utils/vis.py that match the legacy reconstruction approach.

Usage:
    python visualize_brain_reconstruction.py --data_file <path_to_h5_file>
    python visualize_brain_reconstruction.py --data_file <path_to_h5_file> --output_prefix my_recon
"""
import argparse
import numpy as np
from utils.vis import (load_brain_h5, compute_coil_combined_reconstructions,
                       visualize_reconstruction, visualize_all_echoes, 
                       find_best_slice)


def main():
    parser = argparse.ArgumentParser(
        description='Reconstruct and visualize brain MRI data using legacy approach'
    )
    parser.add_argument('--data_file', type=str, required=True,
                       help='Path to HDF5 file containing brain data')
    parser.add_argument('--output_prefix', type=str, default='recon',
                       help='Output filename prefix for visualizations')
    parser.add_argument('--slice_idx', type=int, default=None,
                       help='Specific slice to visualize (default: auto-detect best slice)')
    parser.add_argument('--echo_idx', type=int, default=0,
                       help='Echo index for single-echo visualization (default: 0)')
    args = parser.parse_args()
    
    print("="*60)
    print("Brain MRI Reconstruction and Visualization")
    print("="*60)
    
    # Load data
    print(f"\nLoading data from: {args.data_file}")
    kspace, sens_maps, y_shift = load_brain_h5(args.data_file)
    
    print(f"\nData shapes:")
    print(f"  K-space:      {kspace.shape}")
    print(f"  Sens maps:    {sens_maps.shape}")
    print(f"  Y-shift:      {y_shift}")
    
    # Perform reconstruction
    print("\nPerforming coil-combined reconstruction...")
    img_recon = compute_coil_combined_reconstructions(
        kspace, sens_maps, y_shift, remove_oversampling=True
    )
    
    print(f"Reconstructed image shape: {img_recon.shape}")
    print(f"Image magnitude range: {np.abs(img_recon).min():.2e} to {np.abs(img_recon).max():.2e}")
    
    # Find best slice if not specified
    if args.slice_idx is None:
        best_slice = find_best_slice(img_recon)
        print(f"\nAuto-detected best slice: {best_slice}")
        slice_to_viz = best_slice
    else:
        slice_to_viz = args.slice_idx
        print(f"\nUsing user-specified slice: {slice_to_viz}")
    
    # Check if valid slice
    if img_recon.ndim == 4:  # Multi-slice
        if slice_to_viz >= img_recon.shape[0]:
            print(f"Warning: Slice {slice_to_viz} out of range. Using slice 0.")
            slice_to_viz = 0
    
    # Generate visualizations
    print("\n" + "="*60)
    print("Generating visualizations...")
    print("="*60)
    
    print(f"\n1. Single slice/echo visualization (slice {slice_to_viz}, echo {args.echo_idx}):")
    visualize_reconstruction(img_recon, slice_idx=slice_to_viz, echo_idx=args.echo_idx,
                            output_prefix=args.output_prefix)
    
    print(f"\n2. Multi-echo visualization (slice {slice_to_viz}, all echoes):")
    visualize_all_echoes(img_recon, slice_idx=slice_to_viz,
                        output_prefix=f"{args.output_prefix}_all_echoes")
    
    print("\n" + "="*60)
    print("Done! Check the generated PNG files.")
    print("="*60)


if __name__ == '__main__':
    main()

