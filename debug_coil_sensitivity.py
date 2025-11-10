"""Debug script to visualize coil sensitivity maps and understand spatial bias."""

import numpy as np
import matplotlib.pyplot as plt
from utils.basic import parse_config
from datasets.brain import BrainDataset

def visualize_coil_sensitivity_maps():
    """Visualize coil sensitivity maps to see spatial distribution."""
    config = parse_config('configs/config_brain.yml')
    
    # Create dataset with selection
    config['is_selected'] = True
    dataset_selected = BrainDataset(config)
    csm_selected = dataset_selected.csm  # (slices, 1, coils, kx, ky)
    
    # Create dataset without selection
    config['is_selected'] = False
    dataset_all = BrainDataset(config)
    csm_all = dataset_all.csm  # (slices, echoes, coils, kx, ky)
    
    # Pick a slice to visualize
    slice_idx = 0
    echo_idx = 0
    
    # For selected: shape is (slices, 1, 4_coils, kx, ky)
    # For all: shape is (slices, 1, 31_coils, kx, ky) 
    csm_sel = csm_selected[slice_idx, 0]  # (4, kx, ky)
    csm_all_slice = csm_all[slice_idx, echo_idx]  # (31, kx, ky)
    
    # Compute RSS for each case
    from utils.medutils_compat import rss
    rss_selected = np.abs(rss(csm_sel, coil_axis=0))
    rss_all = np.abs(rss(csm_all_slice, coil_axis=0))
    
    # Visualize individual coils for selected set
    num_coils_selected = csm_sel.shape[0]
    fig, axes = plt.subplots(2, num_coils_selected + 1, figsize=(3*(num_coils_selected+1), 6))
    
    # First row: magnitude of each coil
    for coil_idx in range(num_coils_selected):
        axes[0, coil_idx].imshow(np.abs(csm_sel[coil_idx]).T, cmap='hot', origin='lower')
        axes[0, coil_idx].set_title(f'Selected Coil {coil_idx}\nMagnitude')
        axes[0, coil_idx].axis('off')
    
    # Show RSS combination
    axes[0, num_coils_selected].imshow(rss_selected.T, cmap='hot', origin='lower')
    axes[0, num_coils_selected].set_title('RSS of 4 Selected Coils')
    axes[0, num_coils_selected].axis('off')
    
    # Second row: compare with all coils
    axes[1, 0].imshow(rss_selected.T, cmap='hot', origin='lower')
    axes[1, 0].set_title('RSS: 4 Selected Coils [0,1,2,3]')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(rss_all.T, cmap='hot', origin='lower')
    axes[1, 1].set_title('RSS: All 31 Coils')
    axes[1, 1].axis('off')
    
    # Difference
    diff = rss_selected - rss_all
    im = axes[1, 2].imshow(diff.T, cmap='seismic', origin='lower', vmin=-np.max(np.abs(diff)), vmax=np.max(np.abs(diff)))
    axes[1, 2].set_title('Difference (Selected - All)')
    axes[1, 2].axis('off')
    plt.colorbar(im, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.savefig('coil_sensitivity_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved coil sensitivity analysis to coil_sensitivity_analysis.png")
    
    # Print statistics
    print(f"\nCoil Sensitivity Statistics for Slice {slice_idx}:")
    print(f"Selected coils [0,1,2,3] - RSS max: {rss_selected.max():.4f}, mean: {rss_selected.mean():.4f}")
    print(f"All 31 coils - RSS max: {rss_all.max():.4f}, mean: {rss_all.mean():.4f}")
    print(f"Ratio (selected/all) max: {(rss_selected/rss_all).max():.4f}, min: {(rss_selected/rss_all).min():.4f}")

if __name__ == '__main__':
    visualize_coil_sensitivity_maps()

