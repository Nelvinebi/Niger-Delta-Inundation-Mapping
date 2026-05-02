import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import numpy as np

def compare_temporal(dry_path, flood_path, change_path, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    with rasterio.open(dry_path) as dry_src:
        dry = dry_src.read(1)
        transform = dry_src.transform
    
    with rasterio.open(flood_path) as flood_src:
        flood = flood_src.read(1)
    
    with rasterio.open(change_path) as change_src:
        change = change_src.read(1)
    
    # Dry season
    im1 = axes[0,0].imshow(dry[::5, ::5], cmap='gray', vmin=-25, vmax=0)
    axes[0,0].set_title('Dry Season (Jan 2022)')
    plt.colorbar(im1, ax=axes[0,0], fraction=0.046)
    
    # Flood period
    im2 = axes[0,1].imshow(flood[::5, ::5], cmap='gray', vmin=-25, vmax=0)
    axes[0,1].set_title('Flood Period (Sep-Oct 2022)')
    plt.colorbar(im2, ax=axes[0,1], fraction=0.046)
    
    # Difference
    diff = flood - dry
    im3 = axes[1,0].imshow(diff[::5, ::5], cmap='RdBu_r', vmin=-10, vmax=10)
    axes[1,0].set_title('SAR Change (Flood - Dry)')
    plt.colorbar(im3, ax=axes[1,0], fraction=0.046)
    
    # True flood (change detection)
    im4 = axes[1,1].imshow(change[::5, ::5], cmap='Reds', vmin=0, vmax=1)
    axes[1,1].set_title('True Flood (Change Detection)')
    plt.colorbar(im4, ax=axes[1,1], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/temporal_comparison.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/temporal_comparison.png")
    plt.close()

if __name__ == "__main__":
    compare_temporal(
        "data/raw/sar_dry_2022.tif",
        "data/raw/sar_flood_2022.tif",
        "data/raw/flood_change_2022.tif",
        "outputs/figures"
    )