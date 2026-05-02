import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_maps(feature_path, prediction_path, output_dir, lga_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    with rasterio.open(feature_path) as src:
        sar = src.read(1)
        rain = src.read(2)
        elev = src.read(3)
    
    with rasterio.open(prediction_path) as src:
        pred = src.read(1)
    
    # SAR
    im1 = axes[0,0].imshow(sar[::5, ::5], cmap='gray', vmin=-25, vmax=0)
    axes[0,0].set_title('Sentinel-1 SAR Backscatter (VV)')
    plt.colorbar(im1, ax=axes[0,0], fraction=0.046)
    
    # Rainfall
    im2 = axes[0,1].imshow(rain[::5, ::5], cmap='Blues')
    axes[0,1].set_title('CHIRPS Rainfall (mm)')
    plt.colorbar(im2, ax=axes[0,1], fraction=0.046)
    
    # Elevation
    im3 = axes[1,0].imshow(elev[::5, ::5], cmap='terrain')
    axes[1,0].set_title('SRTM Elevation (m)')
    plt.colorbar(im3, ax=axes[1,0], fraction=0.046)
    
    # Prediction with optional LGA overlay
    im4 = axes[1,1].imshow(pred[::5, ::5], cmap='RdYlBu_r', vmin=0, vmax=1)
    axes[1,1].set_title('Inundation Prediction')
    
    if lga_path and os.path.exists(lga_path):
        import geopandas as gpd
        lg = gpd.read_file(lga_path)
        lg.boundary.plot(ax=axes[1,1], color='black', linewidth=0.5)
    
    plt.colorbar(im4, ax=axes[1,1], fraction=0.046)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/inundation_analysis.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/inundation_analysis.png")
    plt.close()

if __name__ == "__main__":
    generate_maps(
        "data/raw/features_stacked.tif",
        "outputs/figures/inundation_map.tif",
        "outputs/figures",
        lga_path="data/raw/nigeria_lga_boundaries.shp"
    )