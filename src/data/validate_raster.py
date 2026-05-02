import rasterio
import numpy as np

def validate_raster(path):
    with rasterio.open(path) as src:
        print(f"File: {path}")
        print(f"Shape: {src.shape}")
        print(f"CRS: {src.crs}")
        print(f"Bands: {src.count}")
        print(f"Descriptions: {src.descriptions}")
        
        for i in range(1, src.count + 1):
            band = src.read(i)
            sample = band[::10, ::10].flatten()
            sample = sample[~np.isnan(sample)]
            print(f"\n{src.descriptions[i-1]}:")
            print(f"  min={sample.min():.2f}, max={sample.max():.2f}")
            print(f"  mean={sample.mean():.2f}, std={sample.std():.2f}")

if __name__ == "__main__":
    validate_raster("data/raw/features_stacked.tif")