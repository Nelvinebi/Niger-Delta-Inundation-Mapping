import rasterio
from rasterio.warp import reproject, Resampling
import pandas as pd
import numpy as np
import os

def build_change_dataset(dry_path, flood_path, change_label_path, feature_path, output_csv, chunk_size=5000):
    """
    Build dataset for change detection.
    Features: dry season SAR, flood period SAR, SAR change, rainfall, elevation
    Label: true flood (water that appeared during flood but wasn't there in dry season)
    """
    with rasterio.open(dry_path) as dry_src, \
         rasterio.open(flood_path) as flood_src, \
         rasterio.open(change_label_path) as label_src, \
         rasterio.open(feature_path) as feat_src:
        
        # Verify dimensions match
        print(f"Flood SAR shape: {flood_src.shape}")
        print(f"Feature stack shape: {feat_src.shape}")
        
        # Reproject dry SAR and label to match flood SAR
        dry_sar = np.zeros(flood_src.shape, dtype=np.float32)
        reproject(
            source=dry_src.read(1),
            destination=dry_sar,
            src_transform=dry_src.transform,
            src_crs=dry_src.crs,
            dst_transform=flood_src.transform,
            dst_crs=flood_src.crs,
            resampling=Resampling.bilinear
        )
        
        label = np.zeros(flood_src.shape, dtype=np.uint8)
        reproject(
            source=label_src.read(1),
            destination=label,
            src_transform=label_src.transform,
            src_crs=label_src.crs,
            dst_transform=flood_src.transform,
            dst_crs=flood_src.crs,
            resampling=Resampling.nearest
        )
        
        # Read flood SAR
        flood_sar = flood_src.read(1)
        
        chunks = []
        for start_row in range(0, flood_src.shape[0], chunk_size):
            end_row = min(start_row + chunk_size, flood_src.shape[0])
            
            dry_chunk = dry_sar[start_row:end_row, :].flatten()
            flood_chunk = flood_sar[start_row:end_row, :].flatten()
            lab_chunk = label[start_row:end_row, :].flatten()
            
            # Read rainfall (band 2) and elevation (band 3) from stacked features
            rain_chunk = feat_src.read(2, window=((start_row, end_row), (0, feat_src.shape[1]))).flatten()
            elev_chunk = feat_src.read(3, window=((start_row, end_row), (0, feat_src.shape[1]))).flatten()
            
            df_chunk = pd.DataFrame({
                "sar_dry": dry_chunk,
                "sar_flood": flood_chunk,
                "sar_change": flood_chunk - dry_chunk,
                "rainfall": rain_chunk,
                "elevation": elev_chunk,
                "label": (lab_chunk > 0).astype(int)
            })
            
            df_chunk = df_chunk.replace([np.inf, -np.inf], np.nan).dropna()
            chunks.append(df_chunk)
            
            if start_row % 10000 == 0:
                print(f"  Processed rows {start_row}-{end_row}, kept {len(df_chunk)} pixels")
        
        df = pd.concat(chunks, ignore_index=True)
    
    df.to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv}")
    print(f"Shape: {df.shape}")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    print(f"Flood ratio: {df['label'].mean():.4f}")

if __name__ == "__main__":
    build_change_dataset(
        "data/raw/sar_dry_2022.tif",
        "data/raw/sar_flood_2022.tif",
        "data/raw/flood_change_2022.tif",
        "data/raw/features_stacked.tif",
        "data/processed/dataset_change_full.csv"
    )