import pandas as pd
import numpy as np

# Load change dataset
change_df = pd.read_csv("data/processed/dataset_change.csv")

# Load original features (rainfall, elevation) — need to rebuild or extract from raster
# Since we don't have lat/lon, we'll need to rebuild from the stacked raster
# Alternative: use a sample with spatial coordinates

# For now, take a random sample and accept approximate rainfall/elevation
# In practice, rebuild build_change_dataset.py to include these from the start

print(f"Change dataset: {change_df.shape}")
print(f"Need to add rainfall and elevation features")
print("Modify build_change_dataset.py to read from features_stacked.tif alongside SAR images")