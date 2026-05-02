#!/usr/bin/env python3
"""
Niger Delta Inundation Mapping System (NDIMS)
Main pipeline entry point.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data.validate_raster import validate_raster
from models.train_model import train_model
from models.predict_raster import predict_raster
from visualization.generate_maps import generate_maps
from visualization.compare_temporal import compare_temporal

def run_full_pipeline():
    print("=" * 60)
    print("Niger Delta Inundation Mapping System (NDIMS)")
    print("=" * 60)
    
    # Step 1: Validate input data
    print("\n[1/4] Validating input raster...")
    validate_raster("data/raw/features_stacked.tif")
    
    # Step 2: Train model
    print("\n[2/4] Training XGBoost model...")
    train_model("data/processed/dataset_final_balanced.csv", "outputs/models/xgboost_inundation.pkl")
    
    # Step 3: Generate prediction map
    print("\n[3/4] Predicting inundation on full raster...")
    predict_raster(
        "data/raw/features_stacked.tif",
        "outputs/models/xgboost_inundation.pkl",
        "outputs/figures/inundation_map.tif"
    )

    # After generate_maps() call:
    print("\n[5/5] Generating temporal comparison...")
    compare_temporal(
        "data/raw/sar_dry_2022.tif",
        "data/raw/sar_flood_2022.tif",
        "data/raw/flood_change_2022.tif",
        "outputs/figures"
    )
    
    # Step 4: Generate visualization
    print("\n[4/4] Generating analysis maps...")
    generate_maps(
        "data/raw/features_stacked.tif",
        "outputs/figures/inundation_map.tif",
        "outputs/figures"
    )
    
    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("Outputs:")
    print("  - Model: outputs/models/xgboost_inundation.pkl")
    print("  - Prediction: outputs/figures/inundation_map.tif")
    print("  - Figures: outputs/figures/inundation_analysis.png")
    print("=" * 60)

if __name__ == "__main__":
    run_full_pipeline()