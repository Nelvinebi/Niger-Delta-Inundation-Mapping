import rasterio
import numpy as np
import joblib

def predict_raster(feature_path, model_path, output_path):
    model = joblib.load(model_path)
    
    with rasterio.open(feature_path) as src:
        meta = src.meta.copy()
        meta.update(count=1, dtype='uint8', compress='lzw')
        
        # Process in chunks to save memory
        chunk_size = 5000
        height, width = src.shape
        
        with rasterio.open(output_path, 'w', **meta) as dst:
            for row in range(0, height, chunk_size):
                end_row = min(row + chunk_size, height)
                
                rainfall = src.read(1, window=((row, end_row), (0, width)))
                elevation = src.read(2, window=((row, end_row), (0, width)))
                flood_signal = src.read(3, window=((row, end_row), (0, width)))
                
                # Flatten and predict
                n_pixels = rainfall.size
                X = np.column_stack([
                    rainfall.flatten(),
                    elevation.flatten(),
                    flood_signal.flatten()
                ])
                
                # Handle nodata
                valid_mask = ~np.isnan(X).any(axis=1)
                predictions = np.zeros(n_pixels, dtype=np.uint8)
                
                if valid_mask.sum() > 0:
                    predictions[valid_mask] = model.predict(X[valid_mask])
                
                # Reshape and write
                pred_2d = predictions.reshape(rainfall.shape)
                dst.write(pred_2d, 1, window=((row, end_row), (0, width)))
                
                print(f"Processed rows {row}-{end_row}")
    
    print(f"Saved prediction map: {output_path}")

if __name__ == "__main__":
    predict_raster(
        "data/raw/features_stacked.tif",
        "outputs/models/xgboost_inundation.pkl",
        "outputs/figures/inundation_map.tif"
    )