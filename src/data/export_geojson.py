import rasterio
from rasterio.features import shapes
import geopandas as gpd
import numpy as np

def raster_to_geojson(raster_path, output_path, mask_value=1):
    with rasterio.open(raster_path) as src:
        image = src.read(1)
        mask = image == mask_value
        
        results = (
            {'properties': {'inundation': int(v)}, 'geometry': s}
            for i, (s, v) in enumerate(shapes(image, mask=mask, transform=src.transform))
        )
        
        gdf = gpd.GeoDataFrame.from_features(list(results), crs=src.crs)
        gdf.to_file(output_path, driver="GeoJSON")
        print(f"Saved: {output_path}")
        print(f"Features: {len(gdf)}")

if __name__ == "__main__":
    raster_to_geojson("outputs/figures/inundation_map.tif", "outputs/figures/inundation_polygons.geojson")