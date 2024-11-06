#!/usr/bin/env python3

import os
import re
import numpy as np
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.transform import from_bounds
from shapely.geometry import box
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# Global variable to hold land polygons for multiprocessing workers
global_land_polygons_gdf = None
output_dir = "resampled_2/"
# Define the LAND_COVER_TYPES mapping
LAND_COVER_TYPES = {
    1: "Land",
    80: "Permanent water bodies",
}

def parse_tile_filename(filename):
    pattern = r"Ai2_WorldCover_10m_2024_v1_([NS])(\d{2})([EW])(\d{3})_Map\.tif"
    match = re.match(pattern, filename)
    if match:
        lat_prefix, lat_str, lon_prefix, lon_str = match.groups()
        lat = int(lat_str)
        lon = int(lon_str)
        if lat_prefix == "S":
            lat = -lat
        if lon_prefix == "W":
            lon = -lon
        return lat, lon
    else:
        return None

def get_tile_bounds(lat, lon):
    return (lon, lat, lon + 1, lat + 1)

def create_tile_name(lat, lon):
    return f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}{'E' if lon >= 0 else 'W'}{abs(lon):03d}"

def load_land_polygons(shapefile_path):
    global global_land_polygons_gdf
    land_polygons = gpd.read_file(shapefile_path)
    if land_polygons.crs != "EPSG:4326":
        land_polygons = land_polygons.to_crs("EPSG:4326")
    global_land_polygons_gdf = land_polygons
    return global_land_polygons_gdf

def find_missing_tiles(resampled_dir, tiles_with_land_set):
    existing_tiles = set()
    for f in os.listdir(resampled_dir):
        if f.endswith('.tif'):
            tile_coords = parse_tile_filename(f)
            if tile_coords:
                existing_tiles.add(tile_coords)
    missing_tiles = tiles_with_land_set - existing_tiles
    return sorted(list(missing_tiles))

def process_single_tile(tile):
    global global_land_polygons_gdf
    lat, lon = tile
    tile_name = create_tile_name(lat, lon)
    tile_bounds = get_tile_bounds(lat, lon)
    tile_geom = box(*tile_bounds)
    tile_gdf = gpd.GeoDataFrame({"geometry": [tile_geom]}, crs="EPSG:4326")

    # Find polygons intersecting the tile
    possible_matches_index = list(global_land_polygons_gdf.sindex.intersection(tile_geom.bounds))
    possible_matches = global_land_polygons_gdf.iloc[possible_matches_index]
    land_in_tile = gpd.overlay(possible_matches, tile_gdf, how="intersection")

    if land_in_tile.empty:
        return f"No land found in tile {tile_name}, skipping rasterization."

    # Clean geometries
    land_in_tile["geometry"] = land_in_tile["geometry"].buffer(0)
    land_in_tile = land_in_tile[land_in_tile.is_valid]

    shapes = ((geom, 1) for geom in land_in_tile.geometry)
    width = 12000  # 10m resolution over 1 degree
    height = 12000
    transform = from_bounds(*tile_bounds, width, height)

    # Rasterize
    burned = features.rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=80,  # Assuming water is 80
        all_touched=True,
        dtype=np.uint8,
    )

    unique_values = np.unique(burned)
    if unique_values.size == 1 and unique_values[0] == 80:
        return f"Rasterization resulted in all water for tile {tile_name}, skipping save."

    output_filename = f"Ai2_WorldCover_10m_2024_v1_{tile_name}_Map.tif"
    output_path = os.path.join(output_dir, output_filename)

    out_meta = {
        "driver": "GTiff",
        "height": burned.shape[0],
        "width": burned.shape[1],
        "count": 1,
        "dtype": "uint8",
        "crs": "EPSG:4326",
        "transform": transform,
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "nodata": None,  # Set to None or an appropriate value
    }

    with rasterio.open(output_path, "w", **out_meta) as dst:
        dst.write(burned, 1)

    return f"Raster saved to {output_path}"

def process_missing_tiles_parallel(missing_tiles_list, max_workers=None):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(process_single_tile, missing_tiles_list),
            total=len(missing_tiles_list),
            desc="Processing Missing Tiles in Parallel"
        ))
    for result in results:
        if result:
            print(result)

def main():
    shapefile_path = "land-polygons-split-4326/land_polygons.shp"
    resampled_dir = "data/resampled/"
    output_dir = "resampled_2/"
    os.makedirs(output_dir, exist_ok=True)

    # Generate set of tiles from 80°N to 90°N inclusive
    latitudes = range(80, 91, 1)
    longitudes = range(-180, 180, 1)
    tiles_with_land_set = {
        (lat, lon) for lat in latitudes for lon in longitudes
    }

    # Load global land polygons for multiprocessing
    load_land_polygons(shapefile_path)

    # Find missing tiles
    missing_tiles_list = find_missing_tiles(resampled_dir, tiles_with_land_set)
    print(f"Missing tiles count: {len(missing_tiles_list)}")

    if not missing_tiles_list:
        print("No missing tiles to process.")
        return

    # Process missing tiles in parallel
    process_missing_tiles_parallel(missing_tiles_list)
    print("Completed processing missing tiles.")

if __name__ == "__main__":
    main()
