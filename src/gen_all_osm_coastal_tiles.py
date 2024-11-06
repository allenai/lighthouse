#!/usr/bin/env python3

import os
import re
import multiprocessing
from functools import partial
from math import floor

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from rasterio.transform import from_origin
from shapely.geometry import box
from pyproj import CRS, Transformer

# Define the LAND_COVER_TYPES mapping
LAND_COVER_TYPES = {
    1: "Land",        # Assign value 1 to land
    80: "Water",      # Assign value 80 to water/outside land polygons
}

# Function to parse tile filename and extract lat, lon
def parse_tile_filename(filename):
    """
    Parses the tile filename to extract latitude and longitude.

    Expected filename formats:
    - ESA_WorldCover_10m_2021_v200_N00E000_Map.tif
    - Ai2_WorldCover_10m_2024_v1_N00E000_Map.tif

    Returns:
        (lat, lon) tuple if parsing is successful, None otherwise.
    """
    pattern = r"(?:ESA_WorldCover_10m_2021_v200_|Ai2_WorldCover_10m_2024_v1_)?([NS])(\d{2})([EW])(\d{3})_Map\.tif"
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

# Function to create tile name based on lat and lon
def create_tile_name(lat, lon):
    lat_prefix = "N" if lat >= 0 else "S"
    lon_prefix = "E" if lon >= 0 else "W"
    lat_str = f"{abs(lat):02d}"
    lon_str = f"{abs(lon):03d}"
    tile_name = f"{lat_prefix}{lat_str}{lon_prefix}{lon_str}"
    return tile_name

# Function to get tile bounds
def get_tile_bounds(lat, lon):
    """
    Returns the bounding box for a given tile.

    Args:
        lat (int): Latitude of the southwest corner.
        lon (int): Longitude of the southwest corner.

    Returns:
        tuple: (lon_min, lat_min, lon_max, lat_max)
    """
    lat_min = lat
    lat_max = lat + 3
    lon_min = lon
    lon_max = lon + 3
    return (lon_min, lat_min, lon_max, lat_max)

# Function to determine UTM zone based on longitude
def get_utm_crs(lon, lat):
    """
    Determines the appropriate UTM CRS for a given longitude and latitude.

    Args:
        lon (float): Longitude in degrees.
        lat (float): Latitude in degrees.

    Returns:
        CRS: PyProj CRS object for the appropriate UTM zone.
    """
    utm_zone = floor((lon + 180) / 6) + 1
    is_northern = lat >= 0
    return CRS.from_dict({
        'proj': 'utm',
        'zone': utm_zone,
        'datum': 'WGS84',
        'units': 'm',
        'south' if not is_northern else 'north': True
    })

# Function to process a single tile
def process_tile(tile_name, land_polygons_gdf, output_dir, desired_resolution=10):
    """
    Processes a single tile: reprojects to UTM, rasterizes land polygons,
    and saves the raster if it's not 100% land or water.

    Args:
        tile_name (str): Name of the tile (e.g., N00E000).
        land_polygons_gdf (GeoDataFrame): GeoDataFrame of land polygons.
        output_dir (str): Directory to save the output raster.
        desired_resolution (float): Desired resolution in meters/pixel.
    """
    try:
        # Extract lat and lon from tile name
        lat_prefix = tile_name[0]
        lat_str = tile_name[1:3]
        lon_prefix = tile_name[3]
        lon_str = tile_name[4:7]
        lat = int(lat_str) * (1 if lat_prefix == "N" else -1)
        lon = int(lon_str) * (1 if lon_prefix == "E" else -1)

        # Get tile bounds
        tile_bounds = get_tile_bounds(lat, lon)
        tile_geom = box(*tile_bounds)

        # Clip land polygons to tile
        possible_matches_index = list(land_polygons_gdf.sindex.intersection(tile_geom.bounds))
        possible_matches = land_polygons_gdf.iloc[possible_matches_index]
        land_in_tile = gpd.overlay(possible_matches, gpd.GeoDataFrame({"geometry": [tile_geom]}, crs=land_polygons_gdf.crs), how="intersection")

        if land_in_tile.empty:
            print(f"No land found in tile {tile_name}, skipping rasterization.")
            return

        # Fix invalid geometries
        land_in_tile["geometry"] = land_in_tile["geometry"].buffer(0)
        land_in_tile = land_in_tile[land_in_tile.is_valid]

        if land_in_tile.empty:
            print(f"All geometries invalid in tile {tile_name}, skipping rasterization.")
            return

        # Determine appropriate UTM CRS
        center_lat = (tile_bounds[1] + tile_bounds[3]) / 2
        center_lon = (tile_bounds[0] + tile_bounds[2]) / 2
        utm_crs = get_utm_crs(center_lon, center_lat)

        # Reproject land polygons to UTM
        land_in_tile_utm = land_in_tile.to_crs(utm_crs)

        # Reproject tile bounds to UTM
        tile_utm = gpd.GeoDataFrame({"geometry": [tile_geom]}, crs="EPSG:4326").to_crs(utm_crs)
        tile_utm_bounds = tile_utm.total_bounds  # (minx, miny, maxx, maxy)

        # Calculate raster dimensions based on desired resolution
        minx, miny, maxx, maxy = tile_utm_bounds
        width = int((maxx - minx) / desired_resolution)
        height = int((maxy - miny) / desired_resolution)

        # Define affine transform
        transform = from_origin(minx, maxy, desired_resolution, desired_resolution)

        # Prepare shapes for rasterization
        shapes = ((geom, 1) for geom in land_in_tile_utm.geometry)

        # Rasterize
        burned = features.rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform,
            fill=80,          # Assign value 80 to water
            all_touched=True,
            dtype=np.uint8,
        )

        # Check unique values in raster
        unique_values = np.unique(burned)

        # Determine if the tile is 100% land or 100% water
        if unique_values.size == 1:
            if unique_values[0] == 1:
                print(f"Tile {tile_name} is 100% land, skipping save.")
                return
            elif unique_values[0] == 80:
                print(f"Tile {tile_name} is 100% water, skipping save.")
                return

        # Define output filename
        output_filename = f"Ai2_WorldCover_10m_2024_v1_{tile_name}_Map.tif"
        output_path = os.path.join(output_dir, output_filename)

        # Define the metadata
        out_meta = {
            "driver": "GTiff",
            "height": burned.shape[0],
            "width": burned.shape[1],
            "count": 1,
            "dtype": "uint8",
            "crs": utm_crs.to_wkt(),
            "transform": transform,
            "nodata": None,
        }

        # Save the raster
        with rasterio.open(output_path, "w", **out_meta) as dst:
            dst.write(burned, 1)

        print(f"Raster saved to {output_path}")

    except Exception as e:
        print(f"Error processing tile {tile_name}: {e}")

# Main function to set up multiprocessing
def main():
    import time

    # Path to land polygons shapefile
    land_shapefile = "land-polygons-split-4326/land_polygons.shp"

    # Load land polygons
    print("Loading land polygons...")
    land_polygons_gdf = gpd.read_file(land_shapefile)
    # Ensure CRS is WGS84
    land_polygons_gdf = land_polygons_gdf.to_crs("EPSG:4326")

    # Build spatial index
    land_polygons_gdf.sindex  # Ensures the spatial index is built before multiprocessing

    # Step 2: Read the List of Tiles to Process

    # Paths to lists
    list_of_files_file = "src/list_of_files.txt"
    missing_files_file = "src/missing_tiles.txt"

    # Read the list of tiles from list_of_files.txt
    with open(list_of_files_file, "r") as f:
        list_of_files = [line.strip() for line in f if line.strip()]
    print(f"Number of tiles in list_of_files: {len(list_of_files)}")

    # Read the list of tiles from missing_files.txt
    with open(missing_files_file, "r") as f:
        missing_files = [line.strip() for line in f if line.strip()]
    print(f"Number of tiles in missing_files: {len(missing_files)}")

    # Combine the lists
    combined_files = list_of_files + missing_files

    # Extract tile names from filenames
    tile_names = []
    for filepath in combined_files:
        filename = os.path.basename(filepath)
        coords = parse_tile_filename(filename)
        if coords:
            lat, lon = coords
            tile_name = create_tile_name(lat, lon)
            tile_names.append(tile_name)
        else:
            print(f"Could not parse tile filename: {filename}")

    # Remove duplicates
    tile_names = list(set(tile_names))
    print(f"Total number of unique tiles to process: {len(tile_names)}")

    # Step 3: Create Output Directory
    output_dir = "v1/2024/map/"
    os.makedirs(output_dir, exist_ok=True)

    # Number of processes (cores) to use
    num_processes = multiprocessing.cpu_count()  # Use all available cores
    print(f"Using {num_processes} processes for parallel processing.")

    # Partial function to pass constant arguments
    process_tile_partial = partial(
        process_tile,
        land_polygons_gdf=land_polygons_gdf,
        output_dir=output_dir,
        desired_resolution=10,  # 10 meters per pixel
    )

    start_time = time.time()

    # Create a pool of worker processes
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Map the function over the list of tile names
        pool.map(process_tile_partial, tile_names)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Processing completed in {elapsed_time / 60:.2f} minutes.")

if __name__ == "__main__":
    main()
