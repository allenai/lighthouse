#!/usr/bin/env python3

import os
import re

import geopandas as gpd
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio import features
from rasterio.transform import from_bounds
from shapely.geometry import box

# Step 1: Define Functions to Parse Tile Names and Create Polygons


def parse_tile_filename(filename):
    pattern = r"v200/2021/map/ESA_WorldCover_10m_2021_v200_([NS])(\d{2})([EW])(\d{3})_Map\.tif"
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
    # Each tile is 3x3 degrees
    lat_min = lat
    lat_max = lat + 3
    lon_min = lon
    lon_max = lon + 3
    return (lon_min, lat_min, lon_max, lat_max)


def create_tile_name(lat, lon):
    lat_prefix = "N" if lat >= 0 else "S"
    lon_prefix = "E" if lon >= 0 else "W"
    lat_str = f"{abs(lat):02d}"
    lon_str = f"{abs(lon):03d}"
    tile_name = f"{lat_prefix}{lat_str}{lon_prefix}{lon_str}"
    return tile_name


# Step 2: Load Land Polygons

# Path to land polygons shapefile
land_shapefile = "land-polygons-split-4326/land_polygons.shp"

# Load land polygons
print("Loading land polygons...")
land_polygons_gdf = gpd.read_file(land_shapefile)
# Ensure CRS is WGS84
land_polygons_gdf = land_polygons_gdf.to_crs("EPSG:4326")

# Step 3: Load Missing Tiles List

missing_tiles_file = "missing_land_tiles.txt"

# Read the list of missing tiles
print(f"Reading missing tiles from {missing_tiles_file}...")
with open(missing_tiles_file, "r") as f:
    missing_tiles_list = [line.strip() for line in f if line.strip()]

# Step 4: Create Output Directory

output_dir = "v1/2024/map/"
os.makedirs(output_dir, exist_ok=True)

# Step 5: Process Each Missing Tile

for tile_index, tile_path in enumerate(missing_tiles_list):
    print(f"\nProcessing tile {tile_index + 1}/{len(missing_tiles_list)}: {tile_path}")

    # Parse tile filename to get lat and lon
    coords = parse_tile_filename(tile_path)
    if coords:
        lat, lon = coords
        tile_name = create_tile_name(lat, lon)
        tile_bounds = get_tile_bounds(lat, lon)
    else:
        print(f"Could not parse tile name: {tile_path}")
        continue

    # Create tile geometry
    tile_geom = box(*tile_bounds)

    # Create GeoDataFrame for tile geometry
    tile_gdf = gpd.GeoDataFrame({"geometry": [tile_geom]}, crs="EPSG:4326")

    # Clip land polygons to tile
    possible_matches_index = list(
        land_polygons_gdf.sindex.intersection(tile_geom.bounds)
    )
    possible_matches = land_polygons_gdf.iloc[possible_matches_index]
    land_in_tile = gpd.overlay(possible_matches, tile_gdf, how="intersection")

    print(f"Number of land polygons in tile: {len(land_in_tile)}")

    if land_in_tile.empty:
        print(f"No land found in tile {tile_name}, skipping rasterization.")
        continue

    # Fix invalid geometries
    land_in_tile["geometry"] = land_in_tile["geometry"].buffer(0)
    land_in_tile = land_in_tile[land_in_tile.is_valid]

    # Prepare shapes for rasterization
    shapes = ((geom, 1) for geom in land_in_tile.geometry)
    num_shapes = len(land_in_tile)
    print(f"Number of shapes to rasterize: {num_shapes}")

    # Define raster resolution (adjust as needed)
    width = 1800  # Number of pixels in x-direction (longitude)
    height = 1800  # Number of pixels in y-direction (latitude)
    transform = from_bounds(*tile_bounds, width, height)
    print(f"Transform: {transform}")
    print(f"Raster size: width={width}, height={height}")

    # Rasterize
    burned = features.rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        all_touched=True,
        dtype=np.uint8,
    )

    # Check unique values in raster
    unique_values = np.unique(burned)
    print(f"Unique values in raster: {unique_values}")

    if unique_values.size == 1 and unique_values[0] == 0:
        print(
            f"Rasterization resulted in all zeros for tile {tile_name}, skipping save."
        )
        continue

    # Output filename with dynamic tile name
    output_filename = f"Ai2_WorldCover_10m_2024_v1_{tile_name}_Map.tif"
    output_path = os.path.join(output_dir, output_filename)

    # Define the metadata
    out_meta = {
        "driver": "GTiff",
        "height": burned.shape[0],
        "width": burned.shape[1],
        "count": 1,
        "dtype": "uint8",
        "crs": "EPSG:4326",
        "transform": transform,
    }

    # Save the raster
    with rasterio.open(output_path, "w", **out_meta) as dst:
        dst.write(burned, 1)

    print(f"Raster saved to {output_path}")

    # Optional Visualization
    # Uncomment the following block to visualize each tile
    """
    # Define a colormap
    cmap = colors.ListedColormap(['blue', 'green'])  # 0: blue (water), 1: green (land)
    bounds = [0, 0.5, 1]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(6, 6))
    plt.imshow(burned, cmap=cmap, norm=norm, extent=tile_bounds, origin='upper')
    plt.title(f"Tile {tile_name}: Land (green) and Water (blue)")
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    """
