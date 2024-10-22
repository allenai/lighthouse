#!/usr/bin/env python3

import re

import geopandas as gpd
from shapely.geometry import Polygon

# Step 1: Generate the Full Tile Grid


def generate_tile_grid():
    latitudes = list(range(-90, 90, 3))  # From -90 to 87 degrees
    longitudes = list(range(-180, 180, 3))  # From -180 to 177 degrees
    tiles = []

    for lat in latitudes:
        for lon in longitudes:
            # Latitude string
            if lat >= 0:
                lat_prefix = "N"
                lat_str = f"{lat:02d}"
            else:
                lat_prefix = "S"
                lat_str = f"{abs(lat):02d}"

            # Longitude string
            if lon >= 0:
                lon_prefix = "E"
                lon_str = f"{lon:03d}"
            else:
                lon_prefix = "W"
                lon_str = f"{abs(lon):03d}"

            tile_name = f"v200/2021/map/ESA_WorldCover_10m_2021_v200_{lat_prefix}{lat_str}{lon_prefix}{lon_str}_Map.tif"

            # Create tile polygon
            lat_min = lat
            lat_max = lat + 3
            lon_min = lon
            lon_max = lon + 3

            tile_polygon = Polygon(
                [
                    (lon_min, lat_min),
                    (lon_max, lat_min),
                    (lon_max, lat_max),
                    (lon_min, lat_max),
                    (lon_min, lat_min),
                ]
            )

            tiles.append({"tile_name": tile_name, "geometry": tile_polygon})

    # Create a GeoDataFrame for the tiles
    tiles_gdf = gpd.GeoDataFrame(tiles, crs="EPSG:4326")
    return tiles_gdf


# Step 2: Load Land Polygons


def load_land_polygons(shapefile_path):
    land_polygons = gpd.read_file(shapefile_path)
    # Ensure the CRS is WGS84
    if land_polygons.crs != "EPSG:4326":
        land_polygons = land_polygons.to_crs("EPSG:4326")
    return land_polygons


# Step 3: Determine Tiles Covering Land


def get_tiles_covering_land(tiles_gdf, land_polygons_gdf):
    # Spatial join to find tiles that intersect with land polygons
    tiles_with_land = gpd.sjoin(
        tiles_gdf, land_polygons_gdf, how="inner", op="intersects"
    )
    # Drop duplicates in case a tile intersects multiple land polygons
    tiles_with_land = tiles_with_land.drop_duplicates(subset="tile_name")
    return tiles_with_land


# Step 4: Compare with Existing Tiles


def get_missing_tiles(tiles_with_land_gdf, existing_tiles_list):
    existing_tiles_set = set(existing_tiles_list)
    tiles_with_land_set = set(tiles_with_land_gdf["tile_name"])

    missing_tiles_set = tiles_with_land_set - existing_tiles_set
    missing_tiles_list = sorted(list(missing_tiles_set))
    return missing_tiles_list


# Main Function


def main():
    # Paths to your files
    shapefile_path = "land-polygons-split-4326/land_polygons.shp"
    existing_tiles_file = "list_of_files.txt"
    output_missing_tiles_file = "missing_land_tiles.txt"

    print("Generating tile grid...")
    tiles_gdf = generate_tile_grid()
    print(f"Total tiles generated: {len(tiles_gdf)}")

    print("Loading land polygons...")
    land_polygons_gdf = load_land_polygons(shapefile_path)
    print(f"Total land polygons loaded: {len(land_polygons_gdf)}")

    print("Identifying tiles that cover land...")
    tiles_with_land_gdf = get_tiles_covering_land(tiles_gdf, land_polygons_gdf)
    print(f"Tiles covering land: {len(tiles_with_land_gdf)}")

    print("Loading existing tile filenames...")
    with open(existing_tiles_file, "r") as f:
        existing_tiles_list = [line.strip() for line in f if line.strip()]
    print(f"Existing tiles count: {len(existing_tiles_list)}")

    print("Determining missing tiles that cover land...")
    missing_tiles_list = get_missing_tiles(tiles_with_land_gdf, existing_tiles_list)
    print(f"Missing tiles that cover land: {len(missing_tiles_list)}")

    # Save missing tiles to a file
    with open(output_missing_tiles_file, "w") as f:
        for tile in missing_tiles_list:
            f.write(f"{tile}\n")

    print(f"Missing tiles saved to '{output_missing_tiles_file}'.")


if __name__ == "__main__":
    main()
