#!/usr/bin/env python3

import geopandas as gpd
from shapely.geometry import Polygon


def generate_tiles(lat_min, lat_max, lon_min, lon_max):
    latitudes = list(range(lat_min, lat_max, 3))
    longitudes = list(range(lon_min, lon_max, 3))
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
            tile_polygon = Polygon(
                [
                    (lon, lat),
                    (lon + 3, lat),
                    (lon + 3, lat + 3),
                    (lon, lat + 3),
                    (lon, lat),
                ]
            )

            tiles.append({"tile_name": tile_name, "geometry": tile_polygon})

    tiles_gdf = gpd.GeoDataFrame(tiles, crs="EPSG:4326")
    return tiles_gdf


# Define Micronesia extent
lat_min = 0
lat_max = 13  # Up to 12°N
lon_min = 135
lon_max = 153  # Up to 150°E

# Generate tiles for Micronesia
micronesia_tiles_gdf = generate_tiles(lat_min, lat_max, lon_min, lon_max)

# Load existing tiles
with open("list_of_files.txt", "r") as f:
    existing_tiles = [line.strip() for line in f if line.strip()]
existing_tiles_set = set(existing_tiles)

# Identify missing tiles in Micronesia
micronesia_tile_names = set(micronesia_tiles_gdf["tile_name"])
missing_micronesia_tiles = micronesia_tile_names - existing_tiles_set
print(f"Missing tiles in Micronesia: {missing_micronesia_tiles}")

# Filter missing tiles GeoDataFrame
missing_tiles_gdf = micronesia_tiles_gdf[
    micronesia_tiles_gdf["tile_name"].isin(missing_micronesia_tiles)
]

# Save missing tiles to a file
with open("missing_micronesia_tiles.txt", "w") as f:
    for tile_name in missing_micronesia_tiles:
        f.write(f"{tile_name}\n")

print("Missing Micronesia tiles have been saved to 'missing_micronesia_tiles.txt'.")
