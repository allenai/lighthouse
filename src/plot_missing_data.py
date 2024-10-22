#!/usr/bin/env python3

import os
import re

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

# Create the 'plots' directory if it doesn't exist
os.makedirs("plots", exist_ok=True)

# Load land polygons
land_polygons_gdf = gpd.read_file("land-polygons-split-4326/land_polygons.shp")
land_polygons_gdf = land_polygons_gdf.to_crs("EPSG:4326")

# Load missing tiles
with open("missing_land_tiles.txt", "r") as f:
    missing_tiles_list = [line.strip() for line in f if line.strip()]


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


def create_tile_polygon(lat, lon):
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
    return tile_polygon


for tile_name in missing_tiles_list:
    coords = parse_tile_filename(tile_name)
    if coords:
        lat, lon = coords
        tile_polygon = create_tile_polygon(lat, lon)

        # Create a GeoDataFrame for the tile
        tile_gdf = gpd.GeoDataFrame({"geometry": [tile_polygon]}, crs="EPSG:4326")

        # Find land polygons within the tile
        land_in_tile = land_polygons_gdf[land_polygons_gdf.intersects(tile_polygon)]

        # Skip tiles with no land
        if land_in_tile.empty:
            print(f"Tile {tile_name} does not contain any land polygons.")
            continue

        # Plot the tile and land within it
        fig, ax = plt.subplots(
            figsize=(8, 8), subplot_kw={"projection": ccrs.PlateCarree()}
        )
        ax.set_extent(tile_polygon.bounds, crs=ccrs.PlateCarree())

        # Plot the tile boundary
        tile_gdf.boundary.plot(
            ax=ax, edgecolor="red", linewidth=2, label="Missing Tile"
        )

        # Plot land polygons within the tile
        land_in_tile.plot(
            ax=ax, color="lightgreen", edgecolor="black", alpha=0.7, label="Land Areas"
        )

        # Add gridlines and labels
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5)
        gl.top_labels = gl.right_labels = False

        # Title and legend
        ax.set_title(f"Tile {tile_name} and Land Coverage")
        ax.legend(loc="lower left")

        # Show or save the plot
        # To display the plot
        plt.show()
        # To save the plot instead, uncomment the following lines and comment out plt.show()
        # output_filename = f"plots/{tile_name.replace('/', '_')}.png"
        # plt.savefig(output_filename, dpi=300)
        # plt.close(fig)
    else:
        print(f"Could not parse tile name: {tile_name}")
