#!/usr/bin/env python3

import re

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon


# Function to generate the tile grid
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


# Function to load missing tiles list
def load_missing_tiles(filename):
    with open(filename, "r") as f:
        missing_tiles_list = [line.strip() for line in f if line.strip()]
    return missing_tiles_list


# Function to load existing tiles list
def load_existing_tiles(filename):
    with open(filename, "r") as f:
        existing_tiles_list = [line.strip() for line in f if line.strip()]
    return existing_tiles_list


def main():
    # Generate the tile grid
    print("Generating tile grid...")
    tiles_gdf = generate_tile_grid()

    # Load missing tiles
    print("Loading missing tiles...")
    missing_tiles_list = load_missing_tiles("missing_land_tiles.txt")

    # Load existing tiles
    print("Loading existing tiles...")
    existing_tiles_list = load_existing_tiles("list_of_files.txt")

    # Create GeoDataFrames for existing and missing tiles
    missing_tiles_gdf = tiles_gdf[tiles_gdf["tile_name"].isin(missing_tiles_list)]
    existing_tiles_gdf = tiles_gdf[tiles_gdf["tile_name"].isin(existing_tiles_list)]

    # Plotting
    fig = plt.figure(figsize=(15, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Add base map features
    ax.coastlines()
    ax.stock_img()
    ax.set_global()

    # Plot existing tiles in green
    existing_tiles_gdf.plot(
        ax=ax,
        facecolor="green",
        edgecolor="black",
        alpha=0.5,
        transform=ccrs.PlateCarree(),
        label="Existing Tiles",
    )

    # Plot missing tiles in red
    missing_tiles_gdf.plot(
        ax=ax,
        facecolor="red",
        edgecolor="black",
        alpha=0.5,
        transform=ccrs.PlateCarree(),
        label="Missing Tiles",
    )

    # Add legend
    import matplotlib.patches as mpatches

    existing_patch = mpatches.Patch(color="green", alpha=0.5, label="Existing Tiles")
    missing_patch = mpatches.Patch(color="red", alpha=0.5, label="Missing Tiles")
    plt.legend(handles=[existing_patch, missing_patch], loc="lower left")

    plt.title("Visualization of Existing and Missing Tiles Covering Land Areas")
    plt.show()


if __name__ == "__main__":
    main()
