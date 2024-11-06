#!/usr/bin/env python3

import re
import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import matplotlib.patches as mpatches

# Function to parse tile filename and extract starting latitude and longitude
def parse_tile_filename(filename):
    """
    Parses the tile filename to extract latitude and longitude.

    Returns:
        (lat, lon) tuple if parsing is successful, None otherwise.
    """
    pattern = r"([NS])(\d{2})([EW])(\d{3})_Map\.tif"
    match = re.search(pattern, filename)
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
        print(f"Could not parse tile filename: {filename}")
        return None

# Function to create a polygon for a tile
def create_tile_polygon(lat, lon):
    # Each tile is 1x1 degrees
    lat_min = lat
    lat_max = lat + 1
    lon_min = lon
    lon_max = lon + 1

    # Define the corners of the tile
    corners = [
        (lon_min, lat_min),
        (lon_max, lat_min),
        (lon_max, lat_max),
        (lon_min, lat_max),
        (lon_min, lat_min),
    ]
    return Polygon(corners)

# Read existing tiles
with open("src/list_of_files.txt", "r") as f:
    existing_tiles = [line.strip() for line in f if line.strip()]

# Read missing tiles
with open("src/list_of_files_2.txt", "r") as f:
    missing_tiles = [line.strip() for line in f if line.strip()]

print(f"Number of existing tiles: {len(existing_tiles)}")
print(f"Number of missing tiles: {len(missing_tiles)}")

# Create geometries for existing tiles
existing_geometries = []
for tile in existing_tiles:
    coords = parse_tile_filename(tile)
    if coords:
        lat, lon = coords
        polygon = create_tile_polygon(lat, lon)
        existing_geometries.append(polygon)
    else:
        print(f"Could not parse tile filename: {tile}")

# Create geometries for missing tiles
missing_geometries = []
for tile in missing_tiles:
    coords = parse_tile_filename(tile)
    if coords:
        lat, lon = coords
        polygon = create_tile_polygon(lat, lon)
        missing_geometries.append(polygon)
    else:
        print(f"Could not parse tile filename: {tile}")

print(f"Number of existing geometries: {len(existing_geometries)}")
print(f"Number of missing geometries: {len(missing_geometries)}")

if not existing_geometries and not missing_geometries:
    print("No tiles to plot.")
    exit()

# Create GeoDataFrames with 'geometry' column specified
existing_gdf = gpd.GeoDataFrame({"geometry": existing_geometries}, crs="EPSG:4326")
missing_gdf = gpd.GeoDataFrame({"geometry": missing_geometries}, crs="EPSG:4326")

# Plotting
fig = plt.figure(figsize=(15, 8))
ax = plt.axes(projection=ccrs.PlateCarree())

# Add base map
ax.coastlines()
ax.stock_img()
ax.set_global()

# Plot existing tiles in green
if not existing_gdf.empty:
    existing_gdf.plot(
        ax=ax,
        facecolor="green",
        edgecolor="black",
        alpha=0.5,
        transform=ccrs.PlateCarree(),
        label="Existing Tiles",
    )
else:
    print("No existing tiles to plot.")

# Plot missing tiles in red
if not missing_gdf.empty:
    missing_gdf.plot(
        ax=ax,
        facecolor="red",
        edgecolor="black",
        alpha=0.5,
        transform=ccrs.PlateCarree(),
        label="Missing Tiles",
    )
else:
    print("No missing tiles to plot.")

# Add legend
existing_patch = mpatches.Patch(color="green", alpha=0.5, label="Existing Tiles")
missing_patch = mpatches.Patch(color="red", alpha=0.5, label="Missing Tiles")
plt.legend(handles=[existing_patch, missing_patch], loc="lower left")

plt.title("Visualization of Existing and Missing Tiles")
plt.show()
a