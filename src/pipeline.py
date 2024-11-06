import os
import math
import numpy as np
import h5py
import joblib
from sklearn.neighbors import BallTree
from pathlib import Path
from multiprocessing import cpu_count
import time
import rasterio.transform

# Global variable
coastal_ball_tree = None

# WorldCover class ID to classification mapping
land_water_mapping = {
    1: 'land', 10: 'Tree cover', 20: 'Shrubland', 30: 'Grassland',
    40: 'Cropland', 50: 'Built-up', 60: 'Bare/sparse vegetation',
    70: 'Snow and Ice', 80: 'Permanent water bodies', 90: 'Herbaceous wetland',
    95: 'Mangroves', 100: 'Moss and lichen',
}

def initialize_coastal_ball_tree():
    """Loads the global coastal ball tree if not already loaded."""
    global coastal_ball_tree
    if coastal_ball_tree is None:
        coastal_ball_tree = joblib.load('coastal_ball_tree.joblib')

def coord_to_coastal_point(lat, lon):
    """Finds the nearest coastal point and its distance from given coordinates."""
    global coastal_ball_tree
    initialize_coastal_ball_tree()
    point_rad = np.radians([lat, lon])
    distance_rad, index = coastal_ball_tree.query([point_rad], k=1)
    nearest_point_rad = coastal_ball_tree.data[index[0][0]]
    nearest_point = np.degrees(nearest_point_rad)
    distance_m = distance_rad[0][0] * 6371000.0  # Earth's radius in meters
    return nearest_point, distance_m

def get_filename_from_coordinates(lat, lon):
    """Generates filename and directory strings from lat/lon coordinates."""
    lat_dir = 'N' if lat >= 0 else 'S'
    lat_deg = int(abs(math.floor(lat)))
    lon_dir = 'E' if lon >= 0 else 'W'
    lon_deg = int(abs(math.floor(lon)))
    lat_str = f"{lat_dir}{lat_deg:02d}"
    lon_str = f"{lon_dir}{lon_deg:03d}"
    filename = f"Ai2_WorldCover_10m_2024_v1_{lat_str}{lon_str}_Map.h5"
    return filename, lat_str, lon_str

def get_ball_tree(lat_str, lon_str):
    """Loads a BallTree from a joblib file for a specific region."""
    ball_tree_filename = Path('data/ball_trees') / f"Ai2_WorldCover_10m_2024_v1_{lat_str}{lon_str}_Map_coastal_points_ball_tree.joblib"
    if ball_tree_filename.exists():
        return joblib.load(ball_tree_filename)
    else:
        raise FileNotFoundError(f"No coastal data found for tile {lat_str}{lon_str}")

def h5_to_integer(filename, lon, lat):
    """Retrieves land-water classification for given coordinates from an HDF5 file."""
    filename = Path('data/h5s') / filename
    with h5py.File(filename, 'r') as hdf:
        band_data = hdf['band_data']
        geotransform = hdf['geotransform'][:]

        # Convert geographic coordinates to image coordinates (row, col)
        row, col = ~rasterio.transform.Affine(*geotransform) * (lon, lat)
        return band_data[int(row), int(col)]

def ball_tree_distance(ball_tree, point):
    """Calculates distance from a point to the nearest coastal point in a BallTree."""
    point_rad = np.radians(point)
    distance_rad, index = ball_tree.query([point_rad], k=1)
    nearest_point_rad = ball_tree.data[index[0][0]]
    nearest_point = np.degrees(nearest_point_rad)
    distance_m = distance_rad[0][0] * 6371000.0
    return distance_m, nearest_point

def main(lat, lon):
    """Main function to get nearest coastal point, distance, and land-water classification."""
    filename_h5, lat_str, lon_str = get_filename_from_coordinates(lat, lon)
    land_water = h5_to_integer(filename_h5, lon, lat)

    try:
        tile_ball_tree = get_ball_tree(lat_str, lon_str)
        distance_m, nearest_point = ball_tree_distance(tile_ball_tree, [lat, lon])
    except FileNotFoundError:
        nearest_point, distance_m = coord_to_coastal_point(lat, lon)

    return distance_m, land_water, nearest_point

if __name__ == "__main__":

    latitude = 47.642492
    longitude = -122.336008

    # Measure performance of the first and second runs
    start = time.perf_counter()
    distance_m, land_or_water, nearest_point = main(latitude, longitude)

    start = time.perf_counter()
    distance_m, land_or_water, nearest_point = main(latitude, longitude)
    print(f"{distance_m=} meters to coast, land_cover_class={land_water_mapping[land_or_water]}, {nearest_point=}")
    print(f"Second run time: {(time.perf_counter() - start) * 1000:.2f} ms")
