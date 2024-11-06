import h5py
import rasterio.transform
import math
import numpy as np
from sklearn.neighbors import BallTree
import joblib
from pathlib import Path
import time

# Global variables
coastal_ball_tree = None

# WorldCover class ID to classification mapping
land_water_mapping = {
    1: 'land',
    10: 'Tree cover',
    20: 'Shrubland',
    30: 'Grassland',
    40: 'Cropland',
    50: 'Built-up',
    60: 'Bare/sparse vegetation',
    70: 'Snow and Ice',
    80: 'Permanent water bodies',
    90: 'Herbaceous wetland',
    95: 'Mangroves',
    100: 'Moss and lichen',
}

def initialize_coastal_ball_tree():
    start_time = time.perf_counter()
    global coastal_ball_tree
    coastal_ball_tree = joblib.load('coastal_ball_tree.joblib')
    end_time = time.perf_counter()
    print(f"initialize_coastal_ball_tree took {end_time - start_time:.4f} seconds")

def coord_to_coastal_point(lat, lon):
    start_time = time.perf_counter()
    global coastal_ball_tree
    if coastal_ball_tree is None:
        initialize_coastal_ball_tree()
    point_rad = np.radians([lat, lon])
    distance_rad, index = coastal_ball_tree.query([point_rad], k=1)
    nearest_point_rad = coastal_ball_tree.data[index[0][0]]
    nearest_point = np.degrees(nearest_point_rad)
    distance_m = distance_rad[0][0] * 6371000.0  # Earth's radius in meters
    end_time = time.perf_counter()
    print(f"coord_to_coastal_point took {end_time - start_time:.4f} seconds")
    return nearest_point, distance_m

def get_filename_from_coordinates(lat, lon):
    start_time = time.perf_counter()
    lat_dir = 'N' if lat >= 0 else 'S'
    lat_deg = int(abs(math.floor(lat)))
    lon_dir = 'E' if lon >= 0 else 'W'
    lon_deg = int(abs(math.floor(lon)))
    lat_str = f"{lat_dir}{lat_deg:02d}"
    lon_str = f"{lon_dir}{lon_deg:03d}"
    filename = f"Ai2_WorldCover_10m_2024_v1_{lat_str}{lon_str}_Map.h5"
    end_time = time.perf_counter()
    print(f"get_filename_from_coordinates took {end_time - start_time:.4f} seconds")
    return filename, lat_str, lon_str

def get_ball_tree(lat_str, lon_str):
    start_time = time.perf_counter()
    ball_tree_filename = Path('data/ball_trees') / f"Ai2_WorldCover_10m_2024_v1_{lat_str}{lon_str}_Map_coastal_points_ball_tree.joblib"
    if ball_tree_filename.exists():
        tile_ball_tree = joblib.load(ball_tree_filename)
        end_time = time.perf_counter()
        print(f"get_ball_tree took {end_time - start_time:.4f} seconds")
        return tile_ball_tree
    else:
        end_time = time.perf_counter()
        print(f"get_ball_tree (file not found) took {end_time - start_time:.4f} seconds")
        raise FileNotFoundError(f"No coastal data found for tile {lat_str}{lon_str}")

def h5_to_integer(filename, lon, lat):
    start_time = time.perf_counter()
    filename = Path('data/h5s') / filename
    with h5py.File(filename, 'r') as hdf:
        band_data = hdf['band_data'][:]
        geotransform = hdf['geotransform'][:]
        transform = rasterio.transform.Affine(*geotransform)
        col, row = ~transform * (lon, lat)
        row, col = int(row), int(col)
        if 0 <= row < band_data.shape[0] and 0 <= col < band_data.shape[1]:
            class_id = band_data[row, col]
            land_water = land_water_mapping.get(class_id, 'unknown')
        else:
            land_water = 'out_of_bounds'
    end_time = time.perf_counter()
    print(f"h5_to_integer took {end_time - start_time:.4f} seconds")
    return land_water

def ball_tree_distance(ball_tree, point):
    start_time = time.perf_counter()
    point_rad = np.radians(point)
    distance_rad, index = ball_tree.query([point_rad], k=1)
    nearest_point_rad = ball_tree.data[index[0][0]]
    nearest_point = np.degrees(nearest_point_rad)
    distance_m = distance_rad[0][0] * 6371000.0
    end_time = time.perf_counter()
    print(f"ball_tree_distance took {end_time - start_time:.4f} seconds")
    return distance_m, nearest_point

def main(lat, lon):
    start_time = time.perf_counter()
    filename_h5, lat_str, lon_str = get_filename_from_coordinates(lat, lon)
    land_water = h5_to_integer(filename_h5, lon, lat)
    try:
        tile_ball_tree = get_ball_tree(lat_str, lon_str)
        distance_m, nearest_point = ball_tree_distance(tile_ball_tree, [lat, lon])
    except FileNotFoundError:
        nearest_point, distance_m = coord_to_coastal_point(lat, lon)
    end_time = time.perf_counter()
    print(f"main function took {end_time - start_time:.4f} seconds")
    return distance_m, land_water, nearest_point

# Example usage:
if __name__ == "__main__":

    latitude = 47.174229
    longitude =  -129.808710

    print("First run:")
    start = time.perf_counter()
    distance_m, land_or_water, nearest_point = main(latitude, longitude)
    print(f"Total time for first run: {time.perf_counter() - start:.4f} seconds\n")

    print("Second run:")
    start = time.perf_counter()
    distance_m, land_or_water, nearest_point = main(latitude, longitude)
    print(f"Total time for second run: {time.perf_counter() - start:.4f} seconds")
