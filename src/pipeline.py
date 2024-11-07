import json
import logging
import math
import os
import time
from pathlib import Path

import h5py
import joblib
import numpy as np
import rasterio.transform
from sklearn.neighbors import BallTree

# Global variables
coastal_ball_tree = None
ball_tree_cache = {}  # Cache for Ball Trees

# Load the saved bounds dictionary
with open("bounds_dictionary.json", "r") as f:
    bounds_dict = json.load(f)

# WorldCover class ID to classification mapping
land_water_mapping = {
    0: "Permanent water bodies",
    1: "Land",
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare/sparse vegetation",
    70: "Snow and Ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}


def initialize_coastal_ball_tree():
    """Loads the global coastal ball tree if not already loaded."""
    global coastal_ball_tree
    if coastal_ball_tree is None:
        coastal_ball_tree = joblib.load("coastal_ball_tree.joblib")


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


def get_filename_for_coordinates(lat, lon, bounds_dict):
    """Retrieves the filename of the HDF5 file that contains the given coordinates."""
    for filename, bounds in bounds_dict.items():
        if (
            bounds["latmin"] <= lat < bounds["latmax"]
            and bounds["lonmin"] <= lon < bounds["lonmax"]
        ):
            print(f"Correct match found: {filename}")
            print(f"{bounds=}")
            return filename
    print(f"No matching file found for coordinates ({lat}, {lon}).")
    return None


def get_ball_tree(filename_ball_tree):
    """Loads a BallTree from a joblib file for a specific region, using caching to avoid reloading."""
    filename = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "ball_trees"
        / filename_ball_tree
    )
    if filename.exists():
        tile_ball_tree = joblib.load(filename)
        return tile_ball_tree
    else:
        raise FileNotFoundError(f"No coastal data found for tile {filename}")


def h5_to_integer(filename, lon, lat):
    """Retrieves land-water classification for given coordinates from an HDF5 file."""
    filename = (
        Path(__file__).resolve().parent.parent / "data" / "resampled_h5s" / filename
    )
    with h5py.File(filename, "r") as hdf:
        band_data = hdf["band_data"]
        geotransform = hdf["geotransform"][:]
        # Convert geographic coordinates to image coordinates (row, col)
        row, col = ~rasterio.transform.Affine(*geotransform) * (lon, lat)
        print(f"{row=}, {col=}")
        return band_data[int(col), int(row)]


def ball_tree_distance(ball_tree, point):
    """Calculates distance from a point to the nearest coastal point in a BallTree."""
    point_rad = np.radians(point)
    distance_rad, index = ball_tree.query([point_rad], k=1)
    nearest_point_rad = ball_tree.data[index[0][0]]
    nearest_point = np.degrees(nearest_point_rad)
    distance_m = distance_rad[0][0] * 6371000.0  # convert to meters
    return distance_m, nearest_point


def main(lat, lon):
    """Main function to handle two pathways based on whether the point falls within a tile or not."""

    # Pathway 1: Check if the point falls within an HDF5 tile
    filename_h5 = get_filename_for_coordinates(lat, lon, bounds_dict)

    if filename_h5:
        print("Pathway 1: Point is within a tile.")
        # Load HDF5 to get land/water classification
        land_class = h5_to_integer(filename_h5, lon, lat)

        # Get BallTree distance
        filename_ball_tree = filename_h5.replace(
            ".h5", "_coastal_points_ball_tree.joblib"
        )
        filename_ball_tree = filename_ball_tree.replace("resampled_h5s", "ball_trees")
        try:
            print(filename_ball_tree)
            tile_ball_tree = get_ball_tree(filename_ball_tree)
            distance_m, nearest_point = ball_tree_distance(tile_ball_tree, [lat, lon])
            return distance_m, land_class, nearest_point
        except FileNotFoundError:
            print(
                "Pathway 1: Point is within a tile. but there are no points near coast within this tile "
            )
            print("ball tree does not exist, tile is all land")
            nearest_point, distance_m = coord_to_coastal_point(lat, lon)
            filename_h5 = get_filename_for_coordinates(
                nearest_point[0], nearest_point[1], bounds_dict
            )
            filename_ball_tree = filename_h5.replace(
                ".h5", "_coastal_points_ball_tree.joblib"
            )
            tile_ball_tree = get_ball_tree(filename_ball_tree)
            distance_m, nearest_point = ball_tree_distance(tile_ball_tree, [lat, lon])
            return distance_m, land_class, nearest_point

    else:
        print("Pathway 2: Point is not within any tile, assuming it's in the ocean.")
        # Find nearest coastal point using the global coastal BallTree
        nearest_point, distance_m = coord_to_coastal_point(lat, lon)

        return distance_m, 0, nearest_point  # 0 represents water in mapping


if __name__ == "__main__":

    latitude = 47.636895
    longitude = -122.334984

    start = time.perf_counter()
    distance_m, land_or_water, nearest_point = main(latitude, longitude)
    print(
        f"Result: {distance_m} meters to coast, land cover class: {land_water_mapping[land_or_water]}, nearest coastal point: {nearest_point}"
    )
    print(f"{time.perf_counter() - start}")
