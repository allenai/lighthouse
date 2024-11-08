"""
Pipeline module for coastal distance and land cover classification.

This module provides functionality to:
1. Find the nearest coastal point to any given coordinates
2. Determine land cover classification for a location
3. Calculate distances to coastlines using BallTree structures
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import joblib
import numpy as np
import rasterio.transform
from numpy.typing import NDArray
from sklearn.neighbors import BallTree

# Configure logging
logger = logging.getLogger(__name__)

# Global variables
coastal_ball_tree: Optional[BallTree] = None
ball_tree_cache: Dict[str, BallTree] = {}  # Cache for Ball Trees

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


def initialize_coastal_ball_tree() -> BallTree:
    """Load and initialize the global coastal BallTree."""
    global coastal_ball_tree
    if coastal_ball_tree is None:
        coastal_ball_tree = joblib.load("coastal_ball_tree.joblib")
    return coastal_ball_tree


def coord_to_coastal_point(lat: float, lon: float) -> Tuple[NDArray[np.float64], float]:
    """Find nearest coastal point and distance from coordinates."""
    tree: BallTree = initialize_coastal_ball_tree()
    point_rad = np.radians([lat, lon])
    distance_rad, index = tree.query([point_rad], k=1)
    nearest_point_rad = tree.data[index[0][0]]
    nearest_point = np.degrees(nearest_point_rad)
    distance_m = distance_rad[0][0] * 6371000.0  # Earth's radius in meters
    return nearest_point, distance_m


def get_filename_for_coordinates(
    lat: float, lon: float, bounds_dict: Dict[str, Dict[str, float]]
) -> Optional[str]:
    """Get filename of HDF5 file containing the given coordinates."""
    for filename, bounds in bounds_dict.items():
        if (
            bounds["latmin"] <= lat < bounds["latmax"]
            and bounds["lonmin"] <= lon < bounds["lonmax"]
        ):
            logger.debug("Found matching file: %s with bounds %s", filename, bounds)
            return filename
    logger.debug("No matching file found for coordinates (%f, %f)", lat, lon)
    return None


def get_ball_tree(filename_ball_tree: str) -> BallTree:
    """Load a BallTree from a joblib file for a specific region."""
    filename = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "ball_trees"
        / filename_ball_tree
    )
    if filename.exists():
        tile_ball_tree = joblib.load(str(filename))
        return tile_ball_tree
    else:
        raise FileNotFoundError(f"No coastal data found for tile {filename}")


def h5_to_integer(filename: str, lon: float, lat: float) -> int:
    """Get land-water classification for coordinates from HDF5 file."""
    filepath = (
        Path(__file__).resolve().parent.parent / "data" / "resampled_h5s" / filename
    )
    with h5py.File(str(filepath), "r") as hdf:
        band_data = hdf["band_data"]
        geotransform = hdf["geotransform"][:]
        row, col = ~rasterio.transform.Affine(*geotransform) * (lon, lat)
        logger.debug("Image coordinates: row=%f, col=%f", row, col)
        return int(band_data[int(col), int(row)])


def ball_tree_distance(
    ball_tree: BallTree, point: List[float]
) -> Tuple[float, NDArray[np.float64]]:
    """Calculate distance from point to nearest coastal point in BallTree."""
    point_rad = np.radians(point)
    distance_rad, index = ball_tree.query([point_rad], k=1)
    nearest_point_rad = ball_tree.data[index[0][0]]
    nearest_point = np.degrees(nearest_point_rad)
    distance_m = distance_rad[0][0] * 6371000.0  # convert to meters
    return distance_m, nearest_point


def main(lat: float, lon: float) -> Tuple[float, int, NDArray[np.float64]]:
    """Process coordinates and return coastal information."""
    filename_h5 = get_filename_for_coordinates(lat, lon, bounds_dict)

    if filename_h5:
        logger.info("Processing point within tile")
        land_class = h5_to_integer(filename_h5, lon, lat)

        ball_tree_suffix = "_coastal_points_ball_tree.joblib"
        filename_ball_tree = filename_h5.replace(".h5", ball_tree_suffix)
        filename_ball_tree = filename_ball_tree.replace("resampled_h5s", "ball_trees")
        try:
            logger.debug("Using ball tree: %s", filename_ball_tree)
            tile_ball_tree = get_ball_tree(filename_ball_tree)
            distance_m, nearest_point = ball_tree_distance(tile_ball_tree, [lat, lon])
            return distance_m, land_class, nearest_point
        except FileNotFoundError:
            logger.info("Point is within tile but no coastal points nearby")
            logger.debug("Ball tree does not exist, tile is all land")
            nearest_point, distance_m = coord_to_coastal_point(lat, lon)
            new_filename_h5 = get_filename_for_coordinates(
                nearest_point[0], nearest_point[1], bounds_dict
            )
            if new_filename_h5 is None:
                raise ValueError("Could not find tile for nearest coastal point")
            filename_ball_tree = new_filename_h5.replace(".h5", ball_tree_suffix)
            tile_ball_tree = get_ball_tree(filename_ball_tree)
            distance_m, nearest_point = ball_tree_distance(tile_ball_tree, [lat, lon])
            return distance_m, land_class, nearest_point

    else:
        logger.info("Point is not within any tile (ocean)")
        nearest_point, distance_m = coord_to_coastal_point(lat, lon)
        return distance_m, 0, nearest_point  # 0 represents water


if __name__ == "__main__":
    # Configure logging for command line usage
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    latitude = 47.636895
    longitude = -122.334984

    start = time.perf_counter()
    distance_m, land_or_water, nearest_point = main(latitude, longitude)
    result = (
        f"Result: {distance_m} meters to coast, "
        f"land cover class: {land_water_mapping[land_or_water]}, "
        f"nearest coastal point: {nearest_point}"
    )
    logger.info(result)
    logger.info("Processing time: %f seconds", time.perf_counter() - start)
