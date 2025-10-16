"""
Pipeline module for coastal distance and land cover classification.

This module provides functionality to:
1. Find the nearest coastal point to any given coordinates
2. Determine land cover classification for a location. These are provided by either
   ESA or OSM.
3. Calculate distances to coastlines using BallTree structures
In batch mode, the pipeline can process multiple queries at once.
"""

import json
import logging
import os
import time
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import joblib
import numpy as np
import pandas as pd
import rasterio.transform
from numpy.typing import NDArray
from sklearn.neighbors import BallTree

from metrics import time_operation, TimerOperations

# Configure logging
logger = logging.getLogger(__name__)

# Global variables
coastal_ball_tree: Optional[BallTree] = None
ball_tree_cache: Dict[str, BallTree] = {}  # Cache for Ball Trees

# Load the saved bounds dictionary
DATA_DIR = Path(__file__).parent / "data"
bounds_file = DATA_DIR / "bounds_dictionary.json"

BALL_TREE_CACHE_SIZE = int(os.environ.get("BALL_TREE_CACHE_SIZE", 5))

try:
    with open(bounds_file, "r") as f:
        bounds_dict = json.load(f)
except FileNotFoundError:
    logger.error(f"Could not find bounds dictionary at {bounds_file}")
    raise
except json.JSONDecodeError:
    logger.error(f"Could not parse bounds dictionary at {bounds_file}")
    raise


class LandCoverClass(StrEnum):
    PermanentWaterBodies = "Permanent water bodies"
    Land = "Land"
    TreeCover = "Tree cover"
    Shrubland = "Shrubland"
    Grassland = "Grassland"
    Cropland = "Cropland"
    BuiltUp = "Built-up"
    BareSparseVegetation = "Bare/sparse vegetation"
    SnowAndIce = "Snow and Ice"
    HerbaceousWetland = "Herbaceous wetland"
    Mangroves = "Mangroves"
    MossAndLichen = "Moss and lichen"
    Unknown = "Unknown"


# WorldCover class ID to classification mapping
land_water_mapping = {
    0: LandCoverClass.PermanentWaterBodies,
    1: LandCoverClass.Land,
    10: LandCoverClass.TreeCover,
    20: LandCoverClass.Shrubland,
    30: LandCoverClass.Grassland,
    40: LandCoverClass.Cropland,
    50: LandCoverClass.BuiltUp,
    60: LandCoverClass.BareSparseVegetation,
    70: LandCoverClass.SnowAndIce,
    80: LandCoverClass.PermanentWaterBodies,
    90: LandCoverClass.HerbaceousWetland,
    95: LandCoverClass.Mangroves,
    100: LandCoverClass.MossAndLichen,
}


def initialize_coastal_ball_tree() -> BallTree:
    """Load and initialize the global coastal BallTree."""
    global coastal_ball_tree
    if coastal_ball_tree is None:
        ball_tree_path = DATA_DIR / "coastal_ball_tree.joblib"
        try:
            coastal_ball_tree = joblib.load(str(ball_tree_path))
        except FileNotFoundError:
            logger.error(f"Could not find coastal ball tree at {ball_tree_path}")
            raise
    return coastal_ball_tree


def coord_to_coastal_point(lat: float, lon: float) -> Tuple[float, NDArray[np.float64]]:
    """Find nearest coastal point and distance from coordinates."""
    tree: BallTree = initialize_coastal_ball_tree()
    return ball_tree_distance(tree, [lat, lon])


def get_filename_for_coordinates(
    lat: float, lon: float, bounds_dict: Dict[str, Dict[str, float]]
) -> Optional[str]:
    """
    Get the filename of the HDF5 file containing the given coordinates.
    Adds a small buffer to bounds to handle edge cases.
    """
    boundary_buffer = 1e-5

    for filename, bounds in bounds_dict.items():
        latmin, latmax = (
            bounds["latmin"] - boundary_buffer,
            bounds["latmax"] + boundary_buffer,
        )
        lonmin, lonmax = (
            bounds["lonmin"] - boundary_buffer,
            bounds["lonmax"] + boundary_buffer,
        )
        if latmin <= lat <= latmax and lonmin <= lon <= lonmax:
            return filename
    return None


def get_filename_for_coordinates_vectorized(
    lats: np.ndarray, lons: np.ndarray, bounds_dict: Dict[str, Dict[str, float]]
) -> List[Optional[str]]:
    """gets file name in vectorized way for batch processing."""
    # This can be optimized with spatial indexing.
    # For now, just loop. Still better than calling repeatedly if we have many points.
    results = []
    for lat, lon in zip(lats, lons):
        results.append(get_filename_for_coordinates(lat, lon, bounds_dict))
    return results


@lru_cache(maxsize=BALL_TREE_CACHE_SIZE)
def get_ball_tree(filename_ball_tree: str) -> BallTree:
    """Load a BallTree from a joblib file for a specific region."""
    filename = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "ball_trees"
        / filename_ball_tree
    )
    with time_operation(TimerOperations.GetBallTree):
        if filename.exists():
            tile_ball_tree = joblib.load(str(filename))
            return tile_ball_tree
        else:
            raise FileNotFoundError(f"No coastal data found for tile {filename}")


def h5_to_landcover(
    filename: str,
    lats: np.ndarray,
    lons: np.ndarray,
) -> list[LandCoverClass]:
    """Get land-water classification for coordinates from HDF5 file."""
    filepath = (
        Path(__file__).resolve().parent.parent / "data" / "resampled_h5s" / filename
    )
    with h5py.File(str(filepath), "r") as hdf:
        band_data = hdf["band_data"]
        geotransform = hdf["geotransform"][:]

        rows, cols = ~rasterio.transform.Affine(*geotransform) * (lons, lats)
        # Convert to integer indices for lookup in hdf5 file.
        # Clamp to valid range [0, 11999] to handle boundary conditions
        row_int = np.clip(np.round(rows).astype(int), 0, 11999)
        col_int = np.clip(np.round(cols).astype(int), 0, 11999)

        class_ids = [band_data[c, r] for r, c in zip(row_int, col_int)]
        return [
            land_water_mapping.get(class_id, LandCoverClass.Unknown)
            for class_id in class_ids
        ]


def ball_tree_distance(
    ball_tree: BallTree, point: List[float]
) -> Tuple[float, NDArray[np.float64]]:
    """Calculate distance from point to nearest coastal point in BallTree."""
    with time_operation(TimerOperations.LookupNearestCoast):
        point_rad = np.radians(point)
        distance_rad, index = ball_tree.query([point_rad], k=1)
        nearest_point_rad = ball_tree.data[index[0][0]]
        nearest_point = np.degrees(nearest_point_rad)
        distance_m = distance_rad[0][0] * 6371000.0
    return distance_m, nearest_point


def ball_tree_distance_batch(
    tree: BallTree, lats: np.ndarray, lons: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    with time_operation(TimerOperations.LookupNearestCoast):
        points_rad = np.radians(np.column_stack((lats, lons)))
        distance_rad, indices = tree.query(points_rad, k=1)
        distances_m = distance_rad.ravel() * 6371000.0
        indices_int = indices.ravel().astype(int)
        tile_ball_tree_data = np.asarray(tree.data)
        nearest_points_rad = tile_ball_tree_data[indices_int]
        nearest_points = np.degrees(nearest_points_rad)
    return distances_m, nearest_points


def validate_coordinates(lat: float, lon: float) -> None:
    """Validate lat/lon."""
    if not -90 <= lat <= 90:
        raise ValueError(f"Latitude {lat} is outside valid range [-90, 90]")
    if not -180 <= lon <= 180:
        raise ValueError(f"Longitude {lon} is outside valid range [-180, 180]")


def process_batch(
    tile_file: Optional[str],
    balltree_file: Optional[str],
    lats: np.ndarray,
    lons: np.ndarray,
) -> Tuple[np.ndarray, list[LandCoverClass], np.ndarray]:
    if tile_file is None:
        # Ocean case
        tree = initialize_coastal_ball_tree()
        distances_m, nearest_points = ball_tree_distance_batch(tree, lats, lons)
        # Explicitly handle ocean case
        land_classes = np.full_like(
            distances_m, LandCoverClass.PermanentWaterBodies, dtype=int
        ).tolist()
        return distances_m, land_classes, nearest_points

    # Load HDF5 data
    with time_operation(TimerOperations.LoadH5File):
        land_classes = h5_to_landcover(tile_file, lats, lons)

    # BallTree query
    if balltree_file is not None:
        try:
            tree = get_ball_tree(balltree_file)
        except FileNotFoundError:
            # Fallback to global coastal ball tree
            tree = initialize_coastal_ball_tree()
    else:
        # No tile-specific balltree
        tree = initialize_coastal_ball_tree()

    distances_m, nearest_points = ball_tree_distance_batch(tree, lats, lons)
    return distances_m, land_classes, nearest_points


def batch_main(lat: np.ndarray, lon: np.ndarray) -> pd.DataFrame:
    # Batch mode
    lats = np.asarray(lat)
    lons = np.asarray(lon)

    # Validate coordinates
    for la, lo in zip(lats, lons):
        validate_coordinates(la, lo)

    # Get filenames in bulk
    filenames_h5 = get_filename_for_coordinates_vectorized(lats, lons, bounds_dict)
    balltree_files: List[Optional[str]] = []
    for fn in filenames_h5:
        if fn is not None:
            balltree_file = fn.replace(".h5", "_coastal_points_ball_tree.joblib")
            balltree_file = balltree_file.replace("resampled_h5s", "ball_trees")
            balltree_files.append(balltree_file)
        else:
            # ocean fallback
            balltree_files.append(None)

    df = pd.DataFrame(
        {
            "lat": lats,
            "lon": lons,
            "tile_file": filenames_h5,
            "balltree_file": balltree_files,
        }
    )

    # Separate rows with tile_file == None (ocean)
    ocean_df = df[df["tile_file"].isna()]
    tile_df = df[~df["tile_file"].isna()]

    results = []

    # Process batch queries for tiles
    if not tile_df.empty:
        with time_operation(TimerOperations.BatchLandTileLookups):
            grouped = tile_df.groupby(["tile_file", "balltree_file"])
            for (tile_file, balltree_file), group in grouped:
                dists, lc, npnts = process_batch(
                    tile_file, balltree_file, group["lat"].values, group["lon"].values
                )
                result_df = pd.DataFrame(
                    {
                        "distance_m": dists,
                        "land_class": lc,
                        "nearest_lat": npnts[:, 0],
                        "nearest_lon": npnts[:, 1],
                    },
                    index=group.index,
                )
                results.append(result_df)

    # Process ocean points individually
    if not ocean_df.empty:
        # We'll loop through each coordinate and handle them one by one
        dist_list: list[float] = []
        lc_list: list[LandCoverClass] = []
        nearest_lat_list = []
        nearest_lon_list = []

        for idx, row in ocean_df.iterrows():
            la, lo = row["lat"], row["lon"]
            distance_m, nearest_point = coord_to_coastal_point(la, lo)
            # Ocean => land_class = 0
            dist_list.append(distance_m)
            lc_list.append(LandCoverClass.PermanentWaterBodies)
            nearest_lat_list.append(nearest_point[0])
            nearest_lon_list.append(nearest_point[1])

        ocean_result_df = pd.DataFrame(
            {
                "distance_m": dist_list,
                "land_class": lc_list,
                "nearest_lat": nearest_lat_list,
                "nearest_lon": nearest_lon_list,
            },
            index=ocean_df.index,
        )
        results.append(ocean_result_df)

    if results:
        final_results = pd.concat(results).sort_index()
    else:
        # No results at all
        final_results = pd.DataFrame()

    return final_results


def main(lat: float, lon: float) -> tuple[float, LandCoverClass, NDArray[np.float64]]:
    # Single point logic unchanged
    validate_coordinates(lat, lon)
    filename_h5 = get_filename_for_coordinates(lat, lon, bounds_dict)
    if filename_h5:
        land_class = h5_to_landcover(filename_h5, np.array([lat]), np.array([lon]))[0]
        ball_tree_suffix = "_coastal_points_ball_tree.joblib"
        filename_ball_tree = filename_h5.replace(".h5", ball_tree_suffix)
        filename_ball_tree = filename_ball_tree.replace("resampled_h5s", "ball_trees")
        try:
            tile_ball_tree = get_ball_tree(filename_ball_tree)
            distance_m, nearest_point = ball_tree_distance(tile_ball_tree, [lat, lon])
            return distance_m, land_class, nearest_point
        except FileNotFoundError:
            distance_m, nearest_point = coord_to_coastal_point(lat, lon)
            new_filename_h5 = get_filename_for_coordinates(
                nearest_point[0], nearest_point[1], bounds_dict
            )
            if new_filename_h5 is None:
                # No tile for nearest coastal point, ocean fallback
                return distance_m, LandCoverClass.PermanentWaterBodies, nearest_point
            filename_ball_tree = new_filename_h5.replace(".h5", ball_tree_suffix)
            tile_ball_tree = get_ball_tree(filename_ball_tree)
            distance_m, nearest_point = ball_tree_distance(tile_ball_tree, [lat, lon])
            return distance_m, land_class, nearest_point
    else:
        # Ocean fallback for single point
        distance_m, nearest_point = coord_to_coastal_point(lat, lon)
        return distance_m, LandCoverClass.PermanentWaterBodies, nearest_point


if __name__ == "__main__":
    # Example usage:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Single point usage:
    latitude = 47.636895
    longitude = -122.334984
    start = time.perf_counter()
    distance_m, land_or_water, nearest_point = main(latitude, longitude)
    logger.info(
        f"Single query result: {distance_m} meters to coast, "
        f"land cover class: {land_or_water}, "
        f"nearest coastal point: {nearest_point}"
    )
    logger.info("Processing time: %f seconds", time.perf_counter() - start)

    # Batch usage:
    # Suppose we have a list of coordinates
    coords = [
        (47.636895, -122.334984),
        (47.637000, -122.335000),
        (0.0, 0.0),
    ]  # Just as an example
    lats = np.array([c[0] for c in coords])
    lons = np.array([c[1] for c in coords])

    start = time.perf_counter()
    batch_results = batch_main(lats, lons)
    logger.info("Batch query results:\n%s", batch_results)
    logger.info("Batch processing time: %f seconds", time.perf_counter() - start)
