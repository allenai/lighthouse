import os
import requests
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import logging
from typing import Dict, List, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)

input_directory = "/home/patrickb/litus/tracks"
output_directory = "/home/patrickb/litus/track_with_d2c"

# Coastal Detection Service URL
COASTAL_DETECTION_URL = "http://0.0.0.0:8000/detect"


def get_distance_to_coast_single(lat, lon):
    """Fetch distance to coast from the Coastal Detection Service for a single point."""
    payload = {"lat": lat, "lon": lon}
    try:
        response = requests.post(COASTAL_DETECTION_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("distance_to_coast_m", -1)
    except requests.RequestException as e:
        logger.info(f"Error fetching distance for lat: {lat}, lon: {lon} - {e}")
        return -1


def get_distances_to_coast_batch(lats, lons):
    """Fetch distances to coast in batch from the Coastal Detection Service."""
    payload = {
        "batch_mode": True,
        "lat": lats,
        "lon": lons,
    }
    try:
        response = requests.post(COASTAL_DETECTION_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        distances = [item["distance_to_coast_m"] for item in data]
        return distances
    except requests.RequestException as e:
        logger.info(f"Error fetching batch distances: {e}")
        return [-1] * len(lats)


def process_csv(file_path: str, output_path: str):
    """Add distance_to_coast column to a CSV file."""
    df = pd.read_csv(file_path)

    if "lat" not in df.columns or "lon" not in df.columns:
        logger.info(f"Skipping {file_path}: Missing 'lat' or 'lon' columns.")
        return

    lats = df["lat"].tolist()
    lons = df["lon"].tolist()
    n_points = len(lats)

    logger.info(f"Processing {os.path.basename(file_path)} with {n_points} points...")

    if n_points < 1000:
        # Too small for batch mode, do single requests for each point
        logger.info("Under 1000 points, sending individual requests.")
        distances = []
        for lat, lon in tqdm(
            zip(lats, lons), total=n_points, desc="Processing individual points"
        ):
            dist = get_distance_to_coast_single(lat, lon)
            distances.append(dist)
    else:
        # Use batch mode
        logger.info(
            f"Sending batch request for {n_points} points from {os.path.basename(file_path)}..."
        )
        distances = get_distances_to_coast_batch(lats, lons)

    df["distance_to_coast"] = distances
    os.makedirs(output_directory, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Finished processing {file_path}. Output saved to {output_path}.")


def process_noaa_batch(
    vessel_ids: List[str],
    data: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """Process multiple NOAA vessel tracks in batch.

    Args:
        vessel_ids: List of vessel identifiers to process
        data: DataFrame containing track data for multiple vessels

    Returns:
        Dictionary mapping vessel IDs to their processed track DataFrames
    """
    # ... rest of function implementation


def split_tracks(
    data: pd.DataFrame,
    time_threshold: Optional[float] = None,
) -> List[pd.DataFrame]:
    """Split vessel tracks based on time gaps.

    Args:
        data: DataFrame containing track data
        time_threshold: Optional threshold in hours for splitting tracks

    Returns:
        List of DataFrames, each containing a separate track segment
    """
    # ... rest of function implementation


def aggregate_results(
    track_results: List[Dict[str, Union[np.ndarray, float]]],
) -> pd.DataFrame:
    """Aggregate results from multiple track processings.

    Args:
        track_results: List of dictionaries containing track processing results

    Returns:
        DataFrame containing aggregated results for all tracks
    """
    # ... rest of function implementation


if __name__ == "__main__":
    # Collect all CSV files
    input_files = [f for f in os.listdir(input_directory) if f.endswith(".csv")]

    # Create a list of tasks as tuples (input_path, output_path)
    tasks = [
        (
            os.path.join(input_directory, file_name),
            os.path.join(output_directory, file_name),
        )
        for file_name in input_files
    ]
    logger.info(cpu_count())
    # Use multiprocessing to handle one file per core
    with Pool(processes=cpu_count()) as pool:
        pool.starmap(process_csv, tasks)
