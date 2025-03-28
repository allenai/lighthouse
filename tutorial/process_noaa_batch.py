"""Process NOAA track data to add distance to coast information."""

import logging
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, List, Union

import pandas as pd
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Define paths relative to script location
ROOT_DIR = Path(__file__).resolve().parent.parent
input_directory = ROOT_DIR / "data" / "tracks"
output_directory = ROOT_DIR / "data" / "track_with_d2c"

# Coastal Detection Service URL
COASTAL_DETECTION_URL = "http://0.0.0.0:8000/detect"


def get_distance_to_coast_single(lat: float, lon: float) -> Any:
    """Fetch distance to coast from the Coastal Detection Service for a single point.

    Args:
        lat: Latitude of the point
        lon: Longitude of the point

    Returns:
        Distance to coast in meters, or -1 if request fails
    """
    payload = {"lat": lat, "lon": lon}
    try:
        response = requests.post(COASTAL_DETECTION_URL, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get("distance_to_coast_m", -1)
    except requests.RequestException as e:
        logger.error(f"Error fetching distance for lat: {lat}, lon: {lon} - {e}")
        return -1


def get_distances_to_coast_batch(lats: List[float], lons: List[float]) -> List[float]:
    """Fetch distances to coast in batch from the Coastal Detection Service.

    Args:
        lats: List of latitudes
        lons: List of longitudes

    Returns:
        List of distances to coast in meters, or -1 for failed points
    """
    payload = {
        "batch_mode": True,
        "lat": lats,
        "lon": lons,
    }
    try:
        response = requests.post(COASTAL_DETECTION_URL, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        distances = [item["distance_to_coast_m"] for item in data]
        return distances
    except requests.RequestException as e:
        logger.error(f"Error fetching batch distances: {e}")
        return [-1] * len(lats)


def process_csv(file_path: Union[str, Path], output_path: Union[str, Path]) -> None:
    """Add distance_to_coast column to a CSV file.

    Args:
        file_path: Path to input CSV file
        output_path: Path to save processed CSV file
    """
    df = pd.read_csv(file_path)

    if "lat" not in df.columns or "lon" not in df.columns:
        logger.warning(f"Skipping {file_path}: Missing 'lat' or 'lon' columns")
        return

    lats = df["lat"].tolist()
    lons = df["lon"].tolist()
    n_points = len(lats)

    logger.info(f"Processing {Path(file_path).name} with {n_points} points...")

    if n_points < 1000:
        # Too small for batch mode, do single requests for each point
        logger.info("Under 1000 points, sending individual requests.")
        distances = []
        for lat, lon in tqdm(
            zip(lats, lons),
            total=n_points,
            desc="Processing individual points",
        ):
            dist = get_distance_to_coast_single(lat, lon)
            distances.append(dist)
    else:
        # Use batch mode
        logger.info(
            f"Sending batch request for {n_points} points from {Path(file_path).name}..."
        )
        distances = get_distances_to_coast_batch(lats, lons)

    df["distance_to_coast"] = distances
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Finished processing {file_path}. Output saved to {output_path}")


def main() -> None:
    """Process all CSV files in parallel."""

    input_files = list(input_directory.glob("*.csv"))

    # Create a list of tasks as tuples (input_path, output_path)
    tasks = [
        (str(input_file), str(output_directory / input_file.name))
        for input_file in input_files
    ]

    logger.info(f"Using {cpu_count()} CPU cores for processing")

    # Use multiprocessing to handle one file per core
    with Pool(processes=cpu_count()) as pool:
        pool.starmap(process_csv, tasks)


if __name__ == "__main__":
    main()
