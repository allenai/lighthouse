"""Process NOAA track data to add distance to coast information (single mode)."""

import logging
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Union

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
logger = logging.getLogger(__name__)


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
        logger.error(
            f"Error fetching distance for lat: {lat:.6f}, lon: {lon:.6f} - {e}"
        )
        return -1


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
    logger.info("Sending individual requests in parallel.")

    # Prepare arguments as tuples for starmap
    args = zip(lats, lons)
    with Pool(processes=cpu_count()) as p:
        distances = list(
            tqdm(
                p.starmap(get_distance_to_coast_single, args),
                total=n_points,
                desc="Processing individual points",
            )
        )

    df["distance_to_coast"] = distances
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Finished processing {file_path}. Output saved to {output_path}")


def main() -> None:
    """Process each CSV file sequentially."""

    input_files = list(input_directory.glob("*.csv"))

    for input_file in input_files:
        process_csv(
            str(input_file),
            str(output_directory / input_file.name),
        )


if __name__ == "__main__":
    main()
