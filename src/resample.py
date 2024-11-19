#!/usr/bin/env python3

import logging
import re
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Match, Optional, Pattern

import numpy as np
import rasterio
from numpy.typing import NDArray
from rasterio.windows import Window
from tqdm import tqdm

from utils.log_utils import configure_logging

# Configure logging
logger = logging.getLogger(__name__)

# Directory paths
ROOT_DIR = Path(__file__).resolve().parent.parent
input_dir = ROOT_DIR / "data" / "ESA_WORLDCOVER"
output_dir = ROOT_DIR / "data" / "resampled"
output_dir.mkdir(parents=True, exist_ok=True)
WATER_VALUE: int = 80

# Regular expression to extract latitude and longitude
pattern: Pattern[str] = re.compile(
    r"ESA_WorldCover_10m_2021_v200_([NS]\d{2})([EW]\d{3})_Map\.tif"
)


def split_and_resample(file_path: str) -> None:
    """Split a 3x3 GeoTIFF into nine 1x1 GeoTIFFs and save them.

    Args:
        file_path: Path to the input GeoTIFF file
    """
    file_name = Path(file_path).name
    match: Optional[Match[str]] = pattern.match(file_name)

    if not match:
        logger.warning("Filename does not match expected format: %s", file_name)
        return

    lat_prefix, lon_prefix = match.groups()
    lat = int(lat_prefix[1:]) * (1 if lat_prefix[0] == "N" else -1)
    lon = int(lon_prefix[1:]) * (1 if lon_prefix[0] == "E" else -1)

    with rasterio.open(file_path) as src:
        # Calculate pixel dimensions for each 1x1 degree tile
        width, height = src.width // 3, src.height // 3

        for i in range(3):
            for j in range(3):
                # Adjusted latitude calculation
                new_lat = lat + 2 - i
                new_lon = lon + j

                # Set up the window for this 1x1 tile
                window = Window(j * width, i * height, width, height)
                transform = src.window_transform(window)

                # Read the data to check for NoData or water-only tiles
                tile_data: NDArray = src.read(1, window=window)

                # Check if all data is NoData
                if src.nodata is not None and np.all(tile_data == src.nodata):
                    logger.debug(
                        "Tile %d, %d is all NoData. Skipping.",
                        new_lat,
                        new_lon,
                    )
                    continue

                # Check if all valid data is water
                valid_data: NDArray = tile_data[tile_data != src.nodata]
                if valid_data.size > 0 and np.all(valid_data == WATER_VALUE):
                    logger.debug(
                        "Tile %d, %d is all water. Skipping.",
                        new_lat,
                        new_lon,
                    )
                    continue

                # Create the new filename
                lat_prefix_new = "N" if new_lat >= 0 else "S"
                lon_prefix_new = "E" if new_lon >= 0 else "W"
                tile_name = (
                    f"Ai2_WorldCover_10m_2024_v1_"
                    f"{lat_prefix_new}{abs(new_lat):02d}"
                    f"{lon_prefix_new}{abs(new_lon):03d}_Map.tif"
                )
                output_path = output_dir / tile_name

                # Write the new 1x1 GeoTIFF
                with rasterio.open(
                    output_path,
                    "w",
                    driver="GTiff",
                    height=height,
                    width=width,
                    count=src.count,
                    dtype=src.dtypes[0],
                    crs=src.crs,
                    transform=transform,
                    compress="lzw",
                    tiled=True,
                    blockxsize=256,
                    blockysize=256,
                    nodata=src.nodata,
                ) as dst:
                    for band in range(1, src.count + 1):
                        dst.write(src.read(band, window=window), band)

    logger.info("Resampled and saved 1x1 tiles from %s", file_name)


def main() -> None:
    """Process and resample all GeoTIFF files in parallel."""
    # Configure logging using shared utility
    configure_logging()

    # Gather all GeoTIFF files in the input directory
    tif_files: List[str] = [str(f) for f in Path(input_dir).glob("*.tif")]

    # Process files in parallel with a progress bar
    logger.info("Starting resampling process...")
    with Pool(processes=cpu_count()) as pool:
        list(
            tqdm(
                pool.imap_unordered(split_and_resample, tif_files),
                total=len(tif_files),
            )
        )

    logger.info("All tiles have been resampled and saved.")


if __name__ == "__main__":
    main()
