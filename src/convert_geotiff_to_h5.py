"""
GeoTIFF to HDF5 Converter Module.

This module provides functionality to convert GeoTIFF files to HDF5 format while
preserving geospatial metadata. It includes parallel processing capabilities for
handling multiple files efficiently.

Key Features:
    - Converts GeoTIFF files to HDF5 format
    - Preserves geotransform and CRS information
    - Maintains band data with compression
    - Stores boundary information for each tile
    - Supports parallel processing for multiple files
    - Creates a bounds dictionary for fast coordinate lookups

Directory Structure Assumptions:
    The module expects the following directory structure:
    project_root/
    ├── data/
    │   ├── resampled/           # Input GeoTIFF files
    │   ├── resampled_h5s/       # Output HDF5 files
    │   └── bounds_dictionary.json
    └── src/
        └── convert_geotiff_to_h5.py

    Input GeoTIFFs should be in data/resampled/
    Output HDF5s will be saved to data/resampled_h5s/
    Bounds dictionary will be saved to data/bounds_dictionary.json
"""

import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Union

import h5py
import numpy as np
import rasterio
from numpy.typing import NDArray
from rasterio.crs import CRS
from rasterio.transform import Affine
from tqdm import tqdm

from utils.log_utils import configure_logging

# Configure logging
logger = logging.getLogger(__name__)

# Type alias for bounds dictionary
BoundsDict = Dict[str, Dict[str, float]]

# Define paths relative to script location
ROOT_DIR = Path(__file__).resolve().parent.parent
geotiff_directory = ROOT_DIR / "data" / "resampled"
h5_output_directory = ROOT_DIR / "data" / "resampled_h5s"


def convert_geotiff_to_h5(
    geotiff_path: Union[str, Path],
    output_dir: Union[str, Path],
) -> BoundsDict:
    """
    Convert a GeoTIFF file to HDF5 format, preserving geotransform and CRS.

    Args:
        geotiff_path: Path to the GeoTIFF file
        output_dir: Directory to save the HDF5 file

    Returns:
        Dictionary containing filename and bounds
    """
    bounds_info: BoundsDict = {}

    try:
        with rasterio.open(geotiff_path) as src:
            # Read band data
            band_data: NDArray = src.read(1)
            geotransform: Affine = src.transform
            crs: CRS = src.crs

            # Define output HDF5 filename
            filename: str = Path(geotiff_path).stem + ".h5"
            output_path: Path = Path(output_dir) / filename

            # Write data to HDF5
            with h5py.File(output_path, "w") as hdf:
                # Store band data and geotransform
                hdf.create_dataset(
                    "band_data",
                    data=band_data,
                    compression="gzip",
                )
                hdf.create_dataset(
                    "geotransform",
                    data=np.array(geotransform).flatten(),
                )
                hdf.attrs["crs"] = crs.to_string()

            # Store bounds information
            bounds = src.bounds
            bounds_info[filename] = {
                "latmin": bounds.bottom,
                "latmax": bounds.top,
                "lonmin": bounds.left,
                "lonmax": bounds.right,
            }

    except Exception as e:
        logger.error("Error processing %s: %s", geotiff_path, e)

    return bounds_info


def process_directory(
    geotiff_dir: Union[str, Path],
    output_dir: Union[str, Path],
    num_workers: int = 4,
) -> BoundsDict:
    """
    Process all GeoTIFF files in a directory, converting them to HDF5 format.

    Args:
        geotiff_dir: Directory containing GeoTIFF files
        output_dir: Directory to save HDF5 files
        num_workers: Number of parallel processes

    Returns:
        A dictionary with bounds for each processed file
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    bounds_dict: BoundsDict = {}

    geotiff_files: List[str] = [str(f) for f in Path(geotiff_dir).glob("*.tif")]

    logger.info(
        "Found %d GeoTIFF files to process",
        len(geotiff_files),
    )

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                convert_geotiff_to_h5,
                gt_path,
                output_dir,
            ): gt_path
            for gt_path in geotiff_files
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Converting GeoTIFFs",
        ):
            result = future.result()
            if result:
                bounds_dict.update(result)

    return bounds_dict


if __name__ == "__main__":
    # Configure logging using shared utility
    configure_logging()

    # Process directory and get bounds dictionary
    bounds_dictionary: BoundsDict = process_directory(
        geotiff_directory,
        h5_output_directory,
        num_workers=150,  # Adjust based on available cores
    )

    # Save bounds dictionary to JSON file for fast lookup
    bounds_file = ROOT_DIR / "data" / "bounds_dictionary.json"
    with open(bounds_file, "w") as f:
        json.dump(bounds_dictionary, f)

    logger.info("Conversion completed")
