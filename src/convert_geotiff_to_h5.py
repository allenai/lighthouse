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

Usage:
    Run as a script to process all GeoTIFF files in a directory:
    $ python convert_geotiff_to_h5.py

    Or import functions for use in other modules:
    >>> from convert_geotiff_to_h5 import convert_geotiff_to_h5
    >>> bounds = convert_geotiff_to_h5('input.tif', 'output_dir')
"""

import json
import logging
import os
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

# Configure logging
logger = logging.getLogger(__name__)

# Type alias for bounds dictionary
BoundsDict = Dict[str, Dict[str, float]]


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
    os.makedirs(output_dir, exist_ok=True)
    bounds_dict: BoundsDict = {}

    geotiff_files: List[str] = [
        os.path.join(geotiff_dir, f)
        for f in os.listdir(geotiff_dir)
        if f.endswith(".tif")
    ]

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
    # Configure logging for command line usage
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    geotiff_directory = "data/resampled/"
    h5_output_directory = "data/resampled_h5s/"
    num_cores = 150  # Adjust based on available cores

    # Process directory and get bounds dictionary
    bounds_dictionary: BoundsDict = process_directory(
        geotiff_directory,
        h5_output_directory,
        num_workers=num_cores,
    )

    # Save bounds dictionary to JSON file for fast lookup
    with open("bounds_dictionary.json", "w") as f:
        json.dump(bounds_dictionary, f)

    logger.info("Conversion completed")
