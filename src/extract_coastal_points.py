"""

"""

import concurrent.futures
import json
import logging.config
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import tqdm
from osgeo import gdal
from PIL import Image
from scipy import ndimage

logger = logging.getLogger(__name__)

NUM_PROCESSES = 1 if (os.cpu_count() or 4) == 2 else max(1, (os.cpu_count() or 4) - 2)


OUT_DIR = "coastal_data_points"

current_dir = Path(__file__).resolve().parent.parent
data_in_dir = Path("data") / "resampled"

# The Path object will automatically handle the parent directory references
DATA_IN_DIR = str(data_in_dir)


logger = logging.getLogger(__name__)
Image.MAX_IMAGE_PIXELS = 933120000  # to avoid PIL thinking the tifs are bombs
DEBUG = True
LAND_COVER_TYPES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare/sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}

# Bins used if distances are going to be digitized (save_compresed = True)
BINS = (
    np.arange(1, 10, 1),  # note that 1 pixel is 10m, so this is 10-100m
    np.arange(10, 100, 10),  #  100-1000m, in 100m increments
    np.arange(100, 1000, 100),  #  1km-10km, in 1km increments
    np.arange(1000, 2689000, 1000),  # point nemo :)
)


class NumpyFloatValuesEncoder(json.JSONEncoder):
    """convert numpy float32 values to float"""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


def binarize_input_image(input_image: np.ndarray) -> np.ndarray:
    """binarizes input image into water and land pixels

    Permanent water bodies have value 80 in the ESA WorldCover dataset.


    """

    input_image[input_image == 0] = 80  # non mapped pixels (=0) are water
    input_image[input_image < 80] = 0
    input_image[input_image > 80] = 0
    input_image[input_image == 80] = 1

    return input_image


def sobel_edges_scipy(img: np.ndarray) -> np.ndarray:
    """fast version of edge detection_ using simple 2d convolution"
    notee that the input to this image should be a binarized image
    """
    logger.info(f"Running edge detection with scipy on img with size: {img.shape}")

    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # Apply Sobel operator
    img_x = ndimage.convolve(img, kernel_x)
    img_y = ndimage.convolve(img, kernel_y)

    # Magnitude of gradient
    magnitude = np.hypot(img_x, img_y)
    magnitude = magnitude / magnitude.max() * 255
    magnitude = magnitude.astype(np.uint8)

    return magnitude


def get_geo_arrays(
    image_path: Path, edges: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """gets latitude and longitude locations of coastal pixels"""

    raster = gdal.Open(str(image_path))
    gt = raster.GetGeoTransform()

    # Get the coordinates of the edge pixels
    edge_coords = np.column_stack(np.where(edges != 0))

    # Initialize arrays for latitude and longitude with zeros
    lat_array = np.zeros(edges.shape, dtype=np.float32)
    lon_array = np.zeros(edges.shape, dtype=np.float32)

    # Populate the arrays with latitudes and longitudes at the edge locations
    for pix in tqdm.tqdm(edge_coords):
        x_geo = gt[0] + pix[1] * gt[1]
        y_geo = gt[3] + pix[0] * gt[5]
        lon_array[pix[0], pix[1]] = x_geo
        lat_array[pix[0], pix[1]] = y_geo
    return lon_array, lat_array


def digitize_distances(distances: np.ndarray) -> np.ndarray:
    """save space by digitizing distances into integer bins with decreasing precision further from shore"""
    bins = np.concatenate(BINS).flatten()
    return np.digitize(distances, bins)


def extract_coastal_coordinates(
    lon: np.ndarray, lat: np.ndarray, indices: Tuple[np.ndarray, np.ndarray]
) -> Dict[str, List[float]]:
    """Extracts coastal coordinates."""
    coordinates: Dict[str, List[float]] = {"longitude": [], "latitude": []}

    for xy in zip(*indices):
        x, y = xy
        coordinates["longitude"].append(lon[x, y])
        coordinates["latitude"].append(lat[x, y])
    return coordinates


def save_coastal_data_to_csv(dictionary: Dict[str, List[float]], out_csv: str) -> None:
    """Saves coastal data to a CSV file."""
    try:
        df = pd.DataFrame.from_dict(dictionary)
        df.to_csv(out_csv, index=None)
    except (IOError, pd.errors.EmptyDataError, Exception):
        try:
            os.remove(out_csv)
        except OSError:
            logger.exception("Exception removing csv", exc_info=True)
        raise


def write_coastal_distances_from_sentinel2(
    image_path: Path, out_dir: str, save_compressed: bool = False
) -> None:
    """Writes coastal distances from Sentinel-2 image."""
    converted = np.array(Image.open(image_path))
    binarized = binarize_input_image(converted)
    # Ensure the binarized image is of type unsigned integer

    edges = sobel_edges_scipy(binarized)
    lon, lat = get_geo_arrays(image_path, edges)
    indices = np.nonzero(lon)  # any pixel non zero is a coastal pixel
    dictionary = extract_coastal_coordinates(lon, lat, indices)
    out_csv = os.path.join(out_dir, f"{image_path.stem}_coastal_points.csv")
    save_coastal_data_to_csv(dictionary, out_csv)


def main() -> None:
    """
    Orchestrates the processing of Sentinel-2 (or OSM origin) image files to calculate coastal distances.

    This function finds all TIFF image files in a specified directory, `DATA_IN_DIR.

    Raises:
        Exception: Propagates any exception raised during the processing of individual images.

    Note:
        This function assumes the existence of global constants `DATA_IN_DIR`, `OUT_DIR`, and
        `NUM_PROCESSES`, which should be defined externally. `DATA_IN_DIR` is the directory
        containing the input TIFF files, `OUT_DIR` is the directory where the output will be saved,
        and `NUM_PROCESSES` defines the number of worker processes in the pool.
    """
    out_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), OUT_DIR)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    image_paths = list(Path(DATA_IN_DIR).glob("*.tif"))
    successes = []
    failures = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
        # Using a dictionary to associate each image path with its future
        futures_to_path = {
            executor.submit(
                write_coastal_distances_from_sentinel2, img_path, out_dir
            ): img_path
            for img_path in image_paths
        }

        for future in concurrent.futures.as_completed(futures_to_path):
            img_path = futures_to_path[future]
            try:
                future.result()  # This will raise an exception if the function encountered an error
                successes.append(img_path)
            except Exception as e:
                logger.info(f"Error processing {img_path}: {e}")
                failures.append(img_path)

    logger.info(f"Successfully processed {len(successes)} files.")
    logger.info(f"Failed to process {len(failures)} files.")


if __name__ == "__main__":
    main()
