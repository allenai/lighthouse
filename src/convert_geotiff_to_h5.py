import os
import h5py
import rasterio
from rasterio.transform import Affine
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import json
from tqdm import tqdm

def convert_geotiff_to_h5(geotiff_path, output_dir):
    """
    Convert a GeoTIFF file to HDF5 format, preserving geotransform and CRS.

    Args:
        geotiff_path (str): Path to the GeoTIFF file.
        output_dir (str): Directory to save the HDF5 file.

    Returns:
        dict: Dictionary containing filename and bounds.
    """
    bounds_info = {}

    try:
        with rasterio.open(geotiff_path) as src:
            # Read band data
            band_data = src.read(1)
            geotransform = src.transform
            crs = src.crs

            # Define output HDF5 filename
            filename = Path(geotiff_path).stem + ".h5"
            output_path = os.path.join(output_dir, filename)

            # Write data to HDF5
            with h5py.File(output_path, 'w') as hdf:
                # Store band data and geotransform
                hdf.create_dataset("band_data", data=band_data, compression="gzip")
                hdf.create_dataset("geotransform", data=np.array(geotransform).flatten())
                hdf.attrs["crs"] = crs.to_string()

            # Store bounds information
            bounds = src.bounds
            bounds_info[filename] = {
                "latmin": bounds.bottom,
                "latmax": bounds.top,
                "lonmin": bounds.left,
                "lonmax": bounds.right
            }

    except Exception as e:
        print(f"Error processing {geotiff_path}: {e}")

    return bounds_info

def process_directory(geotiff_dir, output_dir, num_workers=4):
    """
    Process all GeoTIFF files in a directory, converting them to HDF5 format in parallel.

    Args:
        geotiff_dir (str): Directory containing GeoTIFF files.
        output_dir (str): Directory to save HDF5 files.
        num_workers (int): Number of parallel processes.

    Returns:
        dict: A dictionary with bounds for each processed file.
    """
    os.makedirs(output_dir, exist_ok=True)
    bounds_dict = {}

    geotiff_files = [os.path.join(geotiff_dir, f) for f in os.listdir(geotiff_dir) if f.endswith('.tif')]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(convert_geotiff_to_h5, gt_path, output_dir): gt_path for gt_path in geotiff_files}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Converting GeoTIFFs"):
            result = future.result()
            if result:
                bounds_dict.update(result)

    return bounds_dict

if __name__ == "__main__":
    geotiff_directory = "data/resampled/"
    h5_output_directory = "data/resampled_h5s/"
    num_cores = 150  # Adjust based on available cores

    # Process directory and get bounds dictionary
    bounds_dictionary = process_directory(geotiff_directory, h5_output_directory, num_workers=num_cores)

    # Save bounds dictionary to JSON file for fast lookup
    with open("bounds_dictionary.json", "w") as f:
        json.dump(bounds_dictionary, f)

    print("Conversion completed.")
