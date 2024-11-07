import os
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
import joblib
from multiprocessing import Pool, cpu_count

# Define input and output directories
input_dir = '/home/patrickb/litus/data/coastal_data_points'  # Replace with your actual directory
output_dir = '/home/patrickb/litus/data/ball_trees/'
os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

def create_and_save_ball_tree(file_path):
    """
    Reads a coastal_points.csv file, creates a BallTree, and saves it.
    """
    try:
        # Load the coastal data
        data = pd.read_csv(file_path, header=None, names=['longitude', 'latitude'])
        data['longitude'] = pd.to_numeric(data['longitude'], errors='coerce')
        data['latitude'] = pd.to_numeric(data['latitude'], errors='coerce')
        data.dropna(inplace=True)

        # Convert data to radians
        coastal_data_rad = np.radians(data[['latitude', 'longitude']].values)

        # Create BallTree
        tree = BallTree(coastal_data_rad, metric='haversine')

        # Define output path
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_ball_tree.joblib")

        # Save the BallTree using joblib
        joblib.dump(tree, output_path, compress=0, protocol=5)
        print(f"Saved BallTree for {file_path} at {output_path}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def process_directory_in_parallel():
    """
    Finds all coastal_points.csv files in the input directory and processes them in parallel.
    """
    # List all CSV files in the input directory
    csv_files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith('coastal_points.csv')]

    # Use all available cores for parallel processing
    with Pool(cpu_count()) as pool:
        pool.map(create_and_save_ball_tree, csv_files)

# Run the parallel processing function
if __name__ == "__main__":
    process_directory_in_parallel()
