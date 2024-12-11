import os
import requests
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

input_directory = "/home/patrickb/litus/tracks"
output_directory = "/home/patrickb/litus/track_with_d2c"

# Coastal Detection Service URL
COASTAL_DETECTION_URL = "http://0.0.0.0:8000/detect"


def get_distance_to_coast_single(lat, lon):
    """Fetch distance to coast from the Coastal Detection Service for a single point."""
    payload = {
        "lat": lat,
        "lon": lon
    }
    try:
        response = requests.post(COASTAL_DETECTION_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("distance_to_coast_m", -1)
    except requests.RequestException as e:
        print(f"Error fetching distance for lat: {lat}, lon: {lon} - {e}")
        return -1


def process_csv(file_path: str, output_path: str):
    """Add distance_to_coast column to a CSV file by processing each point individually in parallel."""
    df = pd.read_csv(file_path)

    if "lat" not in df.columns or "lon" not in df.columns:
        print(f"Skipping {file_path}: Missing 'lat' or 'lon' columns.")
        return

    lats = df["lat"].tolist()
    lons = df["lon"].tolist()
    n_points = len(lats)

    print(f"Processing {os.path.basename(file_path)} with {n_points} points...")
    print("Sending individual requests in parallel.")

    # Prepare arguments as tuples for starmap
    args = zip(lats, lons)
    with Pool(processes=cpu_count()) as p:
        distances = list(
            tqdm(
                p.starmap(get_distance_to_coast_single, args),
                total=n_points,
                desc="Processing individual points"
            )
        )

    df["distance_to_coast"] = distances
    os.makedirs(output_directory, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Finished processing {file_path}. Output saved to {output_path}.")


if __name__ == "__main__":
    # Process each CSV file sequentially
    input_files = [
        f for f in os.listdir(input_directory) if f.endswith(".csv")
    ]

    for file_name in input_files:
        input_path = os.path.join(input_directory, file_name)
        output_path = os.path.join(output_directory, file_name)
        process_csv(input_path, output_path)
