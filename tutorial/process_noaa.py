import os
import requests
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

input_directory = "/home/patrickb/litus/tracks"
output_directory = "/home/patrickb/litus/track_with_d2c"

# Coastal Detection Service URL
COASTAL_DETECTION_URL = "http://0.0.0.0:8000/detect"

def get_distance_to_coast(lat_lon: tuple) -> int:
    """Fetch distance to coast from the Coastal Detection Service."""
    lat, lon = lat_lon
    try:
        response = requests.post(
            COASTAL_DETECTION_URL,
            json={"lat": lat, "lon": lon},
        )
        response.raise_for_status()
        return response.json().get("distance_to_coast_m", -1)
    except requests.RequestException as e:
        print(f"Error fetching distance for lat: {lat}, lon: {lon} - {e}")
        return -1

def process_csv(file_path: str, output_path: str):
    """Add distance_to_coast column to a CSV file."""
    df = pd.read_csv(file_path)

    if "lat" not in df.columns or "lon" not in df.columns:
        print(f"Skipping {file_path}: Missing 'lat' or 'lon' columns.")
        return

    # Prepare latitude and longitude pairs
    lat_lon_pairs = list(zip(df["lat"], df["lon"]))

    # Process in parallel using Pool
    with Pool(cpu_count()) as pool:
        distances = list(tqdm(pool.imap(get_distance_to_coast, lat_lon_pairs), total=len(lat_lon_pairs), desc=f"Processing {os.path.basename(file_path)}"))

    df["distance_to_coast"] = distances

    # Save the updated DataFrame to the output directory
    os.makedirs(output_directory, exist_ok=True)
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    for file_name in os.listdir(input_directory):
        if file_name.endswith(".csv"):
            input_path = os.path.join(input_directory, file_name)
            output_path = os.path.join(output_directory, file_name)

            print(f"Processing {input_path}...")
            process_csv(input_path, output_path)
            print(f"Finished processing {input_path}. Output saved to {output_path}.")
