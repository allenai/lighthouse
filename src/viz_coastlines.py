"""Visualize Coastal Points on a World Map."""

from pathlib import Path
from typing import List

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from geopandas import GeoDataFrame
from pandas import DataFrame

from utils.log_utils import configure_logging, get_logger

# Configure logging
logger = get_logger(__name__)

# Define paths relative to script location
ROOT_DIR = Path(__file__).resolve().parent.parent
coastal_data_dir = ROOT_DIR / "data" / "coastal_data_points"
csv_files = list(coastal_data_dir.glob("*_coast.csv"))
coastal_points_list: List[DataFrame] = []

for csv_file in csv_files[1:10]:
    df = pd.read_csv(csv_file)
    coastal_points_list.append(df)

coastal_points_df = pd.concat(coastal_points_list, ignore_index=True)

coastal_points_gdf: GeoDataFrame = gpd.GeoDataFrame(
    coastal_points_df,
    geometry=gpd.points_from_xy(
        coastal_points_df.longitude,
        coastal_points_df.latitude,
    ),
    crs="EPSG:4326",
)

logger.info(
    "Total number of coastal points loaded: %d",
    len(coastal_points_gdf),
)

# Sample 10 Random Points
sampled_points = coastal_points_gdf.sample(n=10, random_state=42)

logger.info("Sampled Points:")
logger.info(
    "\n%s",
    sampled_points[["longitude", "latitude"]].to_string(),
)

# Read the world map from Natural Earth
world_url = (
    "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
)

# Read the dataset
logger.info("Loading world map data...")
world = gpd.read_file(world_url)

# Ensure the CRS matches your coastal points GeoDataFrame
world = world.to_crs(coastal_points_gdf.crs)

# Create the plot
fig, ax = plt.subplots(figsize=(15, 10))
world.plot(ax=ax, color="lightgrey", edgecolor="white")
sampled_points.plot(
    ax=ax,
    color="blue",
    markersize=50,
    label="Coastal Points",
)

ax.set_title("Sampled Coastal Points on World Map", fontsize=20)
ax.set_xlabel("Longitude", fontsize=15)
ax.set_ylabel("Latitude", fontsize=15)
ax.legend()

logger.info("Displaying plot...")
plt.show()


if __name__ == "__main__":
    configure_logging()
