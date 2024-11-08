""" Visualize Coastal Points on a World Map """

import glob
import os

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

# Step 1: Load the Entire Coastal Dataset

coastal_data_dir = "coastal_data_points/"
csv_files = glob.glob(os.path.join(coastal_data_dir, "*_coast.csv"))
coastal_points_list = []

for csv_file in csv_files[1:10]:
    df = pd.read_csv(csv_file)
    coastal_points_list.append(df)

coastal_points_df = pd.concat(coastal_points_list, ignore_index=True)

coastal_points_gdf = gpd.GeoDataFrame(
    coastal_points_df,
    geometry=gpd.points_from_xy(
        coastal_points_df.longitude, coastal_points_df.latitude
    ),
    crs="EPSG:4326",
)

print(f"Total number of coastal points loaded: {len(coastal_points_gdf)}")

# Step 2: Sample 10 Random Points

sampled_points = coastal_points_gdf.sample(n=10, random_state=42)

print("Sampled Points:")
print(sampled_points[["longitude", "latitude"]])

# Step 3: Plot the Sampled Points on a World Map
# Read the world map directly from the URL
world_url = (
    "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
)

# Read the dataset
world = gpd.read_file(world_url)

# Ensure the CRS matches your coastal points GeoDataFrame
world = world.to_crs(coastal_points_gdf.crs)

fig, ax = plt.subplots(figsize=(15, 10))
world.plot(ax=ax, color="lightgrey", edgecolor="white")
sampled_points.plot(ax=ax, color="blue", markersize=50, label="Coastal Points")

ax.set_title("Sampled Coastal Points on World Map", fontsize=20)
ax.set_xlabel("Longitude", fontsize=15)
ax.set_ylabel("Latitude", fontsize=15)
ax.legend()

plt.show()
