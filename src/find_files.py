#!/usr/bin/env python3

# Step 1: Generate all possible tile names
latitudes = list(range(-90, 90, 3))  # -90 to +87
longitudes = list(range(-180, 180, 3))  # -180 to +177

all_tiles = []

for lat in latitudes:
    for lon in longitudes:
        # Latitude string
        if lat >= 0:
            lat_prefix = "N"
            lat_str = f"{lat:02d}"
        else:
            lat_prefix = "S"
            lat_str = f"{abs(lat):02d}"

        # Longitude string
        if lon >= 0:
            lon_prefix = "E"
            lon_str = f"{lon:03d}"
        else:
            lon_prefix = "W"
            lon_str = f"{abs(lon):03d}"

        tile_name = f"v200/2021/map/ESA_WorldCover_10m_2021_v200_{lat_prefix}{lat_str}{lon_prefix}{lon_str}_Map.tif"
        all_tiles.append(tile_name)

# Save all tile names to a file
with open("all_possible_tiles.txt", "w") as f:
    for tile in all_tiles:
        f.write(f"{tile}\n")

print(f"Total possible tiles: {len(all_tiles)}")

# Step 2: Read existing tiles
with open(
    "/Users/patrickb/skylight/eai/ais/src/atlantes/data/list_of_files.txt", "r"
) as f:
    existing_tiles = set(line.strip() for line in f)

# Step 3: Find missing tiles
all_tiles_set = set(all_tiles)
missing_tiles = all_tiles_set - existing_tiles

# Save missing tiles to a file
with open("missing_tiles.txt", "w") as f:
    for tile in sorted(missing_tiles):
        f.write(f"{tile}\n")

print(f"Total missing tiles: {len(missing_tiles)}")
print("Missing tiles have been saved to 'missing_tiles.txt'")
