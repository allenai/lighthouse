# litus

## Development
1. Download worldcover from European Space Agency
bash src/download_worldcover.sh
2. Download land polygons from open street map
wget -P /path/to/your/directory https://osmdata.openstreetmap.de/download/land-polygons-split-4326.zip
unzip osm/land-polygons-split-4326.zip
3. generate the land-sea geotiffs from osm's land polygon data to match worldcover geotiffs.
python src/gen_all_missing_tiles.py
4.