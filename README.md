# Lighthouse: Fast and precise distance to shoreline calculations from anywhere on earth
Layered Iterative Geospatial Hierarchical Terrain-Oriented Unified Search Engine

##  Overview
Ligthhouse is a library for efficiently querying a 10 meter distance to coast dataset. Lighthouse is a hierarchical search algorithm  that leverages a pre-computed spherical Voronoi tesselation of the whole planet's coastalines (at low resolution) and ball trees constructed from high resolution coastal datasets. The ball trees were generated from a hybrid dataset of satellite imagery based annotations from two sources:
- [ESA WorldCover V2](https://esa-worldcover.org/en): 10m resolution global land cover data
- [OpenStreetMap](https://www.openstreetmap.org) land-water polygon labels

Key Features:
- 10-meter resolution land/water classification
- millisecond distance-to-coast calculations from anywhere on earth
- global coverage -- includes inland bodies of water (rivers, lakes, bays, etc)


## Requirements
- Docker 24.0+
- 500GB storage space (recommended*)
- 4GB RAM
- gcloud CLI (for downloading dataset)

*While you can retrieve the required files on demand, doing so will result in slower query times. For streaming/real-time use cases, it is recommended to download the entire dataset to disk.

## Quick Start
1. Download the dataset
```bash
docker pull ghcr.io/allenai/lighthouse
docker run -d \
  --name litus \
  -p 8000:8000 \
  -v path/to/data:/src/data \
  ghcr.io/allenai/lighthouse
```

## Example Usage

```python
import requests

response = requests.post(
    "http://localhost:8000/detect",
    json={"lat": 47.636895, "lon": -122.334984},
    timeout=30
)

print(response.json())
```

Expected output:
```json
{
  "distance_to_coast_m": 275,
  "land_cover_class": "Permanent water bodies",
  "nearest_coastal_point": [47.63742, -122.33858],
  "version": "2024-11-12T00:25:16.667195"
}
```

## Installation

### Dataset Download

1. Install gcloud CLI:
   <details>
   <summary>Debian/Ubuntu</summary>

   ```bash
   echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] \
     https://packages.cloud.google.com/apt cloud-sdk main" | \
     sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list

   curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
     sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

   sudo apt-get update && sudo apt-get install google-cloud-cli
   ```
   </details>

   <details>
   <summary>macOS</summary>

   ```bash
   brew install --cask google-cloud-sdk
   ```
   </details>

2. Authenticate:
   ```bash
   gcloud auth login
   ```

3. Download dataset:
   ```bash
   mkdir -p data
   gcloud alpha storage cp -r gs://litus/data/ data/
   ```

### Deployment Options

#### Option 1: Pre-built Image (Recommended)
```bash
docker pull ghcr.io/allenai/litus:sha-30b4d50
docker run -d \
  --name litus \
  -p 8000:8000 \
  -v path/to/data:/src/data \
  ghcr.io/allenai/litus:sha-30b4d50
```

#### Option 2: Build from Source
```bash
git clone https://github.com/allenai/litus.git
cd litus
docker build -t litus .
docker run -d \
  --name litus \
  -p 8000:8000 \
  -v path/to/data:/src/data \
  litus
```

## Development

### Local Setup
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements/requirements.txt
pip install -r requirements/requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### Running Tests
```bash
pytest tests/
```

### Code Quality
- Black for formatting
- Ruff for linting
- MyPy for type checking
- Pre-commit hooks for automated checks

### Building Dataset from Scratch

<details>
<summary>Click to expand</summary>

1. Download ESA WorldCover data:
   ```bash
   bash src/download_worldcover.sh
   ```

2. Download OSM land polygons:
   ```bash
   wget -P data/osm \
     https://osmdata.openstreetmap.de/download/land-polygons-split-4326.zip
   unzip data/osm/land-polygons-split-4326.zip -d data/osm
   ```

3. Process data:
   ```bash
   python src/gen_all_missing_tiles.py
   python src/convert_geotiff_to_h5.py
   python src/extract_coastal_points.py
   python src/convert_coastal_points_to_ball_trees.py
   ```
</details>

## License

### Code
Apache 2.0

### Dataset:
- License: Open Database License (ODbL) v1.0
  - http://opendatacommons.org/licenses/odbl/1.0/
  - http://opendatacommons.org/licenses/dbcl/1.0/

## References
### ESA WorldCover 2021
- Source: https://esa-worldcover.org/en
- Citation:
```bibtex
@article{zanaga2021esa,
  title={ESA WorldCover 10 m 2021 v200},
  author={Zanaga, D and Van De Kerchove, R and De Keersmaecker, W and Souverijns, N and Brockmann, C and Quast, R and Wevers, J and Grosu, A and Paccini, A and Vergnaud, S and others},
  year={2021},
  publisher={ESA},
  doi={10.5281/zenodo.5571936}
}
```

### OpenStreetMap
- Source: https://www.openstreetmap.org
- We used the land polygon data to generate land-sea masks: https://osmdata.openstreetmap.de/data/land-polygons.html

## Acknowledgments

We gratefully acknowledge:
- The European Space Agency (ESA) for creating the WorldCover land cover map and for making it openly accessible
- The OpenStreetMap community for their invaluable contributions to pubclily available global maps

## Citation

```bibtex
@software{Lighthouse2024,
  title = {Lighthouse: High-Precision Coastal Distance Calculator},
  author = {{Allen Institute for AI}},
  year = {2024},
  url = {https://github.com/allenai/litus},
  note = {Uses ESA WorldCover 2021 and OpenStreetMap data}
}
```
