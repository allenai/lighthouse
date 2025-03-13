# Lighthouse
Fast and precise distance to shoreline calculations from anywhere on earth (AoE).

Key Features:
- 10-meter resolution land/water classification
- millisecond distance-to-coast calculations from anywhere on earth
- global coverage -- includes inland bodies of water (rivers, lakes, bays, etc)

## Requirements
- Docker 24.0+
- 4GB RAM
Optional
- gcloud CLI (for downloading dataset)
- 500GB storage (recommended)
For streaming/real-time use cases, it is recommended to download the entire dataset to disk.

## Quick Start
(without downloading full dataset)
```bash
docker pull ghcr.io/allenai/lighthouse
docker run -d \
  --name lighthouse \
  -p 8000:8000 \
  -v path/to/data:/data \
  ghcr.io/allenai/lighthouse
```
See ## Installation for downloading dataset from gcp.

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
Note that this dataset is large. approximately 500 GB storage space is required for the full dataset. Individual tiles (1 degree by 1 degree) can also be downloaded rather than whole dataset.
The dataset is stored in a public Google Cloud Storage bucket at:

```
gs://ai2-coastlines/v1/data
```


```bash
mkdir -p data
# using gcloud (see: https://cloud.google.com/sdk/docs/install)
gcloud storage cp --recursive gs://ai2-coastlines/v1/data /path/to/local/data

# using gsutil (see https://cloud.google.com/storage/docs/gsutil_install)
gutil -m cp -r gs://ai2-coastlines/v1/data /path/to/local/data

# using wget (
wget -r -np -nH --cut-dirs=3 -P data https://storage.googleapis.com/ai2-coastlines/v1/data/
```

The above command will download two types of files:

a. **Ball Trees:** (`ai2-coastlines/v1/data/ball_trees`)
   *Example:*
   `ai2-coastlines/v1/data/ball_trees/Ai2_WorldCover_10m_2024_v1_N00E006_Map_coastal_points_ball_tree.joblib` (1.4 MB)

b. **Resampled H5s:** (`ai2-coastlines/v1/data/resampled_h5s`)
   *Example:*
   `ai2-coastlines/v1/data/resampled_h5s/Ai2_WorldCover_10m_2024_v1_N00E006_Map.h5` (584.2 KB)

### Deployment
(requires downloading dataset above)
Note that some sample inferences/examples can run without the full dataset.

#### Option 1: Using Pre-built Image (Recommended)
```bash
docker pull ghcr.io/allenai/lighthouse:sha-X
docker run -d \
  --name lighthouse \
  -p 8000:8000 \
  -v path/to/data:/src/data \
  ghcr.io/allenai/lighthouse:sha-X
```

#### Option 2: Building from Source
```bash
git clone https://github.com/allenai/lighthouse.git
cd lighthouse
docker build -t lighthouse .
docker run -d \
  --name lighthouse \
  -p 8000:8000 \
  -v path/to/data:/src/data \
  lighthouse
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
pytest tests
```

### Code Conventions
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

### How does this algorithm work?
Lighthouse** (Layered Iterative Geospatial Hierarchical Terrain-Oriented Unified Search Engine) leverages
1. pre-computed spherical Voronoi tesselation of the whole planet's coastlines (at low resolution) and
2. ball trees (at high resolution) to produce very fast computations with minimal resources.

The ball trees were generated from a hybrid dataset of satellite imagery based annotations from two sources:
- [ESA WorldCover V2](https://esa-worldcover.org/en): 10m resolution global land cover data
- [OpenStreetMap](https://www.openstreetmap.org) land-water polygon labels

![voronoi (1)](https://github.com/user-attachments/assets/4e91968d-714e-451e-bf04-24e4016e2db5)

^^ that's the Voronoi.

![triplet_of_fun](https://github.com/user-attachments/assets/035f797d-fa94-42e8-bb3b-7f89b077a9ee)
^^ that's a depiction of the method.

See the paper ([todo: add link arXiv]) for details.

## License
Code: Apache 2.0
Dataset: Open Database License (ODbL) v1.0
  - http://opendatacommons.org/licenses/odbl/1.0/
  - http://opendatacommons.org/licenses/dbcl/1.0/

## Acknowledgments

We gratefully acknowledge:
- The European Space Agency (ESA) for creating the WorldCover land cover map and for making it openly accessible
- The OpenStreetMap community for their invaluable contributions to global mapping

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


## Citation

```bibtex
@software{Lighthouse2024,
  title = {Lighthouse: High-Precision Coastal Distance Calculator},
  author = {{Allen Institute for AI}},
  year = {2024},
  url = {https://github.com/allenai/lighthouse},
  note = {Uses ESA WorldCover 2021 and OpenStreetMap data}
}
```

**Also Lighthouse is an excellent coffee shop in Seattle.
