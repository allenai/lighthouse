# Litus

**Litus** (Latin for "beach, shore, or coast") provides highly precise (10 meters) and fast (~milliseconds) distance-to-shoreline calculations from any point on Earth (on land or on the high seas).

## Requirements

### System Requirements
- Docker 24.0 or higher
- 500GB+ storage space for dataset
- 4GB+ RAM recommended
---

## Setup

### Installation
Litus requires copying a global dataset (~500 GB) and then building and running the service.

1. **Copy the Dataset**:
   ```bash
   # Create data directory
   mkdir -p data

   # Copy data from GCS bucket (requires gcloud authentication)
   gcloud alpha storage cp -r gs://litus/data/ data/
   ```

2. **Build the Docker Image**:
   ```bash
   docker build -t coastal_image_service .
   ```

3. **Run the Docker Container**:
   ```bash
   docker run -d \
     --name coastal_image_service \
     -p 8000:8000 \
     -v ~/litus/data:/src/data \
     coastal_image_service
   ```

---

## Development
(in progress)

### Local Development Setup
1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   # or
   .venv\Scripts\activate  # Windows
   ```

2. Install development dependencies:
   ```bash
   pip install -r requirements/requirements.txt
   ```

3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

### Running Tests
```bash
pytest tests/
```

### Code Conventions
- Black for code formatting
- Ruff for linting
- MyPy for type checking
- Pre-commit hooks for automated checks

### Building Dataset from Scratch
1. Download worldcover from European Space Agency:
   ```bash
   bash src/download_worldcover.sh
   ```

2. Download land polygons from OpenStreetMap:
   ```bash
   wget -P data/osm \
     https://osmdata.openstreetmap.de/download/land-polygons-split-4326.zip
   unzip data/osm/land-polygons-split-4326.zip -d data/osm
   ```

3. Generate the land-sea geotiffs:
   ```bash
   python src/gen_all_missing_tiles.py
   ```

4. Convert GeoTIFFs to HDF5:
   ```bash
   python src/convert_geotiff_to_h5.py
   ```

5. Extract coastal points:
   ```bash
   python src/extract_coastal_points.py
   ```

6. Convert coastal points to BallTrees:
   ```bash
   python src/convert_coastal_points_to_ball_trees.py
   ```

---

## License
Apache 2.0

---
