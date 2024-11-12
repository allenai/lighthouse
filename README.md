# Litus

**Litus** (Latin for "beach, shore, or coast") provides highly precise (10 meters) and fast (~milliseconds) distance-to-shoreline calculations from any point on Earth (on land or on the high seas).

## Requirements

### System Requirements
- Docker 24.0 or higher
- 500GB+ storage space for dataset
- 4GB+ RAM recommended
- gcloud CLI

### Installing gcloud CLI

#### Debian/Ubuntu
```bash
# Add the Cloud SDK distribution URI as a package source
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list

# Import the Google Cloud public key
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

# Update and install the Cloud SDK
sudo apt-get update && sudo apt-get install google-cloud-cli
```

#### macOS
```bash
# Using Homebrew
brew install --cask google-cloud-sdk
```

#### Windows
1. Download the [Google Cloud SDK installer](https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe)
2. Launch the installer and follow the prompts

### Authenticate gcloud
```bash
# Login to your Google Cloud account
gcloud auth login

# Set your project (if needed)
gcloud config set project YOUR_PROJECT_ID
```

## Setup
### Download the dataset from GCP
Required gcloud installation

```bash
# Create data directory
mkdir -p path/to/data

# Copy data from GCS bucket (requires gcloud authentication)
gcloud alpha storage cp -r gs://litus/data/ path/to/data
```

### Installation

There are two ways to install and run Litus:

#### Option 1: Using Pre-built Image (Recommended)

1. **Pull the Docker Image**:
   ```bash
   docker pull ghcr.io/allenai/litus:sha-30b4d50
   ```

2. **Run the Docker Container**:
   ```bash
   docker run -d \
     --name coastal_image_service \
     -p 8000:8000 \
     -v path/to/data:/src/data \ # see above for datapath
     ghcr.io/allenai/litus:sha-30b4d50
   ```

### Example
See:
   ```bash
   python example/sample_request.py
   ```
or:
   ```bash
   curl -X POST "http://0.0.0.0:8000/detect" \
    -H "Content-Type: application/json" \
    -d '{"lat": 47.636895, "lon": -122.334984}'
   ```

expected output is:
```json
{
  "distance_to_coast_m": 275,
  "land_cover_class": "Permanent water bodies",
  "nearest_coastal_point": [47.63742, -122.33858],
  "version": "2024-11-12T00:25:16.667195"
}
```


#### Option 2: Building from Source

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/allenai/litus.git
   cd litus
   ```

2. **Copy the Dataset** (~500 GB):
   ```bash
   # Create data directory
   mkdir -p data

   # Copy data from GCS bucket (requires gcloud authentication)
   gcloud alpha storage cp -r gs://litus/data/ path/to/data
   ```

3. **Build the Docker Image**:
   ```bash
   docker build -t coastal_image_service .
   ```

4. **Run the Docker Container**:
   ```bash
   docker run -d \
     --name coastal_image_service \
     -p 8000:8000 \
     -v ~/litus/data:/src/data \
     coastal_image_service
   ```

### Verifying Installation

Once running, you can verify the service is working by accessing:
```bash
curl http://localhost:8000/health
```

The service should return a 200 OK response.

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
