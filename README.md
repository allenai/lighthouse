# litus
Highly precise and computationally efficient distance-to-shoreline calculations from any point on earth.

# Setup
Installation requires copying the dataset and then building/running the service.
Note that this is is ~500 GB
1. gcloud alpha storage cp -r gs://litus/data/ data/
2. docker build -t coastal_image_service .
3. docker run -d \
  --name coastal_image_service \
  -p 8000:8000 \
  -v ~/litus/data:/src/data \

  coastal_image_service


## Development (In progress)
