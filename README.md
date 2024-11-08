
---

# Litus

**Litus** (Latin for "beach, shore, or coast") provides highly precise (10 meters) and fast (~milliseconds) distance-to-shoreline calculations from any point on Earth.

---

## Setup

### Installation
Litus requires copying a global dataset (~500 GB) and then building and running the service.

1. **Copy the Dataset**:
   ```bash
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
**In progress**

---
