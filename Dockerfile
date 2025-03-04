# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/src

# Install system dependencies for GDAL and other geospatial libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    libproj-dev \
    libgeos-dev \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set GDAL environment variables
ENV GDAL_VERSION=3.6.4
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Copy requirements to leverage Docker cache if requirements haven't changed
COPY requirements/requirements.txt /tmp/requirements.txt

# Install Python packages
RUN pip install --no-cache-dir pip==24.0 \
    && pip install --no-cache-dir GDAL==${GDAL_VERSION} \
    && pip install --no-cache-dir -r /tmp/requirements.txt

# Set the working directory
WORKDIR /src

# Copy the source code
COPY ./src /src
COPY ./tests/ tests/

# copy these two files for CI and unit/integration test
COPY ./data/ball_trees/Ai2_WorldCover_10m_2024_v1_N47W123_Map_coastal_points_ball_tree.joblib /data/ball_trees/
COPY ./data/resampled_h5s/Ai2_WorldCover_10m_2024_v1_N47W123_Map.h5 /data/resampled_h5s/

# Expose the default FastAPI port
EXPOSE 8000

# Specify the default command to run
CMD ["python", "-m", "main"]
