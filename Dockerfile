# Use Python 3.12 slim image for better compatibility with GDAL
FROM python:3.12-slim

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

# Set environment variables for GDAL
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Copy requirements to leverage Docker cache if requirements haven't changed
COPY requirements/requirements.txt /tmp/requirements.txt

# Install Python packages
RUN pip install --no-cache-dir --upgrade -r /tmp/requirements.txt

# Set the working directory
WORKDIR /src

# Copy the source code
COPY ./src /src

# Expose the default FastAPI port
EXPOSE 8000

# Specify the default command to run the application
CMD ["python", "main.py"]
