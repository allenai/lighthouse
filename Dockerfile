# Use Python 3.12 Alpine image
FROM python:3.12-alpine

# Install necessary packages for FastAPI and data processing
RUN apk update && apk add --no-cache \
    ffmpeg \
    libsm \
    libxext \
    hdf5-dev \
    netcdf \
    libnetcdf \
    gcc \
    g++ \
    musl-dev \
    make \
    && rm -rf /var/cache/apk/*

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

# Specify the default command to run
CMD ["python", "main.py"]