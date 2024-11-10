"""Example request to the FastAPI endpoint."""

import logging

import requests

logger = logging.getLogger(__name__)

# Define the request data directly in the script
request_data = {"lat": 47.636895, "lon": -122.334984}

# Send the POST request to the FastAPI endpoint with a timeout
response = requests.post(
    "http://0.0.0.0:8000/detect",
    json=request_data,
    timeout=30,  # Add a 30-second timeout
)

# Check and print the response
if response.status_code == 200:
    logger.info("Response from API:", response.json())
else:
    logger.info("Error:", response.status_code, response.text)
