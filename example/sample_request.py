import requests

# Define the request data directly in the script
request_data = {
    "lat": 47.636895,
    "lon": -122.334984
}

# Send the POST request to the FastAPI endpoint
response = requests.post("http://0.0.0.0:8000/detect", json=request_data)

# Check and print the response
if response.status_code == 200:
    print("Response from API:", response.json())
else:
    print("Error:", response.status_code, response.text)
