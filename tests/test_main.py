"""Tests for the Coastal Detection Service."""

from datetime import datetime
from typing import Any, Dict, List, Set

import pytest
import requests
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)

# Test coordinates (Seattle coordinates)
TEST_LAT: float = 47.636895
TEST_LON: float = -122.334984

# Expected response structure
EXPECTED_KEYS: Set[str] = {
    "distance_to_coast_m",
    "land_cover_class",
    "nearest_coastal_point",
    "version",
}

# Sample valid response for testing
SAMPLE_RESPONSE: Dict[str, Any] = {
    "distance_to_coast_m": 275,
    "land_cover_class": "Permanent water bodies",
    "nearest_coastal_point": [47.63742, -122.33858],
    "version": "2024-11-07T21:14:59.848461",
}


def test_home_endpoint() -> None:
    """Test the health check endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Coastal Detection Service is running"}


def test_detect_endpoint_valid_request() -> None:
    """Test the detect endpoint with valid coordinates."""
    request_data: Dict[str, float] = {"lat": TEST_LAT, "lon": TEST_LON}
    response = client.post("/detect", json=request_data)

    assert response.status_code == 200

    response_data: Dict[str, Any] = response.json()
    # Check response structure
    assert set(response_data.keys()) == EXPECTED_KEYS

    # Check data types and ranges
    assert isinstance(response_data["distance_to_coast_m"], int)
    assert isinstance(response_data["land_cover_class"], str)
    assert isinstance(response_data["nearest_coastal_point"], list)
    assert len(response_data["nearest_coastal_point"]) == 2
    assert all(isinstance(x, float) for x in response_data["nearest_coastal_point"])

    # Check value ranges based on sample response
    assert 0 <= response_data["distance_to_coast_m"] <= 1000  # Within 1km
    assert response_data["land_cover_class"] in [
        "Permanent water bodies",
        "Land",
        "Built-up",
    ]
    nearest_point: List[float] = response_data["nearest_coastal_point"]
    assert 47.0 <= nearest_point[0] <= 48.0  # Latitude range
    assert -123.0 <= nearest_point[1] <= -122.0  # Longitude range
    assert isinstance(datetime.fromisoformat(response_data["version"]), datetime)


def test_detect_endpoint_invalid_coordinates() -> None:
    """Test the detect endpoint with invalid coordinates."""
    invalid_requests = [
        {"lat": 91, "lon": 0},  # Invalid latitude
        {"lat": 0, "lon": 181},  # Invalid longitude
        {"lat": "invalid", "lon": 0},  # Invalid type
        {"lat": 0},  # Missing longitude
        {"lon": 0},  # Missing latitude
    ]

    for request_data in invalid_requests:
        response = client.post("/detect", json=request_data)
        assert response.status_code in {400, 422}  # Validation error


def test_detect_endpoint_ocean_point() -> None:
    """Test the detect endpoint with a point in the ocean."""
    # Pacific Ocean coordinates
    request_data: Dict[str, float] = {"lat": 0.0, "lon": -160.0}
    response = client.post("/detect", json=request_data)

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["land_cover_class"] == "Permanent water bodies"


def test_detect_endpoint_land_point() -> None:
    """Test the detect endpoint with a point on land."""
    # Kansas coordinates (definitely on land)
    request_data: Dict[str, float] = {"lat": 47.654290, "lon": -122.336381}
    response = client.post("/detect", json=request_data)
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["land_cover_class"] == "Built-up"


@pytest.mark.integration
def test_live_api_request() -> None:
    """Test the live API endpoint (marked as integration test)."""
    request_data: Dict[str, float] = {"lat": TEST_LAT, "lon": TEST_LON}

    try:
        response = requests.post(
            "http://0.0.0.0:8000/detect", json=request_data, timeout=30
        )
        assert response.status_code == 200
        response_data = response.json()
        assert set(response_data.keys()) == EXPECTED_KEYS

        # Compare with sample response structure
        assert isinstance(response_data["distance_to_coast_m"], int)
        assert isinstance(response_data["land_cover_class"], str)
        assert isinstance(response_data["nearest_coastal_point"], list)
        assert len(response_data["nearest_coastal_point"]) == 2
        assert datetime.fromisoformat(response_data["version"])

    except requests.exceptions.ConnectionError:
        pytest.skip("API server not running")


if __name__ == "__main__":
    pytest.main([__file__])
