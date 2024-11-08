"""Tests for the Coastal Detection Service."""

from typing import Any, Dict, Set

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

    # Check data types
    assert isinstance(response_data["distance_to_coast_m"], int)
    assert isinstance(response_data["land_cover_class"], str)
    assert isinstance(response_data["nearest_coastal_point"], list)
    assert len(response_data["nearest_coastal_point"]) == 2
    assert all(isinstance(x, float) for x in response_data["nearest_coastal_point"])


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
    request_data: Dict[str, float] = {"lat": 39.0997, "lon": -94.5786}
    response = client.post("/detect", json=request_data)

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["land_cover_class"] != "Permanent water bodies"


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

    except requests.exceptions.ConnectionError:
        pytest.skip("API server not running")


if __name__ == "__main__":
    pytest.main([__file__])
