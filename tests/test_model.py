"""Parameterized unit tests for the pipeline module using only coordinates."""

import os

import numpy as np
import pytest

from src.pipeline import main

# Define parameterized test cases with coordinates and expected results
# (latitude, longitude, expected_distance, expected_land_class)
TEST_CASES = [
    (47.636895, -122.334984, 275, 80),  # lake union
    (-89.98833055, -170.05309276, 1251009, 1),  # Antarctica, Ice/land
    (45.00, -122.50, 1057, 10),  # Bioko, Tree Cover
    (83.579602, -34.101947, 6676, 1),
    (82.348767, -54.057979, 7841, 80),
    (-69.716681, -0.785526, 11881, 1),
    (-70.667115, -0.603568, 63637, 1),
    (-83.762443, 25.924445, 1156189, 1),
]


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true", reason="Skipped on GitHub Actions runner"
)
@pytest.mark.parametrize("lat, lon, expected_distance, expected_land_class", TEST_CASES)
def test_main_pipeline(
    lat: float, lon: float, expected_distance: float, expected_land_class: int
) -> None:
    """
    Test the main pipeline function using parameterized coordinates and expected results.
    Only provides latitude and longitude as input.
    """

    # Run main function with given coordinates
    distance, land_class, nearest = main(lat, lon)

    # Assertions to check that the results match expected values
    assert isinstance(distance, float)
    assert distance == pytest.approx(
        expected_distance, rel=1e-1
    )  # Allow small margin for floating-point
    assert land_class == expected_land_class
    assert isinstance(nearest, np.ndarray)
    assert len(nearest) == 2
