"""Unit tests for the pipeline module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sklearn.neighbors import BallTree

from src.pipeline import (
    ball_tree_distance,
    coord_to_coastal_point,
    get_ball_tree,
    get_filename_for_coordinates,
    h5_to_integer,
    initialize_coastal_ball_tree,
    main,
)

# Test data
TEST_LAT = 47.636895
TEST_LON = -122.334984
TEST_BOUNDS_DICT = {
    "test_tile.h5": {
        "latmin": 47.0,
        "latmax": 48.0,
        "lonmin": -123.0,
        "lonmax": -122.0,
    }
}


@pytest.fixture
def mock_ball_tree() -> BallTree:
    """Create a mock BallTree with test data."""
    points = np.array([[47.6, -122.3]])  # Single test point
    return BallTree(np.radians(points), metric="haversine")


@pytest.fixture
def mock_h5_file() -> MagicMock:
    """Create a mock HDF5 file with test data."""
    mock = MagicMock()
    mock["band_data"] = np.array([[50]])  # Test land class
    mock["geotransform"] = np.array(
        [
            0.0,  # x-coordinate of upper-left corner
            0.001,  # pixel width
            0.0,  # rotation (usually 0)
            90.0,  # y-coordinate of upper-left corner
            0.0,  # rotation (usually 0)
            -0.001,  # pixel height (negative because origin is upper-left)
        ]
    )
    return mock


def test_initialize_coastal_ball_tree(mock_ball_tree: BallTree) -> None:
    """Test initialization of coastal ball tree."""
    with patch("joblib.load", return_value=mock_ball_tree):
        tree = initialize_coastal_ball_tree()
        assert isinstance(tree, BallTree)


def test_coord_to_coastal_point(mock_ball_tree: BallTree) -> None:
    """Test finding nearest coastal point."""
    with patch("joblib.load", return_value=mock_ball_tree):
        nearest_point, distance = coord_to_coastal_point(TEST_LAT, TEST_LON)
        assert isinstance(nearest_point, np.ndarray)
        assert isinstance(distance, float)
        assert distance >= 0


def test_get_filename_for_coordinates() -> None:
    """Test finding HDF5 file for coordinates."""
    # Test coordinates within bounds
    filename = get_filename_for_coordinates(TEST_LAT, TEST_LON, TEST_BOUNDS_DICT)
    assert filename == "test_tile.h5"

    # Test coordinates outside bounds
    filename = get_filename_for_coordinates(0.0, 0.0, TEST_BOUNDS_DICT)
    assert filename is None


@patch("pathlib.Path.exists")
@patch("joblib.load")
def test_get_ball_tree(
    mock_load: MagicMock, mock_exists: MagicMock, mock_ball_tree: BallTree
) -> None:
    """Test loading BallTree from file."""
    mock_exists.return_value = True
    mock_load.return_value = mock_ball_tree

    tree = get_ball_tree("test_ball_tree.joblib")
    assert isinstance(tree, BallTree)

    # Test file not found
    mock_exists.return_value = False
    with pytest.raises(FileNotFoundError):
        get_ball_tree("nonexistent.joblib")


@patch("h5py.File")
def test_h5_to_integer(mock_h5py: MagicMock, mock_h5_file: MagicMock) -> None:
    """Test getting land-water classification."""
    mock_h5py.return_value.__enter__.return_value = mock_h5_file

    land_class = h5_to_integer("test.h5", TEST_LON, TEST_LAT)
    assert isinstance(land_class, int)
    assert 0 <= land_class <= 100


def test_ball_tree_distance(mock_ball_tree: BallTree) -> None:
    """Test calculating distance to nearest coastal point."""
    point = [TEST_LAT, TEST_LON]
    distance, nearest_point = ball_tree_distance(mock_ball_tree, point)

    assert isinstance(distance, float)
    assert isinstance(nearest_point, np.ndarray)
    assert distance >= 0
    assert len(nearest_point) == 2


@patch("src.pipeline.get_filename_for_coordinates")
@patch("src.pipeline.h5_to_integer")
@patch("src.pipeline.get_ball_tree")
@patch("src.pipeline.ball_tree_distance")
def test_main(
    mock_distance: MagicMock,
    mock_get_tree: MagicMock,
    mock_h5: MagicMock,
    mock_filename: MagicMock,
    mock_ball_tree: BallTree,
) -> None:
    """Test main pipeline function."""
    # Setup mocks
    mock_filename.return_value = "test.h5"
    mock_h5.return_value = 50  # Built-up area
    mock_get_tree.return_value = mock_ball_tree
    mock_distance.return_value = (100.0, np.array([47.6, -122.3]))

    # Test point in tile
    result = main(TEST_LAT, TEST_LON)
    assert isinstance(result, tuple)
    assert len(result) == 3
    distance, land_class, nearest = result
    assert isinstance(distance, float)
    assert isinstance(land_class, int)
    assert isinstance(nearest, np.ndarray)

    # Test point in ocean
    mock_filename.return_value = None
    result = main(0.0, -160.0)
    assert result[1] == 0  # Should be water


if __name__ == "__main__":
    pytest.main([__file__])
