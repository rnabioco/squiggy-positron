"""Pytest configuration and shared fixtures."""
from pathlib import Path
import pytest


@pytest.fixture
def test_data_dir():
    """Return the path to the test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def sample_pod5_file(test_data_dir):
    """Return the path to the sample POD5 file.

    Note: You'll need to add an actual POD5 file to tests/data/
    """
    pod5_path = test_data_dir / "sample.pod5"
    if not pod5_path.exists():
        pytest.skip(f"Sample POD5 file not found at {pod5_path}")
    return pod5_path
