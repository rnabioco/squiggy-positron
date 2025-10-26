"""Pytest configuration and shared fixtures."""

from pathlib import Path
import pytest


@pytest.fixture
def test_data_dir():
    """Return the path to the test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def sample_pod5_file(test_data_dir):
    """Return the path to a sample POD5 file.

    Prefers sample.pod5, but will use any available POD5 file in tests/data/
    """
    # First try sample.pod5 (may be a symlink)
    pod5_path = test_data_dir / "sample.pod5"
    if pod5_path.exists():
        return pod5_path

    # Otherwise, find any POD5 file
    pod5_files = list(test_data_dir.glob("*.pod5"))
    if pod5_files:
        return pod5_files[0]

    pytest.skip(f"No POD5 files found in {test_data_dir}")
    return None
