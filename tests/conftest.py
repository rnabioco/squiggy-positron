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


@pytest.fixture
def sample_bam_file(test_data_dir):
    """Return the path to a sample BAM file."""
    # Look for simplex BAM file
    bam_path = test_data_dir / "simplex_reads_mapped.bam"
    if bam_path.exists():
        return bam_path

    # Otherwise, find any BAM file
    bam_files = list(test_data_dir.glob("*.bam"))
    if bam_files:
        return bam_files[0]

    pytest.skip(f"No BAM files found in {test_data_dir}")
    return None


@pytest.fixture
def indexed_bam_file(sample_bam_file):
    """Return path to BAM file with index, creating index if needed."""
    from squiggy.utils import index_bam_file

    # Check if index exists
    index_path = Path(str(sample_bam_file) + ".bai")
    if not index_path.exists():
        # Create index
        index_bam_file(sample_bam_file)

    return sample_bam_file
