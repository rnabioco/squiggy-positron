"""Pytest configuration and shared fixtures."""

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def clean_squiggy_state():
    """Automatically clean squiggy global state before and after each test."""
    # Clean state before test
    from squiggy import close_all_samples, close_bam, close_pod5
    from squiggy.io import squiggy_kernel

    close_pod5()
    close_bam()
    # Also clean all samples
    squiggy_kernel.close_all()

    yield

    # Clean state after test
    close_pod5()
    close_bam()
    close_all_samples()


@pytest.fixture
def test_data_dir():
    """Return the path to the test data directory."""
    from squiggy import get_test_data_path

    return Path(get_test_data_path())


@pytest.fixture
def sample_pod5_file(test_data_dir):
    """Return the path to a sample POD5 file.

    Uses yeast_trna_reads.pod5 as the standard test data file.
    """
    # Use yeast_trna_reads.pod5 as the standard test file
    pod5_path = test_data_dir / "yeast_trna_reads.pod5"
    if pod5_path.exists():
        return pod5_path

    # Fallback: find any POD5 file
    pod5_files = list(test_data_dir.glob("*.pod5"))
    if pod5_files:
        return pod5_files[0]

    pytest.skip(f"No POD5 files found in {test_data_dir}")
    return None


@pytest.fixture
def sample_bam_file(test_data_dir):
    """Return the path to a sample BAM file.

    Uses yeast_trna_mappings.bam as the standard test data file.
    """
    # Use yeast_trna_mappings.bam as the standard test file
    bam_path = test_data_dir / "yeast_trna_mappings.bam"
    if bam_path.exists():
        return bam_path

    # Fallback: find any BAM file
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
