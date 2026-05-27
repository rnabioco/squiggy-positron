"""Pytest configuration and shared fixtures."""

from pathlib import Path

import pytest


@pytest.fixture
def test_data_dir():
    """Return the path to the test data directory."""
    from squiggy import get_test_data_path

    return Path(get_test_data_path())


@pytest.fixture
def sample_pod5_file(test_data_dir):
    """Return the path to a sample POD5 file."""
    pod5_path = test_data_dir / "ecoli_trna_wt_reads.pod5"
    if pod5_path.exists():
        return pod5_path

    pod5_files = list(test_data_dir.glob("*.pod5"))
    if pod5_files:
        return pod5_files[0]

    pytest.skip(f"No POD5 files found in {test_data_dir}")
    return None


@pytest.fixture
def sample_bam_file(test_data_dir):
    """Return the path to a sample BAM file."""
    bam_path = test_data_dir / "ecoli_trna_wt_mappings.bam"
    if bam_path.exists():
        return bam_path

    bam_files = list(test_data_dir.glob("*.bam"))
    if bam_files:
        return bam_files[0]

    pytest.skip(f"No BAM files found in {test_data_dir}")
    return None


@pytest.fixture
def indexed_bam_file(sample_bam_file):
    """Return path to BAM file with index, creating index if needed."""
    index_path = Path(str(sample_bam_file) + ".bai")
    if not index_path.exists():
        import pysam

        pysam.index(str(sample_bam_file))

    return sample_bam_file
