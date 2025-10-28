"""Tests for main application functionality."""

import pod5


class TestPOD5Reading:
    """Tests for POD5 file reading functionality."""

    def test_can_open_sample_file(self, sample_pod5_file):
        """Test that we can open and read the sample POD5 file."""
        with pod5.Reader(sample_pod5_file) as reader:
            reads = list(reader.reads())
            assert len(reads) > 0

    def test_read_has_expected_attributes(self, sample_pod5_file):
        """Test that reads have the expected attributes."""
        with pod5.Reader(sample_pod5_file) as reader:
            read = next(reader.reads())

            assert hasattr(read, "read_id")
            assert hasattr(read, "signal")
            assert hasattr(read, "run_info")
            assert hasattr(read.run_info, "sample_rate")
            assert len(read.signal) > 0
            assert read.run_info.sample_rate > 0

    def test_get_read_ids(self, sample_pod5_file):
        """Test extracting all read IDs from a POD5 file."""
        with pod5.Reader(sample_pod5_file) as reader:
            read_ids = [str(read.read_id) for read in reader.reads()]

            # Verify we got some read IDs
            assert len(read_ids) > 0
            # Verify they're unique
            assert len(read_ids) == len(set(read_ids))
