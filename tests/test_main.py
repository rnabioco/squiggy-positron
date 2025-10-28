"""Tests for main application functionality."""

import numpy as np
import pod5
import pytest


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

    def test_read_multiple_reads(self, sample_pod5_file):
        """Test reading multiple reads from POD5 file."""
        with pod5.Reader(sample_pod5_file) as reader:
            reads = list(reader.reads())

            # Should be able to read multiple reads
            assert len(reads) >= 1

            # Each read should have valid data
            for read in reads:
                assert len(read.signal) > 0
                assert read.run_info.sample_rate > 0

    def test_signal_is_numpy_array(self, sample_pod5_file):
        """Test that signal data is a numpy array."""
        with pod5.Reader(sample_pod5_file) as reader:
            read = next(reader.reads())

            assert isinstance(read.signal, np.ndarray)

    def test_signal_has_reasonable_values(self, sample_pod5_file):
        """Test that signal values are in reasonable range."""
        with pod5.Reader(sample_pod5_file) as reader:
            read = next(reader.reads())

            # Signal should be numeric
            assert np.issubdtype(read.signal.dtype, np.number)

            # Signal values should be finite (no NaN or Inf)
            assert np.all(np.isfinite(read.signal))

            # Raw nanopore signal is typically in picoamperes range
            # Values should be reasonable (typically 0-300 pA)
            assert np.min(read.signal) >= -100  # Allow some negative noise
            assert np.max(read.signal) <= 2000  # Allow some positive outliers

    def test_signal_statistics(self, sample_pod5_file):
        """Test signal statistical properties."""
        with pod5.Reader(sample_pod5_file) as reader:
            read = next(reader.reads())

            signal = read.signal

            # Calculate basic statistics
            mean = np.mean(signal)
            std = np.std(signal)
            min_val = np.min(signal)
            max_val = np.max(signal)

            # Statistics should be reasonable
            assert mean > 0  # Signal typically positive
            assert std > 0  # Signal should have variance
            assert max_val > min_val  # Should have range

    def test_sample_rate_is_valid(self, sample_pod5_file):
        """Test that sample rate is a valid value."""
        with pod5.Reader(sample_pod5_file) as reader:
            read = next(reader.reads())

            sample_rate = read.run_info.sample_rate

            # Sample rate should be positive integer
            assert sample_rate > 0
            assert isinstance(sample_rate, (int, np.integer))

            # Common sample rates: 3000, 4000, 5000 Hz
            # Allow any reasonable value
            assert 1000 <= sample_rate <= 10000

    def test_read_id_is_valid_uuid(self, sample_pod5_file):
        """Test that read IDs can be converted to strings."""
        with pod5.Reader(sample_pod5_file) as reader:
            read = next(reader.reads())

            read_id_str = str(read.read_id)

            # Should be a non-empty string
            assert isinstance(read_id_str, str)
            assert len(read_id_str) > 0

            # UUID format: 8-4-4-4-12 hex digits
            # Just check it's a reasonable length
            assert len(read_id_str) >= 32  # At least 32 hex chars

    def test_read_context_manager(self, sample_pod5_file):
        """Test that POD5 Reader works as context manager."""
        # Open and close via context manager
        with pod5.Reader(sample_pod5_file) as reader:
            read_count = len(list(reader.reads()))
            assert read_count > 0

        # File should be closed after context manager exits
        # (No way to directly test, but shouldn't crash)

    def test_iterate_all_reads(self, sample_pod5_file):
        """Test iterating through all reads in file."""
        with pod5.Reader(sample_pod5_file) as reader:
            read_count = 0
            for read in reader.reads():
                read_count += 1
                # Each read should have valid signal
                assert len(read.signal) > 0

            assert read_count > 0

    def test_signal_length_consistency(self, sample_pod5_file):
        """Test that signal length is consistent with duration."""
        with pod5.Reader(sample_pod5_file) as reader:
            read = next(reader.reads())

            signal_length = len(read.signal)
            sample_rate = read.run_info.sample_rate

            # Duration in seconds
            duration_seconds = signal_length / sample_rate

            # Reads are typically seconds to minutes long
            assert duration_seconds > 0
            assert duration_seconds < 3600  # Less than 1 hour (sanity check)

    def test_multiple_reads_have_different_ids(self, sample_pod5_file):
        """Test that multiple reads have different read IDs."""
        with pod5.Reader(sample_pod5_file) as reader:
            reads = list(reader.reads())

            if len(reads) > 1:
                # Collect all read IDs
                read_ids = [str(read.read_id) for read in reads]

                # All IDs should be unique
                assert len(read_ids) == len(set(read_ids))
            else:
                pytest.skip("Only one read in file, can't test uniqueness")

    def test_run_info_has_expected_attributes(self, sample_pod5_file):
        """Test that run_info has expected attributes."""
        with pod5.Reader(sample_pod5_file) as reader:
            read = next(reader.reads())

            run_info = read.run_info

            # Should have sample_rate
            assert hasattr(run_info, "sample_rate")
            assert run_info.sample_rate > 0


class TestPOD5ErrorHandling:
    """Tests for POD5 error handling."""

    def test_open_nonexistent_file(self):
        """Test that opening non-existent file raises error."""
        from pathlib import Path

        # pod5 raises various exceptions for file errors
        with pytest.raises((FileNotFoundError, OSError, RuntimeError)):
            with pod5.Reader(Path("/nonexistent/file.pod5")) as _reader:
                pass

    def test_open_invalid_file(self, tmp_path):
        """Test that opening invalid file raises error."""
        # Create an empty file that's not a valid POD5
        invalid_file = tmp_path / "invalid.pod5"
        invalid_file.write_text("not a pod5 file")

        # pod5 raises various exceptions for invalid files
        with pytest.raises((ValueError, RuntimeError, OSError)):
            with pod5.Reader(invalid_file) as _reader:
                pass
