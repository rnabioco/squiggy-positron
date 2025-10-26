"""Tests for main application functionality."""

import pytest
from pathlib import Path
import pod5


class TestSquigglePlotter:
    """Tests for the SquigglePlotter class."""

    def test_plot_creation_with_sample_data(self, sample_pod5_file):
        """Test that we can create a plot from sample POD5 data."""
        from squiggy.main import SquigglePlotter

        # Read the first read from the sample file
        with pod5.Reader(sample_pod5_file) as reader:
            read = next(reader.reads())

            # Create a plot
            plot = SquigglePlotter.create_plot(
                signal=read.signal,
                sample_rate=read.sample_rate,
                read_id=str(read.read_id),
            )

            # Verify plot was created
            assert plot is not None
            assert hasattr(plot, "save")

    def test_signal_to_dataframe(self, sample_pod5_file):
        """Test conversion of signal data to time-series DataFrame."""
        from squiggy.main import SquigglePlotter
        import numpy as np

        with pod5.Reader(sample_pod5_file) as reader:
            read = next(reader.reads())

            df = SquigglePlotter.signal_to_dataframe(read.signal, read.sample_rate)

            # Verify DataFrame structure
            assert "time" in df.columns
            assert "signal" in df.columns
            assert len(df) == len(read.signal)
            assert df["signal"].dtype == np.float64


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
            assert hasattr(read, "sample_rate")
            assert len(read.signal) > 0
            assert read.sample_rate > 0

    def test_get_read_ids(self, sample_pod5_file):
        """Test extracting all read IDs from a POD5 file."""
        with pod5.Reader(sample_pod5_file) as reader:
            read_ids = [str(read.read_id) for read in reader.reads()]

            # Verify we got some read IDs
            assert len(read_ids) > 0
            # Verify they're unique
            assert len(read_ids) == len(set(read_ids))
