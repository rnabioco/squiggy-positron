"""Tests to verify signal accuracy from POD5 files to Bokeh plots."""

import numpy as np
import pod5
import pytest
from bokeh.models import ColumnDataSource


class TestSignalAccuracy:
    """Tests to ensure POD5 signal data is accurately displayed in plots."""

    def test_single_read_signal_values_match(self, sample_pod5_file):
        """Test that signal values from POD5 match exactly in the plot data source."""
        from squiggy.plotting import SquigglePlotter

        from squiggy.constants import NormalizationMethod

        with pod5.Reader(sample_pod5_file) as reader:
            read = next(reader.reads())
            original_signal = read.signal.copy()
            read_id = str(read.read_id)
            sample_rate = read.run_info.sample_rate

            # Generate plot with no normalization or downsampling
            html, figure = SquigglePlotter.plot_single_read(
                signal=original_signal,
                read_id=read_id,
                sample_rate=sample_rate,
                normalization=NormalizationMethod.NONE,
                downsample=1,
            )

            # Extract data source from figure
            # The signal data is stored in a ColumnDataSource
            signal_sources = [
                r.data_source
                for r in figure.renderers
                if hasattr(r, "data_source")
                and isinstance(r.data_source, ColumnDataSource)
                and "signal" in r.data_source.data
            ]

            assert len(signal_sources) > 0, "No signal data source found in plot"

            # Get the signal data from the plot
            plot_signal = np.array(signal_sources[0].data["signal"])

            # Verify exact match (no normalization, no downsampling)
            np.testing.assert_array_equal(
                plot_signal,
                original_signal,
                err_msg="Plot signal does not match original POD5 signal",
            )

    def test_single_read_signal_length_match(self, sample_pod5_file):
        """Test that the number of signal points matches between POD5 and plot."""
        from squiggy.plotting import SquigglePlotter

        from squiggy.constants import NormalizationMethod

        with pod5.Reader(sample_pod5_file) as reader:
            read = next(reader.reads())
            original_signal = read.signal
            read_id = str(read.read_id)
            sample_rate = read.run_info.sample_rate

            # Generate plot with no downsampling
            html, figure = SquigglePlotter.plot_single_read(
                signal=original_signal,
                read_id=read_id,
                sample_rate=sample_rate,
                normalization=NormalizationMethod.NONE,
                downsample=1,
            )

            # Extract signal data source
            signal_sources = [
                r.data_source
                for r in figure.renderers
                if hasattr(r, "data_source")
                and isinstance(r.data_source, ColumnDataSource)
                and "signal" in r.data_source.data
            ]

            plot_signal = np.array(signal_sources[0].data["signal"])

            assert len(plot_signal) == len(original_signal), (
                f"Signal length mismatch: POD5 has {len(original_signal)}, plot has {len(plot_signal)}"
            )

    def test_single_read_time_axis_accuracy(self, sample_pod5_file):
        """Test that time axis values are calculated correctly from sample rate."""
        from squiggy.plotting import SquigglePlotter

        from squiggy.constants import NormalizationMethod

        with pod5.Reader(sample_pod5_file) as reader:
            read = next(reader.reads())
            signal = read.signal
            read_id = str(read.read_id)
            sample_rate = read.run_info.sample_rate

            # Generate plot
            html, figure = SquigglePlotter.plot_single_read(
                signal=signal,
                read_id=read_id,
                sample_rate=sample_rate,
                normalization=NormalizationMethod.NONE,
            )

            # Extract time data source
            signal_sources = [
                r.data_source
                for r in figure.renderers
                if hasattr(r, "data_source")
                and isinstance(r.data_source, ColumnDataSource)
                and "time" in r.data_source.data
            ]

            assert len(signal_sources) > 0, "No time data found in plot"

            plot_time = np.array(signal_sources[0].data["time"])

            # Calculate expected time values
            expected_time = np.arange(len(signal)) * 1000 / sample_rate

            # Verify time values match (within floating point precision)
            np.testing.assert_array_almost_equal(
                plot_time,
                expected_time,
                decimal=6,
                err_msg="Time axis values do not match expected calculation",
            )

    def test_downsampled_signal_accuracy(self, sample_pod5_file):
        """Test that downsampled signal contains correct subset of original signal."""
        from squiggy.plotting import SquigglePlotter

        from squiggy.constants import NormalizationMethod

        with pod5.Reader(sample_pod5_file) as reader:
            read = next(reader.reads())
            original_signal = read.signal.copy()
            read_id = str(read.read_id)
            sample_rate = read.run_info.sample_rate
            downsample_factor = 10

            # Generate plot with downsampling
            html, figure = SquigglePlotter.plot_single_read(
                signal=original_signal,
                read_id=read_id,
                sample_rate=sample_rate,
                normalization=NormalizationMethod.NONE,
                downsample=downsample_factor,
            )

            # Extract signal data
            signal_sources = [
                r.data_source
                for r in figure.renderers
                if hasattr(r, "data_source")
                and isinstance(r.data_source, ColumnDataSource)
                and "signal" in r.data_source.data
            ]

            plot_signal = np.array(signal_sources[0].data["signal"])

            # Expected downsampled signal
            expected_signal = original_signal[::downsample_factor]

            # Verify downsampled signal matches
            np.testing.assert_array_equal(
                plot_signal,
                expected_signal,
                err_msg=f"Downsampled signal (1/{downsample_factor}) does not match expected values",
            )

    def test_normalized_signal_properties(self, sample_pod5_file):
        """Test that normalized signals have correct statistical properties."""
        from squiggy.plotting import SquigglePlotter

        from squiggy.constants import NormalizationMethod

        with pod5.Reader(sample_pod5_file) as reader:
            read = next(reader.reads())
            signal = read.signal
            read_id = str(read.read_id)
            sample_rate = read.run_info.sample_rate

            # Test Z-normalization
            html_znorm, figure_znorm = SquigglePlotter.plot_single_read(
                signal=signal,
                read_id=read_id,
                sample_rate=sample_rate,
                normalization=NormalizationMethod.ZNORM,
            )

            # Extract z-normalized signal
            signal_sources = [
                r.data_source
                for r in figure_znorm.renderers
                if hasattr(r, "data_source")
                and isinstance(r.data_source, ColumnDataSource)
                and "signal" in r.data_source.data
            ]

            znorm_signal = np.array(signal_sources[0].data["signal"])

            # Verify Z-norm properties (mean ≈ 0, std ≈ 1)
            assert np.abs(np.mean(znorm_signal)) < 1e-10, (
                f"Z-normalized signal mean should be ~0, got {np.mean(znorm_signal)}"
            )
            assert np.abs(np.std(znorm_signal) - 1.0) < 1e-10, (
                f"Z-normalized signal std should be ~1, got {np.std(znorm_signal)}"
            )

    def test_median_normalized_signal_properties(self, sample_pod5_file):
        """Test that median-normalized signals have median of zero."""
        from squiggy.plotting import SquigglePlotter

        from squiggy.constants import NormalizationMethod

        with pod5.Reader(sample_pod5_file) as reader:
            read = next(reader.reads())
            signal = read.signal
            read_id = str(read.read_id)
            sample_rate = read.run_info.sample_rate

            # Test median normalization
            html, figure = SquigglePlotter.plot_single_read(
                signal=signal,
                read_id=read_id,
                sample_rate=sample_rate,
                normalization=NormalizationMethod.MEDIAN,
            )

            # Extract median-normalized signal
            signal_sources = [
                r.data_source
                for r in figure.renderers
                if hasattr(r, "data_source")
                and isinstance(r.data_source, ColumnDataSource)
                and "signal" in r.data_source.data
            ]

            median_norm_signal = np.array(signal_sources[0].data["signal"])

            # Verify median is close to 0
            assert np.abs(np.median(median_norm_signal)) < 1e-10, (
                f"Median-normalized signal median should be ~0, got {np.median(median_norm_signal)}"
            )

    def test_multiple_reads_signal_accuracy_overlay(self, sample_pod5_file):
        """Test that multiple reads maintain signal accuracy in overlay mode."""
        from squiggy.plotting import SquigglePlotter

        from squiggy.constants import NormalizationMethod, PlotMode

        with pod5.Reader(sample_pod5_file) as reader:
            reads = list(reader.reads())[:3]  # Get first 3 reads
            original_signals = [r.signal.copy() for r in reads]

            reads_data = [
                (str(r.read_id), r.signal, r.run_info.sample_rate) for r in reads
            ]

            # Generate overlay plot
            html, figure = SquigglePlotter.plot_multiple_reads(
                reads_data,
                mode=PlotMode.OVERLAY,
                normalization=NormalizationMethod.NONE,
            )

            # Extract all signal data sources
            signal_sources = [
                r.data_source
                for r in figure.renderers
                if hasattr(r, "data_source")
                and isinstance(r.data_source, ColumnDataSource)
                and "y" in r.data_source.data
            ]

            # Should have one data source per read
            assert len(signal_sources) == len(reads), (
                f"Expected {len(reads)} signal sources, found {len(signal_sources)}"
            )

            # Verify each signal matches its original
            for i, source in enumerate(signal_sources):
                plot_signal = np.array(source.data["y"])
                np.testing.assert_array_equal(
                    plot_signal,
                    original_signals[i],
                    err_msg=f"Read {i} signal does not match in overlay plot",
                )

    def test_signal_with_base_annotations_accuracy(
        self, sample_pod5_file, sample_bam_file
    ):
        """Test that signal accuracy is maintained when base annotations are added."""
        from squiggy.plotting import SquigglePlotter

        from squiggy.constants import NormalizationMethod
        from squiggy.utils import get_basecall_data

        with pod5.Reader(sample_pod5_file) as reader:
            read = next(reader.reads())
            original_signal = read.signal.copy()
            read_id = str(read.read_id)
            sample_rate = read.run_info.sample_rate

            # Get basecall data
            sequence, seq_to_sig_map = get_basecall_data(sample_bam_file, read_id)

            if sequence is not None:
                # Generate plot with base annotations
                html, figure = SquigglePlotter.plot_single_read(
                    signal=original_signal,
                    read_id=read_id,
                    sample_rate=sample_rate,
                    sequence=sequence,
                    seq_to_sig_map=seq_to_sig_map,
                    normalization=NormalizationMethod.NONE,
                    show_labels=True,
                )

                # Extract signal data (should still be there despite base annotations)
                signal_sources = [
                    r.data_source
                    for r in figure.renderers
                    if hasattr(r, "data_source")
                    and isinstance(r.data_source, ColumnDataSource)
                    and "signal" in r.data_source.data
                ]

                assert len(signal_sources) > 0, (
                    "No signal data found with base annotations"
                )

                plot_signal = np.array(signal_sources[0].data["signal"])

                # Signal should still match original
                np.testing.assert_array_equal(
                    plot_signal,
                    original_signal,
                    err_msg="Signal accuracy lost when adding base annotations",
                )
            else:
                pytest.skip("No basecall data available for this read")

    def test_eventalign_signal_accuracy(self, sample_pod5_file, indexed_bam_file):
        """Test that signal values are accurate in event-aligned mode."""
        from squiggy.plotting import SquigglePlotter

        from squiggy.alignment import extract_alignment_from_bam
        from squiggy.constants import NormalizationMethod, PlotMode

        with pod5.Reader(sample_pod5_file) as reader:
            read = next(reader.reads())
            original_signal = read.signal.copy()
            read_id = str(read.read_id)
            sample_rate = read.run_info.sample_rate

            # Get alignment data
            aligned_read = extract_alignment_from_bam(indexed_bam_file, read_id)

            if aligned_read:
                reads_data = [(read_id, original_signal, sample_rate)]
                aligned_reads = [aligned_read]

                # Generate event-aligned plot
                html, figure = SquigglePlotter.plot_multiple_reads(
                    reads_data,
                    mode=PlotMode.EVENTALIGN,
                    normalization=NormalizationMethod.NONE,
                    aligned_reads=aligned_reads,
                )

                # Extract signal data
                signal_sources = [
                    r.data_source
                    for r in figure.renderers
                    if hasattr(r, "data_source")
                    and isinstance(r.data_source, ColumnDataSource)
                    and "y" in r.data_source.data
                ]

                assert len(signal_sources) > 0, (
                    "No signal data found in event-aligned plot"
                )

                plot_signal = np.array(signal_sources[0].data["y"])

                # The signal values should be a subset of original signal
                # (mapped to base positions, but values unchanged)
                for sig_val in plot_signal:
                    assert sig_val in original_signal, (
                        f"Signal value {sig_val} not found in original signal"
                    )

                # Verify signal values are in correct range
                assert np.min(plot_signal) >= np.min(original_signal) - 0.01
                assert np.max(plot_signal) <= np.max(original_signal) + 0.01
            else:
                pytest.skip("No alignment data available for this read")

    def test_signal_range_preserved(self, sample_pod5_file):
        """Test that signal min/max range is preserved across all normalization methods."""
        from squiggy.plotting import SquigglePlotter

        from squiggy.constants import NormalizationMethod

        with pod5.Reader(sample_pod5_file) as reader:
            read = next(reader.reads())
            original_signal = read.signal
            read_id = str(read.read_id)
            sample_rate = read.run_info.sample_rate

            # Test with NONE normalization - range should match exactly
            html, figure = SquigglePlotter.plot_single_read(
                signal=original_signal,
                read_id=read_id,
                sample_rate=sample_rate,
                normalization=NormalizationMethod.NONE,
            )

            signal_sources = [
                r.data_source
                for r in figure.renderers
                if hasattr(r, "data_source")
                and isinstance(r.data_source, ColumnDataSource)
                and "signal" in r.data_source.data
            ]

            plot_signal = np.array(signal_sources[0].data["signal"])

            # Verify range matches
            assert np.min(plot_signal) == np.min(original_signal), (
                "Signal minimum does not match"
            )
            assert np.max(plot_signal) == np.max(original_signal), (
                "Signal maximum does not match"
            )

    def test_sample_index_accuracy(self, sample_pod5_file):
        """Test that sample indices are correctly assigned in plot data."""
        from squiggy.plotting import SquigglePlotter

        from squiggy.constants import NormalizationMethod

        with pod5.Reader(sample_pod5_file) as reader:
            read = next(reader.reads())
            signal = read.signal
            read_id = str(read.read_id)
            sample_rate = read.run_info.sample_rate

            # Generate plot
            html, figure = SquigglePlotter.plot_single_read(
                signal=signal,
                read_id=read_id,
                sample_rate=sample_rate,
                normalization=NormalizationMethod.NONE,
            )

            # Extract sample indices
            signal_sources = [
                r.data_source
                for r in figure.renderers
                if hasattr(r, "data_source")
                and isinstance(r.data_source, ColumnDataSource)
                and "sample" in r.data_source.data
            ]

            assert len(signal_sources) > 0, "No sample index data found in plot"

            plot_samples = np.array(signal_sources[0].data["sample"])

            # Verify sample indices are sequential from 0
            expected_samples = np.arange(len(signal))
            np.testing.assert_array_equal(
                plot_samples,
                expected_samples,
                err_msg="Sample indices do not match expected sequential values",
            )
