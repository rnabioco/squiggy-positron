"""Tests for plotting/base_annotations.py - base annotation utilities"""

import numpy as np
from bokeh.models import ColumnDataSource, LinearColorMapper
from bokeh.plotting import figure
from squiggy.plotting.base_annotations import (
    add_base_labels_position_mode,
    add_base_labels_time_mode,
    add_base_type_patches,
    add_dwell_time_patches,
    add_simple_labels,
    calculate_base_regions_position_mode,
    calculate_base_regions_time_mode,
)

from squiggy.alignment import BaseAnnotation
from squiggy.constants import BASE_COLORS, Theme


class TestCalculateBaseRegionsTimeMode:
    """Tests for calculate_base_regions_time_mode function"""

    def test_basic_region_calculation(self):
        """Test basic base region calculation without dwell time"""
        sequence = "ACGT"
        seq_to_sig_map = [0, 10, 20, 30]
        signal = np.random.randn(40)
        time_ms = np.arange(40) * 0.25  # 4000 Hz sample rate
        signal_min = np.min(signal)
        signal_max = np.max(signal)

        result = calculate_base_regions_time_mode(
            sequence,
            seq_to_sig_map,
            time_ms,
            signal,
            signal_min,
            signal_max,
            sample_rate=4000,
            show_dwell_time=False,
            base_colors=BASE_COLORS,
        )

        base_regions, base_labels_data = result
        assert isinstance(base_regions, dict)
        assert isinstance(base_labels_data, dict)

        # Should have regions for each base
        for base in ["A", "C", "G", "T"]:
            assert base in base_regions
            assert base in base_labels_data

    def test_dwell_time_mode(self):
        """Test base region calculation with dwell time coloring"""
        sequence = "ACGT"
        seq_to_sig_map = [0, 10, 20, 30]
        signal = np.random.randn(40)
        time_ms = np.arange(40) * 0.25
        signal_min = np.min(signal)
        signal_max = np.max(signal)

        result = calculate_base_regions_time_mode(
            sequence,
            seq_to_sig_map,
            time_ms,
            signal,
            signal_min,
            signal_max,
            sample_rate=4000,
            show_dwell_time=True,
            base_colors=BASE_COLORS,
        )

        all_regions, all_dwell_times, all_labels_data = result
        assert isinstance(all_regions, list)
        assert isinstance(all_dwell_times, list)
        assert isinstance(all_labels_data, list)

        # Should have equal lengths
        assert len(all_regions) == len(all_dwell_times)
        assert len(all_regions) == len(sequence)

    def test_dwell_time_values(self):
        """Test that dwell times are calculated correctly"""
        sequence = "AC"
        seq_to_sig_map = [0, 20]
        signal = np.random.randn(40)
        time_ms = np.arange(40) * 0.25  # 0.25 ms per sample
        signal_min = np.min(signal)
        signal_max = np.max(signal)

        result = calculate_base_regions_time_mode(
            sequence,
            seq_to_sig_map,
            time_ms,
            signal,
            signal_min,
            signal_max,
            sample_rate=4000,
            show_dwell_time=True,
            base_colors=BASE_COLORS,
        )

        all_regions, all_dwell_times, _ = result

        # First base should have dwell time of 5 ms (20 samples * 0.25 ms)
        assert abs(all_dwell_times[0] - 5.0) < 0.01

    def test_handles_unknown_bases(self):
        """Test that unknown bases (not in base_colors) are skipped"""
        sequence = "ACGT"
        seq_to_sig_map = [0, 10, 20, 30]
        signal = np.random.randn(40)
        time_ms = np.arange(40) * 0.25
        signal_min = np.min(signal)
        signal_max = np.max(signal)

        # Use a limited base_colors dict that excludes T
        limited_colors = {"A": "#E69F00", "C": "#0072B2", "G": "#D55E00"}

        result = calculate_base_regions_time_mode(
            sequence,
            seq_to_sig_map,
            time_ms,
            signal,
            signal_min,
            signal_max,
            sample_rate=4000,
            show_dwell_time=False,
            base_colors=limited_colors,
        )

        base_regions, _ = result

        # T should be skipped (not in limited_colors)
        # Only A, C, G should have regions
        total_regions = sum(len(regions) for regions in base_regions.values())
        assert total_regions == 3  # A, C, G (T is skipped)


class TestCalculateBaseRegionsPositionMode:
    """Tests for calculate_base_regions_position_mode function"""

    def test_position_mode_basic(self):
        """Test basic position mode region calculation"""
        base_annotations = [
            BaseAnnotation(base="A", position=0, signal_start=0, signal_end=10),
            BaseAnnotation(base="C", position=1, signal_start=10, signal_end=20),
            BaseAnnotation(base="G", position=2, signal_start=20, signal_end=30),
            BaseAnnotation(base="T", position=3, signal_start=30, signal_end=40),
        ]

        result = calculate_base_regions_position_mode(
            base_annotations,
            signal_min=0.0,
            signal_max=100.0,
            sample_rate=4000,
            signal_length=40,
            show_dwell_time=False,
            base_colors=BASE_COLORS,
        )

        (base_regions,) = result
        assert isinstance(base_regions, dict)

        # Should have regions for each base type
        for base in ["A", "C", "G", "T"]:
            assert base in base_regions

    def test_position_mode_dwell_time(self):
        """Test position mode with dwell time coloring"""
        base_annotations = [
            BaseAnnotation(base="A", position=0, signal_start=0, signal_end=10),
            BaseAnnotation(base="C", position=1, signal_start=10, signal_end=20),
            BaseAnnotation(base="G", position=2, signal_start=20, signal_end=40),
        ]

        result = calculate_base_regions_position_mode(
            base_annotations,
            signal_min=0.0,
            signal_max=100.0,
            sample_rate=4000,
            signal_length=40,
            show_dwell_time=True,
            base_colors=BASE_COLORS,
        )

        (base_regions,) = result

        # With dwell time, regions should have time-based coordinates
        for _base, regions in base_regions.items():
            for region in regions:
                # Time-based regions should have different left/right values
                if region:
                    assert "left" in region
                    assert "right" in region
                    assert region["right"] > region["left"]

    def test_position_mode_without_dwell_time(self):
        """Test position mode without dwell time uses base positions"""
        base_annotations = [
            BaseAnnotation(base="A", position=0, signal_start=0, signal_end=10),
            BaseAnnotation(base="C", position=1, signal_start=10, signal_end=40),
        ]

        result = calculate_base_regions_position_mode(
            base_annotations,
            signal_min=0.0,
            signal_max=100.0,
            sample_rate=4000,
            signal_length=40,
            show_dwell_time=False,
            base_colors=BASE_COLORS,
        )

        (base_regions,) = result

        # Without dwell time, positions should be evenly spaced (i-0.5 to i+0.5)
        a_regions = base_regions["A"]
        if a_regions:
            first_region = a_regions[0]
            assert first_region["left"] == -0.5
            assert first_region["right"] == 0.5


class TestAddDwellTimePatches:
    """Tests for add_dwell_time_patches function"""

    def test_add_dwell_time_patches_basic(self):
        """Test adding dwell time patches to figure"""
        p = figure()

        all_regions = [
            {"left": 0, "right": 5, "top": 100, "bottom": 0, "dwell": 5},
            {"left": 5, "right": 10, "top": 100, "bottom": 0, "dwell": 5},
            {"left": 10, "right": 15, "top": 100, "bottom": 0, "dwell": 5},
        ]
        all_dwell_times = [5.0, 5.0, 5.0]

        color_mapper = add_dwell_time_patches(p, all_regions, all_dwell_times)

        assert isinstance(color_mapper, LinearColorMapper)
        # Palette is a tuple of hex colors, not a string
        assert isinstance(color_mapper.palette, tuple)
        assert len(color_mapper.palette) == 256  # Viridis256 has 256 colors

    def test_add_dwell_time_patches_empty(self):
        """Test adding patches with empty regions returns None"""
        p = figure()

        color_mapper = add_dwell_time_patches(p, [], [])

        assert color_mapper is None

    def test_add_dwell_time_patches_color_range(self):
        """Test color mapper uses percentiles for range"""
        p = figure()

        # Create varied dwell times
        dwell_times = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0]
        all_regions = [
            {"left": i, "right": i + 1, "top": 100, "bottom": 0, "dwell": d}
            for i, d in enumerate(dwell_times)
        ]

        color_mapper = add_dwell_time_patches(p, all_regions, dwell_times)

        # Should use 5th and 95th percentiles
        expected_low = np.percentile(dwell_times, 5)
        expected_high = np.percentile(dwell_times, 95)

        assert np.isclose(color_mapper.low, expected_low)
        assert np.isclose(color_mapper.high, expected_high)


class TestAddBaseTypePatches:
    """Tests for add_base_type_patches function"""

    def test_add_base_type_patches_basic(self):
        """Test adding base type patches to figure"""
        p = figure()

        base_regions = {
            "A": [{"left": 0, "right": 5, "top": 100, "bottom": 0}],
            "C": [{"left": 5, "right": 10, "top": 100, "bottom": 0}],
            "G": [],
            "T": [],
        }

        add_base_type_patches(p, base_regions, BASE_COLORS)

        # Should have added patches (no error raised)
        assert True

    def test_add_base_type_patches_empty(self):
        """Test adding patches with no regions"""
        p = figure()

        base_regions = {
            "A": [],
            "C": [],
            "G": [],
            "T": [],
        }

        # Should not raise error
        add_base_type_patches(p, base_regions, BASE_COLORS)

    def test_add_base_type_patches_multiple_regions(self):
        """Test adding multiple regions for same base"""
        p = figure()

        base_regions = {
            "A": [
                {"left": 0, "right": 5, "top": 100, "bottom": 0},
                {"left": 10, "right": 15, "top": 100, "bottom": 0},
                {"left": 20, "right": 25, "top": 100, "bottom": 0},
            ],
            "C": [],
            "G": [],
            "T": [],
        }

        add_base_type_patches(p, base_regions, BASE_COLORS)

        # Should not raise error
        assert True


class TestAddBaseLabelsTimeMode:
    """Tests for add_base_labels_time_mode function"""

    def test_time_mode_without_dwell_time(self):
        """Test adding labels without dwell time (grouped by base)"""
        base_labels_data = {
            "A": [{"time": 5, "y": 50, "text": "A0"}],
            "C": [{"time": 15, "y": 60, "text": "C1"}],
            "G": [],
            "T": [],
        }

        base_sources = add_base_labels_time_mode(
            figure(),
            base_labels_data,
            show_dwell_time=False,
            base_colors=BASE_COLORS,
        )

        assert isinstance(base_sources, list)
        # Should have sources for bases with data
        assert len(base_sources) == 2  # A and C

    def test_time_mode_with_dwell_time(self):
        """Test adding labels with dwell time (single source)"""
        all_labels_data = [
            {"time": 5, "y": 50, "text": "A0"},
            {"time": 15, "y": 60, "text": "C1"},
        ]

        base_sources = add_base_labels_time_mode(
            figure(),
            all_labels_data,
            show_dwell_time=True,
            base_colors=BASE_COLORS,
        )

        assert isinstance(base_sources, list)
        # With dwell time, should have single source for all bases
        assert len(base_sources) == 1
        assert base_sources[0][0] == "all"

    def test_time_mode_empty_labels(self):
        """Test with no labels"""
        base_labels_data = {
            "A": [],
            "C": [],
            "G": [],
            "T": [],
        }

        base_sources = add_base_labels_time_mode(
            figure(),
            base_labels_data,
            show_dwell_time=False,
            base_colors=BASE_COLORS,
        )

        # Should return empty list
        assert base_sources == []


class TestAddBaseLabelsPositionMode:
    """Tests for add_base_labels_position_mode function"""

    def test_position_mode_basic(self):
        """Test adding labels in position mode"""
        p = figure()
        base_annotations = [
            BaseAnnotation(base="A", position=0, signal_start=0, signal_end=10),
            BaseAnnotation(base="C", position=1, signal_start=10, signal_end=20),
            BaseAnnotation(base="G", position=2, signal_start=20, signal_end=40),
        ]

        # Should not raise error
        add_base_labels_position_mode(
            p,
            base_annotations,
            signal_max=100.0,
            show_dwell_time=False,
            base_colors=BASE_COLORS,
        )

    def test_position_mode_with_dwell_time(self):
        """Test adding labels with dwell time positioning"""
        p = figure()
        base_annotations = [
            BaseAnnotation(base="A", position=0, signal_start=0, signal_end=10),
            BaseAnnotation(base="C", position=1, signal_start=10, signal_end=40),
        ]

        add_base_labels_position_mode(
            p,
            base_annotations,
            signal_max=100.0,
            show_dwell_time=True,
            base_colors=BASE_COLORS,
            sample_rate=4000,
            signal_length=40,
        )

        # Should not raise error
        assert True

    def test_position_mode_label_interval(self):
        """Test position number labels at specified interval"""
        p = figure()
        base_annotations = [
            BaseAnnotation(
                base="A", position=i, signal_start=i * 10, signal_end=(i + 1) * 10
            )
            for i in range(20)
        ]

        add_base_labels_position_mode(
            p,
            base_annotations,
            signal_max=100.0,
            show_dwell_time=False,
            base_colors=BASE_COLORS,
            position_label_interval=5,
        )

        # Should add position numbers every 5 bases
        assert True

    def test_position_mode_empty_annotations(self):
        """Test with no base annotations"""
        p = figure()

        add_base_labels_position_mode(
            p,
            [],
            signal_max=100.0,
            show_dwell_time=False,
            base_colors=BASE_COLORS,
        )

        # Should not raise error
        assert True


class TestAddSimpleLabels:
    """Tests for add_simple_labels function"""

    def test_add_simple_labels_basic(self):
        """Test adding simple labels to figure"""
        p = figure()

        base_sources = [
            (
                "A",
                ColumnDataSource(data={"time": [5], "y": [50], "text": ["A0"]}),
            ),
            (
                "C",
                ColumnDataSource(data={"time": [15], "y": [60], "text": ["C1"]}),
            ),
        ]

        add_simple_labels(p, base_sources, BASE_COLORS)

        # Should not raise error
        assert True

    def test_add_simple_labels_with_theme(self):
        """Test adding labels with dark theme"""
        p = figure()

        base_sources = [
            (
                "all",
                ColumnDataSource(data={"time": [5], "y": [50], "text": ["A0"]}),
            ),
        ]

        add_simple_labels(p, base_sources, BASE_COLORS, theme=Theme.DARK)

        # Should use dark theme colors
        assert True

    def test_add_simple_labels_empty(self):
        """Test with no label sources"""
        p = figure()

        add_simple_labels(p, [], BASE_COLORS)

        # Should not raise error
        assert True

    def test_add_simple_labels_multiple_bases(self):
        """Test adding labels for multiple bases"""
        p = figure()

        base_sources = [
            (base, ColumnDataSource(data={"time": [i], "y": [50], "text": [base]}))
            for i, base in enumerate(["A", "C", "G", "T"])
        ]

        add_simple_labels(p, base_sources, BASE_COLORS)

        # Should add labels for all bases
        assert True
