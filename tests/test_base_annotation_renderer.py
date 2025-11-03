"""
Tests for BaseAnnotationRenderer class
"""

import numpy as np
import pytest
from bokeh.models import LinearColorMapper
from bokeh.plotting import figure

from squiggy.constants import Theme
from squiggy.rendering import BaseAnnotationRenderer


# Mock BaseAnnotation class for testing
class MockBaseAnnotation:
    """Mock base annotation for testing"""

    def __init__(self, base: str, signal_start: int):
        self.base = base
        self.signal_start = signal_start


class TestBaseAnnotationRendererInitialization:
    """Tests for BaseAnnotationRenderer initialization"""

    def test_init_minimal(self):
        """Test initialization with minimal parameters"""
        base_colors = {"A": "#00b388", "C": "#3c8dbc", "G": "#d8ce0d", "T": "#f18033"}

        renderer = BaseAnnotationRenderer(base_colors=base_colors)

        assert renderer.base_colors == base_colors
        assert renderer.show_dwell_time is False
        assert renderer.show_labels is False

    def test_init_with_dwell_time(self):
        """Test initialization with dwell time enabled"""
        base_colors = {"A": "#00b388", "C": "#3c8dbc", "G": "#d8ce0d", "T": "#f18033"}

        renderer = BaseAnnotationRenderer(base_colors=base_colors, show_dwell_time=True)

        assert renderer.show_dwell_time is True

    def test_init_with_labels(self):
        """Test initialization with labels enabled"""
        base_colors = {"A": "#00b388", "C": "#3c8dbc", "G": "#d8ce0d", "T": "#f18033"}

        renderer = BaseAnnotationRenderer(base_colors=base_colors, show_labels=True)

        assert renderer.show_labels is True

    def test_init_all_options(self):
        """Test initialization with all options enabled"""
        base_colors = {"A": "#00b388", "C": "#3c8dbc", "G": "#d8ce0d", "T": "#f18033"}

        renderer = BaseAnnotationRenderer(
            base_colors=base_colors, show_dwell_time=True, show_labels=True
        )

        assert renderer.base_colors == base_colors
        assert renderer.show_dwell_time is True
        assert renderer.show_labels is True


class TestTimeBasedRendering:
    """Tests for time-based rendering (single read mode)"""

    @pytest.fixture
    def base_colors(self):
        """Base color dictionary for testing"""
        return {"A": "#00b388", "C": "#3c8dbc", "G": "#d8ce0d", "T": "#f18033"}

    @pytest.fixture
    def sample_data(self):
        """Sample data for time-based rendering"""
        sequence = "ACGT"
        seq_to_sig_map = [0, 100, 200, 300]
        time_ms = np.arange(0, 400, 1).astype(float)
        signal = np.random.randn(400)
        sample_rate = 4000
        signal_min = -2.5
        signal_max = 2.5

        return {
            "sequence": sequence,
            "seq_to_sig_map": seq_to_sig_map,
            "time_ms": time_ms,
            "signal": signal,
            "sample_rate": sample_rate,
            "signal_min": signal_min,
            "signal_max": signal_max,
        }

    def test_render_time_based_returns_none_without_dwell(
        self, base_colors, sample_data
    ):
        """Test that render_time_based returns None when not using dwell time"""
        renderer = BaseAnnotationRenderer(base_colors=base_colors)
        fig = figure(width=800, height=400)

        result = renderer.render_time_based(fig=fig, **sample_data)

        assert result is None

    def test_render_time_based_returns_mapper_with_dwell(
        self, base_colors, sample_data
    ):
        """Test that render_time_based returns LinearColorMapper with dwell time"""
        renderer = BaseAnnotationRenderer(base_colors=base_colors, show_dwell_time=True)
        fig = figure(width=800, height=400)

        result = renderer.render_time_based(fig=fig, **sample_data)

        assert isinstance(result, LinearColorMapper)

    def test_render_time_based_adds_glyphs_to_figure(self, base_colors, sample_data):
        """Test that render_time_based adds glyphs to figure"""
        renderer = BaseAnnotationRenderer(base_colors=base_colors)
        fig = figure(width=800, height=400)

        initial_glyph_count = len(fig.renderers)
        renderer.render_time_based(fig=fig, **sample_data)

        # Should add quad glyphs for base patches
        assert len(fig.renderers) > initial_glyph_count

    def test_render_time_based_with_labels(self, base_colors, sample_data):
        """Test that render_time_based adds labels when show_labels=True"""
        renderer = BaseAnnotationRenderer(base_colors=base_colors, show_labels=True)
        fig = figure(width=800, height=400)

        renderer.render_time_based(fig=fig, **sample_data)

        # Should have both quad (patches) and text (labels) glyphs
        # Text glyphs are also renderers
        assert len(fig.renderers) > 4  # At least 4 bases worth of glyphs

    def test_render_time_based_with_empty_sequence(self, base_colors):
        """Test that render_time_based handles empty sequence"""
        renderer = BaseAnnotationRenderer(base_colors=base_colors)
        fig = figure(width=800, height=400)

        data = {
            "sequence": "",
            "seq_to_sig_map": [],
            "time_ms": np.array([]),
            "signal": np.array([]),
            "sample_rate": 4000,
            "signal_min": -2.5,
            "signal_max": 2.5,
        }

        # Should not raise error
        result = renderer.render_time_based(fig=fig, **data)
        assert result is None


class TestPositionBasedRendering:
    """Tests for position-based rendering (event-aligned mode)"""

    @pytest.fixture
    def base_colors(self):
        """Base color dictionary for testing"""
        return {
            "A": "#00b388",
            "C": "#3c8dbc",
            "G": "#d8ce0d",
            "T": "#f18033",
            "U": "#ff0000",
        }

    @pytest.fixture
    def sample_annotations(self):
        """Sample base annotations for testing"""
        return [
            MockBaseAnnotation("A", 0),
            MockBaseAnnotation("C", 100),
            MockBaseAnnotation("G", 200),
            MockBaseAnnotation("T", 300),
        ]

    def test_render_position_based_adds_glyphs(self, base_colors, sample_annotations):
        """Test that render_position_based adds glyphs to figure"""
        renderer = BaseAnnotationRenderer(base_colors=base_colors)
        fig = figure(width=800, height=400)

        initial_glyph_count = len(fig.renderers)

        renderer.render_position_based(
            fig=fig,
            base_annotations=sample_annotations,
            sample_rate=4000,
            signal_length=400,
            signal_min=-2.5,
            signal_max=2.5,
        )

        # Should add quad glyphs for base patches
        assert len(fig.renderers) > initial_glyph_count

    def test_render_position_based_with_labels(self, base_colors, sample_annotations):
        """Test that render_position_based adds labels when show_labels=True"""
        renderer = BaseAnnotationRenderer(base_colors=base_colors, show_labels=True)
        fig = figure(width=800, height=400)

        renderer.render_position_based(
            fig=fig,
            base_annotations=sample_annotations,
            sample_rate=4000,
            signal_length=400,
            signal_min=-2.5,
            signal_max=2.5,
        )

        # Should have both patches and labels
        assert len(fig.renderers) > 4

    def test_render_position_based_with_dwell_time(
        self, base_colors, sample_annotations
    ):
        """Test render_position_based with dwell time scaling"""
        renderer = BaseAnnotationRenderer(base_colors=base_colors, show_dwell_time=True)
        fig = figure(width=800, height=400)

        renderer.render_position_based(
            fig=fig,
            base_annotations=sample_annotations,
            sample_rate=4000,
            signal_length=400,
            signal_min=-2.5,
            signal_max=2.5,
        )

        # Should add glyphs successfully
        assert len(fig.renderers) > 0

    def test_render_position_based_with_empty_annotations(self, base_colors):
        """Test render_position_based with empty annotations list"""
        renderer = BaseAnnotationRenderer(base_colors=base_colors)
        fig = figure(width=800, height=400)

        # Should not raise error
        renderer.render_position_based(
            fig=fig,
            base_annotations=[],
            sample_rate=4000,
            signal_length=400,
            signal_min=-2.5,
            signal_max=2.5,
        )

    def test_render_position_based_with_custom_label_interval(
        self, base_colors, sample_annotations
    ):
        """Test render_position_based with custom position label interval"""
        renderer = BaseAnnotationRenderer(base_colors=base_colors, show_labels=True)
        fig = figure(width=800, height=400)

        # Should not raise error with custom interval
        renderer.render_position_based(
            fig=fig,
            base_annotations=sample_annotations,
            sample_rate=4000,
            signal_length=400,
            signal_min=-2.5,
            signal_max=2.5,
            position_label_interval=10,
        )

        assert len(fig.renderers) > 0


class TestPrivateRegionCalculation:
    """Tests for private region calculation methods"""

    @pytest.fixture
    def base_colors(self):
        return {"A": "#00b388", "C": "#3c8dbc", "G": "#d8ce0d", "T": "#f18033"}

    def test_calculate_regions_time_dwell(self, base_colors):
        """Test _calculate_regions_time_dwell returns correct structure"""
        renderer = BaseAnnotationRenderer(base_colors=base_colors, show_dwell_time=True)

        sequence = "ACGT"
        seq_to_sig_map = [0, 100, 200, 300]
        time_ms = np.arange(0, 400, 1).astype(float)
        signal = np.random.randn(400)
        sample_rate = 4000

        regions, dwell_times, labels = renderer._calculate_regions_time_dwell(
            sequence=sequence,
            seq_to_sig_map=seq_to_sig_map,
            time_ms=time_ms,
            signal=signal,
            signal_min=-2.5,
            signal_max=2.5,
            sample_rate=sample_rate,
        )

        # Check structure
        assert isinstance(regions, list)
        assert isinstance(dwell_times, list)
        assert isinstance(labels, list)
        assert len(regions) == len(sequence)
        assert len(dwell_times) == len(sequence)
        assert len(labels) == len(sequence)

        # Check region dict structure
        if regions:
            assert "left" in regions[0]
            assert "right" in regions[0]
            assert "top" in regions[0]
            assert "bottom" in regions[0]
            assert "dwell" in regions[0]

    def test_calculate_regions_time_base_type(self, base_colors):
        """Test _calculate_regions_time_base_type returns correct structure"""
        renderer = BaseAnnotationRenderer(base_colors=base_colors)

        sequence = "ACGT"
        seq_to_sig_map = [0, 100, 200, 300]
        time_ms = np.arange(0, 400, 1).astype(float)
        signal = np.random.randn(400)

        base_regions, base_labels = renderer._calculate_regions_time_base_type(
            sequence=sequence,
            seq_to_sig_map=seq_to_sig_map,
            time_ms=time_ms,
            signal=signal,
            signal_min=-2.5,
            signal_max=2.5,
        )

        # Check structure
        assert isinstance(base_regions, dict)
        assert isinstance(base_labels, dict)
        assert "A" in base_regions
        assert "C" in base_regions
        assert "G" in base_regions
        assert "T" in base_regions

    def test_calculate_regions_position(self, base_colors):
        """Test _calculate_regions_position returns correct structure"""
        renderer = BaseAnnotationRenderer(base_colors=base_colors)

        annotations = [
            MockBaseAnnotation("A", 0),
            MockBaseAnnotation("C", 100),
            MockBaseAnnotation("G", 200),
        ]

        regions = renderer._calculate_regions_position(
            base_annotations=annotations,
            signal_min=-2.5,
            signal_max=2.5,
            sample_rate=4000,
            signal_length=400,
        )

        assert isinstance(regions, dict)
        assert "A" in regions
        assert "C" in regions
        assert "G" in regions
        assert len(regions["A"]) == 1  # One A base
        assert len(regions["C"]) == 1  # One C base
        assert len(regions["G"]) == 1  # One G base


class TestDwellTimeColorMapper:
    """Tests for dwell time color mapper"""

    @pytest.fixture
    def base_colors(self):
        return {"A": "#00b388", "C": "#3c8dbc", "G": "#d8ce0d", "T": "#f18033"}

    def test_dwell_patches_creates_color_mapper(self, base_colors):
        """Test that _add_dwell_patches creates a LinearColorMapper"""
        renderer = BaseAnnotationRenderer(base_colors=base_colors, show_dwell_time=True)
        fig = figure(width=800, height=400)

        regions = [
            {"left": 0, "right": 100, "top": 2.5, "bottom": -2.5, "dwell": 25.0},
            {"left": 100, "right": 200, "top": 2.5, "bottom": -2.5, "dwell": 30.0},
        ]
        dwell_times = [25.0, 30.0]

        color_mapper = renderer._add_dwell_patches(fig, regions, dwell_times)

        assert isinstance(color_mapper, LinearColorMapper)
        # Bokeh expands "Viridis256" into a tuple of 256 hex colors
        assert isinstance(color_mapper.palette, tuple)
        assert len(color_mapper.palette) == 256

    def test_dwell_patches_with_empty_regions(self, base_colors):
        """Test that _add_dwell_patches handles empty regions"""
        renderer = BaseAnnotationRenderer(base_colors=base_colors, show_dwell_time=True)
        fig = figure(width=800, height=400)

        color_mapper = renderer._add_dwell_patches(fig, [], [])

        assert color_mapper is None


class TestBaseAnnotationRendererDocumentation:
    """Tests for BaseAnnotationRenderer documentation"""

    def test_class_has_docstring(self):
        """Test that BaseAnnotationRenderer has comprehensive docstring"""
        assert BaseAnnotationRenderer.__doc__ is not None
        assert len(BaseAnnotationRenderer.__doc__) > 100

    def test_render_time_based_has_docstring(self):
        """Test that render_time_based has detailed docstring"""
        assert BaseAnnotationRenderer.render_time_based.__doc__ is not None
        assert "Args:" in BaseAnnotationRenderer.render_time_based.__doc__
        assert "Returns:" in BaseAnnotationRenderer.render_time_based.__doc__

    def test_render_position_based_has_docstring(self):
        """Test that render_position_based has detailed docstring"""
        assert BaseAnnotationRenderer.render_position_based.__doc__ is not None
        assert "Args:" in BaseAnnotationRenderer.render_position_based.__doc__
        assert "Returns:" in BaseAnnotationRenderer.render_position_based.__doc__


class TestBaseAnnotationRendererIntegration:
    """Integration tests for BaseAnnotationRenderer"""

    @pytest.fixture
    def base_colors(self):
        return {"A": "#00b388", "C": "#3c8dbc", "G": "#d8ce0d", "T": "#f18033"}

    def test_render_complete_workflow_time_based(self, base_colors):
        """Test complete workflow for time-based rendering"""
        renderer = BaseAnnotationRenderer(
            base_colors=base_colors, show_dwell_time=True, show_labels=True
        )

        fig = figure(width=800, height=400)
        sequence = "ACGTACGT"
        seq_to_sig_map = [0, 50, 100, 150, 200, 250, 300, 350]
        time_ms = np.arange(0, 400, 1).astype(float)
        signal = np.random.randn(400)

        color_mapper = renderer.render_time_based(
            fig=fig,
            sequence=sequence,
            seq_to_sig_map=seq_to_sig_map,
            time_ms=time_ms,
            signal=signal,
            sample_rate=4000,
            signal_min=-2.5,
            signal_max=2.5,
        )

        # Should have color mapper
        assert isinstance(color_mapper, LinearColorMapper)

        # Should have glyphs added
        assert len(fig.renderers) > 0

    def test_render_complete_workflow_position_based(self, base_colors):
        """Test complete workflow for position-based rendering"""
        renderer = BaseAnnotationRenderer(
            base_colors=base_colors, show_dwell_time=False, show_labels=True
        )

        fig = figure(width=800, height=400)
        annotations = [
            MockBaseAnnotation("A", 0),
            MockBaseAnnotation("C", 100),
            MockBaseAnnotation("G", 200),
            MockBaseAnnotation("T", 300),
        ]

        renderer.render_position_based(
            fig=fig,
            base_annotations=annotations,
            sample_rate=4000,
            signal_length=400,
            signal_min=-2.5,
            signal_max=2.5,
            theme=Theme.LIGHT,
        )

        # Should have glyphs added
        assert len(fig.renderers) > 0

    def test_reuse_renderer_on_multiple_figures(self, base_colors):
        """Test that same renderer can be used on multiple figures"""
        renderer = BaseAnnotationRenderer(base_colors=base_colors)

        fig1 = figure(width=800, height=400)
        fig2 = figure(width=800, height=400)

        data = {
            "sequence": "ACGT",
            "seq_to_sig_map": [0, 100, 200, 300],
            "time_ms": np.arange(0, 400, 1).astype(float),
            "signal": np.random.randn(400),
            "sample_rate": 4000,
            "signal_min": -2.5,
            "signal_max": 2.5,
        }

        renderer.render_time_based(fig=fig1, **data)
        renderer.render_time_based(fig=fig2, **data)

        # Both figures should have glyphs
        assert len(fig1.renderers) > 0
        assert len(fig2.renderers) > 0
