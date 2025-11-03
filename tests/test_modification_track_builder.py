"""
Tests for ModificationTrackBuilder class
"""

import numpy as np
import pytest
from bokeh.models.plots import Plot

from squiggy.constants import Theme
from squiggy.rendering import ModificationTrackBuilder


# Mock ModificationAnnotation class for testing
class MockModification:
    """Mock modification annotation for testing"""

    def __init__(
        self,
        position: int,
        mod_code: str | int,
        canonical_base: str,
        probability: float,
        signal_start: int,
        signal_end: int,
    ):
        self.position = position
        self.mod_code = mod_code
        self.canonical_base = canonical_base
        self.probability = probability
        self.signal_start = signal_start
        self.signal_end = signal_end


class TestModificationTrackBuilderInitialization:
    """Tests for ModificationTrackBuilder initialization"""

    def test_init_minimal(self):
        """Test initialization with default parameters"""
        builder = ModificationTrackBuilder()

        assert builder.min_probability == 0.5
        assert builder.enabled_types is None
        assert builder.overlay_opacity == 0.8
        assert builder.theme == Theme.LIGHT

    def test_init_with_min_probability(self):
        """Test initialization with custom min_probability"""
        builder = ModificationTrackBuilder(min_probability=0.7)

        assert builder.min_probability == 0.7

    def test_init_with_enabled_types(self):
        """Test initialization with enabled types filter"""
        builder = ModificationTrackBuilder(enabled_types=["m", "h", "a"])

        assert builder.enabled_types == ["m", "h", "a"]

    def test_init_with_opacity(self):
        """Test initialization with custom overlay opacity"""
        builder = ModificationTrackBuilder(overlay_opacity=0.6)

        assert builder.overlay_opacity == 0.6

    def test_init_with_dark_theme(self):
        """Test initialization with DARK theme"""
        builder = ModificationTrackBuilder(theme=Theme.DARK)

        assert builder.theme == Theme.DARK

    def test_init_all_parameters(self):
        """Test initialization with all parameters"""
        builder = ModificationTrackBuilder(
            min_probability=0.9,
            enabled_types=["m"],
            overlay_opacity=0.5,
            theme=Theme.DARK,
        )

        assert builder.min_probability == 0.9
        assert builder.enabled_types == ["m"]
        assert builder.overlay_opacity == 0.5
        assert builder.theme == Theme.DARK


class TestBuildTrack:
    """Tests for build_track method"""

    @pytest.fixture
    def sample_data(self):
        """Sample data for building modification tracks"""
        return {
            "sequence": "ACGTACGT",
            "seq_to_sig_map": [0, 100, 200, 300, 400, 500, 600, 700],
            "time_ms": np.arange(0, 800, 1).astype(float),
            "sample_rate": 4000,
        }

    @pytest.fixture
    def sample_modifications(self):
        """Sample modifications for testing"""
        return [
            MockModification(0, "m", "C", 0.9, 0, 100),
            MockModification(1, "h", "C", 0.8, 100, 200),
            MockModification(2, "a", "A", 0.7, 200, 300),
            MockModification(3, "m", "C", 0.6, 300, 400),
        ]

    def test_build_track_returns_figure(self, sample_data, sample_modifications):
        """Test that build_track returns a Bokeh figure"""
        builder = ModificationTrackBuilder()

        fig = builder.build_track(modifications=sample_modifications, **sample_data)

        assert isinstance(fig, Plot)

    def test_build_track_returns_none_without_modifications(self, sample_data):
        """Test that build_track returns None when no modifications"""
        builder = ModificationTrackBuilder()

        fig = builder.build_track(modifications=None, **sample_data)

        assert fig is None

    def test_build_track_returns_none_with_empty_list(self, sample_data):
        """Test that build_track returns None with empty modifications list"""
        builder = ModificationTrackBuilder()

        fig = builder.build_track(modifications=[], **sample_data)

        assert fig is None

    def test_build_track_returns_none_without_sequence(self, sample_modifications):
        """Test that build_track returns None without sequence"""
        builder = ModificationTrackBuilder()

        fig = builder.build_track(
            sequence=None,
            seq_to_sig_map=[0, 100, 200],
            time_ms=np.arange(0, 300, 1).astype(float),
            sample_rate=4000,
            modifications=sample_modifications,
        )

        assert fig is None

    def test_build_track_returns_none_without_mapping(self, sample_modifications):
        """Test that build_track returns None without seq_to_sig_map"""
        builder = ModificationTrackBuilder()

        fig = builder.build_track(
            sequence="ACGT",
            seq_to_sig_map=None,
            time_ms=np.arange(0, 300, 1).astype(float),
            sample_rate=4000,
            modifications=sample_modifications,
        )

        assert fig is None

    def test_build_track_filters_by_probability(self, sample_data):
        """Test that build_track filters by minimum probability"""
        # Create mods with probabilities: 0.9, 0.8, 0.7, 0.5
        mods = [
            MockModification(0, "m", "C", 0.9, 0, 100),
            MockModification(1, "h", "C", 0.8, 100, 200),
            MockModification(2, "a", "A", 0.7, 200, 300),
            MockModification(3, "m", "C", 0.5, 300, 400),
        ]

        # Filter with min_probability=0.75 (should keep 0.9 and 0.8)
        builder = ModificationTrackBuilder(min_probability=0.75)
        fig = builder.build_track(modifications=mods, **sample_data)

        assert isinstance(fig, Plot)
        # Figure should have glyphs (rectangles)
        assert len(fig.renderers) > 0

    def test_build_track_filters_by_type(self, sample_data):
        """Test that build_track filters by enabled modification types"""
        mods = [
            MockModification(0, "m", "C", 0.9, 0, 100),  # 5mC
            MockModification(1, "h", "C", 0.8, 100, 200),  # 5hmC
            MockModification(2, "a", "A", 0.7, 200, 300),  # 6mA
        ]

        # Only enable 'm' type
        builder = ModificationTrackBuilder(enabled_types=["m"])
        fig = builder.build_track(modifications=mods, **sample_data)

        assert isinstance(fig, Plot)

    def test_build_track_returns_none_when_all_filtered(self, sample_data):
        """Test that build_track returns None when all mods filtered out"""
        mods = [
            MockModification(0, "m", "C", 0.5, 0, 100),
            MockModification(1, "h", "C", 0.6, 100, 200),
        ]

        # Set threshold too high
        builder = ModificationTrackBuilder(min_probability=0.9)
        fig = builder.build_track(modifications=mods, **sample_data)

        assert fig is None

    def test_build_track_figure_properties(self, sample_data, sample_modifications):
        """Test that built figure has expected properties"""
        builder = ModificationTrackBuilder()

        fig = builder.build_track(modifications=sample_modifications, **sample_data)

        # Check figure properties
        assert fig.height == 80
        assert fig.sizing_mode == "stretch_width"
        assert fig.toolbar_location is None
        assert fig.yaxis.visible is False
        assert fig.y_range.start == 0
        assert fig.y_range.end == 1


class TestUpdateFilters:
    """Tests for update_filters method"""

    def test_update_min_probability(self):
        """Test updating minimum probability threshold"""
        builder = ModificationTrackBuilder(min_probability=0.5)

        builder.update_filters(min_probability=0.8)

        assert builder.min_probability == 0.8

    def test_update_enabled_types(self):
        """Test updating enabled modification types"""
        builder = ModificationTrackBuilder(enabled_types=None)

        builder.update_filters(enabled_types=["m", "h"])

        assert builder.enabled_types == ["m", "h"]

    def test_update_both_filters(self):
        """Test updating both filters at once"""
        builder = ModificationTrackBuilder(min_probability=0.5, enabled_types=None)

        builder.update_filters(min_probability=0.7, enabled_types=["m"])

        assert builder.min_probability == 0.7
        assert builder.enabled_types == ["m"]

    def test_update_with_none_preserves_value(self):
        """Test that passing None preserves existing values"""
        builder = ModificationTrackBuilder(min_probability=0.6, enabled_types=["m"])

        builder.update_filters(min_probability=None, enabled_types=None)

        # Values should remain unchanged
        assert builder.min_probability == 0.6
        assert builder.enabled_types == ["m"]


class TestGetModificationSummary:
    """Tests for get_modification_summary method"""

    def test_summary_with_no_modifications(self):
        """Test summary with None modifications"""
        builder = ModificationTrackBuilder()

        summary = builder.get_modification_summary(None)

        assert summary["total_mods"] == 0
        assert summary["filtered_mods"] == 0
        assert summary["enabled_mods"] == 0
        assert summary["mod_types"] == set()

    def test_summary_with_empty_list(self):
        """Test summary with empty modifications list"""
        builder = ModificationTrackBuilder()

        summary = builder.get_modification_summary([])

        assert summary["total_mods"] == 0
        assert summary["filtered_mods"] == 0
        assert summary["enabled_mods"] == 0
        assert summary["mod_types"] == set()

    def test_summary_with_modifications(self):
        """Test summary with sample modifications"""
        mods = [
            MockModification(0, "m", "C", 0.9, 0, 100),
            MockModification(1, "h", "C", 0.8, 100, 200),
            MockModification(2, "a", "A", 0.7, 200, 300),
            MockModification(3, "m", "C", 0.4, 300, 400),  # Below default threshold
        ]

        builder = ModificationTrackBuilder(min_probability=0.5)
        summary = builder.get_modification_summary(mods)

        assert summary["total_mods"] == 4
        assert summary["filtered_mods"] == 3  # 3 above 0.5
        assert summary["enabled_mods"] == 3  # All types enabled
        assert summary["mod_types"] == {"m", "h", "a"}

    def test_summary_with_type_filter(self):
        """Test summary with type filtering enabled"""
        mods = [
            MockModification(0, "m", "C", 0.9, 0, 100),
            MockModification(1, "h", "C", 0.8, 100, 200),
            MockModification(2, "a", "A", 0.7, 200, 300),
        ]

        builder = ModificationTrackBuilder(enabled_types=["m"])
        summary = builder.get_modification_summary(mods)

        assert summary["total_mods"] == 3
        assert summary["filtered_mods"] == 3
        assert summary["enabled_mods"] == 1  # Only 'm' type
        assert summary["mod_types"] == {"m", "h", "a"}

    def test_summary_with_probability_filter(self):
        """Test summary with probability filtering"""
        mods = [
            MockModification(0, "m", "C", 0.9, 0, 100),
            MockModification(1, "m", "C", 0.8, 100, 200),
            MockModification(2, "m", "C", 0.6, 200, 300),
            MockModification(3, "m", "C", 0.4, 300, 400),
        ]

        builder = ModificationTrackBuilder(min_probability=0.7)
        summary = builder.get_modification_summary(mods)

        assert summary["total_mods"] == 4
        assert summary["filtered_mods"] == 2  # 2 above 0.7
        assert summary["enabled_mods"] == 2
        assert summary["mod_types"] == {"m"}


class TestPrivateDataPreparation:
    """Tests for private _prepare_modification_data method"""

    @pytest.fixture
    def builder(self):
        return ModificationTrackBuilder(min_probability=0.5)

    def test_prepare_modification_data_structure(self, builder):
        """Test that _prepare_modification_data returns correct structure"""
        mods = [
            MockModification(0, "m", "C", 0.9, 0, 100),
        ]

        sequence = "ACGT"
        seq_to_sig_map = [0, 100, 200, 300]
        time_ms = np.arange(0, 400, 1).astype(float)

        data = builder._prepare_modification_data(
            sequence=sequence,
            seq_to_sig_map=seq_to_sig_map,
            time_ms=time_ms,
            modifications=mods,
        )

        # Check all required keys present
        assert "x" in data
        assert "y" in data
        assert "left" in data
        assert "right" in data
        assert "width" in data
        assert "mod_type" in data
        assert "mod_name" in data
        assert "probability" in data
        assert "base" in data
        assert "position" in data
        assert "color" in data

    def test_prepare_filters_by_probability(self, builder):
        """Test that data preparation filters by probability"""
        mods = [
            MockModification(0, "m", "C", 0.9, 0, 100),
            MockModification(1, "m", "C", 0.3, 100, 200),  # Below threshold
        ]

        data = builder._prepare_modification_data(
            sequence="ACGT",
            seq_to_sig_map=[0, 100, 200, 300],
            time_ms=np.arange(0, 400, 1).astype(float),
            modifications=mods,
        )

        # Should only have 1 modification (0.9)
        assert len(data["x"]) == 1
        assert data["probability"][0] == 0.9

    def test_prepare_filters_by_type(self):
        """Test that data preparation filters by modification type"""
        builder = ModificationTrackBuilder(enabled_types=["m"])

        mods = [
            MockModification(0, "m", "C", 0.9, 0, 100),
            MockModification(1, "h", "C", 0.9, 100, 200),  # Wrong type
        ]

        data = builder._prepare_modification_data(
            sequence="ACGT",
            seq_to_sig_map=[0, 100, 200, 300],
            time_ms=np.arange(0, 400, 1).astype(float),
            modifications=mods,
        )

        # Should only have 1 modification ('m')
        assert len(data["x"]) == 1
        assert data["mod_type"][0] == "m"


class TestModificationTrackBuilderDocumentation:
    """Tests for ModificationTrackBuilder documentation"""

    def test_class_has_docstring(self):
        """Test that ModificationTrackBuilder has comprehensive docstring"""
        assert ModificationTrackBuilder.__doc__ is not None
        assert len(ModificationTrackBuilder.__doc__) > 100

    def test_build_track_has_docstring(self):
        """Test that build_track has detailed docstring"""
        assert ModificationTrackBuilder.build_track.__doc__ is not None
        assert "Args:" in ModificationTrackBuilder.build_track.__doc__
        assert "Returns:" in ModificationTrackBuilder.build_track.__doc__

    def test_update_filters_has_docstring(self):
        """Test that update_filters has detailed docstring"""
        assert ModificationTrackBuilder.update_filters.__doc__ is not None
        assert "Args:" in ModificationTrackBuilder.update_filters.__doc__

    def test_get_modification_summary_has_docstring(self):
        """Test that get_modification_summary has detailed docstring"""
        assert ModificationTrackBuilder.get_modification_summary.__doc__ is not None
        assert "Args:" in ModificationTrackBuilder.get_modification_summary.__doc__
        assert "Returns:" in ModificationTrackBuilder.get_modification_summary.__doc__


class TestModificationTrackBuilderIntegration:
    """Integration tests for ModificationTrackBuilder"""

    @pytest.fixture
    def sample_data(self):
        return {
            "sequence": "ACGTACGTACGT",
            "seq_to_sig_map": list(range(0, 1200, 100)),
            "time_ms": np.arange(0, 1200, 1).astype(float),
            "sample_rate": 4000,
        }

    @pytest.fixture
    def sample_modifications(self):
        return [
            MockModification(0, "m", "C", 0.9, 0, 100),
            MockModification(2, "a", "A", 0.8, 200, 300),
            MockModification(4, "h", "C", 0.85, 400, 500),
            MockModification(6, "m", "C", 0.75, 600, 700),
        ]

    def test_complete_workflow(self, sample_data, sample_modifications):
        """Test complete workflow from initialization to figure creation"""
        builder = ModificationTrackBuilder(
            min_probability=0.7,
            enabled_types=["m", "a", "h"],
            overlay_opacity=0.8,
            theme=Theme.LIGHT,
        )

        # Get summary
        summary = builder.get_modification_summary(sample_modifications)
        assert summary["total_mods"] == 4
        assert summary["enabled_mods"] == 4

        # Build track
        fig = builder.build_track(modifications=sample_modifications, **sample_data)

        # Should have figure
        assert isinstance(fig, Plot)
        assert len(fig.renderers) > 0

    def test_reuse_builder_with_different_data(self, sample_data):
        """Test that same builder can be used with different modification data"""
        builder = ModificationTrackBuilder(min_probability=0.7)

        # First dataset
        mods1 = [MockModification(0, "m", "C", 0.9, 0, 100)]
        fig1 = builder.build_track(modifications=mods1, **sample_data)

        # Second dataset
        mods2 = [MockModification(1, "h", "C", 0.8, 100, 200)]
        fig2 = builder.build_track(modifications=mods2, **sample_data)

        # Both should create figures
        assert isinstance(fig1, Plot)
        assert isinstance(fig2, Plot)

    def test_update_filters_and_rebuild(self, sample_data, sample_modifications):
        """Test updating filters and rebuilding track"""
        builder = ModificationTrackBuilder(min_probability=0.5)

        # Build with low threshold
        fig1 = builder.build_track(modifications=sample_modifications, **sample_data)
        assert isinstance(fig1, Plot)

        # Update to higher threshold
        builder.update_filters(min_probability=0.85)

        # Rebuild - should have fewer mods
        fig2 = builder.build_track(modifications=sample_modifications, **sample_data)
        assert isinstance(fig2, Plot)

    def test_theme_variations(self, sample_data, sample_modifications):
        """Test that both themes work correctly"""
        # Light theme
        builder_light = ModificationTrackBuilder(theme=Theme.LIGHT)
        fig_light = builder_light.build_track(
            modifications=sample_modifications, **sample_data
        )

        # Dark theme
        builder_dark = ModificationTrackBuilder(theme=Theme.DARK)
        fig_dark = builder_dark.build_track(
            modifications=sample_modifications, **sample_data
        )

        # Both should create figures
        assert isinstance(fig_light, Plot)
        assert isinstance(fig_dark, Plot)
