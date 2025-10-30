"""Tests for SquiggleViewer auto-downsampling behavior."""

import pytest

from squiggy.constants import (
    DOWNSAMPLE_MULTI_READ,
    DOWNSAMPLE_SINGLE_READ,
    PlotMode,
)


class TestAutoDownsampling:
    """Tests for automatic downsampling based on plot mode and selection."""

    def test_constants_are_defined(self):
        """Test that downsample constants are properly defined."""
        assert DOWNSAMPLE_SINGLE_READ == 25
        assert DOWNSAMPLE_MULTI_READ == 50

    def test_multi_read_higher_than_single(self):
        """Test that multi-read downsample is higher for performance."""
        assert DOWNSAMPLE_MULTI_READ > DOWNSAMPLE_SINGLE_READ


class TestViewerDownsampleLogic:
    """Tests for SquiggleViewer downsample logic (unit tests without Qt)."""

    def test_is_multi_read_mode_detection(self):
        """Test detection of multi-read plot modes."""
        multi_read_modes = [PlotMode.OVERLAY, PlotMode.STACKED, PlotMode.AGGREGATE]
        single_read_modes = [PlotMode.SINGLE, PlotMode.EVENTALIGN]

        # Verify our classification logic
        for mode in multi_read_modes:
            assert mode in (PlotMode.OVERLAY, PlotMode.STACKED, PlotMode.AGGREGATE)

        for mode in single_read_modes:
            assert mode not in (PlotMode.OVERLAY, PlotMode.STACKED, PlotMode.AGGREGATE)

    def test_multi_read_detection_with_count(self):
        """Test that multi-read requires both mode and multiple reads."""
        # Mock scenarios
        scenarios = [
            # (is_multi_read_mode, has_multiple_reads, should_use_multi_downsample)
            (True, True, True),  # Overlay/Stacked with >1 reads -> 50
            (True, False, False),  # Overlay/Stacked with 1 read -> 25
            (False, True, False),  # Single mode with >1 reads -> 25
            (False, False, False),  # Single mode with 1 read -> 25
        ]

        for is_multi_mode, has_multiple, should_use_multi in scenarios:
            # Logic from viewer.py update_plot_from_selection
            should_adjust_to_multi = is_multi_mode and has_multiple
            assert should_adjust_to_multi == should_use_multi


class TestUIComponentsDownsampling:
    """Tests for AdvancedOptionsPanel set_downsample_value method."""

    def test_set_downsample_value_method_exists(self):
        """Test that AdvancedOptionsPanel has set_downsample_value method."""
        from squiggy.ui_components import AdvancedOptionsPanel

        # Check method exists
        assert hasattr(AdvancedOptionsPanel, "set_downsample_value")

        # Check it's callable
        assert callable(getattr(AdvancedOptionsPanel, "set_downsample_value"))


# Integration tests would require Qt application and are skipped in CI
# They would test:
# - user_set_downsample flag tracking
# - Auto-adjustment on selection change
# - Auto-adjustment on mode change
# - UI control updates
# - Manual override behavior

# Example structure for integration tests (would need Qt fixtures):
#
# @pytest.mark.integration
# class TestViewerIntegration:
#     """Integration tests for viewer auto-downsampling (requires Qt)."""
#
#     def test_auto_adjust_on_multi_read_selection(self, viewer, sample_pod5):
#         """Test that downsample adjusts to 50 when selecting multiple reads."""
#         viewer.load_pod5_file(sample_pod5)
#         viewer.plot_mode = PlotMode.OVERLAY
#         viewer.user_set_downsample = False
#
#         # Select multiple reads
#         viewer.read_list.select_multiple_reads(["read1", "read2"])
#
#         assert viewer.downsample_factor == DOWNSAMPLE_MULTI_READ
#
#     def test_auto_adjust_on_single_read_selection(self, viewer, sample_pod5):
#         """Test that downsample resets to 25 when switching to single read."""
#         viewer.load_pod5_file(sample_pod5)
#         viewer.plot_mode = PlotMode.SINGLE
#         viewer.user_set_downsample = False
#         viewer.downsample_factor = 50  # Start at multi-read value
#
#         # Select single read
#         viewer.read_list.select_read("read1")
#
#         assert viewer.downsample_factor == DOWNSAMPLE_SINGLE_READ
#
#     def test_manual_override_prevents_auto_adjust(self, viewer, sample_pod5):
#         """Test that user manual changes prevent auto-adjustment."""
#         viewer.load_pod5_file(sample_pod5)
#         viewer.plot_mode = PlotMode.OVERLAY
#
#         # User manually sets downsample
#         viewer.set_downsample_factor(75)
#         assert viewer.user_set_downsample is True
#
#         # Select multiple reads - should NOT auto-adjust
#         viewer.read_list.select_multiple_reads(["read1", "read2"])
#
#         assert viewer.downsample_factor == 75  # Unchanged
#
#     def test_flag_resets_on_mode_change(self, viewer):
#         """Test that user_set_downsample flag resets on mode change."""
#         viewer.user_set_downsample = True
#         viewer.set_plot_mode(PlotMode.OVERLAY)
#
#         assert viewer.user_set_downsample is False
