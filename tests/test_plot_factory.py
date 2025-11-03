"""
Tests for PlotFactory
"""

import pytest

from squiggy.constants import PlotMode, Theme
from squiggy.plot_factory import create_plot_strategy
from squiggy.plot_strategies.aggregate import AggregatePlotStrategy
from squiggy.plot_strategies.base import PlotStrategy
from squiggy.plot_strategies.eventalign import EventAlignPlotStrategy
from squiggy.plot_strategies.overlay import OverlayPlotStrategy
from squiggy.plot_strategies.single_read import SingleReadPlotStrategy
from squiggy.plot_strategies.stacked import StackedPlotStrategy


class TestPlotFactoryBasic:
    """Tests for basic PlotFactory functionality"""

    def test_create_single_read_strategy(self):
        """Test creating SingleReadPlotStrategy"""
        strategy = create_plot_strategy(PlotMode.SINGLE, Theme.LIGHT)

        assert isinstance(strategy, SingleReadPlotStrategy)
        assert isinstance(strategy, PlotStrategy)
        assert strategy.theme == Theme.LIGHT

    def test_create_overlay_strategy(self):
        """Test creating OverlayPlotStrategy"""
        strategy = create_plot_strategy(PlotMode.OVERLAY, Theme.LIGHT)

        assert isinstance(strategy, OverlayPlotStrategy)
        assert isinstance(strategy, PlotStrategy)
        assert strategy.theme == Theme.LIGHT

    def test_create_stacked_strategy(self):
        """Test creating StackedPlotStrategy"""
        strategy = create_plot_strategy(PlotMode.STACKED, Theme.LIGHT)

        assert isinstance(strategy, StackedPlotStrategy)
        assert isinstance(strategy, PlotStrategy)
        assert strategy.theme == Theme.LIGHT

    def test_create_eventalign_strategy(self):
        """Test creating EventAlignPlotStrategy"""
        strategy = create_plot_strategy(PlotMode.EVENTALIGN, Theme.LIGHT)

        assert isinstance(strategy, EventAlignPlotStrategy)
        assert isinstance(strategy, PlotStrategy)
        assert strategy.theme == Theme.LIGHT

    def test_create_aggregate_strategy(self):
        """Test creating AggregatePlotStrategy"""
        strategy = create_plot_strategy(PlotMode.AGGREGATE, Theme.LIGHT)

        assert isinstance(strategy, AggregatePlotStrategy)
        assert isinstance(strategy, PlotStrategy)
        assert strategy.theme == Theme.LIGHT


class TestPlotFactoryWithThemes:
    """Tests for PlotFactory with different themes"""

    def test_create_with_dark_theme(self):
        """Test creating strategies with DARK theme"""
        strategy = create_plot_strategy(PlotMode.SINGLE, Theme.DARK)

        assert isinstance(strategy, SingleReadPlotStrategy)
        assert strategy.theme == Theme.DARK

    def test_create_with_light_theme(self):
        """Test creating strategies with LIGHT theme"""
        strategy = create_plot_strategy(PlotMode.OVERLAY, Theme.LIGHT)

        assert isinstance(strategy, OverlayPlotStrategy)
        assert strategy.theme == Theme.LIGHT

    def test_all_modes_with_dark_theme(self):
        """Test all plot modes work with DARK theme"""
        for plot_mode in PlotMode:
            strategy = create_plot_strategy(plot_mode, Theme.DARK)
            assert isinstance(strategy, PlotStrategy)
            assert strategy.theme == Theme.DARK

    def test_all_modes_with_light_theme(self):
        """Test all plot modes work with LIGHT theme"""
        for plot_mode in PlotMode:
            strategy = create_plot_strategy(plot_mode, Theme.LIGHT)
            assert isinstance(strategy, PlotStrategy)
            assert strategy.theme == Theme.LIGHT


class TestPlotFactoryErrors:
    """Tests for PlotFactory error handling"""

    def test_invalid_plot_mode_raises_error(self):
        """Test that invalid plot mode raises ValueError"""
        # Create a mock invalid mode (not using PlotMode enum)
        with pytest.raises(ValueError, match="Unknown plot mode"):
            # Force a value that's not in the map
            create_plot_strategy("invalid_mode", Theme.LIGHT)  # type: ignore

    def test_error_message_lists_valid_modes(self):
        """Test that error message includes valid modes"""
        try:
            create_plot_strategy("invalid", Theme.LIGHT)  # type: ignore
        except ValueError as e:
            error_msg = str(e)
            # Check that valid modes are mentioned
            assert "single" in error_msg
            assert "overlay" in error_msg
            assert "stacked" in error_msg
            assert "eventalign" in error_msg
            assert "aggregate" in error_msg


class TestPlotFactoryStrategyInterface:
    """Tests that all strategies have required interface"""

    @pytest.mark.parametrize(
        "plot_mode",
        [
            PlotMode.SINGLE,
            PlotMode.OVERLAY,
            PlotMode.STACKED,
            PlotMode.EVENTALIGN,
            PlotMode.AGGREGATE,
        ],
    )
    def test_all_strategies_have_create_plot(self, plot_mode):
        """Test all strategies have create_plot method"""
        strategy = create_plot_strategy(plot_mode, Theme.LIGHT)

        assert hasattr(strategy, "create_plot")
        assert callable(strategy.create_plot)

    @pytest.mark.parametrize(
        "plot_mode",
        [
            PlotMode.SINGLE,
            PlotMode.OVERLAY,
            PlotMode.STACKED,
            PlotMode.EVENTALIGN,
            PlotMode.AGGREGATE,
        ],
    )
    def test_all_strategies_have_validate_data(self, plot_mode):
        """Test all strategies have validate_data method"""
        strategy = create_plot_strategy(plot_mode, Theme.LIGHT)

        assert hasattr(strategy, "validate_data")
        assert callable(strategy.validate_data)

    @pytest.mark.parametrize(
        "plot_mode",
        [
            PlotMode.SINGLE,
            PlotMode.OVERLAY,
            PlotMode.STACKED,
            PlotMode.EVENTALIGN,
            PlotMode.AGGREGATE,
        ],
    )
    def test_all_strategies_have_theme_attribute(self, plot_mode):
        """Test all strategies have theme attribute"""
        strategy = create_plot_strategy(plot_mode, Theme.DARK)

        assert hasattr(strategy, "theme")
        assert strategy.theme == Theme.DARK


class TestPlotFactoryReusability:
    """Tests for PlotFactory reusability"""

    def test_create_multiple_strategies(self):
        """Test creating multiple strategies independently"""
        strategy1 = create_plot_strategy(PlotMode.SINGLE, Theme.LIGHT)
        strategy2 = create_plot_strategy(PlotMode.OVERLAY, Theme.DARK)
        strategy3 = create_plot_strategy(PlotMode.STACKED, Theme.LIGHT)

        assert isinstance(strategy1, SingleReadPlotStrategy)
        assert isinstance(strategy2, OverlayPlotStrategy)
        assert isinstance(strategy3, StackedPlotStrategy)

        assert strategy1.theme == Theme.LIGHT
        assert strategy2.theme == Theme.DARK
        assert strategy3.theme == Theme.LIGHT

    def test_create_same_strategy_multiple_times(self):
        """Test creating same strategy multiple times returns new instances"""
        strategy1 = create_plot_strategy(PlotMode.SINGLE, Theme.LIGHT)
        strategy2 = create_plot_strategy(PlotMode.SINGLE, Theme.LIGHT)

        assert isinstance(strategy1, SingleReadPlotStrategy)
        assert isinstance(strategy2, SingleReadPlotStrategy)
        # Should be different instances
        assert strategy1 is not strategy2

    def test_create_all_strategies_sequentially(self):
        """Test creating all strategies in sequence"""
        strategies = []
        for plot_mode in PlotMode:
            strategy = create_plot_strategy(plot_mode, Theme.LIGHT)
            strategies.append(strategy)

        # All should be PlotStrategy instances
        assert all(isinstance(s, PlotStrategy) for s in strategies)

        # All should be different types
        types = [type(s) for s in strategies]
        assert len(types) == len(set(types))  # All unique types
