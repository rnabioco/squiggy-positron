"""
Plot factory for creating plot strategies

This module provides a factory function for instantiating the appropriate
plot strategy based on the plot mode.
"""

from .constants import PlotMode, Theme
from .plot_strategies.aggregate import AggregatePlotStrategy
from .plot_strategies.aggregate_comparison import AggregateComparisonStrategy
from .plot_strategies.base import PlotStrategy
from .plot_strategies.delta import DeltaPlotStrategy
from .plot_strategies.eventalign import EventAlignPlotStrategy
from .plot_strategies.overlay import OverlayPlotStrategy
from .plot_strategies.signal_overlay_comparison import (
    SignalOverlayComparisonStrategy,
)
from .plot_strategies.single_read import SingleReadPlotStrategy
from .plot_strategies.stacked import StackedPlotStrategy


def create_plot_strategy(plot_mode: PlotMode, theme: Theme) -> PlotStrategy:
    """
    Factory function to create the appropriate plot strategy

    Args:
        plot_mode: PlotMode enum specifying which plot type to create
        theme: Theme enum (LIGHT or DARK)

    Returns:
        PlotStrategy instance for the specified mode

    Raises:
        ValueError: If plot_mode is not recognized

    Examples:
        >>> from squiggy.plot_factory import create_plot_strategy
        >>> from squiggy.constants import PlotMode, Theme
        >>>
        >>> strategy = create_plot_strategy(PlotMode.SINGLE, Theme.LIGHT)
        >>> html, fig = strategy.create_plot(data, options)
    """
    strategy_map = {
        PlotMode.SINGLE: SingleReadPlotStrategy,
        PlotMode.OVERLAY: OverlayPlotStrategy,
        PlotMode.STACKED: StackedPlotStrategy,
        PlotMode.EVENTALIGN: EventAlignPlotStrategy,
        PlotMode.AGGREGATE: AggregatePlotStrategy,
        PlotMode.DELTA: DeltaPlotStrategy,
        PlotMode.SIGNAL_OVERLAY_COMPARISON: SignalOverlayComparisonStrategy,
        PlotMode.AGGREGATE_COMPARISON: AggregateComparisonStrategy,
    }

    strategy_class = strategy_map.get(plot_mode)
    if strategy_class is None:
        valid_modes = ", ".join(m.value for m in PlotMode)
        raise ValueError(f"Unknown plot mode: {plot_mode}. Valid modes: {valid_modes}")

    return strategy_class(theme)
