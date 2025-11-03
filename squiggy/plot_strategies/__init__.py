"""
Plot strategies for different visualization modes

This package implements the Strategy Pattern for squiggy plot generation.
Each plot mode (SINGLE, EVENTALIGN, OVERLAY, etc.) has its own strategy class
that implements the PlotStrategy interface.

Usage:
    >>> from squiggy.plot_strategies.base import PlotStrategy
    >>> from squiggy.constants import Theme
    >>>
    >>> # Strategies will be imported here as they're implemented
    >>> # from squiggy.plot_strategies.single_read import SingleReadPlotStrategy
    >>> # strategy = SingleReadPlotStrategy(Theme.LIGHT)
    >>> # html, fig = strategy.create_plot(data, options)
"""

from .base import PlotStrategy

__all__ = ["PlotStrategy"]
