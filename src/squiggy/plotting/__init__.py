"""Plotting package for nanopore squiggle visualization

This package provides modular plotting functionality organized by plot mode,
with a backward-compatible SquigglePlotter class facade.
"""

import numpy as np

from ..constants import (
    CoordinateSpace,
    DEFAULT_POSITION_LABEL_INTERVAL,
    NormalizationMethod,
    PlotMode,
    Theme,
)
from .aggregate import plot_aggregate
from .base import MULTI_READ_COLORS, normalize_signal
from .eventalign import plot_eventalign
from .overlay import plot_overlay
from .single import plot_single_read
from .stacked import plot_stacked

# Re-export commonly used items for convenience
__all__ = [
    "SquigglePlotter",
    "MULTI_READ_COLORS",
    "normalize_signal",
    "plot_single_read",
    "plot_multiple_reads",
    "plot_aggregate",
]


class SquigglePlotter:
    """Interactive plotter for nanopore squiggle visualization

    This class provides a backward-compatible API that delegates to the
    modular plotting functions. All methods are static to maintain
    compatibility with existing code.
    """

    # Color palette for multi-read plots
    MULTI_READ_COLORS = MULTI_READ_COLORS

    @staticmethod
    def normalize_signal(signal: np.ndarray, method: NormalizationMethod) -> np.ndarray:
        """Normalize signal data using specified method"""
        return normalize_signal(signal, method)

    @staticmethod
    def plot_single_read(
        signal: np.ndarray,
        read_id: str,
        sample_rate: int,
        sequence: str | None = None,
        seq_to_sig_map: list[int] | None = None,
        normalization: NormalizationMethod = NormalizationMethod.NONE,
        downsample: int = 1,
        show_dwell_time: bool = False,
        show_labels: bool = True,
        show_signal_points: bool = False,
        position_label_interval: int = DEFAULT_POSITION_LABEL_INTERVAL,
        use_reference_positions: bool = False,
        theme: Theme = Theme.LIGHT,
    ):
        """Plot a single nanopore read with optional base annotations"""
        return plot_single_read(
            signal=signal,
            read_id=read_id,
            sample_rate=sample_rate,
            sequence=sequence,
            seq_to_sig_map=seq_to_sig_map,
            normalization=normalization,
            downsample=downsample,
            show_dwell_time=show_dwell_time,
            show_labels=show_labels,
            show_signal_points=show_signal_points,
            position_label_interval=position_label_interval,
            use_reference_positions=use_reference_positions,
            theme=theme,
        )

    @staticmethod
    def plot_multiple_reads(
        reads_data: list[tuple[str, np.ndarray, int]],
        mode: PlotMode,
        normalization: NormalizationMethod = NormalizationMethod.NONE,
        aligned_reads: list | None = None,
        downsample: int = 1,
        coordinate_space: CoordinateSpace = CoordinateSpace.SIGNAL,
        show_dwell_time: bool = False,
        show_labels: bool = True,
        show_signal_points: bool = False,
        position_label_interval: int = DEFAULT_POSITION_LABEL_INTERVAL,
        use_reference_positions: bool = False,
        theme: Theme = Theme.LIGHT,
    ):
        """Plot multiple reads in overlay, stacked, or event-aligned mode"""
        if mode == PlotMode.OVERLAY:
            return plot_overlay(
                reads_data=reads_data,
                normalization=normalization,
                downsample=downsample,
                coordinate_space=coordinate_space,
                aligned_reads=aligned_reads,
                show_signal_points=show_signal_points,
                theme=theme,
            )
        elif mode == PlotMode.STACKED:
            return plot_stacked(
                reads_data=reads_data,
                normalization=normalization,
                downsample=downsample,
                show_signal_points=show_signal_points,
                theme=theme,
            )
        elif mode == PlotMode.EVENTALIGN:
            return plot_eventalign(
                reads_data=reads_data,
                normalization=normalization,
                aligned_reads=aligned_reads,
                downsample=downsample,
                show_dwell_time=show_dwell_time,
                show_labels=show_labels,
                show_signal_points=show_signal_points,
                position_label_interval=position_label_interval,
                use_reference_positions=use_reference_positions,
                theme=theme,
            )
        else:
            raise ValueError(f"Unsupported plot mode: {mode}")

    @staticmethod
    def plot_aggregate(
        aggregate_stats: dict,
        pileup_stats: dict,
        quality_stats: dict,
        reference_name: str,
        num_reads: int,
        normalization: NormalizationMethod = NormalizationMethod.NONE,
        theme: Theme = Theme.LIGHT,
    ):
        """Plot aggregate multi-read visualization with three synchronized tracks"""
        return plot_aggregate(
            aggregate_stats=aggregate_stats,
            pileup_stats=pileup_stats,
            quality_stats=quality_stats,
            reference_name=reference_name,
            num_reads=num_reads,
            normalization=normalization,
            theme=theme,
        )


# Convenience function for backward compatibility
def plot_multiple_reads(
    reads_data: list[tuple[str, np.ndarray, int]],
    mode: PlotMode,
    normalization: NormalizationMethod = NormalizationMethod.NONE,
    aligned_reads: list | None = None,
    downsample: int = 1,
    coordinate_space: CoordinateSpace = CoordinateSpace.SIGNAL,
    show_dwell_time: bool = False,
    show_labels: bool = True,
    show_signal_points: bool = False,
    position_label_interval: int = DEFAULT_POSITION_LABEL_INTERVAL,
    use_reference_positions: bool = False,
    theme: Theme = Theme.LIGHT,
):
    """Convenience function that delegates to SquigglePlotter.plot_multiple_reads"""
    return SquigglePlotter.plot_multiple_reads(
        reads_data=reads_data,
        mode=mode,
        normalization=normalization,
        aligned_reads=aligned_reads,
        downsample=downsample,
        coordinate_space=coordinate_space,
        show_dwell_time=show_dwell_time,
        show_labels=show_labels,
        show_signal_points=show_signal_points,
        position_label_interval=position_label_interval,
        use_reference_positions=use_reference_positions,
        theme=theme,
    )
