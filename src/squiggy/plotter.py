"""Squiggle plot generation using plotnine"""

# IMPORTANT: Set matplotlib backend BEFORE importing plotnine
# This fixes "Starting a Matplotlib GUI outside of the main thread" warning
# when plots are generated in background threads via asyncio.to_thread()
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend (thread-safe)

# Suppress harmless plotnine warnings
import warnings

warnings.filterwarnings("ignore", message="Saving .* in image", category=UserWarning)
warnings.filterwarnings("ignore", message="Filename: .*BytesIO", category=UserWarning)

# ruff: noqa: E402
# Note: Imports below must come after matplotlib.use() to set backend correctly
import numpy as np
import pandas as pd
from plotnine import (
    aes,
    element_text,
    geom_line,
    geom_rect,
    geom_text,
    ggplot,
    labs,
    scale_color_manual,
    scale_fill_manual,
    theme,
    theme_minimal,
)

from .constants import (
    BASE_ANNOTATION_ALPHA,
    BASE_COLORS,
    BASE_LABEL_SIZE,
    DEFAULT_DOWNSAMPLE_FACTOR,
    MAX_OVERLAY_READS,
    MIN_POINTS_FOR_DOWNSAMPLING,
    MULTI_READ_COLORS,
    SIGNAL_LINE_COLOR,
    SIGNAL_LINE_WIDTH,
    SIGNAL_PERCENTILE_HIGH,
    SIGNAL_PERCENTILE_LOW,
    STACKED_VERTICAL_SPACING,
    NormalizationMethod,
    PlotMode,
)
from .normalization import normalize_signal
from .utils import downsample_signal


class SquigglePlotter:
    """Handle squiggle plot generation using plotnine"""

    @staticmethod
    def signal_to_dataframe(signal, sample_rate, downsample_factor=None):
        """Convert signal array to time-series DataFrame

        Args:
            signal: Raw signal array (numpy array)
            sample_rate: Sampling rate in Hz
            downsample_factor: Downsampling factor (None = auto)

        Returns:
            pandas DataFrame with 'time' and 'signal' columns
        """
        # Auto-determine downsampling factor
        if downsample_factor is None:
            if len(signal) > MIN_POINTS_FOR_DOWNSAMPLING:
                downsample_factor = DEFAULT_DOWNSAMPLE_FACTOR
            else:
                downsample_factor = 1

        # Downsample signal if needed
        if downsample_factor > 1:
            signal = downsample_signal(signal, downsample_factor)

        time = np.arange(len(signal)) / sample_rate
        return pd.DataFrame({"time": time, "signal": signal.astype(np.float64)})

    @staticmethod
    def create_plot(signal, sample_rate, read_id):
        """Create a squiggle plot from signal data (simplified interface)

        Args:
            signal: Raw signal array (numpy array)
            sample_rate: Sampling rate in Hz
            read_id: Read identifier for title

        Returns:
            plotnine plot object
        """
        return SquigglePlotter.plot_squiggle(
            signal=signal,
            read_id=read_id,
            sample_rate=sample_rate,
            sequence=None,
            seq_to_sig_map=None,
        )

    @staticmethod
    def plot_squiggle(
        signal,
        read_id,
        sample_rate=4000,
        sequence=None,
        seq_to_sig_map=None,
        downsample_factor=None,
    ):
        """Generate a squiggle plot from signal data

        Args:
            signal: Raw signal array (numpy array)
            read_id: Read identifier for title
            sample_rate: Sampling rate in Hz (default: 4000)
            sequence: Optional basecalled sequence string
            seq_to_sig_map: Optional mapping from sequence positions to signal indices
            downsample_factor: Downsampling factor (None = auto)

        Returns:
            plotnine plot object
        """
        # Auto-determine downsampling factor
        if downsample_factor is None:
            if len(signal) > MIN_POINTS_FOR_DOWNSAMPLING:
                downsample_factor = DEFAULT_DOWNSAMPLE_FACTOR
            else:
                downsample_factor = 1

        # Downsample signal and adjust seq_to_sig_map if needed
        if downsample_factor > 1:
            signal = downsample_signal(signal, downsample_factor)
            if seq_to_sig_map is not None:
                # Adjust the mapping indices to account for downsampling
                seq_to_sig_map = seq_to_sig_map // downsample_factor

        # Create time axis
        time = np.arange(len(signal)) / sample_rate

        # Create DataFrame for plotting
        df = pd.DataFrame({"time": time, "signal": signal})

        # Calculate signal range for base annotations
        sig_min = np.percentile(signal, SIGNAL_PERCENTILE_LOW)
        sig_max = np.percentile(signal, SIGNAL_PERCENTILE_HIGH)

        # Create base plot
        plot = (
            ggplot(df, aes(x="time", y="signal"))
            + geom_line(color=SIGNAL_LINE_COLOR, size=SIGNAL_LINE_WIDTH)
            + labs(title=f"Squiggle Plot: {read_id}", x="Time (s)", y="Signal (pA)")
            + theme_minimal()
            + theme(
                plot_title=element_text(size=12, weight="bold"),
                axis_title=element_text(size=10),
            )
        )

        # Add base annotations if available
        if sequence is not None and seq_to_sig_map is not None:
            base_coords = SquigglePlotter._create_base_coords(
                sequence, seq_to_sig_map, time, sig_min, sig_max
            )

            if len(base_coords) > 0:
                # Add colored rectangles for bases
                plot = plot + geom_rect(
                    aes(
                        xmin="base_st",
                        xmax="base_en",
                        fill="base",
                        ymin=sig_min,
                        ymax=sig_max,
                    ),
                    data=base_coords,
                    alpha=BASE_ANNOTATION_ALPHA,
                    show_legend=False,
                )

                # Add base labels
                plot = plot + geom_text(
                    aes(x="base_st", label="base", color="base", y=sig_min),
                    data=base_coords,
                    va="bottom",
                    ha="left",
                    size=BASE_LABEL_SIZE,
                    show_legend=False,
                )

                # Apply color scales
                plot = (
                    plot
                    + scale_fill_manual(BASE_COLORS)
                    + scale_color_manual(BASE_COLORS)
                )

        return plot

    @staticmethod
    def _create_base_coords(sequence, seq_to_sig_map, time, sig_min, sig_max):
        """Create DataFrame with base coordinate information for plotting

        Args:
            sequence: Basecalled sequence string
            seq_to_sig_map: Array mapping sequence positions to signal indices
            time: Time array corresponding to signal
            sig_min: Minimum signal value for rectangle placement
            sig_max: Maximum signal value for rectangle placement

        Returns:
            pandas DataFrame with columns: base, base_st, base_en
        """
        base_data = []

        for i, base in enumerate(sequence):
            if i < len(seq_to_sig_map):
                sig_idx = seq_to_sig_map[i]
                # Get end position (next base's start or end of signal)
                if i + 1 < len(seq_to_sig_map):
                    next_sig_idx = seq_to_sig_map[i + 1]
                else:
                    next_sig_idx = len(time) - 1

                if sig_idx < len(time) and next_sig_idx < len(time):
                    base_data.append(
                        {
                            "base": base,
                            "base_st": time[sig_idx],
                            "base_en": time[next_sig_idx],
                        }
                    )

        return pd.DataFrame(base_data) if base_data else pd.DataFrame()

    @staticmethod
    def plot_multiple_reads(
        reads_data,
        mode=PlotMode.OVERLAY,
        normalization=NormalizationMethod.ZNORM,
        title=None,
        aligned_reads=None,
        downsample_factor=None,
    ):
        """Plot multiple reads in overlay, stacked, or event-aligned mode

        Args:
            reads_data: List of tuples (read_id, signal, sample_rate)
            mode: PlotMode (OVERLAY, STACKED, or EVENTALIGN)
            normalization: NormalizationMethod for signal preprocessing
            title: Optional plot title
            aligned_reads: List of AlignedRead objects (required for EVENTALIGN mode)
            downsample_factor: Downsampling factor (None = auto)

        Returns:
            plotnine plot object
        """
        if mode == PlotMode.OVERLAY:
            return SquigglePlotter._plot_overlay(
                reads_data, normalization, title, downsample_factor
            )
        elif mode == PlotMode.STACKED:
            return SquigglePlotter._plot_stacked(
                reads_data, normalization, title, downsample_factor
            )
        elif mode == PlotMode.EVENTALIGN:
            if aligned_reads is None:
                raise ValueError("aligned_reads required for EVENTALIGN mode")
            return SquigglePlotter._plot_eventalign(
                reads_data, aligned_reads, normalization, title, downsample_factor
            )
        else:
            raise ValueError(f"Unsupported multi-read mode: {mode}")

    @staticmethod
    def _plot_overlay(reads_data, normalization, title, downsample_factor=None):
        """Plot multiple reads overlaid on same axes

        Args:
            reads_data: List of tuples (read_id, signal, sample_rate)
            normalization: NormalizationMethod
            title: Plot title
            downsample_factor: Downsampling factor (None = auto)

        Returns:
            plotnine plot object
        """
        # Limit number of reads
        if len(reads_data) > MAX_OVERLAY_READS:
            reads_data = reads_data[:MAX_OVERLAY_READS]

        # Prepare data
        all_data = []
        for idx, (read_id, signal, sample_rate) in enumerate(reads_data):
            # Normalize signal
            norm_signal = normalize_signal(signal, normalization)

            # Auto-determine downsampling factor
            if downsample_factor is None:
                if len(norm_signal) > MIN_POINTS_FOR_DOWNSAMPLING:
                    ds_factor = DEFAULT_DOWNSAMPLE_FACTOR
                else:
                    ds_factor = 1
            else:
                ds_factor = downsample_factor

            # Downsample signal
            if ds_factor > 1:
                norm_signal = downsample_signal(norm_signal, ds_factor)

            # Create time axis
            time = np.arange(len(norm_signal)) / sample_rate

            # Add to dataframe
            df = pd.DataFrame(
                {
                    "time": time,
                    "signal": norm_signal,
                    "read_id": read_id,
                    "color_idx": idx % len(MULTI_READ_COLORS),
                }
            )
            all_data.append(df)

        combined_df = pd.concat(all_data, ignore_index=True)

        # Create color mapping
        color_map = {
            i: MULTI_READ_COLORS[i % len(MULTI_READ_COLORS)]
            for i in range(len(reads_data))
        }

        # Create plot
        plot_title = title or f"Overlaid Reads (n={len(reads_data)})"
        y_label = f"Signal ({normalization.value})"

        plot = (
            ggplot(combined_df, aes(x="time", y="signal", color="factor(color_idx)"))
            + geom_line(size=SIGNAL_LINE_WIDTH)
            + scale_color_manual(values=color_map, guide=False)
            + labs(title=plot_title, x="Time (s)", y=y_label)
            + theme_minimal()
            + theme(
                plot_title=element_text(size=12, weight="bold"),
                axis_title=element_text(size=10),
            )
        )

        return plot

    @staticmethod
    def _plot_stacked(reads_data, normalization, title, downsample_factor=None):
        """Plot multiple reads stacked vertically (squigualiser-style)

        Args:
            reads_data: List of tuples (read_id, signal, sample_rate)
            normalization: NormalizationMethod
            title: Plot title
            downsample_factor: Downsampling factor (None = auto)

        Returns:
            plotnine plot object
        """
        # Prepare data
        all_data = []
        for idx, (read_id, signal, sample_rate) in enumerate(reads_data):
            # Normalize signal
            norm_signal = normalize_signal(signal, normalization)

            # Auto-determine downsampling factor
            if downsample_factor is None:
                if len(norm_signal) > MIN_POINTS_FOR_DOWNSAMPLING:
                    ds_factor = DEFAULT_DOWNSAMPLE_FACTOR
                else:
                    ds_factor = 1
            else:
                ds_factor = downsample_factor

            # Downsample signal
            if ds_factor > 1:
                norm_signal = downsample_signal(norm_signal, ds_factor)

            # Apply vertical offset for stacking
            offset = -idx * STACKED_VERTICAL_SPACING
            stacked_signal = norm_signal + offset

            # Create time axis
            time = np.arange(len(stacked_signal)) / sample_rate

            # Add to dataframe
            df = pd.DataFrame(
                {
                    "time": time,
                    "signal": stacked_signal,
                    "read_id": read_id,
                    "color_idx": idx % len(MULTI_READ_COLORS),
                }
            )
            all_data.append(df)

        combined_df = pd.concat(all_data, ignore_index=True)

        # Create color mapping
        color_map = {
            i: MULTI_READ_COLORS[i % len(MULTI_READ_COLORS)]
            for i in range(len(reads_data))
        }

        # Create plot
        plot_title = title or f"Stacked Reads (n={len(reads_data)})"
        y_label = f"Signal ({normalization.value})"

        plot = (
            ggplot(combined_df, aes(x="time", y="signal", color="factor(color_idx)"))
            + geom_line(size=SIGNAL_LINE_WIDTH)
            + scale_color_manual(values=color_map, guide=False)
            + labs(title=plot_title, x="Time (s)", y=y_label)
            + theme_minimal()
            + theme(
                plot_title=element_text(size=12, weight="bold"),
                axis_title=element_text(size=10),
            )
        )

        return plot

    @staticmethod
    def _plot_eventalign(
        reads_data, aligned_reads, normalization, title, downsample_factor=None
    ):
        """Plot multiple reads with event alignment (squigualiser-style)

        Shows bases with fixed width and colored base annotations.

        Args:
            reads_data: List of tuples (read_id, signal, sample_rate)
            aligned_reads: List of AlignedRead objects with base annotations
            normalization: NormalizationMethod
            title: Plot title
            downsample_factor: Downsampling factor (None = auto)

        Returns:
            plotnine plot object
        """
        # Create signal and base annotation data
        all_signal_data = []
        all_base_data = []

        for idx, ((read_id, signal, _sample_rate), aligned_read) in enumerate(
            zip(reads_data, aligned_reads)
        ):
            # Normalize signal
            norm_signal = normalize_signal(signal, normalization)

            # Auto-determine downsampling factor
            if downsample_factor is None:
                if len(norm_signal) > MIN_POINTS_FOR_DOWNSAMPLING:
                    ds_factor = DEFAULT_DOWNSAMPLE_FACTOR
                else:
                    ds_factor = 1
            else:
                ds_factor = downsample_factor

            # Downsample signal (note: we don't downsample base annotations)
            if ds_factor > 1:
                norm_signal = downsample_signal(norm_signal, ds_factor)

            # Apply vertical offset for stacking
            offset = -idx * STACKED_VERTICAL_SPACING
            stacked_signal = norm_signal + offset

            # Create base position index (fixed width per base)
            base_positions = []
            signal_values = []

            for base_ann in aligned_read.bases:
                # Get signal for this base (adjust indices if downsampled)
                sig_start = base_ann.signal_start // ds_factor
                sig_end = min(base_ann.signal_end // ds_factor, len(stacked_signal))

                # Average signal for this base region
                if sig_start < len(stacked_signal) and sig_end > sig_start:
                    base_signal = stacked_signal[sig_start:sig_end]
                    # Create one point per base at its position
                    base_positions.append(base_ann.position)
                    signal_values.append(np.mean(base_signal))

            # Add signal data
            if base_positions:
                df_signal = pd.DataFrame(
                    {
                        "base_position": base_positions,
                        "signal": signal_values,
                        "read_id": read_id,
                        "read_idx": idx,
                    }
                )
                all_signal_data.append(df_signal)

            # Add base annotation rectangles
            sig_min = np.min(stacked_signal)
            sig_max = np.max(stacked_signal)

            for base_ann in aligned_read.bases:
                all_base_data.append(
                    {
                        "base": base_ann.base,
                        "base_pos_start": base_ann.position,
                        "base_pos_end": base_ann.position + 1,
                        "ymin": sig_min,
                        "ymax": sig_max,
                        "read_idx": idx,
                    }
                )

        # Combine data
        combined_signal = pd.concat(all_signal_data, ignore_index=True)
        base_df = pd.DataFrame(all_base_data)

        # Create color mapping for reads
        color_map = {
            i: MULTI_READ_COLORS[i % len(MULTI_READ_COLORS)]
            for i in range(len(reads_data))
        }

        # Create plot
        plot_title = title or f"Event-Aligned Reads (n={len(reads_data)})"
        y_label = f"Signal ({normalization.value})"

        plot = (
            ggplot()
            + geom_rect(
                aes(
                    xmin="base_pos_start",
                    xmax="base_pos_end",
                    ymin="ymin",
                    ymax="ymax",
                    fill="base",
                ),
                data=base_df,
                alpha=BASE_ANNOTATION_ALPHA,
            )
            + geom_line(
                aes(x="base_position", y="signal", color="factor(read_idx)"),
                data=combined_signal,
                size=SIGNAL_LINE_WIDTH,
            )
            + scale_color_manual(values=color_map, guide=False)
            + scale_fill_manual(values=BASE_COLORS)
            + labs(title=plot_title, x="Base Position", y=y_label)
            + theme_minimal()
            + theme(
                plot_title=element_text(size=12, weight="bold"),
                axis_title=element_text(size=10),
                legend_position="none",
            )
        )

        return plot
