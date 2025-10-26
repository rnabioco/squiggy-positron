"""Squiggle plot generation using plotnine"""

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
    SIGNAL_LINE_COLOR,
    SIGNAL_LINE_WIDTH,
    SIGNAL_PERCENTILE_HIGH,
    SIGNAL_PERCENTILE_LOW,
)


class SquigglePlotter:
    """Handle squiggle plot generation using plotnine"""

    @staticmethod
    def signal_to_dataframe(signal, sample_rate):
        """Convert signal array to time-series DataFrame

        Args:
            signal: Raw signal array (numpy array)
            sample_rate: Sampling rate in Hz

        Returns:
            pandas DataFrame with 'time' and 'signal' columns
        """
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
        signal, read_id, sample_rate=4000, sequence=None, seq_to_sig_map=None
    ):
        """Generate a squiggle plot from signal data

        Args:
            signal: Raw signal array (numpy array)
            read_id: Read identifier for title
            sample_rate: Sampling rate in Hz (default: 4000)
            sequence: Optional basecalled sequence string
            seq_to_sig_map: Optional mapping from sequence positions to signal indices

        Returns:
            plotnine plot object
        """
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
