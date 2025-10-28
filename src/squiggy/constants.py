"""Constants and configuration for Squiggy application"""

from enum import Enum

# Application metadata
APP_NAME = "Squiggy"
APP_VERSION = "0.1.0"
APP_DESCRIPTION = "Nanopore Squiggle Viewer"

# Window settings
DEFAULT_WINDOW_WIDTH = 1200
DEFAULT_WINDOW_HEIGHT = 800
PLOT_MIN_WIDTH = 800
PLOT_MIN_HEIGHT = 600

# Splitter proportions (read list : plot area)
SPLITTER_RATIO = (1, 3)

# Plot settings
PLOT_DPI = 100
PLOT_WIDTH = 10
PLOT_HEIGHT = 6
SIGNAL_LINE_COLOR = "#000000"
SIGNAL_LINE_WIDTH = 0.5

# Signal downsampling settings
DEFAULT_DOWNSAMPLE_FACTOR = 10  # Sample every Nth point (10 = 10% of data)
MIN_POINTS_FOR_DOWNSAMPLING = (
    10000  # Only downsample if signal has more than this many points
)

# Base colors for visualization
# Using Okabe-Ito colorblind-friendly palette
BASE_COLORS = {
    "A": "#009E73",  # Green (purine)
    "C": "#F0E442",  # Yellow (pyrimidine)
    "G": "#0072B2",  # Blue (purine)
    "T": "#D55E00",  # Orange (pyrimidine)
    "U": "#D55E00",  # Same as T
    "N": "#808080",  # Gray (unknown)
}

# Base annotation settings
BASE_ANNOTATION_ALPHA = 0.2
BASE_LABEL_SIZE = 8
SIGNAL_PERCENTILE_LOW = 2.5
SIGNAL_PERCENTILE_HIGH = 97.5


# Plot modes
class PlotMode(Enum):
    """Available plotting modes for signal visualization"""

    SINGLE = "single"  # Single read (default)
    OVERLAY = "overlay"  # Multiple reads overlaid on same axes
    STACKED = "stacked"  # Multiple reads stacked vertically (squigualiser-style)
    EVENTALIGN = "eventalign"  # Event-aligned with base annotations


# Normalization methods
class NormalizationMethod(Enum):
    """Signal normalization methods"""

    NONE = "none"  # Raw signal (no normalization)
    ZNORM = "znorm"  # Z-score normalization (mean=0, std=1)
    MEDIAN = "median"  # Median normalization
    MAD = "mad"  # Median absolute deviation


# Multi-read plotting settings
MAX_OVERLAY_READS = 10  # Maximum reads to overlay
STACKED_VERTICAL_SPACING = 20  # Vertical spacing between stacked reads (in pA)

# Color palette for multi-read plots (colorblind-friendly)
MULTI_READ_COLORS = [
    "#0173B2",  # Blue
    "#DE8F05",  # Orange
    "#029E73",  # Green
    "#CC78BC",  # Purple
    "#CA9161",  # Tan
    "#FBAFE4",  # Pink
    "#949494",  # Gray
    "#ECE133",  # Yellow
    "#56B4E9",  # Light blue
    "#D55E00",  # Red-orange
]


# Theme modes
class Theme(Enum):
    """Application theme modes"""

    LIGHT = "light"  # Light mode (default)
    DARK = "dark"  # Dark mode


# Theme color palettes
LIGHT_THEME = {
    # Qt application colors
    "window_bg": "#ffffff",
    "window_text": "#000000",
    "base_bg": "#ffffff",
    "base_text": "#000000",
    "button_bg": "#f0f0f0",
    "button_text": "#000000",
    "highlight_bg": "#308cc6",
    "highlight_text": "#ffffff",
    # Bokeh plot colors
    "plot_bg": "#ffffff",
    "plot_border": "#ffffff",
    "grid_line": "#e6e6e6",
    "axis_line": "#000000",
    "axis_text": "#000000",
    "title_text": "#000000",
    "signal_line": "#000000",
}

DARK_THEME = {
    # Qt application colors
    "window_bg": "#2b2b2b",
    "window_text": "#ffffff",
    "base_bg": "#1e1e1e",
    "base_text": "#ffffff",
    "button_bg": "#3c3c3c",
    "button_text": "#ffffff",
    "highlight_bg": "#0d5d9f",
    "highlight_text": "#ffffff",
    # Bokeh plot colors
    "plot_bg": "#2b2b2b",
    "plot_border": "#1e1e1e",
    "grid_line": "#3c3c3c",
    "axis_line": "#cccccc",
    "axis_text": "#cccccc",
    "title_text": "#ffffff",
    "signal_line": "#e0e0e0",  # Bright light gray for good visibility
}

# Base colors for dark mode (adjusted for better visibility)
BASE_COLORS_DARK = {
    "A": "#00d9a3",  # Brighter green (purine)
    "C": "#fff34d",  # Brighter yellow (pyrimidine)
    "G": "#4da6ff",  # Brighter blue (purine)
    "T": "#ff8c42",  # Brighter orange (pyrimidine)
    "U": "#ff8c42",  # Same as T
    "N": "#999999",  # Lighter gray (unknown)
}
