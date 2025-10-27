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
# Purines (A, G) = orange/warm colors, Pyrimidines (C, T) = blue/cool colors
BASE_COLORS = {
    "A": "#E69F00",  # Orange (purine)
    "C": "#0072B2",  # Blue (pyrimidine)
    "G": "#D55E00",  # Vermillion/reddish-orange (purine)
    "T": "#56B4E9",  # Sky blue (pyrimidine)
    "U": "#56B4E9",  # Sky blue (for RNA, same as T)
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
