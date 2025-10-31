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

# Signal point visualization
SIGNAL_POINT_SIZE = 6  # Point size in pixels
SIGNAL_POINT_ALPHA = 0.6  # Point transparency
SIGNAL_POINT_COLOR = "#E74C3C"  # Light red for contrast with signal line

# Position label settings
DEFAULT_POSITION_LABEL_INTERVAL = 10  # Show position label every N bases


# Plot modes
class PlotMode(Enum):
    """Available plotting modes for signal visualization"""

    SINGLE = "single"  # Single read (default)
    OVERLAY = "overlay"  # Multiple reads overlaid on same axes
    STACKED = "stacked"  # Multiple reads stacked vertically (squigualiser-style)
    EVENTALIGN = "eventalign"  # Event-aligned with base annotations
    AGGREGATE = "aggregate"  # Multi-read aggregate with pileup statistics


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

# Aggregate mode settings
DEFAULT_AGGREGATE_SAMPLE_SIZE = 100  # Default number of reads to sample for aggregate
MIN_AGGREGATE_SAMPLE_SIZE = 10  # Minimum reads for aggregate mode
MAX_AGGREGATE_SAMPLE_SIZE = 10000  # Maximum reads for aggregate mode
AGGREGATE_CONFIDENCE_LEVEL = 1  # Standard deviations for confidence bands (±1 std dev)

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

# ==============================================================================
# Base Modification (modBAM) Settings
# ==============================================================================

# Common modification codes and display names
# Based on https://github.com/samtools/hts-specs/blob/master/SAMtags.pdf
# Complete list from modkit (https://github.com/nanoporetech/modkit)
# Supports both single-letter codes and ChEBI numeric codes
MODIFICATION_CODES = {
    # Cytosine modifications
    "m": "5mC",  # 5-methylcytosine
    "h": "5hmC",  # 5-hydroxymethylcytosine
    "f": "5fC",  # 5-formylcytosine
    "c": "5caC",  # 5-carboxylcytosine
    21839: "4mC",  # 4-methylcytosine
    19228: "Cm",  # 2'-O-methylcytosine
    "C": "C*",  # any cytosine modification
    # Adenine modifications
    "a": "6mA",  # N6-methyladenine
    17596: "I",  # inosine
    69426: "Am",  # 2'-O-methyladenine
    "A": "A*",  # any adenine modification
    # Thymine/Uracil modifications
    "g": "5hmU",  # 5-hydroxymethyluracil
    "e": "5fU",  # 5-formyluracil
    "b": "5caU",  # 5-carboxyuracil
    17802: "Ψ",  # pseudouridine
    16450: "dU",  # deoxyuridine
    19227: "Um",  # 2'-O-methyluracil
    "T": "T*",  # any thymine modification
    # Guanine modifications
    "o": "8oxoG",  # 8-oxoguanine
    19229: "Gm",  # 2'-O-methylguanine
    "G": "G*",  # any guanine modification
}

# Modification type colors (colorblind-friendly, distinct from base colors)
# Uses Okabe-Ito palette colors organized by base type
MODIFICATION_COLORS = {
    # Cytosine modifications (blue/cyan hues)
    "m": "#0072B2",  # Blue (5mC)
    "h": "#56B4E9",  # Sky blue (5hmC)
    "f": "#4682B4",  # Steel blue (5fC)
    "c": "#87CEEB",  # Sky blue light (5caC)
    21839: "#1E90FF",  # Dodger blue (4mC)
    19228: "#00CED1",  # Dark turquoise (Cm)
    "C": "#0072B2",  # Blue (any C*)
    # Adenine modifications (orange/red hues)
    "a": "#D55E00",  # Vermillion (6mA)
    17596: "#CC6600",  # Dark orange (inosine)
    69426: "#E69F00",  # Orange (Am)
    "A": "#D55E00",  # Vermillion (any A*)
    # Thymine/Uracil modifications (purple/magenta hues)
    "g": "#CC79A7",  # Reddish purple (5hmU)
    "e": "#9370DB",  # Medium purple (5fU)
    "b": "#BA55D3",  # Medium orchid (5caU)
    17802: "#8B008B",  # Dark magenta (pseudouridine Ψ)
    16450: "#DA70D6",  # Orchid (dU)
    19227: "#DDA0DD",  # Plum (Um)
    "T": "#CC79A7",  # Reddish purple (any T*)
    # Guanine modifications (green/teal hues)
    "o": "#009E73",  # Bluish green (8oxoG)
    19229: "#20B2AA",  # Light sea green (Gm)
    "G": "#009E73",  # Bluish green (any G*)
    # Default for unknown modifications
    "default": "#000000",  # Black
}

# Modification overlay settings
DEFAULT_MOD_OVERLAY_OPACITY = 0.6  # Default opacity for modification shading (0-1)
MOD_OVERLAY_MIN_OPACITY = 0.1  # Minimum overlay opacity
MOD_OVERLAY_MAX_OPACITY = 0.9  # Maximum overlay opacity

# Modification threshold settings
DEFAULT_MOD_THRESHOLD = 0.5  # Default probability threshold (tau)
MOD_THRESHOLD_MIN = 0.0  # Minimum threshold
MOD_THRESHOLD_MAX = 1.0  # Maximum threshold
MOD_THRESHOLD_STEP = 0.05  # Threshold slider step size
