"""Constants and configuration for Squiggy application"""

from enum import Enum

# ==============================================================================
# Signal Processing Settings
# ==============================================================================

# Signal downsampling settings
DEFAULT_DOWNSAMPLE = 5  # Default downsampling factor for all plot functions

# ==============================================================================
# Base Annotation Settings
# ==============================================================================

# Base colors for visualization (light mode)
# Using Okabe-Ito colorblind-friendly palette
BASE_COLORS = {
    "A": "#009E73",  # Green (purine)
    "C": "#F0E442",  # Yellow (pyrimidine)
    "G": "#0072B2",  # Blue (purine)
    "T": "#D55E00",  # Orange (pyrimidine)
    "U": "#D55E00",  # Same as T
    "N": "#808080",  # Gray (unknown)
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

# Base annotation settings
BASE_ANNOTATION_ALPHA = 0.2

# Position label settings
DEFAULT_POSITION_LABEL_INTERVAL = 100  # Show position label every N bases

# ==============================================================================
# Plot Modes and Normalization
# ==============================================================================


# Plot modes
class PlotMode(Enum):
    """Available plotting modes for signal visualization"""

    SINGLE = "single"  # Single read (default)
    OVERLAY = "overlay"  # Multiple reads overlaid on same axes
    STACKED = "stacked"  # Multiple reads stacked vertically (squigualiser-style)
    EVENTALIGN = "eventalign"  # Event-aligned with base annotations
    AGGREGATE = "aggregate"  # Multi-read aggregate with pileup statistics
    DELTA = "delta"  # Delta track comparing two samples
    SIGNAL_OVERLAY_COMPARISON = (
        "signal_overlay_comparison"  # Multi-sample signal overlay
    )
    AGGREGATE_COMPARISON = (
        "aggregate_comparison"  # Multi-sample aggregate statistics comparison
    )


# Normalization methods
class NormalizationMethod(Enum):
    """Signal normalization methods"""

    NONE = "none"  # Raw signal (no normalization)
    ZNORM = "znorm"  # Z-score normalization (mean=0, std=1)
    MEDIAN = "median"  # Median normalization
    MAD = "mad"  # Median absolute deviation


# ==============================================================================
# File I/O Settings
# ==============================================================================

# BAM file I/O settings
BAM_SAMPLE_SIZE = 100  # Number of reads to sample when checking BAM features

# Motif window settings
DEFAULT_MOTIF_WINDOW_UPSTREAM = 10  # Bases upstream of motif center
DEFAULT_MOTIF_WINDOW_DOWNSTREAM = 10  # Bases downstream of motif center

# ==============================================================================
# Multi-Read Plot Settings
# ==============================================================================

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

# ==============================================================================
# Theme Settings
# ==============================================================================


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
    # Aggregate plot band colors
    "signal_band": "#56B4E9",  # Light blue for signal confidence bands
    "quality_band": "#FFA500",  # Orange for quality bands
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
    # Aggregate plot band colors
    "signal_band": "#0072B2",  # Darker blue for signal confidence bands
    "quality_band": "#FF8C00",  # Dark orange for quality bands
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

# Modification type colors (shades matching base colors from eventalign view)
# Each modification uses shades within the same color family as its canonical base
# BASE_COLORS reference: C=#F0E442 (yellow), A=#009E73 (green), G=#0072B2 (blue), T/U=#D55E00 (orange)
MODIFICATION_COLORS = {
    # Cytosine modifications (yellow family - C=#F0E442)
    "m": "#F0E442",  # Base yellow (5mC)
    "h": "#E6D835",  # Dark yellow (5hmC)
    "f": "#DCC728",  # Darker yellow (5fC)
    "c": "#FFF78A",  # Light yellow (5caC)
    21839: "#FFFC9E",  # Very light yellow (4mC)
    19228: "#D4BC1F",  # Deep yellow (Cm)
    "C": "#F0E442",  # Base yellow (any C*)
    # Adenine modifications (green family - A=#009E73)
    "a": "#009E73",  # Base green (6mA)
    17596: "#00C490",  # Light green (inosine)
    69426: "#007A57",  # Dark green (Am)
    "A": "#009E73",  # Base green (any A*)
    # Guanine modifications (blue family - G=#0072B2)
    "o": "#0072B2",  # Base blue (8oxoG)
    19229: "#4DA6E0",  # Light blue (Gm)
    "G": "#0072B2",  # Base blue (any G*)
    # Thymine/Uracil modifications (orange family - T/U=#D55E00)
    "g": "#D55E00",  # Base orange (5hmU)
    "e": "#FF7518",  # Light orange (5fU)
    "b": "#B34C00",  # Dark orange (5caU)
    17802: "#FF9447",  # Lighter orange (pseudouridine Ψ)
    16450: "#8F3D00",  # Deep orange (dU)
    19227: "#FFB880",  # Very light orange (Um)
    "T": "#D55E00",  # Base orange (any T*)
    # Default for unknown modifications
    "default": "#000000",  # Black
}

# ==============================================================================
# Delta Comparison Plot Settings
# ==============================================================================

# Delta plot dimensions
DELTA_SIGNAL_HEIGHT = 300  # Delta signal track height
DELTA_STATS_HEIGHT = 200  # Delta stats track height

# Delta visualization colors
DELTA_POSITIVE_COLOR = "#E74C3C"  # Red for positive deltas (B > A)
DELTA_NEGATIVE_COLOR = "#3498DB"  # Blue for negative deltas (B < A)
DELTA_NEUTRAL_COLOR = "#95A5A6"  # Gray for near-zero deltas
DELTA_ZERO_LINE_COLOR = "#34495E"  # Dark gray for zero reference line
DELTA_BAND_ALPHA = 0.3  # Alpha for delta confidence bands
DELTA_LINE_WIDTH = 1.5  # Line width for delta tracks
