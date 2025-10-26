"""Constants and configuration for Squiggy application"""

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
SIGNAL_LINE_COLOR = "#2E86AB"
SIGNAL_LINE_WIDTH = 0.5

# Base colors from remora visualization
BASE_COLORS = {
    "A": "#00CC00",  # Green
    "C": "#0000CC",  # Blue
    "G": "#FFB300",  # Orange
    "T": "#CC0000",  # Red
    "U": "#CC0000",  # Red (for RNA)
    "N": "#FFFFFF",  # White (unknown)
}

# Base annotation settings
BASE_ANNOTATION_ALPHA = 0.1
BASE_LABEL_SIZE = 8
SIGNAL_PERCENTILE_LOW = 2.5
SIGNAL_PERCENTILE_HIGH = 97.5
