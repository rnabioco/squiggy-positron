"""UI component panels for Squiggy viewer

This module provides UI panel components for the Squiggy application.
Each panel is in a separate file for better maintainability.
"""

from .advanced_options import AdvancedOptionsPanel
from .file_info import FileInfoPanel
from .modifications import ModificationsPanel
from .plot_options import PlotOptionsPanel
from .search import SearchPanel

__all__ = [
    "AdvancedOptionsPanel",
    "FileInfoPanel",
    "ModificationsPanel",
    "PlotOptionsPanel",
    "SearchPanel",
]
