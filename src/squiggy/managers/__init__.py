"""Manager classes for Squiggy viewer

This module provides manager classes that handle specific responsibilities,
extracted from the main SquiggleViewer class for better maintainability.
"""

from .export import ExportManager
from .zoom import ZoomManager

__all__ = [
    "ExportManager",
    "ZoomManager",
]
