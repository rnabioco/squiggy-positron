"""
Rendering utilities for Squiggy plots

This package contains rendering infrastructure used by plot strategies:
- ThemeManager: Bokeh figure theming (light/dark mode)
- BaseAnnotationRenderer: DNA/RNA base annotations on plots
- ModificationTrackBuilder: Base modification visualization tracks
"""

from .base_annotation_renderer import BaseAnnotationRenderer
from .modification_track_builder import ModificationTrackBuilder
from .theme_manager import ThemeManager

__all__ = [
    "ThemeManager",
    "BaseAnnotationRenderer",
    "ModificationTrackBuilder",
]
