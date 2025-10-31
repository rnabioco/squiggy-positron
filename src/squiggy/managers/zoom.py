"""Zoom manager for plot zoom and range operations"""

import json

from PySide6.QtCore import Qt, QTimer


class ZoomManager:
    """Manages plot zoom, pan, and range operations via JavaScript"""

    def __init__(self, viewer):
        """Initialize zoom manager

        Args:
            viewer: SquiggleViewer instance (parent window)
        """
        self.viewer = viewer
        self.saved_x_range = None
        self.saved_y_range = None

    def save_plot_ranges(self):
        """Extract and save current plot ranges via JavaScript"""
        js_code = """
        (function() {
            try {
                // Find the Bokeh plot object
                const plots = Bokeh.documents[0].roots();
                if (plots.length > 0) {
                    // Get the first plot (main figure)
                    let plot = plots[0];

                    // If it's a layout, find the actual plot figure
                    if (plot.constructor.name === 'Column' || plot.constructor.name === 'Row') {
                        for (let child of plot.children) {
                            if (child.constructor.name === 'Figure') {
                                plot = child;
                                break;
                            }
                            // Check nested layouts
                            if (child.children) {
                                for (let nested of child.children) {
                                    if (nested.constructor.name === 'Figure') {
                                        plot = nested;
                                        break;
                                    }
                                }
                            }
                        }
                    }

                    // Return the ranges
                    return JSON.stringify({
                        x_start: plot.x_range.start,
                        x_end: plot.x_range.end,
                        y_start: plot.y_range.start,
                        y_end: plot.y_range.end
                    });
                }
            } catch (e) {
                return null;
            }
            return null;
        })();
        """

        # Execute JavaScript and get the result
        self.viewer.plot_view.page().runJavaScript(js_code, self.on_ranges_extracted)

    def on_ranges_extracted(self, result):
        """Callback when ranges are extracted from JavaScript"""
        if result:
            try:
                ranges = json.loads(result)
                self.saved_x_range = (ranges["x_start"], ranges["x_end"])
                self.saved_y_range = (ranges["y_start"], ranges["y_end"])
            except (json.JSONDecodeError, KeyError, TypeError):
                # If parsing fails, just use None (will reset zoom)
                self.saved_x_range = None
                self.saved_y_range = None

    def restore_plot_ranges(self):
        """Restore saved plot ranges via JavaScript after plot is loaded"""
        if self.saved_x_range is None or self.saved_y_range is None:
            return

        x_start, x_end = self.saved_x_range
        y_start, y_end = self.saved_y_range

        js_code = f"""
        (function() {{
            try {{
                // Wait for Bokeh to be ready
                if (typeof Bokeh === 'undefined') {{
                    return;
                }}

                // Find the Bokeh plot object
                const plots = Bokeh.documents[0].roots();
                if (plots.length > 0) {{
                    // Get the first plot (main figure)
                    let plot = plots[0];

                    // If it's a layout, find the actual plot figure
                    if (plot.constructor.name === 'Column' || plot.constructor.name === 'Row') {{
                        for (let child of plot.children) {{
                            if (child.constructor.name === 'Figure') {{
                                plot = child;
                                break;
                            }}
                            // Check nested layouts
                            if (child.children) {{
                                for (let nested of child.children) {{
                                    if (nested.constructor.name === 'Figure') {{
                                        plot = nested;
                                        break;
                                    }}
                                }}
                            }}
                        }}
                    }}

                    // Restore the ranges
                    plot.x_range.start = {x_start};
                    plot.x_range.end = {x_end};
                    plot.y_range.start = {y_start};
                    plot.y_range.end = {y_end};
                }}
            }} catch (e) {{
                console.error('Error restoring plot ranges:', e);
            }}
        }})();
        """

        # Execute JavaScript with a delay to ensure plot is loaded
        QTimer.singleShot(
            200, lambda: self.viewer.plot_view.page().runJavaScript(js_code)
        )

    def zoom_to_sequence_match(self, item):
        """Zoom plot to show a sequence match

        Args:
            item: QListWidgetItem containing match data (with Qt.UserRole)
        """
        match = item.data(Qt.UserRole)
        if not match:
            return

        # Extract base position range
        base_start = match["base_start"]
        base_end = match["base_end"]

        # Add some padding (show 20 bases before and after)
        padding = 20
        zoom_start = max(0, base_start - padding)
        zoom_end = base_end + padding

        # JavaScript to zoom the plot
        js_code = f"""
        (function() {{
            try {{
                if (typeof Bokeh === 'undefined') {{
                    return;
                }}

                const plots = Bokeh.documents[0].roots();
                if (plots.length > 0) {{
                    let plot = plots[0];

                    // If it's a layout, find the actual plot figure
                    if (plot.constructor.name === 'Column' || plot.constructor.name === 'Row') {{
                        for (let child of plot.children) {{
                            if (child.constructor.name === 'Figure') {{
                                plot = child;
                                break;
                            }}
                            if (child.children) {{
                                for (let nested of child.children) {{
                                    if (nested.constructor.name === 'Figure') {{
                                        plot = nested;
                                        break;
                                    }}
                                }}
                            }}
                        }}
                    }}

                    // Zoom to the region
                    plot.x_range.start = {zoom_start};
                    plot.x_range.end = {zoom_end};
                }}
            }} catch (e) {{
                console.error('Error zooming to sequence:', e);
            }}
        }})();
        """

        self.viewer.plot_view.page().runJavaScript(js_code)
        self.viewer.statusBar().showMessage(
            f"Zoomed to {match['strand']} match at position {match['base_start']}-{match['base_end']}"
        )

    def clear_saved_ranges(self):
        """Clear saved ranges (useful when loading new plot)"""
        self.saved_x_range = None
        self.saved_y_range = None
