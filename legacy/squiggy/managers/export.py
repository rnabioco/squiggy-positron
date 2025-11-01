"""Export manager for plot export operations"""

import json
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QMessageBox

from ..dialogs import ExportDialog


class ExportManager:
    """Manages plot export operations (HTML, PNG, SVG)"""

    def __init__(self, viewer):
        """Initialize export manager

        Args:
            viewer: SquiggleViewer instance (parent window)
        """
        self.viewer = viewer

    def get_current_view_ranges(self):
        """Get current x and y ranges from the displayed Bokeh plot

        Returns:
            tuple: ((x_start, x_end), (y_start, y_end)) or (None, None) if unavailable
        """
        # JavaScript to extract current range values from Bokeh plot
        js_code = """
        (function() {
            try {
                // Get the Bokeh document root
                var root = Bokeh.documents[0].roots()[0];

                // Get x and y ranges
                var x_range = root.x_range;
                var y_range = root.y_range;

                return JSON.stringify({
                    x_start: x_range.start,
                    x_end: x_range.end,
                    y_start: y_range.start,
                    y_end: y_range.end
                });
            } catch(e) {
                return null;
            }
        })();
        """

        # Execute JavaScript and get result synchronously
        result = [None]  # Use list to store result in callback

        def callback(js_result):
            result[0] = js_result

        self.viewer.plot_view.page().runJavaScript(js_code, callback)

        # Process events to wait for JavaScript execution
        QApplication.processEvents()

        if result[0]:
            try:
                ranges = json.loads(result[0])
                x_range = (ranges["x_start"], ranges["x_end"])
                y_range = (ranges["y_start"], ranges["y_end"])
                return x_range, y_range
            except Exception:
                pass

        return None, None

    def export_html(self, file_path):
        """Export plot as HTML file

        Args:
            file_path: Path to save HTML file
        """
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self.viewer.current_plot_html)

    def export_png(self, file_path, width, height, x_range=None, y_range=None):
        """Export plot as PNG image

        Args:
            file_path: Path to save PNG file
            width: Image width in pixels
            height: Image height in pixels
            x_range: Optional tuple (x_start, x_end) to set x-axis range
            y_range: Optional tuple (y_start, y_end) to set y-axis range

        Raises:
            ImportError: If selenium or pillow not installed
        """
        try:
            from bokeh.io import export_png
        except ImportError as e:
            raise ImportError(
                "PNG export requires additional dependencies. "
                "Install with: uv pip install -e '.[export]'"
            ) from e

        # Store original dimensions, sizing mode, and ranges
        original_width = self.viewer.current_plot_figure.width
        original_height = self.viewer.current_plot_figure.height
        original_sizing_mode = self.viewer.current_plot_figure.sizing_mode
        original_x_range = None
        original_y_range = None

        try:
            # Temporarily modify figure dimensions and sizing mode
            # Setting sizing_mode to None allows explicit width/height to work properly
            self.viewer.current_plot_figure.sizing_mode = None
            self.viewer.current_plot_figure.width = width
            self.viewer.current_plot_figure.height = height

            # Apply custom ranges if provided
            if x_range is not None:
                original_x_range = (
                    self.viewer.current_plot_figure.x_range.start,
                    self.viewer.current_plot_figure.x_range.end,
                )
                self.viewer.current_plot_figure.x_range.start = x_range[0]
                self.viewer.current_plot_figure.x_range.end = x_range[1]

            if y_range is not None:
                original_y_range = (
                    self.viewer.current_plot_figure.y_range.start,
                    self.viewer.current_plot_figure.y_range.end,
                )
                self.viewer.current_plot_figure.y_range.start = y_range[0]
                self.viewer.current_plot_figure.y_range.end = y_range[1]

            # Export to PNG
            export_png(self.viewer.current_plot_figure, filename=str(file_path))
        finally:
            # Restore original dimensions and sizing mode
            self.viewer.current_plot_figure.sizing_mode = original_sizing_mode
            self.viewer.current_plot_figure.width = original_width
            self.viewer.current_plot_figure.height = original_height

            # Restore original ranges
            if original_x_range is not None:
                self.viewer.current_plot_figure.x_range.start = original_x_range[0]
                self.viewer.current_plot_figure.x_range.end = original_x_range[1]

            if original_y_range is not None:
                self.viewer.current_plot_figure.y_range.start = original_y_range[0]
                self.viewer.current_plot_figure.y_range.end = original_y_range[1]

    def export_svg(self, file_path, width, height, x_range=None, y_range=None):
        """Export plot as SVG image

        Args:
            file_path: Path to save SVG file
            width: Image width in pixels
            height: Image height in pixels
            x_range: Optional tuple (x_start, x_end) to set x-axis range
            y_range: Optional tuple (y_start, y_end) to set y-axis range

        Raises:
            ImportError: If selenium not installed
        """
        try:
            from bokeh.io import export_svgs
        except ImportError as e:
            raise ImportError(
                "SVG export requires additional dependencies. "
                "Install with: uv pip install -e '.[export]'"
            ) from e

        # Store original dimensions, backend, sizing mode, and ranges
        original_width = self.viewer.current_plot_figure.width
        original_height = self.viewer.current_plot_figure.height
        original_backend = self.viewer.current_plot_figure.output_backend
        original_sizing_mode = self.viewer.current_plot_figure.sizing_mode
        original_x_range = None
        original_y_range = None

        try:
            # Temporarily modify figure dimensions, backend, and sizing mode
            # Setting sizing_mode to None allows explicit width/height to work properly
            self.viewer.current_plot_figure.sizing_mode = None
            self.viewer.current_plot_figure.width = width
            self.viewer.current_plot_figure.height = height
            self.viewer.current_plot_figure.output_backend = "svg"

            # Apply custom ranges if provided
            if x_range is not None:
                original_x_range = (
                    self.viewer.current_plot_figure.x_range.start,
                    self.viewer.current_plot_figure.x_range.end,
                )
                self.viewer.current_plot_figure.x_range.start = x_range[0]
                self.viewer.current_plot_figure.x_range.end = x_range[1]

            if y_range is not None:
                original_y_range = (
                    self.viewer.current_plot_figure.y_range.start,
                    self.viewer.current_plot_figure.y_range.end,
                )
                self.viewer.current_plot_figure.y_range.start = y_range[0]
                self.viewer.current_plot_figure.y_range.end = y_range[1]

            # Export to SVG
            export_svgs(self.viewer.current_plot_figure, filename=str(file_path))
        finally:
            # Restore original dimensions, backend, and sizing mode
            self.viewer.current_plot_figure.sizing_mode = original_sizing_mode
            self.viewer.current_plot_figure.width = original_width
            self.viewer.current_plot_figure.height = original_height
            self.viewer.current_plot_figure.output_backend = original_backend

            # Restore original ranges
            if original_x_range is not None:
                self.viewer.current_plot_figure.x_range.start = original_x_range[0]
                self.viewer.current_plot_figure.x_range.end = original_x_range[1]

            if original_y_range is not None:
                self.viewer.current_plot_figure.y_range.start = original_y_range[0]
                self.viewer.current_plot_figure.y_range.end = original_y_range[1]

    def export_plot(self):
        """Export the current plot with format and dimension options"""
        if (
            self.viewer.current_plot_html is None
            or self.viewer.current_plot_figure is None
        ):
            QMessageBox.warning(
                self.viewer,
                "No Plot",
                "No plot to export. Please display a plot first.",
            )
            return

        # Show export dialog
        dialog = ExportDialog(self.viewer)
        if dialog.exec() != QDialog.Accepted:
            return  # User cancelled

        # Get export settings
        settings = dialog.get_export_settings()
        export_format = settings["format"]
        width = settings["width"]
        height = settings["height"]
        use_current_view = settings["use_current_view"]

        # Get current view ranges if needed
        x_range, y_range = None, None
        if use_current_view:
            x_range, y_range = self.get_current_view_ranges()
            if x_range is None or y_range is None:
                QMessageBox.warning(
                    self.viewer,
                    "Cannot Get View Range",
                    "Unable to extract current zoom level. Exporting full plot instead.",
                )
                use_current_view = False

        # Determine file extension and filter based on format
        if export_format == "html":
            filter_str = "HTML File (*.html);;All Files (*)"
            default_ext = ".html"
        elif export_format == "png":
            filter_str = "PNG Image (*.png);;All Files (*)"
            default_ext = ".png"
        else:  # svg
            filter_str = "SVG Image (*.svg);;All Files (*)"
            default_ext = ".svg"

        # Get file path from user
        file_path, _ = QFileDialog.getSaveFileName(
            self.viewer,
            "Export Plot",
            "",
            filter_str,
        )

        if not file_path:
            return  # User cancelled

        try:
            # Ensure correct extension
            file_path = Path(file_path)
            if not file_path.suffix:
                file_path = file_path.with_suffix(default_ext)

            # Show progress
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.viewer.statusBar().showMessage(
                f"Exporting plot as {export_format.upper()}..."
            )

            try:
                # Export based on format
                if export_format == "html":
                    self.export_html(file_path)
                    view_info = " (current view)" if use_current_view else ""
                    message = (
                        f"Interactive plot successfully exported to:\n{file_path}\n\n"
                        f"Open the file in a web browser to view the interactive plot."
                    )
                elif export_format == "png":
                    self.export_png(file_path, width, height, x_range, y_range)
                    view_info = " (current view)" if use_current_view else ""
                    message = (
                        f"PNG image successfully exported to:\n{file_path}\n\n"
                        f"Dimensions: {width} × {height} pixels{view_info}"
                    )
                else:  # svg
                    self.export_svg(file_path, width, height, x_range, y_range)
                    view_info = " (current view)" if use_current_view else ""
                    message = (
                        f"SVG image successfully exported to:\n{file_path}\n\n"
                        f"Dimensions: {width} × {height} pixels{view_info}\n"
                        f"Edit in vector graphics software like Inkscape or Adobe Illustrator."
                    )

                self.viewer.statusBar().showMessage(
                    f"Plot exported to {file_path.name}"
                )
                QMessageBox.information(self.viewer, "Export Successful", message)

            finally:
                QApplication.restoreOverrideCursor()

        except ImportError as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(
                self.viewer,
                "Missing Dependencies",
                f"{str(e)}\n\n"
                f"PNG and SVG export require additional dependencies:\n"
                f"• selenium (for headless browser rendering)\n"
                f"• pillow (for image handling)\n\n"
                f"Install with:\n"
                f"  uv pip install -e '.[export]'\n\n"
                f"or:\n"
                f"  pip install selenium pillow",
            )
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(
                self.viewer, "Export Failed", f"Failed to export plot:\n{str(e)}"
            )
