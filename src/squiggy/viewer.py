"""Main application window for Squiggy"""

import asyncio
import time
from pathlib import Path

import pod5
import qasync
from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QAction
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from qt_material import apply_stylesheet

from .constants import (
    APP_DESCRIPTION,
    APP_NAME,
    DEFAULT_WINDOW_HEIGHT,
    DEFAULT_WINDOW_WIDTH,
    MAX_OVERLAY_READS,
    PLOT_MIN_HEIGHT,
    PLOT_MIN_WIDTH,
    NormalizationMethod,
    PlotMode,
    Theme,
)
from .dialogs import AboutDialog, ExportDialog, ReferenceBrowserDialog
from .plotter import SquigglePlotter
from .utils import (
    get_bam_references,
    get_basecall_data,
    get_reads_in_region,
    get_sample_data_path,
    index_bam_file,
    parse_region,
    validate_bam_reads_in_pod5,
)


class CollapsibleBox(QWidget):
    """A collapsible widget with a toggle button to show/hide content"""

    def __init__(self, title="", parent=None):
        super().__init__(parent)

        self.toggle_button = QToolButton()
        self.toggle_button.setText(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(False)
        self.toggle_button.setStyleSheet(
            "QToolButton { border: none; font-weight: bold; }"
        )
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.RightArrow)
        self.toggle_button.clicked.connect(self.on_toggle)

        self.content_area = QScrollArea()
        self.content_area.setMaximumHeight(0)
        self.content_area.setMinimumHeight(0)
        self.content_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.content_area.setFrameShape(QFrame.NoFrame)

        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toggle_button)
        layout.addWidget(self.content_area)

    def on_toggle(self):
        """Toggle the collapsible section"""
        checked = self.toggle_button.isChecked()
        self.toggle_button.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        if checked:
            self.content_area.setMaximumHeight(350)
        else:
            self.content_area.setMaximumHeight(0)

    def set_content_layout(self, layout):
        """Set the layout for the content area"""
        widget = QWidget()
        widget.setLayout(layout)
        self.content_area.setWidget(widget)


class SquiggleViewer(QMainWindow):
    """Main application window for nanopore squiggle visualization"""

    def __init__(self):
        super().__init__()
        self.pod5_file = None
        self.bam_file = None
        self.read_dict = {}
        self.alignment_info = {}  # Maps read_id -> alignment metadata
        self.current_read_item = None
        self.show_bases = True  # Default to showing base annotations
        self.plot_mode = (
            PlotMode.EVENTALIGN
        )  # Default to event-aligned mode (primary mode)
        self.normalization_method = NormalizationMethod.MEDIAN
        self.downsample_factor = 25  # Default downsampling for performance
        self.show_dwell_time = False  # Show dwell time coloring
        self.current_plot_html = None  # Store current plot HTML for export
        self.current_plot_figure = None  # Store current plot figure for export
        self.search_mode = "read_id"  # "read_id" or "region"
        self.saved_x_range = None  # Store current x-axis range for zoom preservation
        self.saved_y_range = None  # Store current y-axis range for zoom preservation
        self.show_signal_points = False  # Show individual signal points on plot
        self.position_label_interval = 10  # Show position labels every N bases
        self.use_reference_positions = (
            False  # Use reference positions vs sequence positions
        )
        self.current_theme = Theme.DARK  # Default to dark theme

        self.init_ui()
        self.apply_theme()  # Apply initial theme

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle(f"{APP_NAME} - {APP_DESCRIPTION}")
        self.setGeometry(100, 100, DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)

        # Create menu bar
        self.create_menu_bar()

        # Create toolbar with dark mode toggle
        self.create_toolbar()

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Create splitter for left panel, plot area, and read list
        splitter = QSplitter(Qt.Horizontal)

        # Left panel - file browser and plot options
        left_panel = QWidget()
        left_panel_layout = QVBoxLayout(left_panel)
        left_panel_layout.setContentsMargins(0, 0, 0, 0)

        # File selection section
        file_layout = QHBoxLayout()
        self.file_label = QLabel("No file selected")
        self.file_button = QPushButton("Open POD5 File")
        self.file_button.clicked.connect(self.open_pod5_file)
        file_layout.addWidget(QLabel("POD5 File:"))
        file_layout.addWidget(self.file_label, 1)
        file_layout.addWidget(self.file_button)
        left_panel_layout.addLayout(file_layout)

        # BAM file selection section (optional)
        bam_layout = QHBoxLayout()
        self.bam_label = QLabel("No BAM file (optional)")
        self.bam_button = QPushButton("Open BAM File")
        self.bam_button.clicked.connect(self.open_bam_file)
        bam_layout.addWidget(QLabel("BAM File:"))
        bam_layout.addWidget(self.bam_label, 1)
        bam_layout.addWidget(self.bam_button)
        left_panel_layout.addLayout(bam_layout)

        # Plot options collapsible panel (open by default)
        self.plot_options_box = CollapsibleBox("Plot Options")
        self.create_plot_options_content()
        left_panel_layout.addWidget(self.plot_options_box)
        # Set plot options to be expanded by default
        self.plot_options_box.toggle_button.setChecked(True)
        self.plot_options_box.on_toggle()

        # Advanced options collapsible panel (collapsed by default)
        self.advanced_options_box = CollapsibleBox("Advanced")
        self.create_advanced_options_content()
        left_panel_layout.addWidget(self.advanced_options_box)

        # POD5 file information collapsible panel
        self.file_info_box = CollapsibleBox("POD5 File Information")
        self.create_file_info_content()
        left_panel_layout.addWidget(self.file_info_box)

        # Add stretch to push everything to the top
        left_panel_layout.addStretch()

        splitter.addWidget(left_panel)

        # Plot display area (using QWebEngineView for interactive bokeh plots)
        self.plot_view = QWebEngineView()
        self.plot_view.setMinimumSize(PLOT_MIN_WIDTH, PLOT_MIN_HEIGHT)
        # Will be set by apply_theme() which is called after init_ui()
        splitter.addWidget(self.plot_view)

        # Right panel - Read list widget with multi-selection enabled
        self.read_list = QListWidget()
        self.read_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.read_list.itemSelectionChanged.connect(self.on_read_selection_changed)
        splitter.addWidget(self.read_list)

        # Set splitter proportions (left panel, plot, right panel)
        splitter.setStretchFactor(0, 1)  # Left panel - narrower
        splitter.setStretchFactor(1, 3)  # Plot area - widest
        splitter.setStretchFactor(2, 1)  # Right panel (read list) - narrower

        main_layout.addWidget(splitter, 1)

        # Bottom search panel
        self.create_search_panel()
        main_layout.addWidget(self.search_panel)

        # Status bar
        self.statusBar().showMessage("Ready")

    def create_search_panel(self):
        """Create the bottom search panel with all search modes"""
        self.search_panel = QWidget()
        search_panel_layout = QVBoxLayout(self.search_panel)
        search_panel_layout.setContentsMargins(5, 5, 5, 5)

        # Main search controls row
        search_layout = QHBoxLayout()

        # Search mode selector
        self.search_mode_combo = QComboBox()
        self.search_mode_combo.addItem("Read ID", "read_id")
        self.search_mode_combo.addItem("Reference Region", "region")
        self.search_mode_combo.addItem("Sequence", "sequence")
        self.search_mode_combo.currentIndexChanged.connect(self.on_search_mode_changed)
        self.search_mode_combo.setToolTip(
            "Switch between searching by read ID, genomic region, or sequence"
        )
        search_layout.addWidget(QLabel("Search by:"))
        search_layout.addWidget(self.search_mode_combo)

        # Search input
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search read ID...")
        self.search_input.textChanged.connect(self.on_search_text_changed)
        self.search_input.returnPressed.connect(self.execute_search)
        search_layout.addWidget(self.search_input, 1)

        # Search button
        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self.execute_search)
        search_layout.addWidget(self.search_button)

        # Browse references button (for region search mode)
        self.browse_refs_button = QPushButton("Browse References...")
        self.browse_refs_button.clicked.connect(self.browse_references)
        self.browse_refs_button.setEnabled(False)
        self.browse_refs_button.setVisible(False)  # Hidden by default
        self.browse_refs_button.setToolTip(
            "View available reference sequences in BAM file"
        )
        search_layout.addWidget(self.browse_refs_button)

        # Reverse complement checkbox (for sequence search mode)
        self.revcomp_checkbox = QCheckBox("Include reverse complement")
        self.revcomp_checkbox.setChecked(True)
        self.revcomp_checkbox.setVisible(False)  # Hidden by default
        self.revcomp_checkbox.setToolTip(
            "Also search for the reverse complement of the query sequence"
        )
        search_layout.addWidget(self.revcomp_checkbox)

        search_panel_layout.addLayout(search_layout)

        # Sequence search results area (collapsible, hidden by default)
        self.sequence_results_box = CollapsibleBox("Search Results")
        self.sequence_results_list = QListWidget()
        self.sequence_results_list.itemClicked.connect(self.zoom_to_sequence_match)
        results_layout = QVBoxLayout()
        results_layout.addWidget(self.sequence_results_list)
        self.sequence_results_box.set_content_layout(results_layout)
        self.sequence_results_box.setVisible(False)  # Hidden by default
        search_panel_layout.addWidget(self.sequence_results_box)

    def create_file_info_content(self):
        """Create the content layout for the file information panel"""
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(10, 5, 10, 5)
        content_layout.setSpacing(3)

        # Create labels for file information with alternating title/value layout
        # Title labels are bold, value labels are regular

        # File name
        label_filename = QLabel("File name:")
        label_filename.setStyleSheet("font-size: 9pt; font-weight: bold;")
        content_layout.addWidget(label_filename)
        self.info_filename_label = QLabel("—")
        self.info_filename_label.setStyleSheet("font-size: 9pt; padding-left: 10px;")
        self.info_filename_label.setWordWrap(True)
        content_layout.addWidget(self.info_filename_label)

        # File size
        label_filesize = QLabel("File size:")
        label_filesize.setStyleSheet(
            "font-size: 9pt; font-weight: bold; margin-top: 5px;"
        )
        content_layout.addWidget(label_filesize)
        self.info_filesize_label = QLabel("—")
        self.info_filesize_label.setStyleSheet("font-size: 9pt; padding-left: 10px;")
        self.info_filesize_label.setWordWrap(True)
        content_layout.addWidget(self.info_filesize_label)

        # Number of reads
        label_num_reads = QLabel("Number of reads:")
        label_num_reads.setStyleSheet(
            "font-size: 9pt; font-weight: bold; margin-top: 5px;"
        )
        content_layout.addWidget(label_num_reads)
        self.info_num_reads_label = QLabel("—")
        self.info_num_reads_label.setStyleSheet("font-size: 9pt; padding-left: 10px;")
        self.info_num_reads_label.setWordWrap(True)
        content_layout.addWidget(self.info_num_reads_label)

        # Sample rate
        label_sample_rate = QLabel("Sample rate:")
        label_sample_rate.setStyleSheet(
            "font-size: 9pt; font-weight: bold; margin-top: 5px;"
        )
        content_layout.addWidget(label_sample_rate)
        self.info_sample_rate_label = QLabel("—")
        self.info_sample_rate_label.setStyleSheet("font-size: 9pt; padding-left: 10px;")
        self.info_sample_rate_label.setWordWrap(True)
        content_layout.addWidget(self.info_sample_rate_label)

        # Total samples
        label_total_samples = QLabel("Total samples:")
        label_total_samples.setStyleSheet(
            "font-size: 9pt; font-weight: bold; margin-top: 5px;"
        )
        content_layout.addWidget(label_total_samples)
        self.info_total_samples_label = QLabel("—")
        self.info_total_samples_label.setStyleSheet(
            "font-size: 9pt; padding-left: 10px;"
        )
        self.info_total_samples_label.setWordWrap(True)
        content_layout.addWidget(self.info_total_samples_label)

        self.file_info_box.set_content_layout(content_layout)

    def create_plot_options_content(self):
        """Create the content layout for the plot options panel"""
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(10, 5, 10, 5)
        content_layout.setSpacing(10)

        # Plot mode selection
        mode_label = QLabel("Plot Mode:")
        mode_label.setStyleSheet("font-weight: bold;")
        content_layout.addWidget(mode_label)

        self.mode_button_group = QButtonGroup()

        # Event-aligned mode (top - default when BAM is loaded)
        self.mode_eventalign = QRadioButton("Event-aligned (base annotations)")
        self.mode_eventalign.setToolTip(
            "Show event-aligned reads with base annotations (requires BAM file)"
        )
        self.mode_eventalign.toggled.connect(
            lambda checked: self.set_plot_mode(PlotMode.EVENTALIGN) if checked else None
        )
        self.mode_eventalign.setChecked(
            True
        )  # Checked by default (visual indication of primary mode)
        self.mode_eventalign.setEnabled(False)  # Disabled until BAM file loaded
        self.mode_button_group.addButton(self.mode_eventalign)
        content_layout.addWidget(self.mode_eventalign)

        self.mode_overlay = QRadioButton("Overlay (multiple reads)")
        self.mode_overlay.setToolTip(
            f"Overlay multiple reads on same axes (max {MAX_OVERLAY_READS})"
        )
        self.mode_overlay.toggled.connect(
            lambda checked: self.set_plot_mode(PlotMode.OVERLAY) if checked else None
        )
        self.mode_button_group.addButton(self.mode_overlay)
        content_layout.addWidget(self.mode_overlay)

        self.mode_stacked = QRadioButton("Stacked (squigualiser-style)")
        self.mode_stacked.setToolTip("Stack multiple reads vertically with offset")
        self.mode_stacked.toggled.connect(
            lambda checked: self.set_plot_mode(PlotMode.STACKED) if checked else None
        )
        self.mode_button_group.addButton(self.mode_stacked)
        content_layout.addWidget(self.mode_stacked)

        # Single read mode (bottom - fallback when no BAM)
        self.mode_single = QRadioButton("Single Read")
        self.mode_single.setToolTip("Display one read at a time")
        self.mode_single.toggled.connect(
            lambda checked: self.set_plot_mode(PlotMode.SINGLE) if checked else None
        )
        self.mode_button_group.addButton(self.mode_single)
        content_layout.addWidget(self.mode_single)

        # Normalization method selection
        norm_label = QLabel("Signal Normalization:")
        norm_label.setStyleSheet("font-weight: bold;")
        content_layout.addWidget(norm_label)

        self.norm_combo = QComboBox()
        self.norm_combo.addItem("None (raw signal)", NormalizationMethod.NONE)
        self.norm_combo.addItem("Z-score", NormalizationMethod.ZNORM)
        self.norm_combo.addItem("Median", NormalizationMethod.MEDIAN)
        self.norm_combo.addItem("MAD (robust)", NormalizationMethod.MAD)
        self.norm_combo.setCurrentIndex(2)  # Default to Median
        self.norm_combo.currentIndexChanged.connect(self.set_normalization_method)
        content_layout.addWidget(self.norm_combo)

        # Base annotations toggle
        base_label = QLabel("Base Annotations:")
        base_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        content_layout.addWidget(base_label)

        self.base_checkbox = QCheckBox("Show base annotations")
        self.base_checkbox.setChecked(True)  # Checked by default
        self.base_checkbox.setEnabled(False)  # Disabled until BAM file loaded
        self.base_checkbox.setToolTip(
            "Show base letters on event-aligned plots (requires BAM file)"
        )
        self.base_checkbox.stateChanged.connect(self.toggle_base_annotations)
        content_layout.addWidget(self.base_checkbox)

        # Signal points toggle
        self.points_checkbox = QCheckBox("Show signal points")
        self.points_checkbox.setChecked(False)  # Unchecked by default
        self.points_checkbox.setToolTip(
            "Display individual signal data points as circles on the squiggle line"
        )
        self.points_checkbox.stateChanged.connect(self.toggle_signal_points)
        content_layout.addWidget(self.points_checkbox)

        # Info label
        info_label = QLabel(
            "💡 Tip: Use Ctrl/Cmd+Click or Shift+Click to select multiple reads"
        )
        info_label.setStyleSheet("color: #666; font-size: 9pt;")
        info_label.setWordWrap(True)
        content_layout.addWidget(info_label)

        content_layout.addStretch()

        self.plot_options_box.set_content_layout(content_layout)

    def create_advanced_options_content(self):
        """Create the content layout for the advanced options panel"""
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(10, 5, 10, 5)
        content_layout.setSpacing(10)

        # Downsampling control
        downsample_label = QLabel("Signal Downsampling:")
        downsample_label.setStyleSheet("font-weight: bold;")
        content_layout.addWidget(downsample_label)

        downsample_layout = QHBoxLayout()
        downsample_layout.addWidget(QLabel("Show every"))

        self.downsample_slider = QSlider(Qt.Horizontal)
        self.downsample_slider.setMinimum(1)
        self.downsample_slider.setMaximum(100)
        self.downsample_slider.setValue(25)
        self.downsample_slider.setTickPosition(QSlider.TicksBelow)
        self.downsample_slider.setTickInterval(10)
        self.downsample_slider.setToolTip(
            "Downsample signal for faster rendering\n"
            "1 = show all points (no downsampling)\n"
            "25 = show every 25th point (default)\n"
            "100 = show every 100th point (fastest)"
        )
        self.downsample_slider.valueChanged.connect(
            self.set_downsample_factor_from_slider
        )
        downsample_layout.addWidget(self.downsample_slider)

        # SpinBox for direct value entry
        self.downsample_spinbox = QSpinBox()
        self.downsample_spinbox.setMinimum(1)
        self.downsample_spinbox.setMaximum(100)
        self.downsample_spinbox.setValue(25)
        self.downsample_spinbox.setToolTip("Enter downsample factor (1-100)")
        self.downsample_spinbox.valueChanged.connect(
            self.set_downsample_factor_from_spinbox
        )
        downsample_layout.addWidget(self.downsample_spinbox)

        content_layout.addLayout(downsample_layout)

        # Info label
        info_label = QLabel("💡 Downsampling reduces rendering time for large signals")
        info_label.setStyleSheet("color: #666; font-size: 9pt;")
        info_label.setWordWrap(True)
        content_layout.addWidget(info_label)

        # Dwell time visualization
        dwell_label = QLabel("Dwell Time Visualization:")
        dwell_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        content_layout.addWidget(dwell_label)

        self.dwell_time_checkbox = QCheckBox("Scale x-axis by dwell time")
        self.dwell_time_checkbox.setToolTip(
            "Scale x-axis by actual sequencing time instead of base position\n"
            "Bases with longer dwell times take more horizontal space\n"
            "(requires event-aligned mode with BAM file)"
        )
        self.dwell_time_checkbox.setEnabled(
            False
        )  # Enabled when in EVENTALIGN mode with BAM
        self.dwell_time_checkbox.stateChanged.connect(self.toggle_dwell_time)
        content_layout.addWidget(self.dwell_time_checkbox)

        # Dwell time info label
        dwell_info_label = QLabel(
            "💡 Time-scaled x-axis: bases with longer dwell times take more horizontal space"
        )
        dwell_info_label.setStyleSheet("color: #666; font-size: 9pt;")
        dwell_info_label.setWordWrap(True)
        content_layout.addWidget(dwell_info_label)

        # Position label settings
        position_label = QLabel("Position Labels:")
        position_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        content_layout.addWidget(position_label)

        # Position label interval
        interval_layout = QHBoxLayout()
        interval_layout.addWidget(QLabel("Show every"))

        self.position_interval_spinbox = QSpinBox()
        self.position_interval_spinbox.setMinimum(1)
        self.position_interval_spinbox.setMaximum(100)
        self.position_interval_spinbox.setValue(10)
        self.position_interval_spinbox.setSuffix(" bases")
        self.position_interval_spinbox.setToolTip("Show position number every N bases")
        self.position_interval_spinbox.valueChanged.connect(
            self.on_position_interval_changed
        )
        interval_layout.addWidget(self.position_interval_spinbox)
        interval_layout.addStretch()

        content_layout.addLayout(interval_layout)

        # Position type toggle (sequence vs reference)
        self.position_type_checkbox = QCheckBox(
            "Use reference positions (when available)"
        )
        self.position_type_checkbox.setChecked(False)  # Default to sequence positions
        self.position_type_checkbox.setEnabled(False)  # Enabled when BAM file loaded
        self.position_type_checkbox.setToolTip(
            "Show genomic coordinates instead of sequence positions\n"
            "(requires BAM file with alignment)"
        )
        self.position_type_checkbox.stateChanged.connect(self.toggle_position_type)
        content_layout.addWidget(self.position_type_checkbox)

        # Position label info
        position_info_label = QLabel(
            "💡 Position labels show numbers on base annotations"
        )
        position_info_label.setStyleSheet("color: #666; font-size: 9pt;")
        position_info_label.setWordWrap(True)
        content_layout.addWidget(position_info_label)

        content_layout.addStretch()

        self.advanced_options_box.set_content_layout(content_layout)

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
        self.plot_view.page().runJavaScript(js_code, self.on_ranges_extracted)

    def on_ranges_extracted(self, result):
        """Callback when ranges are extracted from JavaScript"""
        if result:
            import json

            try:
                ranges = json.loads(result)
                self.saved_x_range = (ranges["x_start"], ranges["x_end"])
                self.saved_y_range = (ranges["y_start"], ranges["y_end"])
            except (json.JSONDecodeError, KeyError, TypeError):
                # If parsing fails, just use None (will reset zoom)
                self.saved_x_range = None
                self.saved_y_range = None

    @qasync.asyncSlot()
    async def set_downsample_factor_from_slider(self, value):
        """Set the downsampling factor from slider and update spinbox"""
        self.downsample_factor = value
        # Update spinbox to match (this won't trigger valueChanged if value is same)
        self.downsample_spinbox.blockSignals(True)
        self.downsample_spinbox.setValue(value)
        self.downsample_spinbox.blockSignals(False)
        # Refresh plot if reads are selected
        if self.read_list.selectedItems():
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.statusBar().showMessage("Regenerating plot...")
            try:
                # Save current zoom/pan state before regenerating
                self.save_plot_ranges()
                # Small delay to allow JavaScript to execute
                await self.update_plot_with_delay()
            finally:
                QApplication.restoreOverrideCursor()

    @qasync.asyncSlot()
    async def set_downsample_factor_from_spinbox(self, value):
        """Set the downsampling factor from spinbox and update slider"""
        self.downsample_factor = value
        # Update slider to match (this won't trigger valueChanged if value is same)
        self.downsample_slider.blockSignals(True)
        self.downsample_slider.setValue(value)
        self.downsample_slider.blockSignals(False)
        # Refresh plot if reads are selected
        if self.read_list.selectedItems():
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.statusBar().showMessage("Regenerating plot...")
            try:
                # Save current zoom/pan state before regenerating
                self.save_plot_ranges()
                # Small delay to allow JavaScript to execute
                await self.update_plot_with_delay()
            finally:
                QApplication.restoreOverrideCursor()

    async def update_plot_with_delay(self):
        """Update plot after a short delay to allow JavaScript extraction"""
        await asyncio.sleep(0.1)  # 100ms delay
        await self.update_plot_from_selection()

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
        QTimer.singleShot(200, lambda: self.plot_view.page().runJavaScript(js_code))

    @qasync.asyncSlot()
    async def toggle_dwell_time(self, state):
        """Toggle dwell time visualization"""
        self.show_dwell_time = bool(state)  # state is 0 (unchecked) or 2 (checked)
        # Refresh plot if reads are selected
        if self.read_list.selectedItems():
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.statusBar().showMessage("Regenerating plot...")
            try:
                # Save current zoom/pan state before regenerating
                self.save_plot_ranges()
                # Small delay to allow JavaScript to execute
                await self.update_plot_with_delay()
            finally:
                QApplication.restoreOverrideCursor()

    @qasync.asyncSlot(int)
    async def toggle_signal_points(self, state):
        """Toggle display of individual signal points"""
        self.show_signal_points = bool(state)  # state is 0 (unchecked) or 2 (checked)
        # Refresh plot if reads are selected
        if self.read_list.selectedItems():
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.statusBar().showMessage("Regenerating plot...")
            try:
                # Save current zoom/pan state before regenerating
                self.save_plot_ranges()
                # Small delay to allow JavaScript to execute
                await self.update_plot_with_delay()
            finally:
                QApplication.restoreOverrideCursor()

    @qasync.asyncSlot(int)
    async def toggle_position_type(self, state):
        """Toggle between sequence and reference positions"""
        self.use_reference_positions = bool(state)
        # Refresh plot if reads are selected
        if self.read_list.selectedItems():
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.statusBar().showMessage("Regenerating plot...")
            try:
                # Save current zoom/pan state before regenerating
                self.save_plot_ranges()
                # Small delay to allow JavaScript to execute
                await self.update_plot_with_delay()
            finally:
                QApplication.restoreOverrideCursor()

    @qasync.asyncSlot(int)
    async def on_position_interval_changed(self, value):
        """Handle position label interval change"""
        self.position_label_interval = value
        # Refresh plot if reads are selected
        if self.read_list.selectedItems():
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.statusBar().showMessage("Regenerating plot...")
            try:
                # Save current zoom/pan state before regenerating
                self.save_plot_ranges()
                # Small delay to allow JavaScript to execute
                await self.update_plot_with_delay()
            finally:
                QApplication.restoreOverrideCursor()

    def set_plot_mode(self, mode):
        """Set the plot mode and refresh display"""
        self.plot_mode = mode

        # Enable dwell time checkbox only in EVENTALIGN mode with BAM file
        # Only update checkbox if it exists (may not exist during initialization)
        if hasattr(self, "dwell_time_checkbox"):
            if mode == PlotMode.EVENTALIGN and self.bam_file:
                self.dwell_time_checkbox.setEnabled(True)
            else:
                self.dwell_time_checkbox.setEnabled(False)
                # Uncheck if disabled to avoid confusion
                if self.dwell_time_checkbox.isChecked():
                    self.dwell_time_checkbox.setChecked(False)

        # Refresh plot if reads are selected
        if hasattr(self, "read_list") and self.read_list.selectedItems():
            # Save current zoom/pan state before regenerating
            self.save_plot_ranges()
            # Small delay to allow JavaScript to execute
            asyncio.ensure_future(self.update_plot_with_delay())

    @qasync.asyncSlot()
    async def set_normalization_method(self, index):
        """Set the normalization method and refresh display"""
        self.normalization_method = self.norm_combo.itemData(index)
        # Refresh plot if reads are selected
        if hasattr(self, "read_list") and self.read_list.selectedItems():
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.statusBar().showMessage("Regenerating plot...")
            try:
                # Save current zoom/pan state before regenerating
                self.save_plot_ranges()
                # Small delay to allow JavaScript to execute
                await self.update_plot_with_delay()
            finally:
                QApplication.restoreOverrideCursor()

    def create_menu_bar(self):
        """Create the application menu bar"""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        # Open file action
        open_action = QAction("Open POD5 File...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_pod5_file)
        file_menu.addAction(open_action)

        # Open sample data action
        sample_action = QAction("Open Sample Data", self)
        sample_action.setShortcut("Ctrl+Shift+O")
        sample_action.triggered.connect(self.open_sample_data)
        file_menu.addAction(sample_action)

        file_menu.addSeparator()

        # Export plot action
        self.export_action = QAction("Export Plot...", self)
        self.export_action.setShortcut("Ctrl+E")
        self.export_action.triggered.connect(self.export_plot)
        self.export_action.setEnabled(False)  # Disabled until plot is displayed
        file_menu.addAction(self.export_action)

        file_menu.addSeparator()

        # Exit action
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu
        view_menu = menubar.addMenu("View")

        # Dark mode toggle action
        self.dark_mode_action = QAction("Dark Mode", self)
        self.dark_mode_action.setCheckable(True)
        self.dark_mode_action.setChecked(True)  # Default to dark mode
        self.dark_mode_action.setShortcut("Ctrl+D")
        self.dark_mode_action.triggered.connect(self.toggle_theme)
        view_menu.addAction(self.dark_mode_action)

        # Help menu
        help_menu = menubar.addMenu("Help")

        # About action
        about_action = QAction(f"About {APP_NAME}", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)

    def show_about_dialog(self):
        """Show the About dialog"""
        dialog = AboutDialog(self)
        dialog.exec()

    def create_toolbar(self):
        """Create toolbar with dark mode toggle in top right"""
        self.toolbar = QToolBar("Main Toolbar")
        self.toolbar.setMovable(False)  # Keep toolbar fixed
        self.addToolBar(Qt.TopToolBarArea, self.toolbar)

        # Add spacer to push dark mode toggle to the right
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.toolbar.addWidget(spacer)

        # Add dark mode checkbox
        self.dark_mode_checkbox = QCheckBox("🌙 Dark Mode")
        self.dark_mode_checkbox.setChecked(True)  # Default to dark mode
        self.dark_mode_checkbox.setToolTip(
            "Toggle between dark and light themes (Ctrl/Cmd+D)"
        )
        self.dark_mode_checkbox.toggled.connect(
            lambda checked: asyncio.ensure_future(
                self.on_dark_mode_checkbox_toggled(checked)
            )
        )
        self.toolbar.addWidget(self.dark_mode_checkbox)

    @qasync.asyncSlot()
    async def on_dark_mode_checkbox_toggled(self, checked):
        """Handle dark mode checkbox toggle"""
        # 'checked' is a simple bool - True=dark mode, False=light mode
        new_theme = Theme.DARK if checked else Theme.LIGHT

        # Only apply if theme actually changed
        if self.current_theme != new_theme:
            # Update menu action to match checkbox
            self.dark_mode_action.setChecked(checked)

            # Switch theme directly
            self.current_theme = new_theme
            self.apply_theme()

            # Regenerate plot if one is displayed
            if self.read_list.selectedItems():
                await self._regenerate_plot_async()
            else:
                self.statusBar().showMessage(
                    f"Theme changed to {self.current_theme.value} mode"
                )

    def apply_theme(self):
        """Apply the current theme using qt-material"""
        # Use qt-material themes with compact density
        # density_scale: -2 (more compact) to 2 (more spacious)
        extra = {
            "density_scale": "-2",  # Maximum compactness
        }

        if self.current_theme == Theme.DARK:
            apply_stylesheet(
                QApplication.instance(), theme="dark_amber.xml", extra=extra
            )
        else:
            apply_stylesheet(
                QApplication.instance(), theme="light_blue.xml", extra=extra
            )

        # Update welcome message in plot view if no plot is currently displayed
        if self.current_plot_html is None:
            self._show_welcome_message()

    def _show_welcome_message(self):
        """Display themed welcome message in plot view"""
        # Use simple dark/light colors for welcome message
        if self.current_theme == Theme.DARK:
            bg_color = "#2b2b2b"
            text_color = "#ffffff"
        else:
            bg_color = "#ffffff"
            text_color = "#000000"

        welcome_html = f"""
        <html>
        <body style='display:flex;align-items:center;justify-content:center;
                     height:100vh;margin:0;font-family:sans-serif;
                     background-color:{bg_color};color:{text_color};'>
            <div style='text-align:center;'>
                <h2>Squiggy</h2>
                <p>Select a POD5 file and read to display squiggle plot</p>
            </div>
        </body>
        </html>
        """
        self.plot_view.setHtml(welcome_html)

    async def _regenerate_plot_async(self):
        """Helper method to regenerate plot asynchronously"""
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.statusBar().showMessage("Applying theme to plot...")
        try:
            # Save current zoom/pan state before regenerating
            self.save_plot_ranges()
            # Small delay to allow JavaScript to execute
            await self.update_plot_with_delay()
            self.statusBar().showMessage(
                f"Theme changed to {self.current_theme.value} mode"
            )
        finally:
            QApplication.restoreOverrideCursor()

    @qasync.asyncSlot()
    async def toggle_theme(self):
        """Toggle between light and dark themes"""
        # Switch theme
        if self.current_theme == Theme.LIGHT:
            self.current_theme = Theme.DARK
            is_dark = True
        else:
            self.current_theme = Theme.LIGHT
            is_dark = False

        # Update UI controls without triggering signals
        self.dark_mode_action.setChecked(is_dark)
        self.dark_mode_checkbox.blockSignals(True)
        self.dark_mode_checkbox.setChecked(is_dark)
        self.dark_mode_checkbox.blockSignals(False)

        # Apply theme to Qt application
        self.apply_theme()

        # Regenerate plot if one is displayed
        if self.read_list.selectedItems():
            await self._regenerate_plot_async()
        else:
            self.statusBar().showMessage(
                f"Theme changed to {self.current_theme.value} mode"
            )

    @qasync.asyncSlot()
    async def open_sample_data(self):
        """Open the bundled sample POD5 file (async)"""
        try:
            sample_path = get_sample_data_path()
            if not sample_path.exists():
                QMessageBox.warning(
                    self,
                    "Sample Data Not Found",
                    "The sample data file could not be found.\n\n"
                    "This may happen if Squiggy was not installed properly.",
                )
                return

            self.pod5_file = sample_path
            self.file_label.setText(f"{sample_path.name} (sample)")
            await self.load_read_ids()
            self.statusBar().showMessage(
                f"Loaded {len(self.read_dict)} reads from sample data"
            )
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to load sample data:\n{str(e)}"
            )

    @qasync.asyncSlot()
    async def open_pod5_file(self):
        """Open and load a POD5 file (async)"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open POD5 File", "", "POD5 Files (*.pod5);;All Files (*)"
        )

        if file_path:
            try:
                self.pod5_file = Path(file_path)
                self.file_label.setText(self.pod5_file.name)
                await self.load_read_ids()
                self.statusBar().showMessage(f"Loaded {len(self.read_dict)} reads")
            except Exception as e:
                QMessageBox.critical(
                    self, "Error", f"Failed to load POD5 file:\n{str(e)}"
                )

    def _load_read_ids_blocking(self):
        """Blocking function to load read IDs from POD5 file"""
        read_dict = {}
        with pod5.Reader(self.pod5_file) as reader:
            for read in reader.reads():
                read_id = str(read.read_id)
                # Store just the read_id, not the read object (which becomes invalid)
                read_dict[read_id] = read_id
        return read_dict

    async def load_read_ids(self):
        """Load all read IDs from the POD5 file (async)"""
        self.read_dict.clear()
        self.read_list.clear()
        self.statusBar().showMessage("Loading reads...")

        try:
            # Run blocking I/O in thread pool
            read_dict = await asyncio.to_thread(self._load_read_ids_blocking)

            # Update UI on main thread
            self.read_dict = read_dict
            for read_id in read_dict.keys():
                self.read_list.addItem(read_id)

            # Update file information panel
            await self.update_file_info()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read POD5 file:\n{str(e)}")

    def _get_file_stats_blocking(self):
        """Blocking function to get file statistics"""
        sample_rates = set()
        total_samples = 0
        with pod5.Reader(self.pod5_file) as reader:
            for read in reader.reads():
                sample_rates.add(read.run_info.sample_rate)
                total_samples += len(read.signal)
        return sample_rates, total_samples

    async def update_file_info(self):
        """Update the file information panel with POD5 metadata (async)"""
        if not self.pod5_file:
            return

        try:
            # File name
            self.info_filename_label.setText(str(self.pod5_file.name))

            # File size
            file_size_bytes = self.pod5_file.stat().st_size
            file_size_mb = file_size_bytes / (1024 * 1024)
            if file_size_mb < 1:
                file_size_str = f"{file_size_bytes / 1024:.2f} KB"
            elif file_size_mb < 1024:
                file_size_str = f"{file_size_mb:.2f} MB"
            else:
                file_size_str = f"{file_size_mb / 1024:.2f} GB"
            self.info_filesize_label.setText(file_size_str)

            # Number of reads
            num_reads = len(self.read_dict)
            self.info_num_reads_label.setText(f"{num_reads:,}")

            # Sample rate and total samples (run in thread pool)
            sample_rates, total_samples = await asyncio.to_thread(
                self._get_file_stats_blocking
            )

            # Display sample rate (show range if multiple rates exist)
            if len(sample_rates) == 1:
                rate = list(sample_rates)[0]
                self.info_sample_rate_label.setText(f"{rate:,} Hz")
            else:
                min_rate = min(sample_rates)
                max_rate = max(sample_rates)
                self.info_sample_rate_label.setText(
                    f"{min_rate:,} - {max_rate:,} Hz (variable)"
                )

            # Total samples
            self.info_total_samples_label.setText(f"{total_samples:,}")

        except Exception:
            # If there's an error, just show error message
            self.info_filename_label.setText("Error reading file")
            self.info_filesize_label.setText("—")
            self.info_num_reads_label.setText("—")
            self.info_sample_rate_label.setText("—")
            self.info_total_samples_label.setText("—")

    @qasync.asyncSlot()
    async def open_bam_file(self):
        """Open and load a BAM file for base annotations (async with validation)"""
        # Check if POD5 file is loaded first
        if not self.pod5_file:
            QMessageBox.warning(
                self,
                "POD5 File Required",
                "Please load a POD5 file before loading a BAM file.\n\n"
                "The BAM file will be validated against the POD5 file.",
            )
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open BAM File", "", "BAM Files (*.bam);;All Files (*)"
        )

        if file_path:
            try:
                bam_path = Path(file_path)

                # Check for BAM index, create if missing
                bai_path = Path(str(bam_path) + ".bai")
                if not bai_path.exists():
                    # Ask user if they want to create index
                    reply = QMessageBox.question(
                        self,
                        "BAM Index Missing",
                        "The BAM file is not indexed (.bai file not found).\n\n"
                        "Would you like to create an index now?\n"
                        "This may take a few minutes for large files.",
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.Yes,
                    )

                    if reply == QMessageBox.Yes:
                        self.statusBar().showMessage("Indexing BAM file...")
                        try:
                            await asyncio.to_thread(index_bam_file, bam_path)
                            self.statusBar().showMessage(
                                "BAM index created successfully"
                            )
                        except Exception as e:
                            QMessageBox.critical(
                                self,
                                "Indexing Failed",
                                f"Failed to create BAM index:\n{str(e)}",
                            )
                            return
                    else:
                        return

                # Validate BAM file against POD5 file
                self.statusBar().showMessage("Validating BAM file against POD5...")
                validation_result = await asyncio.to_thread(
                    validate_bam_reads_in_pod5, bam_path, self.pod5_file
                )

                if not validation_result["is_valid"]:
                    error_msg = (
                        f"BAM validation failed!\n\n"
                        f"Found {validation_result['bam_read_count']} reads in BAM file.\n"
                        f"Found {validation_result['pod5_read_count']} reads in POD5 file.\n"
                        f"{validation_result['missing_count']} BAM reads are NOT in POD5 file.\n\n"
                        f"This indicates a serious mismatch between files.\n"
                        f"Please ensure the BAM file corresponds to the loaded POD5 file."
                    )
                    if validation_result["missing_reads"]:
                        # Show first few missing reads as examples
                        examples = list(validation_result["missing_reads"])[:5]
                        error_msg += "\n\nExample missing reads:\n" + "\n".join(
                            f"  - {r}" for r in examples
                        )

                    QMessageBox.critical(self, "BAM Validation Failed", error_msg)
                    return

                # Validation passed, load BAM file
                self.bam_file = bam_path
                self.bam_label.setText(bam_path.name)
                self.base_checkbox.setEnabled(True)
                self.base_checkbox.setChecked(True)  # Check by default
                self.mode_eventalign.setEnabled(True)
                self.mode_eventalign.setChecked(True)  # Switch to event-aligned mode
                self.plot_mode = PlotMode.EVENTALIGN  # Explicitly sync internal state
                self.dwell_time_checkbox.setEnabled(True)  # Enable dwell time option
                self.position_type_checkbox.setEnabled(
                    True
                )  # Enable reference positions

                # Enable browse references button if in region search mode
                if self.search_mode == "region":
                    self.browse_refs_button.setEnabled(True)

                self.statusBar().showMessage(
                    f"Loaded and validated BAM file: {bam_path.name} "
                    f"({validation_result['bam_read_count']} reads)"
                )

            except Exception as e:
                QMessageBox.critical(
                    self, "Error", f"Failed to load BAM file:\n{str(e)}"
                )

    @qasync.asyncSlot()
    async def toggle_base_annotations(self, state):
        """Toggle display of base annotations (async)"""
        # Use integer comparison since Qt.CheckState enum comparison may not work
        self.show_bases = state == 2  # Qt.CheckState.Checked = 2
        # Refresh current plot if one is displayed
        if self.read_list.selectedItems():
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.statusBar().showMessage("Regenerating plot...")
            try:
                # Save current zoom/pan state before regenerating
                self.save_plot_ranges()
                # Small delay to allow JavaScript to execute
                await self.update_plot_with_delay()
            finally:
                QApplication.restoreOverrideCursor()

    @qasync.asyncSlot()
    async def on_read_selection_changed(self):
        """Handle read selection changes"""
        await self.update_plot_from_selection()

    @qasync.asyncSlot()
    async def update_plot_from_selection(self):
        """Update plot based on current selection and plot mode"""
        selected_items = self.read_list.selectedItems()

        if not selected_items:
            return

        # Automatic fallback: if event-aligned mode is selected but no BAM is loaded,
        # switch to single read mode for a smoother user experience
        if self.plot_mode == PlotMode.EVENTALIGN and not self.bam_file:
            self.plot_mode = PlotMode.SINGLE
            self.mode_single.setChecked(True)

        # Get selected read IDs
        read_ids = [item.text() for item in selected_items]

        if self.plot_mode == PlotMode.SINGLE:
            # Single read mode: display first selected read
            await self.display_single_read(read_ids[0])
        elif self.plot_mode in (PlotMode.OVERLAY, PlotMode.STACKED):
            # Multi-read modes
            await self.display_multiple_reads(read_ids)
        elif self.plot_mode == PlotMode.EVENTALIGN:
            # Event-aligned mode requires BAM file
            if not self.bam_file:
                QMessageBox.warning(
                    self,
                    "BAM File Required",
                    "Event-aligned mode requires a BAM file with base call information.\n\n"
                    "Please load a BAM file first.",
                )
                return
            await self.display_eventaligned_reads(read_ids)
        else:
            QMessageBox.warning(
                self,
                "Unsupported Mode",
                f"Plot mode {self.plot_mode} not yet implemented",
            )

    def on_search_mode_changed(self, index):
        """Handle search mode change"""
        self.search_mode = self.search_mode_combo.itemData(index)

        # Update placeholder text and visibility based on mode
        if self.search_mode == "read_id":
            self.search_input.setPlaceholderText("Search read ID...")
            self.search_input.setToolTip("Filter reads by read ID (case-insensitive)")
            self.browse_refs_button.setVisible(False)
            self.revcomp_checkbox.setVisible(False)
            self.sequence_results_box.setVisible(False)
        elif self.search_mode == "region":
            self.search_input.setPlaceholderText("e.g., chr1:1000-2000 or chr1")
            self.search_input.setToolTip(
                "Search reads by genomic region (requires BAM file)\n"
                "Format: chr1, chr1:1000, or chr1:1000-2000"
            )
            self.browse_refs_button.setVisible(True)
            self.browse_refs_button.setEnabled(self.bam_file is not None)
            self.revcomp_checkbox.setVisible(False)
            self.sequence_results_box.setVisible(False)
        else:  # sequence
            self.search_input.setPlaceholderText("e.g., ATCGATCG")
            self.search_input.setToolTip(
                "Search for a DNA sequence in the reference (requires BAM file)\n"
                "Enter sequence in 5' to 3' direction"
            )
            self.browse_refs_button.setVisible(False)
            self.revcomp_checkbox.setVisible(True)
            # Results box will be shown when search is performed

        # Clear search
        self.search_input.clear()

    def on_search_text_changed(self):
        """Handle real-time search for read ID mode"""
        if self.search_mode == "read_id":
            # Real-time filtering for read ID mode
            self.filter_reads_by_id()

    @qasync.asyncSlot()
    async def execute_search(self):
        """Execute search based on current mode"""
        if self.search_mode == "read_id":
            self.filter_reads_by_id()
        elif self.search_mode == "region":
            await self.filter_reads_by_region()
        else:  # sequence mode
            await self.search_sequence()

    def filter_reads_by_id(self):
        """Filter the read list based on read ID search input"""
        search_text = self.search_input.text().lower()

        for i in range(self.read_list.count()):
            item = self.read_list.item(i)
            # Extract read ID from item text (may include alignment info)
            item_text = item.text()
            read_id = item_text.split("[")[
                0
            ].strip()  # Remove alignment info if present

            if search_text in read_id.lower():
                item.setHidden(False)
            else:
                item.setHidden(True)

    @qasync.asyncSlot()
    async def filter_reads_by_region(self):
        """Filter reads based on genomic region query (requires BAM file)"""
        region_str = self.search_input.text().strip()

        if not region_str:
            # Clear filter - show all reads
            for i in range(self.read_list.count()):
                item = self.read_list.item(i)
                item.setHidden(False)
            self.statusBar().showMessage("Ready")
            return

        # Check if BAM file is loaded
        if not self.bam_file:
            QMessageBox.warning(
                self,
                "BAM File Required",
                "Reference region search requires a BAM file.\n\n"
                "Please load a BAM file first.",
            )
            return

        # Parse region
        chromosome, start, end = parse_region(region_str)
        if chromosome is None:
            QMessageBox.warning(
                self,
                "Invalid Region",
                f"Could not parse region: {region_str}\n\n"
                "Expected format: chr1, chr1:1000, or chr1:1000-2000",
            )
            return

        # Query BAM file for reads in region
        self.statusBar().showMessage(f"Querying BAM for region {region_str}...")

        try:
            # Run query in background thread
            reads_in_region = await asyncio.to_thread(
                get_reads_in_region, self.bam_file, chromosome, start, end
            )

            # Store alignment info
            self.alignment_info = reads_in_region

            # Update read list to show only reads in region
            reads_found = set(reads_in_region.keys())
            visible_count = 0

            for i in range(self.read_list.count()):
                item = self.read_list.item(i)
                item_text = item.text()
                read_id = item_text.split("[")[0].strip()

                if read_id in reads_found:
                    item.setHidden(False)
                    visible_count += 1

                    # Update item text to show alignment info
                    aln_info = reads_in_region[read_id]
                    item.setText(
                        f"{read_id} [{aln_info['chromosome']}:"
                        f"{aln_info['start']}-{aln_info['end']} "
                        f"{aln_info['strand']}]"
                    )
                else:
                    item.setHidden(True)

            # Update status
            region_desc = f"{chromosome}"
            if start is not None and end is not None:
                region_desc += f":{start}-{end}"
            elif start is not None:
                region_desc += f":{start}"

            self.statusBar().showMessage(
                f"Found {visible_count} reads in region {region_desc}"
            )

        except ValueError as e:
            QMessageBox.critical(self, "Query Failed", str(e))
            self.statusBar().showMessage("Query failed")
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Unexpected error querying BAM file:\n{str(e)}"
            )
            self.statusBar().showMessage("Error")

    @qasync.asyncSlot()
    async def browse_references(self):
        """Open dialog to browse available references in BAM file"""
        if not self.bam_file:
            QMessageBox.warning(
                self,
                "No BAM File",
                "Please load a BAM file first to view available references.",
            )
            return

        try:
            # Get references in background thread
            self.statusBar().showMessage("Loading BAM references...")
            references = await asyncio.to_thread(get_bam_references, self.bam_file)

            # Open dialog
            dialog = ReferenceBrowserDialog(references, self)
            result = dialog.exec()

            if result == ReferenceBrowserDialog.Accepted and dialog.selected_reference:
                # User selected a reference - populate search field
                self.search_input.setText(dialog.selected_reference)
                # Automatically execute search
                await self.execute_search()

            self.statusBar().showMessage("Ready")

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to load BAM references:\n{str(e)}",
            )
            self.statusBar().showMessage("Error")

    @qasync.asyncSlot()
    async def search_sequence(self):
        """Search for a DNA sequence in the reference"""
        query_seq = self.search_input.text().strip().upper()

        if not query_seq:
            self.sequence_results_box.setVisible(False)
            self.statusBar().showMessage("Ready")
            return

        # Check if BAM file is loaded
        if not self.bam_file:
            QMessageBox.warning(
                self,
                "BAM File Required",
                "Sequence search requires a BAM file with reference alignment.\n\n"
                "Please load a BAM file first.",
            )
            return

        # Validate sequence (should be DNA: A, C, G, T, N)
        valid_bases = set("ACGTN")
        if not all(base in valid_bases for base in query_seq):
            QMessageBox.warning(
                self,
                "Invalid Sequence",
                f"Invalid DNA sequence: {query_seq}\n\n"
                "Only A, C, G, T, N characters are allowed.",
            )
            return

        # Check if reads are selected and in event-aligned mode
        if not self.read_list.selectedItems():
            QMessageBox.warning(
                self,
                "No Read Selected",
                "Please select a read first to search its reference sequence.",
            )
            return

        if self.plot_mode != PlotMode.EVENTALIGN:
            QMessageBox.warning(
                self,
                "Event-Aligned Mode Required",
                "Sequence search requires event-aligned mode.\n\n"
                "Please switch to event-aligned mode first.",
            )
            return

        # Get first selected read
        read_id = self.read_list.selectedItems()[0].text().split("[")[0].strip()

        self.statusBar().showMessage(f"Searching for sequence: {query_seq}...")

        try:
            # Search for sequence in background thread
            include_revcomp = self.revcomp_checkbox.isChecked()
            matches = await asyncio.to_thread(
                self._search_sequence_in_reference, read_id, query_seq, include_revcomp
            )

            # Display results
            self.sequence_results_list.clear()

            if not matches:
                self.sequence_results_list.addItem(
                    f"No matches found for '{query_seq}'"
                    + (" (or reverse complement)" if include_revcomp else "")
                )
                self.sequence_results_box.setVisible(True)
                self.sequence_results_box.toggle_button.setChecked(True)
                self.sequence_results_box.on_toggle()
                self.statusBar().showMessage("No matches found")
            else:
                for match in matches:
                    item_text = (
                        f"{match['strand']} strand: position {match['ref_start']}-{match['ref_end']} "
                        f"(base {match['base_start']}-{match['base_end']})"
                    )
                    self.sequence_results_list.addItem(item_text)
                    # Store match data for zoom functionality
                    self.sequence_results_list.item(
                        self.sequence_results_list.count() - 1
                    ).setData(Qt.UserRole, match)

                self.sequence_results_box.setVisible(True)
                self.sequence_results_box.toggle_button.setChecked(True)
                self.sequence_results_box.on_toggle()
                self.statusBar().showMessage(
                    f"Found {len(matches)} match(es) for '{query_seq}'"
                )

        except Exception as e:
            QMessageBox.critical(
                self, "Search Failed", f"Failed to search sequence:\n{str(e)}"
            )
            self.statusBar().showMessage("Search failed")

    def _search_sequence_in_reference(self, read_id, query_seq, include_revcomp=True):
        """Search for sequence in the reference (blocking function)"""
        from .utils import get_reference_sequence_for_read, reverse_complement

        # Get reference sequence for this read
        ref_seq, ref_start, aligned_read = get_reference_sequence_for_read(
            self.bam_file, read_id
        )

        if not ref_seq:
            raise ValueError(f"Could not extract reference sequence for read {read_id}")

        matches = []

        # Search forward strand
        start_pos = 0
        while True:
            pos = ref_seq.find(query_seq, start_pos)
            if pos == -1:
                break

            # Convert reference position to base position in alignment
            ref_pos = ref_start + pos
            ref_end = ref_pos + len(query_seq)

            # Map to base position (0-indexed in the aligned read)
            base_start = pos
            base_end = pos + len(query_seq)

            matches.append(
                {
                    "strand": "Forward",
                    "ref_start": ref_pos,
                    "ref_end": ref_end,
                    "base_start": base_start,
                    "base_end": base_end,
                    "sequence": query_seq,
                }
            )

            start_pos = pos + 1

        # Search reverse complement if requested
        if include_revcomp:
            revcomp_seq = reverse_complement(query_seq)
            if revcomp_seq != query_seq:  # Only search if different
                start_pos = 0
                while True:
                    pos = ref_seq.find(revcomp_seq, start_pos)
                    if pos == -1:
                        break

                    ref_pos = ref_start + pos
                    ref_end = ref_pos + len(revcomp_seq)
                    base_start = pos
                    base_end = pos + len(revcomp_seq)

                    matches.append(
                        {
                            "strand": "Reverse",
                            "ref_start": ref_pos,
                            "ref_end": ref_end,
                            "base_start": base_start,
                            "base_end": base_end,
                            "sequence": revcomp_seq,
                        }
                    )

                    start_pos = pos + 1

        return matches

    def zoom_to_sequence_match(self, item):
        """Zoom plot to show a sequence match"""
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

        self.plot_view.page().runJavaScript(js_code)
        self.statusBar().showMessage(
            f"Zoomed to {match['strand']} match at position {match['base_start']}-{match['base_end']}"
        )

    def _generate_plot_blocking(self, read_id):
        """Blocking function to generate bokeh plot HTML"""
        # Get signal data
        with pod5.Reader(self.pod5_file) as reader:
            # Find the specific read by iterating through all reads
            read = None
            for r in reader.reads():
                if str(r.read_id) == read_id:
                    read = r
                    break

            if read is None:
                raise ValueError(f"Read {read_id} not found in POD5 file")

            signal = read.signal
            sample_rate = read.run_info.sample_rate

        # Get basecall data if available and requested
        sequence = None
        seq_to_sig_map = None
        if self.show_bases and self.bam_file:
            sequence, seq_to_sig_map = get_basecall_data(self.bam_file, read_id)

        # Generate bokeh plot HTML
        html, figure = SquigglePlotter.plot_single_read(
            signal,
            read_id,
            sample_rate,
            sequence=sequence,
            seq_to_sig_map=seq_to_sig_map,
            normalization=self.normalization_method,
            downsample=self.downsample_factor,
            show_dwell_time=self.show_dwell_time,
            show_labels=self.show_bases,
            show_signal_points=self.show_signal_points,
            position_label_interval=self.position_label_interval,
            use_reference_positions=self.use_reference_positions,
            theme=self.current_theme,
        )

        return html, figure, signal, sequence

    async def display_single_read(self, read_id):
        """Display squiggle plot for a single read (async)"""
        self.statusBar().showMessage(f"Generating plot for {read_id}...")

        try:
            # Generate plot in thread pool
            html, figure, signal, sequence = await asyncio.to_thread(
                self._generate_plot_blocking, read_id
            )

            # Store HTML and figure for export
            self.current_plot_html = html
            self.current_plot_figure = figure
            self.export_action.setEnabled(True)

            # Display on main thread - use unique URL to force complete reload
            unique_url = QUrl(f"http://localhost/{time.time()}")
            self.plot_view.setHtml(html, baseUrl=unique_url)

            # Restore zoom/pan state if available
            self.restore_plot_ranges()

            status_msg = f"Displaying read: {read_id} ({len(signal)} samples)"
            if sequence:
                status_msg += f" - {len(sequence)} bases"
            self.statusBar().showMessage(status_msg)

        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to display squiggle:\n{str(e)}"
            )

    def _generate_multi_read_plot_blocking(self, read_ids):
        """Blocking function to generate multi-read bokeh plot HTML"""
        reads_data = []

        # Collect signal data for all reads
        with pod5.Reader(self.pod5_file) as reader:
            for r in reader.reads():
                read_id_str = str(r.read_id)
                if read_id_str in read_ids:
                    reads_data.append((read_id_str, r.signal, r.run_info.sample_rate))
                    if len(reads_data) == len(read_ids):
                        break

        if not reads_data:
            raise ValueError("No matching reads found in POD5 file")

        # Generate bokeh multi-read plot HTML
        html, figure = SquigglePlotter.plot_multiple_reads(
            reads_data,
            mode=self.plot_mode,
            normalization=self.normalization_method,
            downsample=self.downsample_factor,
            show_dwell_time=self.show_dwell_time,
            show_labels=self.show_bases,
            show_signal_points=self.show_signal_points,
            position_label_interval=self.position_label_interval,
            use_reference_positions=self.use_reference_positions,
            theme=self.current_theme,
        )

        return html, figure, reads_data

    async def display_multiple_reads(self, read_ids):
        """Display multiple reads in overlay or stacked mode (async)"""
        self.statusBar().showMessage(f"Generating plot for {len(read_ids)} reads...")

        try:
            # Generate plot in thread pool
            html, figure, reads_data = await asyncio.to_thread(
                self._generate_multi_read_plot_blocking, read_ids
            )

            # Store HTML and figure for export
            self.current_plot_html = html
            self.current_plot_figure = figure
            self.export_action.setEnabled(True)

            # Display on main thread - use unique URL to force complete reload
            unique_url = QUrl(f"http://localhost/{time.time()}")
            self.plot_view.setHtml(html, baseUrl=unique_url)

            # Restore zoom/pan state if available
            self.restore_plot_ranges()

            mode_name = "overlaid" if self.plot_mode == PlotMode.OVERLAY else "stacked"
            self.statusBar().showMessage(
                f"Displaying {len(reads_data)} reads ({mode_name}, {self.normalization_method.value} normalization)"
            )

        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to display multi-read plot:\n{str(e)}"
            )

    def _generate_eventalign_plot_blocking(self, read_ids):
        """Blocking function to generate event-aligned bokeh plot HTML"""
        from .alignment import extract_alignment_from_bam

        reads_data = []
        aligned_reads = []

        # Collect signal data and alignment info for all reads
        with pod5.Reader(self.pod5_file) as reader:
            for r in reader.reads():
                read_id_str = str(r.read_id)
                if read_id_str in read_ids:
                    # Get signal data
                    reads_data.append((read_id_str, r.signal, r.run_info.sample_rate))

                    # Get alignment info from BAM
                    aligned_read = extract_alignment_from_bam(
                        self.bam_file, read_id_str
                    )
                    if aligned_read is None:
                        raise ValueError(
                            f"No alignment found for read {read_id_str} in BAM file"
                        )
                    aligned_reads.append(aligned_read)

                    if len(reads_data) == len(read_ids):
                        break

        if not reads_data:
            raise ValueError("No matching reads found in POD5 file")

        # Generate bokeh event-aligned plot HTML
        html, figure = SquigglePlotter.plot_multiple_reads(
            reads_data,
            mode=self.plot_mode,
            normalization=self.normalization_method,
            aligned_reads=aligned_reads,
            downsample=self.downsample_factor,
            show_dwell_time=self.show_dwell_time,
            show_labels=self.show_bases,
            show_signal_points=self.show_signal_points,
            position_label_interval=self.position_label_interval,
            use_reference_positions=self.use_reference_positions,
            theme=self.current_theme,
        )

        return html, figure, reads_data, aligned_reads

    async def display_eventaligned_reads(self, read_ids):
        """Display multiple reads in event-aligned mode (async)"""
        self.statusBar().showMessage(
            f"Generating event-aligned plot for {len(read_ids)} reads..."
        )

        try:
            # Generate plot in thread pool
            html, figure, reads_data, aligned_reads = await asyncio.to_thread(
                self._generate_eventalign_plot_blocking, read_ids
            )

            # Store HTML and figure for export
            self.current_plot_html = html
            self.current_plot_figure = figure
            self.export_action.setEnabled(True)

            # Display on main thread - use unique URL to force complete reload
            unique_url = QUrl(f"http://localhost/{time.time()}")
            self.plot_view.setHtml(html, baseUrl=unique_url)

            # Restore zoom/pan state if available
            self.restore_plot_ranges()

            # Build status message
            total_bases = sum(len(ar.bases) for ar in aligned_reads)
            self.statusBar().showMessage(
                f"Displaying {len(reads_data)} reads (event-aligned, {total_bases} bases, {self.normalization_method.value} normalization)"
            )

        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to display event-aligned plot:\n{str(e)}"
            )

    def _get_current_view_ranges(self):
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

        self.plot_view.page().runJavaScript(js_code, callback)

        # Process events to wait for JavaScript execution
        # This is a simple approach - for production might need QEventLoop
        from PySide6.QtWidgets import QApplication

        QApplication.processEvents()

        if result[0]:
            try:
                import json

                ranges = json.loads(result[0])
                x_range = (ranges["x_start"], ranges["x_end"])
                y_range = (ranges["y_start"], ranges["y_end"])
                return x_range, y_range
            except Exception:
                pass

        return None, None

    def _export_html(self, file_path):
        """Export plot as HTML file

        Args:
            file_path: Path to save HTML file
        """
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self.current_plot_html)

    def _export_png(self, file_path, width, height, x_range=None, y_range=None):
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
        original_width = self.current_plot_figure.width
        original_height = self.current_plot_figure.height
        original_sizing_mode = self.current_plot_figure.sizing_mode
        original_x_range = None
        original_y_range = None

        try:
            # Temporarily modify figure dimensions and sizing mode
            # Setting sizing_mode to None allows explicit width/height to work properly
            self.current_plot_figure.sizing_mode = None
            self.current_plot_figure.width = width
            self.current_plot_figure.height = height

            # Apply custom ranges if provided
            if x_range is not None:
                original_x_range = (
                    self.current_plot_figure.x_range.start,
                    self.current_plot_figure.x_range.end,
                )
                self.current_plot_figure.x_range.start = x_range[0]
                self.current_plot_figure.x_range.end = x_range[1]

            if y_range is not None:
                original_y_range = (
                    self.current_plot_figure.y_range.start,
                    self.current_plot_figure.y_range.end,
                )
                self.current_plot_figure.y_range.start = y_range[0]
                self.current_plot_figure.y_range.end = y_range[1]

            # Export to PNG
            export_png(self.current_plot_figure, filename=str(file_path))
        finally:
            # Restore original dimensions and sizing mode
            self.current_plot_figure.sizing_mode = original_sizing_mode
            self.current_plot_figure.width = original_width
            self.current_plot_figure.height = original_height

            # Restore original ranges
            if original_x_range is not None:
                self.current_plot_figure.x_range.start = original_x_range[0]
                self.current_plot_figure.x_range.end = original_x_range[1]

            if original_y_range is not None:
                self.current_plot_figure.y_range.start = original_y_range[0]
                self.current_plot_figure.y_range.end = original_y_range[1]

    def _export_svg(self, file_path, width, height, x_range=None, y_range=None):
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
        original_width = self.current_plot_figure.width
        original_height = self.current_plot_figure.height
        original_backend = self.current_plot_figure.output_backend
        original_sizing_mode = self.current_plot_figure.sizing_mode
        original_x_range = None
        original_y_range = None

        try:
            # Temporarily modify figure dimensions, backend, and sizing mode
            # Setting sizing_mode to None allows explicit width/height to work properly
            self.current_plot_figure.sizing_mode = None
            self.current_plot_figure.width = width
            self.current_plot_figure.height = height
            self.current_plot_figure.output_backend = "svg"

            # Apply custom ranges if provided
            if x_range is not None:
                original_x_range = (
                    self.current_plot_figure.x_range.start,
                    self.current_plot_figure.x_range.end,
                )
                self.current_plot_figure.x_range.start = x_range[0]
                self.current_plot_figure.x_range.end = x_range[1]

            if y_range is not None:
                original_y_range = (
                    self.current_plot_figure.y_range.start,
                    self.current_plot_figure.y_range.end,
                )
                self.current_plot_figure.y_range.start = y_range[0]
                self.current_plot_figure.y_range.end = y_range[1]

            # Export to SVG
            export_svgs(self.current_plot_figure, filename=str(file_path))
        finally:
            # Restore original dimensions, backend, and sizing mode
            self.current_plot_figure.sizing_mode = original_sizing_mode
            self.current_plot_figure.width = original_width
            self.current_plot_figure.height = original_height
            self.current_plot_figure.output_backend = original_backend

            # Restore original ranges
            if original_x_range is not None:
                self.current_plot_figure.x_range.start = original_x_range[0]
                self.current_plot_figure.x_range.end = original_x_range[1]

            if original_y_range is not None:
                self.current_plot_figure.y_range.start = original_y_range[0]
                self.current_plot_figure.y_range.end = original_y_range[1]

    def export_plot(self):
        """Export the current plot with format and dimension options"""
        if self.current_plot_html is None or self.current_plot_figure is None:
            QMessageBox.warning(
                self,
                "No Plot",
                "No plot to export. Please display a plot first.",
            )
            return

        # Show export dialog
        dialog = ExportDialog(self)
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
            x_range, y_range = self._get_current_view_ranges()
            if x_range is None or y_range is None:
                QMessageBox.warning(
                    self,
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
            self,
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
            self.statusBar().showMessage(
                f"Exporting plot as {export_format.upper()}..."
            )

            try:
                # Export based on format
                if export_format == "html":
                    self._export_html(file_path)
                    view_info = " (current view)" if use_current_view else ""
                    message = (
                        f"Interactive plot successfully exported to:\n{file_path}\n\n"
                        f"Open the file in a web browser to view the interactive plot."
                    )
                elif export_format == "png":
                    self._export_png(file_path, width, height, x_range, y_range)
                    view_info = " (current view)" if use_current_view else ""
                    message = (
                        f"PNG image successfully exported to:\n{file_path}\n\n"
                        f"Dimensions: {width} × {height} pixels{view_info}"
                    )
                else:  # svg
                    self._export_svg(file_path, width, height, x_range, y_range)
                    view_info = " (current view)" if use_current_view else ""
                    message = (
                        f"SVG image successfully exported to:\n{file_path}\n\n"
                        f"Dimensions: {width} × {height} pixels{view_info}\n"
                        f"Edit in vector graphics software like Inkscape or Adobe Illustrator."
                    )

                self.statusBar().showMessage(f"Plot exported to {file_path.name}")
                QMessageBox.information(self, "Export Successful", message)

            finally:
                QApplication.restoreOverrideCursor()

        except ImportError as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(
                self,
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
                self, "Export Failed", f"Failed to export plot:\n{str(e)}"
            )
