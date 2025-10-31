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
    QCheckBox,
    QFileDialog,
    QFrame,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
)
from qt_material import apply_stylesheet

from .constants import (
    APP_DESCRIPTION,
    APP_NAME,
    DEFAULT_AGGREGATE_SAMPLE_SIZE,
    DEFAULT_WINDOW_HEIGHT,
    DEFAULT_WINDOW_WIDTH,
    PLOT_MIN_HEIGHT,
    PLOT_MIN_WIDTH,
    NormalizationMethod,
    PlotMode,
    Theme,
)
from .dialogs import AboutDialog, ReferenceBrowserDialog
from .managers import ExportManager, ZoomManager
from .plotting import SquigglePlotter
from .search import SearchManager
from .ui import (
    AdvancedOptionsPanel,
    FileInfoPanel,
    ModificationsPanel,
    PlotOptionsPanel,
    SearchPanel,
)
from .utils import (
    calculate_aggregate_signal,
    calculate_base_pileup,
    calculate_quality_by_position,
    extract_reads_for_reference,
    get_bam_references,
    get_basecall_data,
    get_read_to_reference_mapping,
    get_sample_bam_path,
    get_sample_data_path,
    index_bam_file,
    validate_bam_reads_in_pod5,
    writable_working_directory,
)
from .widgets import CollapsibleBox, ReadTreeWidget


class SquiggleViewer(QMainWindow):
    """Main application window for nanopore squiggle visualization"""

    def __init__(self, window_width=None, window_height=None):
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
        self.show_signal_points = False  # Show individual signal points on plot
        self.position_label_interval = 10  # Show position labels every N bases
        self.use_reference_positions = (
            False  # Use reference positions vs sequence positions
        )
        # Modification visualization settings
        self.show_modification_overlay = True  # Show modification overlays by default
        self.modification_overlay_opacity = (
            0.6  # Default opacity for modification overlays
        )
        self.modification_type_filter = "all"  # Filter for modification types
        self.modification_threshold_enabled = True  # Threshold always enabled
        self.modification_threshold = 0.5  # Default tau threshold
        self.modification_classification_scope = "position"  # "position" or "any"
        self.current_theme = Theme.DARK  # Default to dark theme
        self.selected_reference = None  # Selected reference for aggregate mode
        self.max_aggregate_reads = (
            DEFAULT_AGGREGATE_SAMPLE_SIZE  # Max reads for aggregate
        )

        # Window dimensions (can be overridden from CLI)
        self.window_width = window_width or DEFAULT_WINDOW_WIDTH
        self.window_height = window_height or DEFAULT_WINDOW_HEIGHT

        # Task tracking for debouncing plot updates
        self._update_plot_task = None

        # Initialize managers
        self.search_manager = SearchManager(self)
        self.export_manager = ExportManager(self)
        self.zoom_manager = ZoomManager(self)

        self.init_ui()
        self.apply_theme()  # Apply initial theme
        self.showMaximized()  # Maximize window on startup

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle(f"{APP_NAME} - {APP_DESCRIPTION}")
        self.setGeometry(100, 100, self.window_width, self.window_height)

        # Create menu bar
        self.create_menu_bar()

        # Create toolbar with dark mode toggle
        self.create_toolbar()

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Create splitter for left panel, plot area, and read list
        self.splitter = QSplitter(Qt.Horizontal)

        # Left panel - file browser and plot options (scrollable)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setFrameShape(QFrame.NoFrame)

        left_panel = QWidget()
        left_panel_layout = QVBoxLayout(left_panel)
        left_panel_layout.setContentsMargins(5, 5, 5, 5)

        # Create UI component panels
        self.plot_options_panel = PlotOptionsPanel()
        self.plot_options_panel.plot_mode_changed.connect(self.set_plot_mode)
        self.plot_options_panel.normalization_changed.connect(
            self.set_normalization_method
        )
        self.plot_options_panel.base_annotations_toggled.connect(
            self.toggle_base_annotations
        )
        self.plot_options_panel.signal_points_toggled.connect(self.toggle_signal_points)
        left_panel_layout.addWidget(self.plot_options_panel)

        self.advanced_options_panel = AdvancedOptionsPanel()
        self.advanced_options_panel.downsample_changed.connect(
            self.set_downsample_factor
        )
        self.advanced_options_panel.dwell_time_toggled.connect(self.toggle_dwell_time)
        self.advanced_options_panel.position_interval_changed.connect(
            self.on_position_interval_changed
        )
        self.advanced_options_panel.position_type_toggled.connect(
            self.toggle_position_type
        )
        left_panel_layout.addWidget(self.advanced_options_panel)

        # Modifications panel (initially hidden until modifications are detected)
        self.modifications_panel = ModificationsPanel()
        self.modifications_panel.modification_overlay_toggled.connect(
            self.toggle_modification_overlay
        )
        self.modifications_panel.overlay_opacity_changed.connect(
            self.set_modification_overlay_opacity
        )
        self.modifications_panel.mod_type_filter_changed.connect(
            self.set_modification_type_filter
        )
        self.modifications_panel.threshold_changed.connect(
            self.on_modification_threshold_changed
        )
        self.modifications_panel.classification_scope_changed.connect(
            self.on_modification_scope_changed
        )
        self.modifications_panel.hide()  # Initially hidden
        left_panel_layout.addWidget(self.modifications_panel)

        self.file_info_panel = FileInfoPanel()
        left_panel_layout.addWidget(self.file_info_panel)

        # Add stretch to push everything to the top
        left_panel_layout.addStretch()

        # Add left panel to scroll area, then add scroll area to splitter
        left_scroll.setWidget(left_panel)
        self.splitter.addWidget(left_scroll)

        # Plot display area (using QWebEngineView for interactive bokeh plots)
        self.plot_view = QWebEngineView()
        self.plot_view.setMinimumSize(PLOT_MIN_WIDTH, PLOT_MIN_HEIGHT)
        # Will be set by apply_theme() which is called after init_ui()
        self.splitter.addWidget(self.plot_view)

        # Right panel - Container for read list and reference list
        right_panel = QWidget()
        right_panel_layout = QVBoxLayout(right_panel)
        right_panel_layout.setContentsMargins(0, 0, 0, 0)

        # Read tree widget with multi-selection enabled (shown in non-aggregate modes)
        # Now displays reads grouped by reference when BAM is loaded
        self.read_list = ReadTreeWidget()
        self.read_list.itemSelectionChanged.connect(self.on_read_selection_changed)
        right_panel_layout.addWidget(self.read_list)

        # Reference list widget (shown in aggregate mode)
        self.reference_list = QListWidget()
        self.reference_list.setSelectionMode(QListWidget.SingleSelection)
        self.reference_list.itemSelectionChanged.connect(
            self.on_reference_selection_changed
        )
        self.reference_list.setVisible(False)  # Hidden by default
        right_panel_layout.addWidget(self.reference_list)

        self.splitter.addWidget(right_panel)

        # Set initial splitter sizes (window is maximized, so plenty of space)
        # Left panel: 300px (for controls)
        # Plot: rest of space
        # Right panel: 0 (hidden initially)
        self.splitter.setSizes([300, 1000, 0])

        # Hide right panel initially - will be shown when data is loaded
        right_panel.hide()
        self.right_panel = right_panel  # Store reference for later

        main_layout.addWidget(self.splitter, 1)

        # Bottom search panel
        self.search_panel = SearchPanel()
        self.search_panel.search_mode_changed.connect(self.on_search_mode_changed)
        self.search_panel.search_requested.connect(self.execute_search)
        self.search_panel.reference_browse_requested.connect(self.browse_references)
        # Connect text changed for real-time Read ID filtering
        self.search_panel.search_input.textChanged.connect(self.on_search_text_changed)
        main_layout.addWidget(self.search_panel)

        # Sequence search results area (collapsible, hidden by default)
        self.sequence_results_box = CollapsibleBox("Search Results")
        self.sequence_results_list = QListWidget()
        self.sequence_results_list.itemClicked.connect(self.zoom_to_sequence_match)
        results_layout = QVBoxLayout()
        results_layout.addWidget(self.sequence_results_list)
        self.sequence_results_box.set_content_layout(results_layout)
        self.sequence_results_box.setVisible(False)  # Hidden by default
        main_layout.addWidget(self.sequence_results_box)

        # Status bar with permanent file labels on the right
        self.statusBar().showMessage("Ready")

        # Create permanent labels for file status (right side of status bar)
        self.pod5_status_label = QLabel("POD5: None")
        self.pod5_status_label.setStyleSheet("margin-right: 10px;")
        self.statusBar().addPermanentWidget(self.pod5_status_label)

        self.bam_status_label = QLabel("BAM: None")
        self.bam_status_label.setStyleSheet("margin-right: 10px;")
        self.statusBar().addPermanentWidget(self.bam_status_label)

    def save_plot_ranges(self):
        """Extract and save current plot ranges via JavaScript"""
        self.zoom_manager.save_plot_ranges()

    def on_ranges_extracted(self, result):
        """Callback when ranges are extracted from JavaScript"""
        self.zoom_manager.on_ranges_extracted(result)

    @qasync.asyncSlot()
    async def set_downsample_factor(self, value):
        """Set the downsampling factor and refresh plot"""
        self.downsample_factor = value
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
        self.zoom_manager.restore_plot_ranges()

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

    # ==============================================================================
    # Modification Handlers
    # ==============================================================================

    @qasync.asyncSlot(bool)
    async def toggle_modification_overlay(self, state):
        """Toggle modification overlay display"""
        self.show_modification_overlay = state
        # Refresh plot if reads are selected
        if self.read_list.selectedItems():
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.statusBar().showMessage("Regenerating plot...")
            try:
                await self.update_plot_with_delay()
            finally:
                QApplication.restoreOverrideCursor()

    @qasync.asyncSlot(float)
    async def set_modification_overlay_opacity(self, opacity):
        """Set modification overlay opacity"""
        self.modification_overlay_opacity = opacity
        # Refresh plot if reads are selected
        if self.read_list.selectedItems():
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.statusBar().showMessage("Regenerating plot...")
            try:
                await self.update_plot_with_delay()
            finally:
                QApplication.restoreOverrideCursor()

    @qasync.asyncSlot(str)
    async def set_modification_type_filter(self, mod_type):
        """Set modification type filter"""
        self.modification_type_filter = mod_type
        # Refresh plot if reads are selected
        if self.read_list.selectedItems():
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.statusBar().showMessage("Regenerating plot...")
            try:
                await self.update_plot_with_delay()
            finally:
                QApplication.restoreOverrideCursor()

    @qasync.asyncSlot(float)
    async def on_modification_threshold_changed(self, tau):
        """Handle modification threshold value change"""
        self.modification_threshold = tau
        # Refresh plot if reads are selected
        if self.read_list.selectedItems():
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.statusBar().showMessage("Regenerating plot...")
            try:
                await self.update_plot_with_delay()
            finally:
                QApplication.restoreOverrideCursor()

    @qasync.asyncSlot(str)
    async def on_modification_scope_changed(self, scope):
        """Handle modification classification scope change"""
        self.modification_classification_scope = scope
        # Refresh plot if reads are selected
        if self.read_list.selectedItems():
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.statusBar().showMessage("Regenerating plot...")
            try:
                await self.update_plot_with_delay()
            finally:
                QApplication.restoreOverrideCursor()

    def set_plot_mode(self, mode):
        """Set the plot mode and refresh display"""
        # Validate that BAM file is loaded for modes that require it
        if mode in (PlotMode.OVERLAY, PlotMode.STACKED) and not self.bam_file:
            QMessageBox.warning(
                self,
                "BAM File Required",
                f"{mode.value} mode requires a BAM file for reference grouping.\n\n"
                "Please load a BAM file first.",
            )
            # Revert to previous mode (or single if no previous mode)
            if hasattr(self, "plot_options_panel"):
                self.plot_options_panel.set_plot_mode(PlotMode.SINGLE)
            return

        self.plot_mode = mode

        # Enable dwell time checkbox only in EVENTALIGN mode with BAM file
        # Only update checkbox if it exists (may not exist during initialization)
        if hasattr(self, "advanced_options_panel"):
            self.advanced_options_panel.set_dwell_time_enabled(
                mode == PlotMode.EVENTALIGN and self.bam_file is not None
            )

        # Toggle between read list and reference list based on mode
        if hasattr(self, "read_list") and hasattr(self, "reference_list"):
            if mode == PlotMode.AGGREGATE:
                # Show reference list, hide read list
                self.read_list.setVisible(False)
                self.reference_list.setVisible(True)
                # Load references if BAM file is available
                if self.bam_file:
                    self.load_references()
            else:
                # Show read list, hide reference list
                self.read_list.setVisible(True)
                self.reference_list.setVisible(False)

        # Refresh plot if reads are selected or if aggregate mode with selected reference
        if mode == PlotMode.AGGREGATE and self.selected_reference:
            # In aggregate mode, display plot for selected reference
            asyncio.ensure_future(self.display_aggregate())
        elif hasattr(self, "read_list") and self.read_list.selectedItems():
            # Save current zoom/pan state before regenerating
            self.save_plot_ranges()
            # Small delay to allow JavaScript to execute
            asyncio.ensure_future(self.update_plot_with_delay())

    @qasync.asyncSlot()
    async def set_normalization_method(self, method):
        """Set the normalization method and refresh display"""
        self.normalization_method = method
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

        # Open POD5 file action
        open_action = QAction("Open POD5 File...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_pod5_file)
        file_menu.addAction(open_action)

        # Open BAM file action
        open_bam_action = QAction("Open BAM File...", self)
        open_bam_action.setShortcut("Ctrl+B")
        open_bam_action.triggered.connect(self.open_bam_file)
        file_menu.addAction(open_bam_action)

        file_menu.addSeparator()

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
            elif self.plot_mode == PlotMode.AGGREGATE and self.selected_reference:
                # Regenerate aggregate plot if in aggregate mode with selected reference
                await self.display_aggregate()
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
        # Check aggregate mode first to avoid triggering read selection warning
        if self.plot_mode == PlotMode.AGGREGATE and self.selected_reference:
            # Regenerate aggregate plot if in aggregate mode with selected reference
            await self.display_aggregate()
        elif self.read_list.selectedItems():
            await self._regenerate_plot_async()
        else:
            self.statusBar().showMessage(
                f"Theme changed to {self.current_theme.value} mode"
            )

    @qasync.asyncSlot()
    async def open_sample_data(self):
        """Open the bundled sample POD5 file and BAM file (async)"""
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
            self.pod5_status_label.setText(f"POD5: {sample_path.name} (sample)")
            await self.load_read_ids()

            # Also load the sample BAM file if available
            sample_bam = get_sample_bam_path()
            if sample_bam and sample_bam.exists():
                self.bam_file = sample_bam
                self.bam_status_label.setText(f"BAM: {sample_bam.name} (sample)")

                # Enable BAM-dependent features
                self.plot_options_panel.set_bam_controls_enabled(True)
                self.plot_options_panel.set_plot_mode(PlotMode.EVENTALIGN)
                self.plot_mode = PlotMode.EVENTALIGN  # Explicitly sync internal state
                self.advanced_options_panel.set_dwell_time_enabled(True)
                self.advanced_options_panel.set_position_type_enabled(True)

                # Detect and show modifications if present
                await self._detect_and_show_modifications(sample_bam)

                # Reload read IDs to populate tree with reference grouping
                await self.load_read_ids()

                self.statusBar().showMessage(
                    f"Loaded {len(self.read_dict)} reads from sample data with BAM file (grouped by reference)"
                )
            else:
                self.statusBar().showMessage(
                    f"Loaded {len(self.read_dict)} reads from sample data"
                )
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to load sample data:\n{str(e)}"
            )

    async def _detect_and_show_modifications(self, bam_path):
        """Detect modifications in BAM file and show ModificationsPanel if present

        Args:
            bam_path: Path to BAM file
        """
        import pysam

        from squiggy.alignment import extract_alignment_from_bam
        from squiggy.modifications import detect_modification_provenance

        # Always get provenance info (even if unknown)
        provenance = await asyncio.to_thread(detect_modification_provenance, bam_path)

        # Scan reads to detect modification types (regardless of provenance)
        def scan_mods():
            mods = set()
            with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
                for i, read in enumerate(bam.fetch(until_eof=True)):
                    if i >= 100:  # Sample first 100 reads
                        break
                    aligned_read = extract_alignment_from_bam(bam_path, read.query_name)
                    if aligned_read and aligned_read.modifications:
                        for mod in aligned_read.modifications:
                            # Store (canonical_base, mod_code) tuple
                            # Get canonical base from position in sequence
                            if mod.position < len(aligned_read.sequence):
                                canonical_base = aligned_read.sequence[mod.position]
                                mods.add((canonical_base, mod.mod_code))
            return mods

        detected_mods = await asyncio.to_thread(scan_mods)

        if detected_mods:
            # BAM file has modifications - populate and show the ModificationsPanel
            self.modifications_panel.set_provenance(provenance)
            self.modifications_panel.set_detected_modifications(detected_mods)
            self.modifications_panel.show()
            self.modifications_panel.updateGeometry()

            self.statusBar().showMessage(
                f"Detected {len(detected_mods)} modification type(s) "
                f"(basecaller: {provenance.get('basecaller', 'unknown')})",
                5000,
            )
        else:
            # No modifications detected - keep panel hidden
            self.modifications_panel.hide()

    @qasync.asyncSlot()
    async def open_pod5_file(self):
        """Open and load a POD5 file (async)"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open POD5 File", "", "POD5 Files (*.pod5);;All Files (*)"
        )

        if file_path:
            try:
                # Convert to absolute path to avoid issues with CWD changes
                self.pod5_file = Path(file_path).resolve()
                self.pod5_status_label.setText(f"POD5: {self.pod5_file.name}")
                await self.load_read_ids()
                self.statusBar().showMessage(f"Loaded {len(self.read_dict)} reads")
            except Exception as e:
                QMessageBox.critical(
                    self, "Error", f"Failed to load POD5 file:\n{str(e)}"
                )

    def _adjust_splitter_for_tree_content(self, reads_by_reference):
        """Show the right panel and size it based on reference names

        Args:
            reads_by_reference: Dict mapping reference name -> list of read IDs
        """
        if not reads_by_reference:
            return

        # Show the right panel first
        self.right_panel.show()

        # Find the longest reference name
        longest_ref_name = max(reads_by_reference.keys(), key=len)

        # Calculate width needed for the longest reference name
        # Use the tree widget's font metrics to measure text width
        font_metrics = self.read_list.fontMetrics()
        text_width = font_metrics.horizontalAdvance(longest_ref_name)

        # Add padding for tree widget decorations (expand arrow, margins, scrollbar)
        padding = 100
        needed_width = max(text_width + padding, 250)  # At least 250px

        # Cap at 400px max
        right_width = min(needed_width, 400)

        # Use QTimer to set sizes after the panel is visible

        def do_resize():
            # Get current sizes
            current_sizes = self.splitter.sizes()
            left_width = current_sizes[0]
            total_width = sum(current_sizes)

            # Calculate plot width as remainder
            plot_width = total_width - left_width - right_width

            # Set the new sizes
            self.splitter.setSizes([left_width, plot_width, right_width])

        QTimer.singleShot(50, do_resize)

    def _load_read_ids_blocking(self):
        """Blocking function to load read IDs from POD5 file"""
        read_dict = {}
        with writable_working_directory():
            with pod5.Reader(self.pod5_file) as reader:
                for read in reader.reads():
                    read_id = str(read.read_id)
                    # Store just the read_id, not the read object (which becomes invalid)
                    read_dict[read_id] = read_id
        return read_dict

    async def load_read_ids(self):
        """Load all read IDs from the POD5 file (async) and group by reference if BAM loaded"""
        self.read_dict.clear()
        self.read_list.clear()
        self.statusBar().showMessage("Loading reads...")

        try:
            # Run blocking I/O in thread pool
            read_dict = await asyncio.to_thread(self._load_read_ids_blocking)

            # Update UI on main thread
            self.read_dict = read_dict

            # If BAM file is loaded, group reads by reference
            if self.bam_file:
                # Get read-to-reference mapping from BAM
                read_to_ref = await asyncio.to_thread(
                    get_read_to_reference_mapping, self.bam_file, list(read_dict.keys())
                )

                # Group reads by reference
                reads_by_reference = {}
                unmapped_reads = []

                for read_id in read_dict.keys():
                    ref_name = read_to_ref.get(read_id)
                    if ref_name:
                        if ref_name not in reads_by_reference:
                            reads_by_reference[ref_name] = []
                        reads_by_reference[ref_name].append(read_id)
                    else:
                        unmapped_reads.append(read_id)

                # Add unmapped reads as a separate group if any exist
                if unmapped_reads:
                    reads_by_reference["Unmapped"] = unmapped_reads

                # Populate tree widget with grouped reads
                self.read_list.populate_with_reads(reads_by_reference)

                # Adjust splitter to fit reference names
                self._adjust_splitter_for_tree_content(reads_by_reference)

            else:
                # No BAM file: Create a single "All Reads" group
                all_reads = list(read_dict.keys())
                self.read_list.populate_with_reads({"All Reads": all_reads})

                # Show right panel with default sizing
                self.right_panel.show()

            # Update file information panel
            await self.update_file_info()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read POD5 file:\n{str(e)}")

    def _get_file_stats_blocking(self):
        """Blocking function to get file statistics"""
        sample_rates = set()
        total_samples = 0
        with writable_working_directory():
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
            filename = str(self.pod5_file.name)

            # File size
            file_size_bytes = self.pod5_file.stat().st_size
            file_size_mb = file_size_bytes / (1024 * 1024)
            if file_size_mb < 1:
                filesize = f"{file_size_bytes / 1024:.2f} KB"
            elif file_size_mb < 1024:
                filesize = f"{file_size_mb:.2f} MB"
            else:
                filesize = f"{file_size_mb / 1024:.2f} GB"

            # Number of reads
            num_reads = f"{len(self.read_dict):,}"

            # Sample rate and total samples (run in thread pool)
            sample_rates, total_samples = await asyncio.to_thread(
                self._get_file_stats_blocking
            )

            # Display sample rate (show range if multiple rates exist)
            if len(sample_rates) == 1:
                rate = list(sample_rates)[0]
                sample_rate = f"{rate:,} Hz"
            else:
                min_rate = min(sample_rates)
                max_rate = max(sample_rates)
                sample_rate = f"{min_rate:,} - {max_rate:,} Hz (variable)"

            # Total samples
            total_samples_str = f"{total_samples:,}"

            # Update panel
            self.file_info_panel.update_info(
                filename, filesize, num_reads, sample_rate, total_samples_str
            )

        except Exception:
            # If there's an error, just show error message
            self.file_info_panel.update_info("Error reading file", "—", "—", "—", "—")

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
                # Convert to absolute path to avoid issues with CWD changes
                bam_path = Path(file_path).resolve()

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
                self.bam_status_label.setText(f"BAM: {bam_path.name}")
                self.plot_options_panel.set_bam_controls_enabled(True)
                self.plot_options_panel.set_plot_mode(PlotMode.EVENTALIGN)
                self.plot_mode = PlotMode.EVENTALIGN  # Explicitly sync internal state
                self.advanced_options_panel.set_dwell_time_enabled(True)
                self.advanced_options_panel.set_position_type_enabled(True)

                # Enable browse references button if in region search mode
                if self.search_panel.get_search_mode() == "region":
                    self.search_panel.set_browse_enabled(True)

                # Detect and show modifications if present
                await self._detect_and_show_modifications(bam_path)

                # Reload read IDs to populate tree with reference grouping
                await self.load_read_ids()

                self.statusBar().showMessage(
                    f"Loaded and validated BAM file: {bam_path.name} "
                    f"({validation_result['bam_read_count']} reads, grouped by reference)"
                )

            except Exception as e:
                QMessageBox.critical(
                    self, "Error", f"Failed to load BAM file:\n{str(e)}"
                )

    def load_references(self):
        """Load available references into the reference list"""
        self.reference_list.clear()

        if not self.bam_file:
            return

        try:
            # Get references with read counts
            references = get_bam_references(self.bam_file)

            # Filter out references with 0 reads
            references_with_reads = [
                ref
                for ref in references
                if ref.get("read_count") is not None and ref["read_count"] > 0
            ]

            # Add to list widget with formatted display
            for ref in references_with_reads:
                display_text = f"{ref['name']} ({ref['read_count']} reads)"
                item = QListWidgetItem(display_text)
                # Store reference name as item data
                item.setData(Qt.UserRole, ref["name"])
                # Store full reference dict
                item.setData(Qt.UserRole + 1, ref)
                self.reference_list.addItem(item)

        except Exception as e:
            self.statusBar().showMessage(f"Error loading references: {str(e)}")

    @qasync.asyncSlot()
    async def on_reference_selection_changed(self):
        """Handle reference selection from the reference list"""
        selected_items = self.reference_list.selectedItems()

        if not selected_items:
            return

        # Get the selected reference
        item = selected_items[0]
        ref_name = item.data(Qt.UserRole)

        # Update selected reference
        self.selected_reference = ref_name

        # Automatically display aggregate plot
        await self.display_aggregate()

    def select_reference_for_aggregate(self):
        """Open dialog to select a reference for aggregate mode"""
        if not self.bam_file:
            QMessageBox.warning(
                self,
                "BAM File Required",
                "Please load a BAM file before selecting a reference.",
            )
            return

        # Get available references from BAM
        try:
            references = get_bam_references(self.bam_file)
            if not references:
                QMessageBox.warning(
                    self,
                    "No References Found",
                    "No reference sequences found in BAM file.",
                )
                return

            # Filter out references with 0 reads
            references_with_reads = [
                ref
                for ref in references
                if ref.get("read_count") is not None and ref["read_count"] > 0
            ]

            if not references_with_reads:
                QMessageBox.warning(
                    self,
                    "No Reads Found",
                    "No reference sequences with aligned reads found in BAM file.",
                )
                return

            # Show reference browser dialog
            dialog = ReferenceBrowserDialog(references_with_reads, parent=self)
            if dialog.exec():
                selected_ref = dialog.get_selected_reference()
                if selected_ref:
                    self.selected_reference = selected_ref["name"]
                    self.selected_reference_label.setText(
                        f"Selected: {self.selected_reference}\n"
                        f"({selected_ref['read_count']} reads)"
                    )
                    # Automatically display aggregate if already in aggregate mode
                    if self.plot_mode == PlotMode.AGGREGATE:
                        asyncio.ensure_future(self.display_aggregate())

        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to retrieve references:\n{str(e)}"
            )

    @qasync.asyncSlot()
    async def display_aggregate(self):
        """Display aggregate plot for selected reference"""
        if not self.selected_reference:
            QMessageBox.warning(
                self,
                "No Reference Selected",
                "Please select a reference sequence first.",
            )
            return

        if not self.pod5_file or not self.bam_file:
            QMessageBox.warning(
                self, "Files Required", "Please load both POD5 and BAM files."
            )
            return

        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.statusBar().showMessage(
                f"Extracting {self.max_aggregate_reads} reads for {self.selected_reference}..."
            )

            # Extract reads in background thread
            reads_data = await asyncio.to_thread(
                extract_reads_for_reference,
                self.pod5_file,
                self.bam_file,
                self.selected_reference,
                self.max_aggregate_reads,
                random_sample=True,
            )

            if not reads_data:
                QMessageBox.warning(
                    self,
                    "No Reads Found",
                    f"No reads found mapping to {self.selected_reference}",
                )
                return

            self.statusBar().showMessage(
                f"Calculating aggregate statistics from {len(reads_data)} reads..."
            )

            # Calculate statistics in background thread
            aggregate_stats = await asyncio.to_thread(
                calculate_aggregate_signal, reads_data, self.normalization_method
            )

            pileup_stats = await asyncio.to_thread(
                calculate_base_pileup,
                reads_data,
                self.bam_file,
                self.selected_reference,
            )

            quality_stats = await asyncio.to_thread(
                calculate_quality_by_position, reads_data
            )

            self.statusBar().showMessage("Generating aggregate plot...")

            # Generate plot in background thread
            html, grid = await asyncio.to_thread(
                SquigglePlotter.plot_aggregate,
                aggregate_stats,
                pileup_stats,
                quality_stats,
                self.selected_reference,
                len(reads_data),
                self.normalization_method,
                self.current_theme,
            )

            # Store HTML and figure for export
            self.current_plot_html = html
            self.current_plot_figure = grid
            self.export_action.setEnabled(True)

            # Display on main thread
            unique_url = QUrl(f"http://localhost/{time.time()}")
            self.plot_view.setHtml(html, baseUrl=unique_url)

            self.statusBar().showMessage(
                f"Aggregate plot: {self.selected_reference} "
                f"({len(reads_data)} reads, {len(aggregate_stats['positions'])} positions)"
            )

        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to generate aggregate plot:\n{str(e)}"
            )
        finally:
            QApplication.restoreOverrideCursor()

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
        """Handle read selection changes with debouncing to prevent race conditions"""
        # Cancel any existing update task to prevent multiple concurrent updates
        if self._update_plot_task and not self._update_plot_task.done():
            self._update_plot_task.cancel()

        # Create a new task for this selection change
        self._update_plot_task = asyncio.create_task(self._debounced_update_plot())

    async def _debounced_update_plot(self):
        """Debounced plot update with cancellation handling"""
        try:
            # Wait for selection to stabilize (200ms debounce)
            await asyncio.sleep(0.2)

            # Update the plot with the current selection
            await self.update_plot_from_selection()
        except asyncio.CancelledError:
            # Task was cancelled by a newer selection change - this is expected behavior
            pass

    @qasync.asyncSlot()
    async def update_plot_from_selection(self):
        """Update plot based on current selection and plot mode"""
        # Get selected read IDs from tree widget
        # This handles both individual read selections and reference group selections
        read_ids = self.read_list.get_selected_read_ids()

        if not read_ids:
            return

        # Automatic fallback: if event-aligned mode is selected but no BAM is loaded,
        # switch to single read mode for a smoother user experience
        if self.plot_mode == PlotMode.EVENTALIGN and not self.bam_file:
            self.plot_mode = PlotMode.SINGLE
            self.plot_options_panel.set_plot_mode(PlotMode.SINGLE)

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
        elif self.plot_mode == PlotMode.AGGREGATE:
            # Aggregate mode uses reference selection, not read selection
            # Show informative message to guide user
            QMessageBox.information(
                self,
                "Aggregate Mode",
                "Aggregate mode displays multi-read pileup for a reference sequence.\n\n"
                "Please use the 'Select Reference' button to choose a reference,\n"
                "or select individual reads and switch to a different plot mode.",
            )
        else:
            QMessageBox.warning(
                self,
                "Unsupported Mode",
                f"Plot mode {self.plot_mode} not yet implemented",
            )

    def on_search_mode_changed(self, mode):
        """Handle search mode change"""
        self.search_mode = mode

        # Enable/disable browse button based on BAM file availability
        if mode == "region":
            self.search_panel.set_browse_enabled(self.bam_file is not None)

        # Hide sequence results box if switching away from sequence mode
        if mode != "sequence":
            self.sequence_results_box.setVisible(False)

    def on_search_text_changed(self):
        """Handle real-time search for read ID mode"""
        if self.search_panel.get_search_mode() == "read_id":
            # Real-time filtering for read ID mode
            self.filter_reads_by_id()

    @qasync.asyncSlot()
    async def execute_search(self):
        """Execute search based on current mode"""
        if self.search_panel.get_search_mode() == "read_id":
            self.filter_reads_by_id()
        elif self.search_panel.get_search_mode() == "region":
            await self.filter_reads_by_region()
        else:  # sequence mode
            await self.search_sequence()

    def filter_reads_by_id(self):
        """Filter the read tree based on read ID search input"""
        search_text = self.search_panel.get_search_text()
        # Use tree widget's built-in filter method
        self.read_list.filter_by_read_id(search_text)

    @qasync.asyncSlot()
    async def filter_reads_by_region(self):
        """Filter reads based on genomic region query (requires BAM file)"""
        region_str = self.search_panel.get_search_text().strip()

        # Query BAM file for reads in region
        self.statusBar().showMessage(
            f"Querying BAM for region {region_str}..." if region_str else "Ready"
        )

        # Use SearchManager to handle the search
        (
            success,
            visible_count,
            message,
            reads_in_region,
        ) = await self.search_manager.filter_by_region(
            self.bam_file, self.read_list, region_str
        )

        if success:
            # Store alignment info
            self.alignment_info = reads_in_region
            self.statusBar().showMessage(message)
        else:
            self.statusBar().showMessage(message)

    @qasync.asyncSlot()
    async def browse_references(self):
        """Open dialog to browse available references in BAM file"""
        # Get references in background thread
        self.statusBar().showMessage("Loading BAM references...")

        # Use SearchManager to get references
        success, references = await self.search_manager.browse_references(self.bam_file)

        if success and references:
            # Open dialog
            dialog = ReferenceBrowserDialog(references, self)
            result = dialog.exec()

            if result == ReferenceBrowserDialog.Accepted and dialog.selected_reference:
                # User selected a reference - populate search field
                self.search_panel.set_search_text(dialog.selected_reference)
                # Automatically execute search
                await self.execute_search()

            self.statusBar().showMessage("Ready")
        else:
            self.statusBar().showMessage("Error")

    @qasync.asyncSlot()
    async def search_sequence(self):
        """Search for a DNA sequence in the reference"""
        query_seq = self.search_panel.get_search_text().strip().upper()

        if not query_seq:
            self.sequence_results_box.setVisible(False)
            self.statusBar().showMessage("Ready")
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

        # Use SearchManager to search for sequence
        include_revcomp = self.search_panel.is_revcomp_checked()
        success, matches, message = await self.search_manager.search_sequence(
            self.bam_file, read_id, query_seq, include_revcomp
        )

        if success:
            # Display results
            self.sequence_results_list.clear()

            if not matches:
                self.sequence_results_list.addItem(message)
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

            # Show results box
            self.sequence_results_box.setVisible(True)
            self.sequence_results_box.toggle_button.setChecked(True)
            self.sequence_results_box.on_toggle()

        self.statusBar().showMessage(message)

    def zoom_to_sequence_match(self, item):
        """Zoom plot to show a sequence match"""
        self.zoom_manager.zoom_to_sequence_match(item)

    def _generate_plot_blocking(self, read_id):
        """Blocking function to generate bokeh plot HTML"""
        # Get signal data
        with writable_working_directory():
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

        # Get modifications if available
        modifications = None
        if self.bam_file:
            from .alignment import extract_alignment_from_bam

            aligned_read = extract_alignment_from_bam(self.bam_file, read_id)
            if aligned_read and aligned_read.modifications:
                modifications = aligned_read.modifications

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
            modifications=modifications,
            show_modification_overlay=self.show_modification_overlay,
            modification_overlay_opacity=self.modification_overlay_opacity,
            modification_type_filter=self.modification_type_filter,
            modification_threshold_enabled=self.modification_threshold_enabled,
            modification_threshold=self.modification_threshold,
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
        with writable_working_directory():
            with pod5.Reader(self.pod5_file) as reader:
                for r in reader.reads():
                    read_id_str = str(r.read_id)
                    if read_id_str in read_ids:
                        reads_data.append(
                            (read_id_str, r.signal, r.run_info.sample_rate)
                        )
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
        with writable_working_directory():
            with pod5.Reader(self.pod5_file) as reader:
                for r in reader.reads():
                    read_id_str = str(r.read_id)
                    if read_id_str in read_ids:
                        # Get signal data
                        reads_data.append(
                            (read_id_str, r.signal, r.run_info.sample_rate)
                        )

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
            show_modification_overlay=self.show_modification_overlay,
            modification_overlay_opacity=self.modification_overlay_opacity,
            modification_type_filter=self.modification_type_filter,
            modification_threshold_enabled=self.modification_threshold_enabled,
            modification_threshold=self.modification_threshold,
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

    def export_plot(self):
        """Export the current plot with format and dimension options"""
        self.export_manager.export_plot()
