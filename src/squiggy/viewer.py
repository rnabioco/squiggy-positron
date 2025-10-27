"""Main application window for Squiggy"""

import asyncio
from io import BytesIO
from pathlib import Path

import pod5
import qasync
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
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
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .constants import (
    APP_DESCRIPTION,
    APP_NAME,
    DEFAULT_WINDOW_HEIGHT,
    DEFAULT_WINDOW_WIDTH,
    MAX_OVERLAY_READS,
    PLOT_DPI,
    PLOT_HEIGHT,
    PLOT_MIN_HEIGHT,
    PLOT_MIN_WIDTH,
    PLOT_WIDTH,
    SPLITTER_RATIO,
    NormalizationMethod,
    PlotMode,
)
from .dialogs import AboutDialog, ReferenceBrowserDialog
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
            self.content_area.setMaximumHeight(200)
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
        self.show_bases = False
        self.plot_mode = PlotMode.SINGLE
        self.normalization_method = NormalizationMethod.NONE
        self.current_plot = None  # Store current plot object for export
        self.search_mode = "read_id"  # "read_id" or "region"

        self.init_ui()

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle(f"{APP_NAME} - {APP_DESCRIPTION}")
        self.setGeometry(100, 100, DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)

        # Create menu bar
        self.create_menu_bar()

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Search section at the top
        search_layout = QHBoxLayout()

        # Search mode selector
        self.search_mode_combo = QComboBox()
        self.search_mode_combo.addItem("Read ID", "read_id")
        self.search_mode_combo.addItem("Reference Region", "region")
        self.search_mode_combo.currentIndexChanged.connect(self.on_search_mode_changed)
        self.search_mode_combo.setToolTip(
            "Switch between searching by read ID or by genomic region"
        )
        search_layout.addWidget(QLabel("Search by:"))
        search_layout.addWidget(self.search_mode_combo)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search read ID...")
        self.search_input.textChanged.connect(self.on_search_text_changed)
        self.search_input.returnPressed.connect(self.execute_search)

        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self.execute_search)

        # Browse references button (for region search mode)
        self.browse_refs_button = QPushButton("Browse References...")
        self.browse_refs_button.clicked.connect(self.browse_references)
        self.browse_refs_button.setEnabled(False)
        self.browse_refs_button.setVisible(False)  # Hidden by default
        self.browse_refs_button.setToolTip(
            "View available reference sequences in BAM file"
        )

        search_layout.addWidget(self.search_input, 1)
        search_layout.addWidget(self.search_button)
        search_layout.addWidget(self.browse_refs_button)

        # Add base annotation toggle
        self.base_checkbox = QCheckBox("Show base annotations")
        self.base_checkbox.setEnabled(False)
        self.base_checkbox.stateChanged.connect(self.toggle_base_annotations)
        search_layout.addWidget(self.base_checkbox)

        main_layout.addLayout(search_layout)

        # Create main horizontal splitter for left panel, center plot, and right sidebar
        main_splitter = QSplitter(Qt.Horizontal)

        # Left panel: File browser and plot options
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # File selection section
        file_layout = QHBoxLayout()
        self.file_label = QLabel("No file selected")
        self.file_button = QPushButton("Open POD5 File")
        self.file_button.clicked.connect(self.open_pod5_file)
        file_layout.addWidget(QLabel("POD5 File:"))
        file_layout.addWidget(self.file_label, 1)
        file_layout.addWidget(self.file_button)
        left_layout.addLayout(file_layout)

        # BAM file selection section (optional)
        bam_layout = QHBoxLayout()
        self.bam_label = QLabel("No BAM file (optional)")
        self.bam_button = QPushButton("Open BAM File")
        self.bam_button.clicked.connect(self.open_bam_file)
        bam_layout.addWidget(QLabel("BAM File:"))
        bam_layout.addWidget(self.bam_label, 1)
        bam_layout.addWidget(self.bam_button)
        left_layout.addLayout(bam_layout)

        # POD5 file information collapsible panel
        self.file_info_box = CollapsibleBox("POD5 File Information")
        self.create_file_info_content()
        left_layout.addWidget(self.file_info_box)

        # Plot options collapsible panel
        self.plot_options_box = CollapsibleBox("Plot Options")
        self.create_plot_options_content()
        left_layout.addWidget(self.plot_options_box)

        # Add stretch to push everything to the top
        left_layout.addStretch(1)

        main_splitter.addWidget(left_panel)

        # Plot display area (center)
        self.plot_label = QLabel("Select a POD5 file and read to display squiggle plot")
        self.plot_label.setAlignment(Qt.AlignCenter)
        self.plot_label.setStyleSheet(
            "border: 1px solid #ccc; background-color: #f9f9f9;"
        )
        self.plot_label.setMinimumSize(PLOT_MIN_WIDTH, PLOT_MIN_HEIGHT)
        main_splitter.addWidget(self.plot_label)

        # Right sidebar: Read list
        self.read_list = QListWidget()
        self.read_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.read_list.itemSelectionChanged.connect(self.on_read_selection_changed)
        main_splitter.addWidget(self.read_list)

        # Set splitter proportions: left panel (1), center plot (3), right sidebar (1)
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 3)
        main_splitter.setStretchFactor(2, 1)

        main_layout.addWidget(main_splitter, 1)

        # Status bar
        self.statusBar().showMessage("Ready")

    def create_file_info_content(self):
        """Create the content layout for the file information panel"""
        content_layout = QGridLayout()
        content_layout.setContentsMargins(10, 5, 10, 5)
        content_layout.setSpacing(5)

        # Create labels for file information
        self.info_filename_label = QLabel("—")
        self.info_filesize_label = QLabel("—")
        self.info_num_reads_label = QLabel("—")
        self.info_sample_rate_label = QLabel("—")
        self.info_total_samples_label = QLabel("—")

        # Add labels to grid layout
        row = 0
        content_layout.addWidget(QLabel("File name:"), row, 0)
        content_layout.addWidget(self.info_filename_label, row, 1)
        row += 1
        content_layout.addWidget(QLabel("File size:"), row, 0)
        content_layout.addWidget(self.info_filesize_label, row, 1)
        row += 1
        content_layout.addWidget(QLabel("Number of reads:"), row, 0)
        content_layout.addWidget(self.info_num_reads_label, row, 1)
        row += 1
        content_layout.addWidget(QLabel("Sample rate:"), row, 0)
        content_layout.addWidget(self.info_sample_rate_label, row, 1)
        row += 1
        content_layout.addWidget(QLabel("Total samples:"), row, 0)
        content_layout.addWidget(self.info_total_samples_label, row, 1)

        # Set column stretch to make labels expand
        content_layout.setColumnStretch(1, 1)

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

        self.mode_single = QRadioButton("Single Read")
        self.mode_single.setChecked(True)
        self.mode_single.setToolTip("Display one read at a time")
        self.mode_single.toggled.connect(
            lambda checked: self.set_plot_mode(PlotMode.SINGLE) if checked else None
        )
        self.mode_button_group.addButton(self.mode_single)
        content_layout.addWidget(self.mode_single)

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

        self.mode_eventalign = QRadioButton("Event-aligned (base annotations)")
        self.mode_eventalign.setToolTip(
            "Show event-aligned reads with base annotations (requires BAM file)"
        )
        self.mode_eventalign.toggled.connect(
            lambda checked: self.set_plot_mode(PlotMode.EVENTALIGN) if checked else None
        )
        self.mode_eventalign.setEnabled(False)  # Disabled until BAM file loaded
        self.mode_button_group.addButton(self.mode_eventalign)
        content_layout.addWidget(self.mode_eventalign)

        # Normalization method selection
        norm_label = QLabel("Signal Normalization:")
        norm_label.setStyleSheet("font-weight: bold;")
        content_layout.addWidget(norm_label)

        self.norm_combo = QComboBox()
        self.norm_combo.addItem("None (raw signal)", NormalizationMethod.NONE)
        self.norm_combo.addItem("Z-score", NormalizationMethod.ZNORM)
        self.norm_combo.addItem("Median", NormalizationMethod.MEDIAN)
        self.norm_combo.addItem("MAD (robust)", NormalizationMethod.MAD)
        self.norm_combo.currentIndexChanged.connect(self.set_normalization_method)
        content_layout.addWidget(self.norm_combo)

        # Info label
        info_label = QLabel(
            "💡 Tip: Use Ctrl/Cmd+Click or Shift+Click to select multiple reads"
        )
        info_label.setStyleSheet("color: #666; font-size: 9pt;")
        info_label.setWordWrap(True)
        content_layout.addWidget(info_label)

        content_layout.addStretch()

        self.plot_options_box.set_content_layout(content_layout)

    def set_plot_mode(self, mode):
        """Set the plot mode and refresh display"""
        self.plot_mode = mode
        # Refresh plot if reads are selected
        if self.read_list.selectedItems():
            self.update_plot_from_selection()

    def set_normalization_method(self, index):
        """Set the normalization method and refresh display"""
        self.normalization_method = self.norm_combo.itemData(index)
        # Refresh plot if reads are selected
        if self.read_list.selectedItems():
            self.update_plot_from_selection()

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
                self.mode_eventalign.setEnabled(True)

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
        self.show_bases = state == Qt.Checked
        # Refresh current plot if one is displayed
        if self.read_list.selectedItems():
            await self.update_plot_from_selection()

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

        # Update placeholder text
        if self.search_mode == "read_id":
            self.search_input.setPlaceholderText("Search read ID...")
            self.search_input.setToolTip("Filter reads by read ID (case-insensitive)")
            # Hide browse button for read ID mode
            self.browse_refs_button.setVisible(False)
        else:  # region
            self.search_input.setPlaceholderText("e.g., chr1:1000-2000 or chr1")
            self.search_input.setToolTip(
                "Search reads by genomic region (requires BAM file)\n"
                "Format: chr1, chr1:1000, or chr1:1000-2000"
            )
            # Show browse button for region mode (enabled only if BAM loaded)
            self.browse_refs_button.setVisible(True)
            self.browse_refs_button.setEnabled(self.bam_file is not None)

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
        else:  # region mode
            await self.filter_reads_by_region()

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

    def _generate_plot_blocking(self, read_id):
        """Blocking function to generate plot"""
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

        # Generate plot
        plot = SquigglePlotter.plot_squiggle(
            signal,
            read_id,
            sample_rate,
            sequence=sequence,
            seq_to_sig_map=seq_to_sig_map,
        )

        # Save plot to buffer
        buffer = BytesIO()
        plot.save(
            buffer, format="png", dpi=PLOT_DPI, width=PLOT_WIDTH, height=PLOT_HEIGHT
        )
        buffer.seek(0)

        return buffer, signal, sequence, plot

    async def display_single_read(self, read_id):
        """Display squiggle plot for a single read (async)"""
        self.statusBar().showMessage(f"Generating plot for {read_id}...")

        try:
            # Generate plot in thread pool
            buffer, signal, sequence, plot = await asyncio.to_thread(
                self._generate_plot_blocking, read_id
            )

            # Store plot for export
            self.current_plot = plot
            self.export_action.setEnabled(True)

            # Display on main thread
            pixmap = QPixmap()
            pixmap.loadFromData(buffer.read())
            self.plot_label.setPixmap(
                pixmap.scaled(
                    self.plot_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )

            status_msg = f"Displaying read: {read_id} ({len(signal)} samples)"
            if sequence:
                status_msg += f" - {len(sequence)} bases"
            self.statusBar().showMessage(status_msg)

        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to display squiggle:\n{str(e)}"
            )

    def _generate_multi_read_plot_blocking(self, read_ids):
        """Blocking function to generate multi-read plot"""
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

        # Generate multi-read plot
        plot = SquigglePlotter.plot_multiple_reads(
            reads_data,
            mode=self.plot_mode,
            normalization=self.normalization_method,
        )

        # Save plot to buffer
        buffer = BytesIO()
        plot.save(
            buffer, format="png", dpi=PLOT_DPI, width=PLOT_WIDTH, height=PLOT_HEIGHT
        )
        buffer.seek(0)

        return buffer, reads_data, plot

    async def display_multiple_reads(self, read_ids):
        """Display multiple reads in overlay or stacked mode (async)"""
        self.statusBar().showMessage(f"Generating plot for {len(read_ids)} reads...")

        try:
            # Generate plot in thread pool
            buffer, reads_data, plot = await asyncio.to_thread(
                self._generate_multi_read_plot_blocking, read_ids
            )

            # Store plot for export
            self.current_plot = plot
            self.export_action.setEnabled(True)

            # Display on main thread
            pixmap = QPixmap()
            pixmap.loadFromData(buffer.read())
            self.plot_label.setPixmap(
                pixmap.scaled(
                    self.plot_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )

            mode_name = "overlaid" if self.plot_mode == PlotMode.OVERLAY else "stacked"
            self.statusBar().showMessage(
                f"Displaying {len(reads_data)} reads ({mode_name}, {self.normalization_method.value} normalization)"
            )

        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to display multi-read plot:\n{str(e)}"
            )

    def _generate_eventalign_plot_blocking(self, read_ids):
        """Blocking function to generate event-aligned plot"""
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

        # Generate event-aligned plot
        plot = SquigglePlotter.plot_multiple_reads(
            reads_data,
            mode=self.plot_mode,
            normalization=self.normalization_method,
            aligned_reads=aligned_reads,
        )

        # Save plot to buffer
        buffer = BytesIO()
        plot.save(
            buffer, format="png", dpi=PLOT_DPI, width=PLOT_WIDTH, height=PLOT_HEIGHT
        )
        buffer.seek(0)

        return buffer, reads_data, aligned_reads, plot

    async def display_eventaligned_reads(self, read_ids):
        """Display multiple reads in event-aligned mode (async)"""
        self.statusBar().showMessage(
            f"Generating event-aligned plot for {len(read_ids)} reads..."
        )

        try:
            # Generate plot in thread pool
            buffer, reads_data, aligned_reads, plot = await asyncio.to_thread(
                self._generate_eventalign_plot_blocking, read_ids
            )

            # Store plot for export
            self.current_plot = plot
            self.export_action.setEnabled(True)

            # Display on main thread
            pixmap = QPixmap()
            pixmap.loadFromData(buffer.read())
            self.plot_label.setPixmap(
                pixmap.scaled(
                    self.plot_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )

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
        """Export the current plot to a file"""
        if self.current_plot is None:
            QMessageBox.warning(
                self,
                "No Plot",
                "No plot to export. Please display a plot first.",
            )
            return

        # Get file path from user
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Plot",
            "",
            "PNG Image (*.png);;PDF Document (*.pdf);;SVG Vector (*.svg);;All Files (*)",
        )

        if not file_path:
            return  # User cancelled

        try:
            # Determine format from extension or filter
            file_path = Path(file_path)
            if not file_path.suffix:
                # Add extension based on filter
                if "PNG" in selected_filter:
                    file_path = file_path.with_suffix(".png")
                elif "PDF" in selected_filter:
                    file_path = file_path.with_suffix(".pdf")
                elif "SVG" in selected_filter:
                    file_path = file_path.with_suffix(".svg")
                else:
                    file_path = file_path.with_suffix(".png")  # Default to PNG

            format_str = file_path.suffix[1:]  # Remove leading dot

            # Save plot
            self.current_plot.save(
                str(file_path),
                format=format_str,
                dpi=300,  # High DPI for export
                width=12,
                height=8,
            )

            self.statusBar().showMessage(f"Plot exported to {file_path.name}")
            QMessageBox.information(
                self,
                "Export Successful",
                f"Plot successfully exported to:\n{file_path}",
            )

        except Exception as e:
            QMessageBox.critical(
                self, "Export Failed", f"Failed to export plot:\n{str(e)}"
            )
