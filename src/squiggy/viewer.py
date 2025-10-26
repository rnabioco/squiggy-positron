"""Main application window for Squiggy"""

import asyncio
from io import BytesIO
from pathlib import Path

import pod5
import qasync

try:
    import pysam

    PYSAM_AVAILABLE = True
except ImportError:
    PYSAM_AVAILABLE = False

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
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
    PLOT_DPI,
    PLOT_HEIGHT,
    PLOT_MIN_HEIGHT,
    PLOT_MIN_WIDTH,
    PLOT_WIDTH,
    SPLITTER_RATIO,
)
from .dialogs import AboutDialog
from .plotter import SquigglePlotter
from .utils import get_basecall_data, get_sample_data_path


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
        self.current_read_item = None
        self.show_bases = False

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

        # File selection section
        file_layout = QHBoxLayout()
        self.file_label = QLabel("No file selected")
        self.file_button = QPushButton("Open POD5 File")
        self.file_button.clicked.connect(self.open_pod5_file)
        file_layout.addWidget(QLabel("POD5 File:"))
        file_layout.addWidget(self.file_label, 1)
        file_layout.addWidget(self.file_button)
        main_layout.addLayout(file_layout)

        # BAM file selection section (optional)
        bam_layout = QHBoxLayout()
        self.bam_label = QLabel("No BAM file (optional)")
        self.bam_button = QPushButton("Open BAM File")
        self.bam_button.clicked.connect(self.open_bam_file)
        if not PYSAM_AVAILABLE:
            self.bam_button.setEnabled(False)
            self.bam_button.setToolTip("Install pysam to enable BAM support")
        bam_layout.addWidget(QLabel("BAM File:"))
        bam_layout.addWidget(self.bam_label, 1)
        bam_layout.addWidget(self.bam_button)
        main_layout.addLayout(bam_layout)

        # POD5 file information collapsible panel
        self.file_info_box = CollapsibleBox("POD5 File Information")
        self.create_file_info_content()
        main_layout.addWidget(self.file_info_box)

        # Search section
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search read ID...")
        self.search_input.textChanged.connect(self.filter_reads)
        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self.filter_reads)
        search_layout.addWidget(QLabel("Search:"))
        search_layout.addWidget(self.search_input, 1)
        search_layout.addWidget(self.search_button)

        # Add base annotation toggle
        self.base_checkbox = QCheckBox("Show base annotations")
        self.base_checkbox.setEnabled(False)
        self.base_checkbox.stateChanged.connect(self.toggle_base_annotations)
        search_layout.addWidget(self.base_checkbox)

        main_layout.addLayout(search_layout)

        # Create splitter for read list and plot area
        splitter = QSplitter(Qt.Horizontal)

        # Read list widget
        self.read_list = QListWidget()
        self.read_list.itemClicked.connect(self.display_squiggle)
        splitter.addWidget(self.read_list)

        # Plot display area
        self.plot_label = QLabel("Select a POD5 file and read to display squiggle plot")
        self.plot_label.setAlignment(Qt.AlignCenter)
        self.plot_label.setStyleSheet(
            "border: 1px solid #ccc; background-color: #f9f9f9;"
        )
        self.plot_label.setMinimumSize(PLOT_MIN_WIDTH, PLOT_MIN_HEIGHT)
        splitter.addWidget(self.plot_label)

        # Set splitter proportions
        splitter.setStretchFactor(0, SPLITTER_RATIO[0])
        splitter.setStretchFactor(1, SPLITTER_RATIO[1])

        main_layout.addWidget(splitter, 1)

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

    def open_bam_file(self):
        """Open and load a BAM file for base annotations"""
        if not PYSAM_AVAILABLE:
            QMessageBox.warning(
                self,
                "pysam Not Available",
                "Please install pysam to use BAM file support:\n\npip install pysam",
            )
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open BAM File", "", "BAM Files (*.bam);;All Files (*)"
        )

        if file_path:
            try:
                bam_path = Path(file_path)
                # Open BAM file to verify it's valid
                bam = pysam.AlignmentFile(str(bam_path), "rb", check_sq=False)
                bam.close()

                self.bam_file = bam_path
                self.bam_label.setText(bam_path.name)
                self.base_checkbox.setEnabled(True)
                self.statusBar().showMessage(f"Loaded BAM file: {bam_path.name}")

            except Exception as e:
                QMessageBox.critical(
                    self, "Error", f"Failed to load BAM file:\n{str(e)}"
                )

    @qasync.asyncSlot()
    async def toggle_base_annotations(self, state):
        """Toggle display of base annotations (async)"""
        self.show_bases = state == Qt.Checked
        # Refresh current plot if one is displayed
        if self.current_read_item:
            await self.display_squiggle(self.current_read_item)

    def filter_reads(self):
        """Filter the read list based on search input"""
        search_text = self.search_input.text().lower()

        for i in range(self.read_list.count()):
            item = self.read_list.item(i)
            if search_text in item.text().lower():
                item.setHidden(False)
            else:
                item.setHidden(True)

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

        return buffer, signal, sequence

    @qasync.asyncSlot()
    async def display_squiggle(self, item):
        """Display squiggle plot for selected read (async)"""
        read_id = item.text()
        self.current_read_item = item
        self.statusBar().showMessage(f"Generating plot for {read_id}...")

        try:
            # Generate plot in thread pool
            buffer, signal, sequence = await asyncio.to_thread(
                self._generate_plot_blocking, read_id
            )

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
