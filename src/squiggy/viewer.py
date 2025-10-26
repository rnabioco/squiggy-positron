"""Main application window for Squiggy"""

from pathlib import Path
from io import BytesIO

import pod5

try:
    import pysam

    PYSAM_AVAILABLE = True
except ImportError:
    PYSAM_AVAILABLE = False

from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLineEdit,
    QLabel,
    QFileDialog,
    QListWidget,
    QSplitter,
    QMessageBox,
    QCheckBox,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QAction

from .constants import (
    APP_NAME,
    APP_DESCRIPTION,
    DEFAULT_WINDOW_WIDTH,
    DEFAULT_WINDOW_HEIGHT,
    PLOT_MIN_WIDTH,
    PLOT_MIN_HEIGHT,
    SPLITTER_RATIO,
    PLOT_DPI,
    PLOT_WIDTH,
    PLOT_HEIGHT,
)
from .plotter import SquigglePlotter
from .dialogs import AboutDialog
from .utils import get_sample_data_path, get_basecall_data


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

    def open_sample_data(self):
        """Open the bundled sample POD5 file"""
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
            self.load_read_ids()
            self.statusBar().showMessage(
                f"Loaded {len(self.read_dict)} reads from sample data"
            )
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to load sample data:\n{str(e)}"
            )

    def open_pod5_file(self):
        """Open and load a POD5 file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open POD5 File", "", "POD5 Files (*.pod5);;All Files (*)"
        )

        if file_path:
            try:
                self.pod5_file = Path(file_path)
                self.file_label.setText(self.pod5_file.name)
                self.load_read_ids()
                self.statusBar().showMessage(f"Loaded {len(self.read_dict)} reads")
            except Exception as e:
                QMessageBox.critical(
                    self, "Error", f"Failed to load POD5 file:\n{str(e)}"
                )

    def load_read_ids(self):
        """Load all read IDs from the POD5 file"""
        self.read_dict.clear()
        self.read_list.clear()

        try:
            with pod5.Reader(self.pod5_file) as reader:
                for read in reader:
                    read_id = str(read.read_id)
                    self.read_dict[read_id] = read
                    self.read_list.addItem(read_id)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read POD5 file:\n{str(e)}")

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

    def toggle_base_annotations(self, state):
        """Toggle display of base annotations"""
        self.show_bases = state == Qt.Checked
        # Refresh current plot if one is displayed
        if self.current_read_item:
            self.display_squiggle(self.current_read_item)

    def filter_reads(self):
        """Filter the read list based on search input"""
        search_text = self.search_input.text().lower()

        for i in range(self.read_list.count()):
            item = self.read_list.item(i)
            if search_text in item.text().lower():
                item.setHidden(False)
            else:
                item.setHidden(True)

    def display_squiggle(self, item):
        """Display squiggle plot for selected read"""
        read_id = item.text()
        self.current_read_item = item

        try:
            # Get signal data
            with pod5.Reader(self.pod5_file) as reader:
                read = reader.get_read(read_id)
                signal = read.signal
                sample_rate = read.sample_rate

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

            # Save plot to buffer and display
            buffer = BytesIO()
            plot.save(
                buffer, format="png", dpi=PLOT_DPI, width=PLOT_WIDTH, height=PLOT_HEIGHT
            )
            buffer.seek(0)

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
