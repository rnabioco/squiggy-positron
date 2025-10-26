import sys
import numpy as np
import pandas as pd
import pod5
from pathlib import Path
from io import BytesIO
import argparse
import importlib.resources

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QLabel, QFileDialog, QListWidget,
    QSplitter, QMessageBox, QMenuBar, QMenu, QCheckBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QAction

from plotnine import (
    ggplot, aes, geom_line, geom_rect, geom_text, geom_vline,
    labs, theme_minimal, theme, element_text, scale_fill_manual,
    scale_color_manual, ylim
)

try:
    import pysam
    PYSAM_AVAILABLE = True
except ImportError:
    PYSAM_AVAILABLE = False

# Base colors from remora
BASE_COLORS = {
    "A": "#00CC00",  # Green
    "C": "#0000CC",  # Blue
    "G": "#FFB300",  # Orange
    "T": "#CC0000",  # Red
    "U": "#CC0000",  # Red
    "N": "#FFFFFF",  # White
}


class SquigglePlotter:
    """Handle squiggle plot generation using plotnine"""

    @staticmethod
    def plot_squiggle(signal, read_id, sample_rate=4000, sequence=None, seq_to_sig_map=None):
        """Generate a squiggle plot from signal data

        Args:
            signal: Raw signal array
            read_id: Read identifier for title
            sample_rate: Sampling rate (Hz)
            sequence: Optional basecalled sequence
            seq_to_sig_map: Optional mapping from sequence positions to signal indices
        """
        # Create time axis
        time = np.arange(len(signal)) / sample_rate

        # Create DataFrame for plotting
        df = pd.DataFrame({
            'time': time,
            'signal': signal
        })

        # Calculate signal range for base annotations
        sig_min = np.percentile(signal, 2.5)
        sig_max = np.percentile(signal, 97.5)

        # Create plot
        plot = (
            ggplot(df, aes(x='time', y='signal'))
            + geom_line(color='#2E86AB', size=0.5)
            + labs(
                title=f'Squiggle Plot: {read_id}',
                x='Time (s)',
                y='Signal (pA)'
            )
            + theme_minimal()
            + theme(
                plot_title=element_text(size=12, weight='bold'),
                axis_title=element_text(size=10)
            )
        )

        # Add base annotations if available
        if sequence is not None and seq_to_sig_map is not None:
            base_coords = SquigglePlotter._create_base_coords(
                sequence, seq_to_sig_map, time, sig_min, sig_max
            )

            if len(base_coords) > 0:
                # Add colored rectangles for bases
                plot = plot + geom_rect(
                    aes(xmin='base_st', xmax='base_en', fill='base', ymin=sig_min, ymax=sig_max),
                    data=base_coords,
                    alpha=0.1,
                    show_legend=False
                )

                # Add base labels
                plot = plot + geom_text(
                    aes(x='base_st', label='base', color='base', y=sig_min),
                    data=base_coords,
                    va='bottom',
                    ha='left',
                    size=8,
                    show_legend=False
                )

                # Apply color scales
                plot = plot + scale_fill_manual(BASE_COLORS) + scale_color_manual(BASE_COLORS)

        return plot

    @staticmethod
    def _create_base_coords(sequence, seq_to_sig_map, time, sig_min, sig_max):
        """Create DataFrame with base coordinate information for plotting"""
        base_data = []

        for i, base in enumerate(sequence):
            if i < len(seq_to_sig_map):
                sig_idx = seq_to_sig_map[i]
                # Get end position (next base's start or end of signal)
                if i + 1 < len(seq_to_sig_map):
                    next_sig_idx = seq_to_sig_map[i + 1]
                else:
                    next_sig_idx = len(time) - 1

                if sig_idx < len(time) and next_sig_idx < len(time):
                    base_data.append({
                        'base': base,
                        'base_st': time[sig_idx],
                        'base_en': time[next_sig_idx]
                    })

        return pd.DataFrame(base_data) if base_data else pd.DataFrame()


class SquiggleViewer(QMainWindow):
    """Main application window for nanopore squiggle visualization"""

    def __init__(self):
        super().__init__()
        self.pod5_file = None
        self.bam_file = None
        self.bam_index = None
        self.read_dict = {}
        self.current_plot = None
        self.show_bases = False

        self.init_ui()

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle('Nanopore Squiggle Viewer')
        self.setGeometry(100, 100, 1200, 800)

        # Create menu bar
        self.create_menu_bar()

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # File selection section
        file_layout = QHBoxLayout()
        self.file_label = QLabel('No file selected')
        self.file_button = QPushButton('Open POD5 File')
        self.file_button.clicked.connect(self.open_pod5_file)
        file_layout.addWidget(QLabel('POD5 File:'))
        file_layout.addWidget(self.file_label, 1)
        file_layout.addWidget(self.file_button)
        main_layout.addLayout(file_layout)

        # BAM file selection section (optional)
        bam_layout = QHBoxLayout()
        self.bam_label = QLabel('No BAM file (optional)')
        self.bam_button = QPushButton('Open BAM File')
        self.bam_button.clicked.connect(self.open_bam_file)
        if not PYSAM_AVAILABLE:
            self.bam_button.setEnabled(False)
            self.bam_button.setToolTip('Install pysam to enable BAM support')
        bam_layout.addWidget(QLabel('BAM File:'))
        bam_layout.addWidget(self.bam_label, 1)
        bam_layout.addWidget(self.bam_button)
        main_layout.addLayout(bam_layout)

        # Search section
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText('Search read ID...')
        self.search_input.textChanged.connect(self.filter_reads)
        self.search_button = QPushButton('Search')
        self.search_button.clicked.connect(self.filter_reads)
        search_layout.addWidget(QLabel('Search:'))
        search_layout.addWidget(self.search_input, 1)
        search_layout.addWidget(self.search_button)

        # Add base annotation toggle
        self.base_checkbox = QCheckBox('Show base annotations')
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
        self.plot_label = QLabel('Select a POD5 file and read to display squiggle plot')
        self.plot_label.setAlignment(Qt.AlignCenter)
        self.plot_label.setStyleSheet('border: 1px solid #ccc; background-color: #f9f9f9;')
        self.plot_label.setMinimumSize(800, 600)
        splitter.addWidget(self.plot_label)

        # Set splitter proportions
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        main_layout.addWidget(splitter, 1)

        # Status bar
        self.statusBar().showMessage('Ready')

    def create_menu_bar(self):
        """Create the application menu bar"""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu('File')

        # Open file action
        open_action = QAction('Open POD5 File...', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.open_pod5_file)
        file_menu.addAction(open_action)

        # Open sample data action
        sample_action = QAction('Open Sample Data', self)
        sample_action.setShortcut('Ctrl+Shift+O')
        sample_action.triggered.connect(self.open_sample_data)
        file_menu.addAction(sample_action)

        file_menu.addSeparator()

        # Exit action
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def get_sample_data_path(self):
        """Get the path to the bundled sample data file"""
        try:
            # For Python 3.9+
            if sys.version_info >= (3, 9):
                import importlib.resources as resources
                files = resources.files('squiggy')
                sample_path = files / 'data' / 'sample.pod5'
                if hasattr(sample_path, 'as_posix'):
                    return Path(sample_path)
                # For traversable objects, we need to extract to temp
                import tempfile
                temp_dir = Path(tempfile.gettempdir()) / 'squiggy_data'
                temp_dir.mkdir(exist_ok=True)
                temp_file = temp_dir / 'sample.pod5'
                if not temp_file.exists():
                    with resources.as_file(sample_path) as f:
                        import shutil
                        shutil.copy(f, temp_file)
                return temp_file
            else:
                # Fallback for older Python
                import pkg_resources
                sample_path = pkg_resources.resource_filename('squiggy', 'data/sample.pod5')
                return Path(sample_path)
        except Exception as e:
            # Fallback: look in installed package directory
            package_dir = Path(__file__).parent
            sample_path = package_dir / 'data' / 'sample.pod5'
            if sample_path.exists():
                return sample_path
            raise FileNotFoundError(f"Sample data not found. Error: {e}")

    def open_sample_data(self):
        """Open the bundled sample POD5 file"""
        try:
            sample_path = self.get_sample_data_path()
            if not sample_path.exists():
                QMessageBox.warning(
                    self,
                    'Sample Data Not Found',
                    'The sample data file could not be found.\n\n'
                    'This may happen if Squiggy was not installed properly.'
                )
                return

            self.pod5_file = sample_path
            self.file_label.setText(f'{sample_path.name} (sample)')
            self.load_read_ids()
            self.statusBar().showMessage(
                f'Loaded {len(self.read_dict)} reads from sample data'
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                'Error',
                f'Failed to load sample data:\n{str(e)}'
            )

    def open_pod5_file(self):
        """Open and load a POD5 file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            'Open POD5 File',
            '',
            'POD5 Files (*.pod5);;All Files (*)'
        )

        if file_path:
            try:
                self.pod5_file = Path(file_path)
                self.file_label.setText(self.pod5_file.name)
                self.load_read_ids()
                self.statusBar().showMessage(f'Loaded {len(self.read_dict)} reads')
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Failed to load POD5 file:\n{str(e)}')

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
            QMessageBox.critical(self, 'Error', f'Failed to read POD5 file:\n{str(e)}')

    def open_bam_file(self):
        """Open and load a BAM file for base annotations"""
        if not PYSAM_AVAILABLE:
            QMessageBox.warning(
                self,
                'pysam Not Available',
                'Please install pysam to use BAM file support:\n\npip install pysam'
            )
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            'Open BAM File',
            '',
            'BAM Files (*.bam);;All Files (*)'
        )

        if file_path:
            try:
                bam_path = Path(file_path)
                # Open BAM file to verify it's valid
                bam = pysam.AlignmentFile(str(bam_path), 'rb', check_sq=False)
                bam.close()

                self.bam_file = bam_path
                self.bam_label.setText(bam_path.name)
                self.base_checkbox.setEnabled(True)
                self.statusBar().showMessage(f'Loaded BAM file: {bam_path.name}')

            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Failed to load BAM file:\n{str(e)}')

    def toggle_base_annotations(self, state):
        """Toggle display of base annotations"""
        self.show_bases = (state == Qt.Checked)
        # Refresh current plot if one is displayed
        if hasattr(self, 'current_read_item') and self.current_read_item:
            self.display_squiggle(self.current_read_item)

    def get_basecall_data(self, read_id):
        """Extract basecall sequence and signal mapping from BAM file

        Returns:
            tuple: (sequence, seq_to_sig_map) or (None, None) if not available
        """
        if not self.bam_file or not PYSAM_AVAILABLE:
            return None, None

        try:
            bam = pysam.AlignmentFile(str(self.bam_file), 'rb', check_sq=False)

            # Find the read in BAM
            for read in bam.fetch(until_eof=True):
                if read.query_name == read_id:
                    # Get sequence
                    sequence = read.query_sequence

                    # Get move table from BAM tags
                    if read.has_tag('mv'):
                        move_table = np.array(read.get_tag('mv'), dtype=np.uint8)

                        # Convert move table to signal-to-sequence mapping
                        seq_to_sig_map = []
                        sig_pos = 0
                        for i, move in enumerate(move_table):
                            if move == 1:
                                seq_to_sig_map.append(sig_pos)
                            sig_pos += 1

                        bam.close()
                        return sequence, np.array(seq_to_sig_map)

            bam.close()

        except Exception as e:
            print(f"Error reading BAM file for {read_id}: {e}")

        return None, None

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
                sequence, seq_to_sig_map = self.get_basecall_data(read_id)

            # Generate plot
            plot = SquigglePlotter.plot_squiggle(
                signal, read_id, sample_rate,
                sequence=sequence,
                seq_to_sig_map=seq_to_sig_map
            )

            # Save plot to buffer and display
            buffer = BytesIO()
            plot.save(buffer, format='png', dpi=100, width=10, height=6)
            buffer.seek(0)

            pixmap = QPixmap()
            pixmap.loadFromData(buffer.read())
            self.plot_label.setPixmap(pixmap.scaled(
                self.plot_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))

            status_msg = f'Displaying read: {read_id} ({len(signal)} samples)'
            if sequence:
                status_msg += f' - {len(sequence)} bases'
            self.statusBar().showMessage(status_msg)

        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to display squiggle:\n{str(e)}')


def main():
    """Main entry point for Squiggy GUI application"""
    parser = argparse.ArgumentParser(
        description='Squiggy - Nanopore Squiggle Viewer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  squiggy                          Launch GUI
  squiggy --file data.pod5         Launch GUI with file pre-loaded

Use File → Open Sample Data in the GUI to load bundled sample data.
        """
    )
    parser.add_argument(
        '--file', '-f',
        type=str,
        help='POD5 file to open on startup'
    )

    args = parser.parse_args()

    app = QApplication(sys.argv)
    viewer = SquiggleViewer()

    # Auto-load file if specified
    if args.file:
        file_path = Path(args.file)
        if file_path.exists():
            try:
                viewer.pod5_file = file_path
                viewer.file_label.setText(file_path.name)
                viewer.load_read_ids()
                viewer.statusBar().showMessage(f'Loaded {len(viewer.read_dict)} reads')
            except Exception as e:
                QMessageBox.critical(viewer, 'Error', f'Failed to load POD5 file:\n{str(e)}')
        else:
            QMessageBox.warning(viewer, 'File Not Found', f'File does not exist:\n{file_path}')

    viewer.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
