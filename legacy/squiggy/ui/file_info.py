"""File information panel for Squiggy viewer"""

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ..widgets import CollapsibleBox


class FileInfoPanel(QWidget):
    """Panel displaying POD5 file information"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.collapsible_box = CollapsibleBox("POD5 File Information")
        self._create_ui()

    def _create_ui(self):
        """Create the file info content layout"""
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

        self.collapsible_box.set_content_layout(content_layout)

        # Set panel layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.collapsible_box)

    def update_info(
        self,
        filename="—",
        filesize="—",
        num_reads="—",
        sample_rate="—",
        total_samples="—",
    ):
        """Update the file information display"""
        self.info_filename_label.setText(filename)
        self.info_filesize_label.setText(filesize)
        self.info_num_reads_label.setText(num_reads)
        self.info_sample_rate_label.setText(sample_rate)
        self.info_total_samples_label.setText(total_samples)
