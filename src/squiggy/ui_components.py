"""UI component panels for Squiggy viewer"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .constants import MAX_OVERLAY_READS, NormalizationMethod, PlotMode
from .widgets import CollapsibleBox


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

    def update_info(self, filename="—", filesize="—", num_reads="—", sample_rate="—", total_samples="—"):
        """Update the file information display"""
        self.info_filename_label.setText(filename)
        self.info_filesize_label.setText(filesize)
        self.info_num_reads_label.setText(num_reads)
        self.info_sample_rate_label.setText(sample_rate)
        self.info_total_samples_label.setText(total_samples)


class PlotOptionsPanel(QWidget):
    """Panel for plot mode and normalization controls"""

    # Signals
    plot_mode_changed = Signal(PlotMode)
    normalization_changed = Signal(NormalizationMethod)
    base_annotations_toggled = Signal(bool)
    signal_points_toggled = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.collapsible_box = CollapsibleBox("Plot Options")
        self._create_ui()

    def _create_ui(self):
        """Create the plot options content layout"""
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(10, 5, 10, 5)
        content_layout.setSpacing(10)

        # Plot mode selection
        mode_label = QLabel("Plot Mode:")
        mode_label.setStyleSheet("font-weight: bold;")
        content_layout.addWidget(mode_label)

        self.mode_button_group = QButtonGroup(self)

        # Event-aligned mode (top - default when BAM is loaded)
        self.mode_eventalign = QRadioButton("Event-aligned (base annotations)")
        self.mode_eventalign.setToolTip(
            "Show event-aligned reads with base annotations (requires BAM file)"
        )
        self.mode_eventalign.toggled.connect(
            lambda checked: self.plot_mode_changed.emit(PlotMode.EVENTALIGN) if checked else None
        )
        self.mode_eventalign.setChecked(
            True
        )  # Checked by default (visual indication of primary mode)
        self.mode_eventalign.setEnabled(False)  # Disabled until BAM file loaded
        self.mode_button_group.addButton(self.mode_eventalign)
        content_layout.addWidget(self.mode_eventalign)

        # Aggregate mode (multi-read pileup)
        self.mode_aggregate = QRadioButton("Aggregate (multi-read pileup)")
        self.mode_aggregate.setToolTip(
            "Show aggregate signal with base pileup and quality tracks (requires BAM file)"
        )
        self.mode_aggregate.toggled.connect(
            lambda checked: self.plot_mode_changed.emit(PlotMode.AGGREGATE) if checked else None
        )
        self.mode_aggregate.setEnabled(False)  # Disabled until BAM file loaded
        self.mode_button_group.addButton(self.mode_aggregate)
        content_layout.addWidget(self.mode_aggregate)

        self.mode_overlay = QRadioButton("Overlay (multiple reads)")
        self.mode_overlay.setToolTip(
            f"Overlay multiple reads on same axes (max {MAX_OVERLAY_READS})"
        )
        self.mode_overlay.toggled.connect(
            lambda checked: self.plot_mode_changed.emit(PlotMode.OVERLAY) if checked else None
        )
        self.mode_button_group.addButton(self.mode_overlay)
        content_layout.addWidget(self.mode_overlay)

        self.mode_stacked = QRadioButton("Stacked (squigualiser-style)")
        self.mode_stacked.setToolTip("Stack multiple reads vertically with offset")
        self.mode_stacked.toggled.connect(
            lambda checked: self.plot_mode_changed.emit(PlotMode.STACKED) if checked else None
        )
        self.mode_button_group.addButton(self.mode_stacked)
        content_layout.addWidget(self.mode_stacked)

        # Single read mode (bottom - fallback when no BAM)
        self.mode_single = QRadioButton("Single Read")
        self.mode_single.setToolTip("Display one read at a time")
        self.mode_single.toggled.connect(
            lambda checked: self.plot_mode_changed.emit(PlotMode.SINGLE) if checked else None
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
        self.norm_combo.currentIndexChanged.connect(self._on_norm_changed)
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
        self.base_checkbox.stateChanged.connect(
            lambda state: self.base_annotations_toggled.emit(state == Qt.Checked)
        )
        content_layout.addWidget(self.base_checkbox)

        # Signal points toggle
        self.points_checkbox = QCheckBox("Show signal points")
        self.points_checkbox.setChecked(False)  # Unchecked by default
        self.points_checkbox.setToolTip(
            "Display individual signal data points as circles on the squiggle line"
        )
        self.points_checkbox.stateChanged.connect(
            lambda state: self.signal_points_toggled.emit(state == Qt.Checked)
        )
        content_layout.addWidget(self.points_checkbox)

        # Info label
        info_label = QLabel(
            "💡 Tip: Use Ctrl/Cmd+Click or Shift+Click to select multiple reads"
        )
        info_label.setStyleSheet("color: #666; font-size: 9pt;")
        info_label.setWordWrap(True)
        content_layout.addWidget(info_label)

        content_layout.addStretch()

        self.collapsible_box.set_content_layout(content_layout)

        # Set panel layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.collapsible_box)

        # Set plot options to be expanded by default
        self.collapsible_box.toggle_button.setChecked(True)
        self.collapsible_box.on_toggle()

    def _on_norm_changed(self, index):
        """Handle normalization method change"""
        method = self.norm_combo.itemData(index)
        self.normalization_changed.emit(method)

    def set_bam_controls_enabled(self, enabled):
        """Enable/disable controls that require BAM file"""
        self.mode_eventalign.setEnabled(enabled)
        self.mode_aggregate.setEnabled(enabled)
        self.base_checkbox.setEnabled(enabled)

    def set_plot_mode(self, mode):
        """Set the current plot mode"""
        if mode == PlotMode.EVENTALIGN:
            self.mode_eventalign.setChecked(True)
        elif mode == PlotMode.AGGREGATE:
            self.mode_aggregate.setChecked(True)
        elif mode == PlotMode.OVERLAY:
            self.mode_overlay.setChecked(True)
        elif mode == PlotMode.STACKED:
            self.mode_stacked.setChecked(True)
        elif mode == PlotMode.SINGLE:
            self.mode_single.setChecked(True)


class AdvancedOptionsPanel(QWidget):
    """Panel for advanced plot settings"""

    # Signals
    downsample_changed = Signal(int)
    dwell_time_toggled = Signal(bool)
    position_interval_changed = Signal(int)
    position_type_toggled = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.collapsible_box = CollapsibleBox("Advanced")
        self._create_ui()

    def _create_ui(self):
        """Create the advanced options content layout"""
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
        self.downsample_slider.valueChanged.connect(self._on_slider_changed)
        downsample_layout.addWidget(self.downsample_slider)

        # SpinBox for direct value entry
        self.downsample_spinbox = QSpinBox()
        self.downsample_spinbox.setMinimum(1)
        self.downsample_spinbox.setMaximum(100)
        self.downsample_spinbox.setValue(25)
        self.downsample_spinbox.setToolTip("Enter downsample factor (1-100)")
        self.downsample_spinbox.valueChanged.connect(self._on_spinbox_changed)
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
        self.dwell_time_checkbox.stateChanged.connect(
            lambda state: self.dwell_time_toggled.emit(state == Qt.Checked)
        )
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
            lambda value: self.position_interval_changed.emit(value)
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
        self.position_type_checkbox.stateChanged.connect(
            lambda state: self.position_type_toggled.emit(state == Qt.Checked)
        )
        content_layout.addWidget(self.position_type_checkbox)

        # Position label info
        position_info_label = QLabel(
            "💡 Position labels show numbers on base annotations"
        )
        position_info_label.setStyleSheet("color: #666; font-size: 9pt;")
        position_info_label.setWordWrap(True)
        content_layout.addWidget(position_info_label)

        content_layout.addStretch()

        self.collapsible_box.set_content_layout(content_layout)

        # Set panel layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.collapsible_box)

    def _on_slider_changed(self, value):
        """Handle slider value change - update spinbox and emit signal"""
        # Block spinbox signals to avoid circular updates
        self.downsample_spinbox.blockSignals(True)
        self.downsample_spinbox.setValue(value)
        self.downsample_spinbox.blockSignals(False)
        self.downsample_changed.emit(value)

    def _on_spinbox_changed(self, value):
        """Handle spinbox value change - update slider and emit signal"""
        # Block slider signals to avoid circular updates
        self.downsample_slider.blockSignals(True)
        self.downsample_slider.setValue(value)
        self.downsample_slider.blockSignals(False)
        self.downsample_changed.emit(value)

    def set_dwell_time_enabled(self, enabled):
        """Enable/disable dwell time checkbox"""
        self.dwell_time_checkbox.setEnabled(enabled)

    def set_position_type_enabled(self, enabled):
        """Enable/disable position type checkbox"""
        self.position_type_checkbox.setEnabled(enabled)


class SearchPanel(QWidget):
    """Panel for searching reads by ID, region, or sequence"""

    # Signals
    search_mode_changed = Signal(str)
    search_requested = Signal()
    reference_browse_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._create_ui()

    def _create_ui(self):
        """Create the search panel UI"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # Main search controls row
        search_layout = QHBoxLayout()

        # Search mode selector
        self.search_mode_combo = QComboBox()
        self.search_mode_combo.addItem("Read ID", "read_id")
        self.search_mode_combo.addItem("Reference Region", "region")
        self.search_mode_combo.addItem("Sequence", "sequence")
        self.search_mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.search_mode_combo.setToolTip(
            "Switch between searching by read ID, genomic region, or sequence"
        )
        search_layout.addWidget(QLabel("Search by:"))
        search_layout.addWidget(self.search_mode_combo)

        # Search input
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search read ID...")
        self.search_input.returnPressed.connect(lambda: self.search_requested.emit())
        search_layout.addWidget(self.search_input, 1)

        # Search button
        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(lambda: self.search_requested.emit())
        search_layout.addWidget(self.search_button)

        # Browse references button (for region search mode)
        self.browse_refs_button = QPushButton("Browse References...")
        self.browse_refs_button.clicked.connect(lambda: self.reference_browse_requested.emit())
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

        main_layout.addLayout(search_layout)

    def _on_mode_changed(self, index):
        """Handle search mode change"""
        mode = self.search_mode_combo.itemData(index)
        self.search_mode_changed.emit(mode)

        # Update placeholder text and visibility based on mode
        if mode == "read_id":
            self.search_input.setPlaceholderText("Search read ID...")
            self.search_input.setToolTip("Filter reads by read ID (case-insensitive)")
            self.browse_refs_button.setVisible(False)
            self.revcomp_checkbox.setVisible(False)
        elif mode == "region":
            self.search_input.setPlaceholderText("e.g., chr1:1000-2000 or chr1")
            self.search_input.setToolTip(
                "Search reads by genomic region (requires BAM file)\n"
                "Format: chr1, chr1:1000, or chr1:1000-2000"
            )
            self.browse_refs_button.setVisible(True)
            self.revcomp_checkbox.setVisible(False)
        else:  # sequence
            self.search_input.setPlaceholderText("e.g., ATCGATCG")
            self.search_input.setToolTip(
                "Search for a DNA sequence in the reference (requires BAM file)\n"
                "Enter sequence in 5' to 3' direction"
            )
            self.browse_refs_button.setVisible(False)
            self.revcomp_checkbox.setVisible(True)

    def get_search_mode(self):
        """Get the current search mode"""
        return self.search_mode_combo.currentData()

    def get_search_text(self):
        """Get the current search text"""
        return self.search_input.text()

    def set_search_text(self, text):
        """Set the search input text"""
        self.search_input.setText(text)

    def clear_search(self):
        """Clear the search input"""
        self.search_input.clear()

    def set_browse_enabled(self, enabled):
        """Enable/disable browse references button"""
        self.browse_refs_button.setEnabled(enabled)

    def is_revcomp_checked(self):
        """Check if reverse complement checkbox is checked"""
        return self.revcomp_checkbox.isChecked()
