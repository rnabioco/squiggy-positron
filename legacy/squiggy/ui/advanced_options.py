"""Advanced options panel for Squiggy viewer"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..widgets import CollapsibleBox


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
        self.dwell_time_checkbox.toggled.connect(self.dwell_time_toggled.emit)
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
        self.position_type_checkbox.toggled.connect(self.position_type_toggled.emit)
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
