"""Plot options panel for Squiggy viewer"""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QLabel,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from ..constants import MAX_OVERLAY_READS, NormalizationMethod, PlotMode
from ..widgets import CollapsibleBox


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
            lambda checked: self.plot_mode_changed.emit(PlotMode.EVENTALIGN)
            if checked
            else None
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
            lambda checked: self.plot_mode_changed.emit(PlotMode.AGGREGATE)
            if checked
            else None
        )
        self.mode_aggregate.setEnabled(False)  # Disabled until BAM file loaded
        self.mode_button_group.addButton(self.mode_aggregate)
        content_layout.addWidget(self.mode_aggregate)

        self.mode_overlay = QRadioButton("Overlay (multiple reads)")
        self.mode_overlay.setToolTip(
            f"Overlay multiple reads on same axes (max {MAX_OVERLAY_READS})"
        )
        self.mode_overlay.toggled.connect(
            lambda checked: self.plot_mode_changed.emit(PlotMode.OVERLAY)
            if checked
            else None
        )
        self.mode_button_group.addButton(self.mode_overlay)
        content_layout.addWidget(self.mode_overlay)

        self.mode_stacked = QRadioButton("Stacked (squigualiser-style)")
        self.mode_stacked.setToolTip("Stack multiple reads vertically with offset")
        self.mode_stacked.toggled.connect(
            lambda checked: self.plot_mode_changed.emit(PlotMode.STACKED)
            if checked
            else None
        )
        self.mode_button_group.addButton(self.mode_stacked)
        content_layout.addWidget(self.mode_stacked)

        # Single read mode (bottom - fallback when no BAM)
        self.mode_single = QRadioButton("Single Read")
        self.mode_single.setToolTip("Display one read at a time")
        self.mode_single.toggled.connect(
            lambda checked: self.plot_mode_changed.emit(PlotMode.SINGLE)
            if checked
            else None
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
        self.base_checkbox.toggled.connect(self.base_annotations_toggled.emit)
        content_layout.addWidget(self.base_checkbox)

        # Signal points toggle
        self.points_checkbox = QCheckBox("Show signal points")
        self.points_checkbox.setChecked(False)  # Unchecked by default
        self.points_checkbox.setToolTip(
            "Display individual signal data points as circles on the squiggle line"
        )
        self.points_checkbox.toggled.connect(self.signal_points_toggled.emit)
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
