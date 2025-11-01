"""Modifications panel for Squiggy viewer"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..constants import (
    DEFAULT_MOD_OVERLAY_OPACITY,
    DEFAULT_MOD_THRESHOLD,
    MOD_OVERLAY_MAX_OPACITY,
    MOD_OVERLAY_MIN_OPACITY,
    MOD_SCOPE_ANY,
    MOD_SCOPE_POSITION,
    MOD_THRESHOLD_MAX,
    MOD_THRESHOLD_MIN,
    MOD_THRESHOLD_STEP,
    MODIFICATION_CODES,
    MODIFICATION_COLORS,
)
from ..widgets import CollapsibleBox


class ModificationsPanel(QWidget):
    """Panel for modification (modBAM) visualization controls"""

    # Signals emitted when settings change
    modification_overlay_toggled = Signal(bool)
    mod_type_filter_changed = Signal(str)
    overlay_opacity_changed = Signal(float)
    threshold_changed = Signal(float)
    classification_scope_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._detected_mods = set()  # Set of (canonical_base, mod_code) tuples
        self._provenance = None  # Provenance dict from detect_modification_provenance()
        self._create_ui()
        # Hide panel initially (shown when modifications detected)
        self.hide()

    def _create_color_icon(self, color_hex: str, size: int = 16) -> QIcon:
        """Create a colored square icon for the modification color

        Args:
            color_hex: Hex color string (e.g., "#FF0000")
            size: Size of the icon in pixels

        Returns:
            QIcon with solid color square
        """
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw colored square with border
        color = QColor(color_hex)
        painter.fillRect(0, 0, size, size, color)
        painter.setPen(QColor("#666666"))
        painter.drawRect(0, 0, size - 1, size - 1)
        painter.end()

        return QIcon(pixmap)

    def _create_ui(self):
        """Create the modifications panel UI"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(10)

        # ==============================================================================
        # Section 1: Status (always visible when panel shown)
        # ==============================================================================
        status_box = CollapsibleBox("Modifications (MM/ML)")

        status_layout = QVBoxLayout()
        status_layout.setContentsMargins(10, 5, 10, 5)
        status_layout.setSpacing(5)

        # Detected modifications with color legend
        label_detected = QLabel("Detected modifications:")
        label_detected.setStyleSheet("font-size: 9pt; font-weight: bold;")
        status_layout.addWidget(label_detected)

        self.detected_mods_label = QLabel("—")
        self.detected_mods_label.setStyleSheet("font-size: 9pt; padding-left: 10px;")
        self.detected_mods_label.setWordWrap(True)
        status_layout.addWidget(self.detected_mods_label)

        # Color legend (no separate header, flows from detected mods)
        self.color_legend_widget = QWidget()
        self.color_legend_layout = QGridLayout(self.color_legend_widget)
        self.color_legend_layout.setContentsMargins(10, 5, 10, 5)
        self.color_legend_layout.setSpacing(8)
        status_layout.addWidget(self.color_legend_widget)
        # Initially hide until mods are detected
        self.color_legend_widget.setVisible(False)

        # Provenance label
        label_provenance = QLabel("Basecaller:")
        label_provenance.setStyleSheet("font-size: 9pt; font-weight: bold;")
        status_layout.addWidget(label_provenance)

        self.provenance_label = QLabel("—")
        self.provenance_label.setStyleSheet("font-size: 9pt; padding-left: 10px;")
        self.provenance_label.setWordWrap(True)
        self.provenance_label.setToolTip(
            "Modification calling provenance from BAM header"
        )
        status_layout.addWidget(self.provenance_label)

        # Status message
        self.status_message_label = QLabel(
            "Thresholding enabled (exploratory visualization only)"
        )
        self.status_message_label.setStyleSheet(
            "font-size: 9pt; font-style: italic; color: #D55E00; padding-left: 10px;"
        )
        self.status_message_label.setWordWrap(True)
        status_layout.addWidget(self.status_message_label)

        status_box.set_content_layout(status_layout)
        # Expand by default
        status_box.toggle_button.setChecked(True)
        status_box.on_toggle()
        main_layout.addWidget(status_box)

        # ==============================================================================
        # Display Controls (directly in main layout, no separate box)
        # ==============================================================================

        # Add separator line
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(separator)

        # Controls container
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(10, 10, 10, 5)
        controls_layout.setSpacing(8)

        # Show modification overlay checkbox
        self.show_overlay_checkbox = QCheckBox("Show modification overlay")
        self.show_overlay_checkbox.setChecked(True)
        self.show_overlay_checkbox.setToolTip(
            "Display modification probabilities as visual overlays on the plot"
        )
        self.show_overlay_checkbox.toggled.connect(self.modification_overlay_toggled)
        controls_layout.addWidget(self.show_overlay_checkbox)

        # Overlay opacity slider
        opacity_label = QLabel("Overlay opacity:")
        opacity_label.setStyleSheet("font-size: 9pt;")
        controls_layout.addWidget(opacity_label)

        opacity_sublayout = QHBoxLayout()
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setMinimum(int(MOD_OVERLAY_MIN_OPACITY * 100))
        self.opacity_slider.setMaximum(int(MOD_OVERLAY_MAX_OPACITY * 100))
        self.opacity_slider.setValue(int(DEFAULT_MOD_OVERLAY_OPACITY * 100))
        self.opacity_slider.setTickPosition(QSlider.TicksBelow)
        self.opacity_slider.setTickInterval(10)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        opacity_sublayout.addWidget(self.opacity_slider)

        self.opacity_value_label = QLabel(f"{DEFAULT_MOD_OVERLAY_OPACITY:.1f}")
        self.opacity_value_label.setStyleSheet("font-size: 9pt;")
        self.opacity_value_label.setMinimumWidth(30)
        opacity_sublayout.addWidget(self.opacity_value_label)

        controls_layout.addLayout(opacity_sublayout)

        # Modification type filter dropdown
        filter_label = QLabel("Filter modification types:")
        filter_label.setStyleSheet("font-size: 9pt;")
        controls_layout.addWidget(filter_label)

        self.mod_type_combo = QComboBox()
        self.mod_type_combo.addItem("All modifications", "all")
        # Additional mod types will be added dynamically when mods are detected
        self.mod_type_combo.currentIndexChanged.connect(
            lambda: self.mod_type_filter_changed.emit(self.mod_type_combo.currentData())
        )
        controls_layout.addWidget(self.mod_type_combo)

        # Add spacing before threshold section
        controls_layout.addSpacing(10)

        # Probability threshold slider (always visible)
        tau_label = QLabel("Probability threshold (τ):")
        tau_label.setStyleSheet("font-size: 9pt; font-weight: bold;")
        tau_label.setToolTip(
            "Filter modifications by probability threshold.\n\n"
            "This is for exploratory visualization only - NOT for making biological conclusions.\n"
            "Use it to:\n"
            "• Explore your data interactively\n"
            "• Understand modification patterns\n"
            "• Generate hypotheses for downstream analysis\n\n"
            "For publication-quality modification calls, use specialized tools."
        )
        controls_layout.addWidget(tau_label)

        tau_sublayout = QHBoxLayout()
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(int(MOD_THRESHOLD_MIN * 100))
        self.threshold_slider.setMaximum(int(MOD_THRESHOLD_MAX * 100))
        self.threshold_slider.setValue(int(DEFAULT_MOD_THRESHOLD * 100))
        self.threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.threshold_slider.setTickInterval(int(MOD_THRESHOLD_STEP * 100))
        self.threshold_slider.valueChanged.connect(self._on_threshold_changed)
        tau_sublayout.addWidget(self.threshold_slider)

        self.threshold_value_label = QLabel(f"{DEFAULT_MOD_THRESHOLD:.2f}")
        self.threshold_value_label.setStyleSheet("font-size: 9pt;")
        self.threshold_value_label.setMinimumWidth(40)
        tau_sublayout.addWidget(self.threshold_value_label)

        controls_layout.addLayout(tau_sublayout)

        # Classification scope radio buttons
        scope_label = QLabel("Classification scope:")
        scope_label.setStyleSheet("font-size: 9pt; font-weight: bold;")
        controls_layout.addWidget(scope_label)

        self.scope_button_group = QButtonGroup(self)
        self.scope_position_radio = QRadioButton("At selected position (per-position)")
        self.scope_position_radio.setChecked(True)
        self.scope_position_radio.setToolTip(
            "Classify each position independently based on probability at that position"
        )
        self.scope_button_group.addButton(self.scope_position_radio)
        controls_layout.addWidget(self.scope_position_radio)

        self.scope_any_radio = QRadioButton("Any site in visible region")
        self.scope_any_radio.setToolTip(
            "Classify entire read as modified if ANY modification >= τ in visible region"
        )
        self.scope_button_group.addButton(self.scope_any_radio)
        controls_layout.addWidget(self.scope_any_radio)

        self.scope_button_group.buttonToggled.connect(self._on_scope_changed)

        # Statistics display
        stats_label = QLabel("Classification statistics:")
        stats_label.setStyleSheet("font-size: 9pt; font-weight: bold;")
        controls_layout.addWidget(stats_label)

        self.stats_label = QLabel("—")
        self.stats_label.setStyleSheet(
            "font-size: 9pt; padding-left: 10px; font-family: Monaco, Menlo, Consolas, 'Courier New', monospace;"
        )
        self.stats_label.setWordWrap(True)
        controls_layout.addWidget(self.stats_label)

        main_layout.addWidget(controls_widget)

        # Add stretch to push everything to the top
        main_layout.addStretch()

    def _on_opacity_changed(self, value):
        """Handle opacity slider change"""
        opacity = value / 100.0
        self.opacity_value_label.setText(f"{opacity:.1f}")
        self.overlay_opacity_changed.emit(opacity)

    def _on_threshold_changed(self, value):
        """Handle threshold slider change"""
        tau = value / 100.0
        self.threshold_value_label.setText(f"{tau:.2f}")
        self.threshold_changed.emit(tau)

    def _on_scope_changed(self):
        """Handle classification scope change"""
        if self.scope_position_radio.isChecked():
            self.classification_scope_changed.emit(MOD_SCOPE_POSITION)
        else:
            self.classification_scope_changed.emit(MOD_SCOPE_ANY)

    def set_detected_modifications(self, mods: set):
        """Set the detected modification types

        Args:
            mods: Set of (canonical_base, mod_code) tuples
                mod_code can be a string (e.g., 'm', 'a') or int (ChEBI code)
        """
        self._detected_mods = mods

        # Format display string
        if not mods:
            display_text = "None detected"
            self.color_legend_widget.setVisible(False)
        else:
            mod_names = []
            for canonical, mod_code in sorted(mods, key=lambda x: (x[0], str(x[1]))):
                mod_name = MODIFICATION_CODES.get(mod_code, str(mod_code))
                # Format: "modification_name (base+code)"
                # e.g., "5mC (C+m)" or "inosine (A+17596)"
                mod_names.append(f"{mod_name} ({canonical}+{mod_code})")
            display_text = ", ".join(mod_names)

            # Update color legend
            self._update_color_legend(mods)
            self.color_legend_widget.setVisible(True)

        self.detected_mods_label.setText(display_text)

        # Update mod type filter dropdown with colored icons
        self.mod_type_combo.clear()
        self.mod_type_combo.addItem("All modifications", "all")
        for canonical, mod_code in sorted(mods, key=lambda x: (x[0], str(x[1]))):
            mod_name = MODIFICATION_CODES.get(mod_code, str(mod_code))
            label = f"{mod_name} ({canonical}+{mod_code})"

            # Get modification color and create icon
            mod_color = MODIFICATION_COLORS.get(
                mod_code, MODIFICATION_COLORS["default"]
            )
            icon = self._create_color_icon(mod_color, size=16)

            # Store as string for consistent lookup
            self.mod_type_combo.addItem(icon, label, f"{canonical}+{mod_code}")

    def _update_color_legend(self, mods: set):
        """Update the color legend grid with detected modifications

        Args:
            mods: Set of (canonical_base, mod_code) tuples
        """
        # Clear existing legend items
        while self.color_legend_layout.count():
            item = self.color_legend_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Add color boxes and labels in a grid (2 columns)
        row = 0
        col = 0
        max_cols = 2

        for canonical, mod_code in sorted(mods, key=lambda x: (x[0], str(x[1]))):
            # Get mod name and color
            mod_name = MODIFICATION_CODES.get(mod_code, str(mod_code))
            color = MODIFICATION_COLORS.get(mod_code, MODIFICATION_COLORS["default"])

            # Create colored box (QFrame with background color)
            color_box = QFrame()
            color_box.setFixedSize(16, 16)
            color_box.setStyleSheet(
                f"background-color: {color}; border: 1px solid #666;"
            )

            # Create label
            label_text = f"{mod_name} ({canonical}+{mod_code})"
            label = QLabel(label_text)
            label.setStyleSheet("font-size: 8pt;")

            # Add to grid
            self.color_legend_layout.addWidget(color_box, row, col * 2)
            self.color_legend_layout.addWidget(label, row, col * 2 + 1)

            # Move to next position
            col += 1
            if col >= max_cols:
                col = 0
                row += 1

    def set_provenance(self, provenance: dict):
        """Set the modification calling provenance

        Args:
            provenance: Dict from detect_modification_provenance()
        """
        self._provenance = provenance

        if provenance["unknown"]:
            display_text = "Unknown (probabilities only)"
        else:
            display_text = f"{provenance['basecaller']} v{provenance['version']}"
            if provenance["model"] != "Unknown":
                display_text += f"\nModel: {provenance['model']}"

        self.provenance_label.setText(display_text)

        # Update tooltip with full info if available
        if provenance["full_info"]:
            self.provenance_label.setToolTip(
                f"Full command line:\n{provenance['full_info']}"
            )

    def update_statistics(self, n_modified: int, n_unmodified: int, coverage: int):
        """Update the classification statistics display

        Args:
            n_modified: Number of reads/positions classified as modified
            n_unmodified: Number of reads/positions classified as unmodified
            coverage: Total coverage
        """
        if coverage == 0:
            self.stats_label.setText("—")
            return

        pct_modified = (n_modified / coverage) * 100
        pct_unmodified = (n_unmodified / coverage) * 100

        stats_text = (
            f"Modified:   {n_modified:4d} ({pct_modified:5.1f}%)\n"
            f"Unmodified: {n_unmodified:4d} ({pct_unmodified:5.1f}%)\n"
            f"Total:      {coverage:4d}"
        )
        self.stats_label.setText(stats_text)

    def get_threshold(self) -> float:
        """Get current threshold value"""
        return self.threshold_slider.value() / 100.0

    def get_opacity(self) -> float:
        """Get current opacity value"""
        return self.opacity_slider.value() / 100.0

    def get_mod_type_filter(self) -> str:
        """Get current modification type filter"""
        return self.mod_type_combo.currentData()

    def is_threshold_enabled(self) -> bool:
        """Check if thresholding is enabled"""
        # Thresholding is now always enabled
        return True

    def is_overlay_enabled(self) -> bool:
        """Check if modification overlay is enabled"""
        return self.show_overlay_checkbox.isChecked()

    def get_classification_scope(self) -> str:
        """Get current classification scope"""
        if self.scope_position_radio.isChecked():
            return MOD_SCOPE_POSITION
        return MOD_SCOPE_ANY
