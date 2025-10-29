"""Dialog windows for Squiggy application"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from .constants import APP_DESCRIPTION, APP_NAME, APP_VERSION
from .utils import get_logo_path


class AboutDialog(QDialog):
    """About dialog showing application information and logo"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"About {APP_NAME}")
        self.setModal(True)
        self.setFixedSize(400, 500)

        layout = QVBoxLayout(self)

        # Add logo if available
        logo_path = get_logo_path()
        if logo_path:
            logo_label = QLabel()
            pixmap = QPixmap(str(logo_path))
            # Scale logo to fit nicely in dialog
            scaled_pixmap = pixmap.scaled(
                200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            logo_label.setPixmap(scaled_pixmap)
            logo_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(logo_label)
        else:
            # Fallback if logo not found
            logo_label = QLabel(f"<h1>{APP_NAME}</h1>")
            logo_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(logo_label)

        # Application name (if logo was shown)
        if logo_path:
            name_label = QLabel(f"<h2>{APP_NAME}</h2>")
            name_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(name_label)

        # Version
        version_label = QLabel(f"Version {APP_VERSION}")
        version_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(version_label)

        # Description
        desc_label = QLabel(APP_DESCRIPTION)
        desc_label.setAlignment(Qt.AlignCenter)
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)

        # License and credits
        credits_text = """
        <p><b>License:</b> MIT</p>
        <p><b>Author:</b> Jay Hesselberth</p>
        <p><b>GitHub:</b> <a href="https://github.com/rnabioco/squiggy">
        github.com/rnabioco/squiggy</a></p>
        """
        credits_label = QLabel(credits_text)
        credits_label.setAlignment(Qt.AlignCenter)
        credits_label.setWordWrap(True)
        credits_label.setOpenExternalLinks(True)
        layout.addWidget(credits_label)

        # Close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)


class ReferenceBrowserDialog(QDialog):
    """Dialog for browsing reference sequences in BAM file"""

    def __init__(self, references, parent=None):
        """Initialize reference browser dialog

        Args:
            references: List of dict with keys: name, length, read_count
            parent: Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle("Browse Reference Sequences")
        self.setModal(True)
        self.resize(600, 400)

        self.references = references  # Store full reference data
        self.selected_reference = None  # Just the name string (for backward compatibility)
        self.selected_reference_dict = None  # Full reference dict

        layout = QVBoxLayout(self)

        # Info label
        info_label = QLabel(
            "Double-click a reference sequence to select it for region search:"
        )
        layout.addWidget(info_label)

        # Create table
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Reference", "Length", "Reads"])
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.itemDoubleClicked.connect(self.on_double_click)

        # Populate table
        self.table.setRowCount(len(references))
        for row, ref in enumerate(references):
            # Reference name
            name_item = QTableWidgetItem(ref["name"])
            self.table.setItem(row, 0, name_item)

            # Length
            length_item = QTableWidgetItem(f"{ref['length']:,}")
            length_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.table.setItem(row, 1, length_item)

            # Read count
            count_item = QTableWidgetItem(f"{ref['read_count']:,}")
            count_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.table.setItem(row, 2, count_item)

        # Resize columns to contents
        self.table.resizeColumnsToContents()

        layout.addWidget(self.table)

        # Search box
        search_layout = QHBoxLayout()
        search_label = QLabel("Filter:")
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Type to filter references...")
        self.search_box.textChanged.connect(self.filter_table)
        search_layout.addWidget(search_label)
        search_layout.addWidget(self.search_box)
        layout.addLayout(search_layout)

        # Buttons
        button_box = QHBoxLayout()

        select_button = QPushButton("Select")
        select_button.clicked.connect(self.on_select)
        button_box.addWidget(select_button)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_box.addWidget(cancel_button)

        layout.addLayout(button_box)

    def filter_table(self, text):
        """Filter table rows based on search text"""
        for row in range(self.table.rowCount()):
            name_item = self.table.item(row, 0)
            if name_item:
                # Show row if search text is in reference name (case insensitive)
                should_show = text.lower() in name_item.text().lower()
                self.table.setRowHidden(row, not should_show)

    def on_select(self):
        """Handle select button click"""
        selected_rows = self.table.selectedItems()
        if selected_rows:
            # Get the reference name from the first column of selected row
            row = selected_rows[0].row()
            name_item = self.table.item(row, 0)
            if name_item:
                ref_name = name_item.text()
                self.selected_reference = ref_name
                # Find and store the full reference dict
                self.selected_reference_dict = next(
                    (ref for ref in self.references if ref["name"] == ref_name),
                    None
                )
                self.accept()

    def on_double_click(self, item):
        """Handle double-click on table row"""
        row = item.row()
        name_item = self.table.item(row, 0)
        if name_item:
            ref_name = name_item.text()
            self.selected_reference = ref_name
            # Find and store the full reference dict
            self.selected_reference_dict = next(
                (ref for ref in self.references if ref["name"] == ref_name),
                None
            )
            self.accept()

    def get_selected_reference(self):
        """Get the selected reference as a dictionary

        Returns:
            dict: Selected reference with keys: name, length, read_count
            None: If no reference selected
        """
        return self.selected_reference_dict


class ExportDialog(QDialog):
    """Dialog for exporting plots with format and dimension options"""

    def __init__(self, parent=None):
        """Initialize export dialog

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle("Export Plot")
        self.setModal(True)
        self.resize(400, 300)

        layout = QVBoxLayout(self)

        # Format selection group
        format_group = QGroupBox("Export Format")
        format_layout = QVBoxLayout()

        self.format_combo = QComboBox()
        self.format_combo.addItem("HTML (Interactive)", "html")
        self.format_combo.addItem("PNG (Raster Image)", "png")
        self.format_combo.addItem("SVG (Vector Graphics)", "svg")
        self.format_combo.currentIndexChanged.connect(self.on_format_changed)

        format_layout.addWidget(self.format_combo)

        # Warning label for PNG/SVG
        self.warning_label = QLabel()
        self.warning_label.setWordWrap(True)
        self.warning_label.setStyleSheet("color: #ff6600; font-style: italic;")
        self.warning_label.setVisible(False)
        format_layout.addWidget(self.warning_label)

        format_group.setLayout(format_layout)
        layout.addWidget(format_group)

        # Dimensions group
        dimensions_group = QGroupBox("Dimensions (for PNG/SVG)")
        dimensions_layout = QFormLayout()

        # Add info label about dimensions
        info_label = QLabel(
            "Note: Dimensions include plot area + margins.\n"
            "Recommended: width ≥ 1200px, height ≥ 800px"
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #666666; font-size: 10px; font-style: italic;")
        dimensions_layout.addRow(info_label)

        # Width
        self.width_spinbox = QSpinBox()
        self.width_spinbox.setRange(100, 10000)
        self.width_spinbox.setValue(1400)
        self.width_spinbox.setSuffix(" px")
        self.width_spinbox.valueChanged.connect(self.on_width_changed)
        dimensions_layout.addRow("Width:", self.width_spinbox)

        # Height
        self.height_spinbox = QSpinBox()
        self.height_spinbox.setRange(100, 10000)
        self.height_spinbox.setValue(900)
        self.height_spinbox.setSuffix(" px")
        self.height_spinbox.valueChanged.connect(self.on_height_changed)
        dimensions_layout.addRow("Height:", self.height_spinbox)

        # Aspect ratio lock
        self.aspect_lock = QCheckBox("Lock Aspect Ratio")
        self.aspect_lock.setChecked(True)
        dimensions_layout.addRow("", self.aspect_lock)

        dimensions_group.setLayout(dimensions_layout)
        layout.addWidget(dimensions_group)

        self.dimensions_group = dimensions_group
        self.dimensions_group.setEnabled(False)  # Initially disabled for HTML

        # View options group
        view_group = QGroupBox("Export Range")
        view_layout = QVBoxLayout()

        self.export_current_view = QCheckBox(
            "Export current zoom level (visible range only)"
        )
        self.export_current_view.setChecked(False)
        self.export_current_view.setToolTip(
            "When checked, exports only the currently visible range.\n"
            "When unchecked, exports the full plot range."
        )
        view_layout.addWidget(self.export_current_view)

        view_group.setLayout(view_layout)
        layout.addWidget(view_group)

        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.aspect_ratio = 1400 / 900  # Default aspect ratio

    def on_format_changed(self):
        """Handle format selection change"""
        format_type = self.format_combo.currentData()
        is_image = format_type in ("png", "svg")

        # Enable/disable dimensions controls
        self.dimensions_group.setEnabled(is_image)

        # Check if export dependencies are available for PNG/SVG
        if is_image:
            try:
                import selenium  # noqa: F401

                self.warning_label.setVisible(False)
            except ImportError:
                self.warning_label.setText(
                    "Note: PNG/SVG export requires additional dependencies. "
                    "Install with: uv pip install selenium pillow"
                )
                self.warning_label.setVisible(True)
        else:
            self.warning_label.setVisible(False)

    def on_width_changed(self, value):
        """Handle width change with aspect ratio lock"""
        if self.aspect_lock.isChecked():
            self.height_spinbox.blockSignals(True)
            self.height_spinbox.setValue(int(value / self.aspect_ratio))
            self.height_spinbox.blockSignals(False)

    def on_height_changed(self, value):
        """Handle height change with aspect ratio lock"""
        if self.aspect_lock.isChecked():
            self.width_spinbox.blockSignals(True)
            self.width_spinbox.setValue(int(value * self.aspect_ratio))
            self.width_spinbox.blockSignals(False)

    def get_export_settings(self):
        """Get export settings from dialog

        Returns:
            dict with keys: format, width, height, use_current_view
        """
        return {
            "format": self.format_combo.currentData(),
            "width": self.width_spinbox.value(),
            "height": self.height_spinbox.value(),
            "use_current_view": self.export_current_view.isChecked(),
        }
