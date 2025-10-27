"""Dialog windows for Squiggy application"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
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
            name_label = QLabel(f"<h1>{APP_NAME}</h1>")
            name_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(name_label)

        # Version info
        version_label = QLabel(f"<p>Version {APP_VERSION}</p>")
        version_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(version_label)

        # Description
        desc_label = QLabel(
            f"<p>{APP_DESCRIPTION}</p>"
            "<p>A desktop application for visualizing<br>"
            "Oxford Nanopore sequencing data.</p>"
        )
        desc_label.setAlignment(Qt.AlignCenter)
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)

        # Features list
        features_label = QLabel(
            '<p style="margin-top: 20px;">'
            "<b>Features:</b><br>"
            "• POD5 file visualization<br>"
            "• Optional BAM base annotations<br>"
            "• Interactive squiggle plots<br>"
            "• Color-coded base visualization<br>"
            "</p>"
        )
        features_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(features_label)

        # Technology info
        tech_label = QLabel(
            '<p style="margin-top: 10px; font-size: 10px; color: #666;">'
            "Built with PySide6, plotnine, and pod5"
            "</p>"
        )
        tech_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(tech_label)

        # Add spacer
        layout.addStretch()

        # Close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        close_button.setDefault(True)
        layout.addWidget(close_button)


class ReferenceBrowserDialog(QDialog):
    """Dialog for browsing available reference sequences in BAM file"""

    def __init__(self, references, parent=None):
        """Initialize reference browser dialog

        Args:
            references: List of reference info dicts from get_bam_references()
            parent: Parent widget
        """
        super().__init__(parent)
        self.references = references
        self.selected_reference = None

        self.setWindowTitle("BAM Reference Sequences")
        self.setModal(True)
        self.setMinimumSize(700, 500)

        layout = QVBoxLayout(self)

        # Info label
        info_label = QLabel(
            f"<b>Available references in BAM file</b><br>"
            f"Found {len(references)} reference sequences. "
            "Click a row to select, then click 'Use Selected' to search."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Search filter
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter:"))
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Type to filter references...")
        self.filter_input.textChanged.connect(self.filter_table)
        filter_layout.addWidget(self.filter_input)
        layout.addLayout(filter_layout)

        # Create table
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Reference", "Length (bp)", "Reads"])
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSortingEnabled(True)
        self.table.itemDoubleClicked.connect(self.on_double_click)

        # Populate table
        self.populate_table(references)

        # Adjust column widths
        self.table.resizeColumnsToContents()
        self.table.setColumnWidth(0, 300)  # Reference name column

        layout.addWidget(self.table)

        # Summary label
        self.summary_label = QLabel()
        self.update_summary()
        layout.addWidget(self.summary_label)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.use_button = QPushButton("Use Selected")
        self.use_button.clicked.connect(self.on_use_selected)
        self.use_button.setEnabled(False)
        button_layout.addWidget(self.use_button)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.reject)
        button_layout.addWidget(close_button)

        layout.addLayout(button_layout)

        # Connect selection change
        self.table.itemSelectionChanged.connect(self.on_selection_changed)

    def populate_table(self, references):
        """Populate table with reference data"""
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(references))

        for row, ref in enumerate(references):
            # Reference name
            name_item = QTableWidgetItem(ref["name"])
            self.table.setItem(row, 0, name_item)

            # Length
            if ref["length"]:
                length_str = f"{ref['length']:,}"
            else:
                length_str = "—"
            length_item = QTableWidgetItem(length_str)
            length_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.table.setItem(row, 1, length_item)

            # Read count
            if ref["read_count"] is not None:
                count_str = f"{ref['read_count']:,}"
            else:
                count_str = "—"
            count_item = QTableWidgetItem(count_str)
            count_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.table.setItem(row, 2, count_item)

        self.table.setSortingEnabled(True)
        # Sort by read count descending (most reads first)
        self.table.sortItems(2, Qt.DescendingOrder)

    def filter_table(self):
        """Filter table rows based on search text"""
        filter_text = self.filter_input.text().lower()

        for row in range(self.table.rowCount()):
            name_item = self.table.item(row, 0)
            if name_item:
                # Show row if filter text is in reference name
                should_show = filter_text in name_item.text().lower()
                self.table.setRowHidden(row, not should_show)

        self.update_summary()

    def update_summary(self):
        """Update summary label with visible/total counts"""
        visible_count = sum(
            1 for row in range(self.table.rowCount()) if not self.table.isRowHidden(row)
        )
        total_count = self.table.rowCount()

        if visible_count == total_count:
            self.summary_label.setText(f"Showing {total_count} references")
        else:
            self.summary_label.setText(
                f"Showing {visible_count} of {total_count} references"
            )

    def on_selection_changed(self):
        """Handle selection change"""
        selected_items = self.table.selectedItems()
        self.use_button.setEnabled(len(selected_items) > 0)

    def on_use_selected(self):
        """Handle 'Use Selected' button click"""
        selected_rows = self.table.selectionModel().selectedRows()
        if selected_rows:
            row = selected_rows[0].row()
            name_item = self.table.item(row, 0)
            if name_item:
                self.selected_reference = name_item.text()
                self.accept()

    def on_double_click(self, item):
        """Handle double-click on table row"""
        row = item.row()
        name_item = self.table.item(row, 0)
        if name_item:
            self.selected_reference = name_item.text()
            self.accept()
