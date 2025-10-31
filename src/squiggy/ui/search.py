"""Search panel for Squiggy viewer"""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


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
        self.browse_refs_button.clicked.connect(
            lambda: self.reference_browse_requested.emit()
        )
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
