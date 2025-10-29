"""Reusable Qt widgets for Squiggy UI"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)


class CollapsibleBox(QWidget):
    """A collapsible widget with a toggle button to show/hide content"""

    def __init__(self, title="", parent=None):
        super().__init__(parent)

        self.toggle_button = QToolButton()
        self.toggle_button.setText(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(False)
        self.toggle_button.setStyleSheet(
            "QToolButton { border: none; font-weight: bold; }"
        )
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.RightArrow)
        self.toggle_button.clicked.connect(self.on_toggle)

        self.content_area = QScrollArea()
        self.content_area.setMaximumHeight(0)
        self.content_area.setMinimumHeight(0)
        self.content_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.content_area.setFrameShape(QFrame.NoFrame)

        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toggle_button)
        layout.addWidget(self.content_area)

    def on_toggle(self):
        """Toggle the collapsible section"""
        checked = self.toggle_button.isChecked()
        self.toggle_button.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        if checked:
            self.content_area.setMaximumHeight(350)
        else:
            self.content_area.setMaximumHeight(0)

    def set_content_layout(self, layout):
        """Set the layout for the content area"""
        widget = QWidget()
        widget.setLayout(layout)
        self.content_area.setWidget(widget)


class ReadTreeWidget(QTreeWidget):
    """Custom tree widget for displaying reads grouped by reference"""

    def __init__(self, parent=None):
        super().__init__(parent)

        # Configure tree widget
        self.setHeaderHidden(True)  # Hide column header
        self.setSelectionMode(QTreeWidget.ExtendedSelection)  # Multi-select
        self.setAlternatingRowColors(True)
        self.setIndentation(15)  # Indent child items

        # Store read-to-reference mapping
        self.read_to_reference = {}
        self.reference_items = {}  # Map reference name -> QTreeWidgetItem

    def populate_with_reads(self, reads_by_reference):
        """
        Populate tree with reads grouped by reference.

        Args:
            reads_by_reference: Dict mapping reference name -> list of read IDs
                Example: {"chr1": ["read1", "read2"], "chr2": ["read3"]}
        """
        self.clear()
        self.read_to_reference.clear()
        self.reference_items.clear()

        # Create top-level items for each reference
        for reference_name, read_ids in sorted(reads_by_reference.items()):
            # Create reference header item
            ref_item = QTreeWidgetItem(self)
            ref_item.setText(0, f"{reference_name} ({len(read_ids)} reads)")
            ref_item.setData(0, Qt.UserRole, reference_name)  # Store reference name
            ref_item.setData(0, Qt.UserRole + 1, "reference")  # Mark as reference item
            ref_item.setExpanded(False)  # Start collapsed

            self.reference_items[reference_name] = ref_item

            # Add read items as children
            for read_id in read_ids:
                read_item = QTreeWidgetItem(ref_item)
                read_item.setText(0, read_id)
                read_item.setData(0, Qt.UserRole, read_id)  # Store read ID
                read_item.setData(0, Qt.UserRole + 1, "read")  # Mark as read item

                # Track mapping
                self.read_to_reference[read_id] = reference_name

    def get_selected_read_ids(self):
        """
        Get all selected read IDs, expanding reference selections to include all reads.

        Returns:
            List of read IDs (strings)
        """
        read_ids = []
        selected_items = self.selectedItems()

        for item in selected_items:
            item_type = item.data(0, Qt.UserRole + 1)

            if item_type == "reference":
                # Reference selected: get all child reads
                for i in range(item.childCount()):
                    child = item.child(i)
                    read_id = child.data(0, Qt.UserRole)
                    if read_id and read_id not in read_ids:
                        read_ids.append(read_id)

            elif item_type == "read":
                # Individual read selected
                read_id = item.data(0, Qt.UserRole)
                if read_id and read_id not in read_ids:
                    read_ids.append(read_id)

        return read_ids

    def filter_by_read_id(self, search_text):
        """
        Filter tree to show only reads matching search text.

        Args:
            search_text: String to search for in read IDs (case-insensitive)
        """
        search_lower = search_text.lower()

        for reference_name, ref_item in self.reference_items.items():
            visible_children = 0

            # Check each child read
            for i in range(ref_item.childCount()):
                child = ref_item.child(i)
                read_id = child.data(0, Qt.UserRole)

                # Show/hide based on match
                matches = search_lower in read_id.lower() if search_text else True
                child.setHidden(not matches)

                if matches:
                    visible_children += 1

            # Hide reference if no children match
            ref_item.setHidden(visible_children == 0)

            # Update count in reference label
            if visible_children > 0:
                total_children = ref_item.childCount()
                if search_text:
                    ref_item.setText(0, f"{reference_name} ({visible_children}/{total_children} reads)")
                else:
                    ref_item.setText(0, f"{reference_name} ({total_children} reads)")

    def iter_all_read_items(self):
        """
        Iterate over all read items in the tree (not reference headers).

        Yields:
            QTreeWidgetItem: Each read item (child of reference items)
        """
        for ref_item in self.reference_items.values():
            for i in range(ref_item.childCount()):
                yield ref_item.child(i)

    def get_read_item_by_id(self, read_id):
        """
        Find a read item by its read ID.

        Args:
            read_id: The read ID to search for

        Returns:
            QTreeWidgetItem or None: The read item if found, None otherwise
        """
        for read_item in self.iter_all_read_items():
            if read_item.data(0, Qt.UserRole) == read_id:
                return read_item
        return None

    def show_all_reads(self):
        """Show all reads and references (clear any filters)"""
        for reference_name, ref_item in self.reference_items.items():
            ref_item.setHidden(False)
            total_children = ref_item.childCount()
            ref_item.setText(0, f"{reference_name} ({total_children} reads)")

            for i in range(ref_item.childCount()):
                child = ref_item.child(i)
                child.setHidden(False)

    def filter_by_region(self, reads_in_region_dict):
        """
        Filter tree to show only reads in a specific region.

        Args:
            reads_in_region_dict: Dict mapping read_id -> alignment info
                                 Only reads with keys in this dict will be shown
        """
        reads_found = set(reads_in_region_dict.keys())
        visible_count = 0

        for reference_name, ref_item in self.reference_items.items():
            visible_children = 0

            for i in range(ref_item.childCount()):
                child = ref_item.child(i)
                read_id = child.data(0, Qt.UserRole)

                if read_id in reads_found:
                    child.setHidden(False)
                    visible_children += 1
                    visible_count += 1

                    # Update item text to show alignment info
                    aln_info = reads_in_region_dict[read_id]
                    child.setText(0,
                        f"{read_id} [{aln_info['chromosome']}:"
                        f"{aln_info['start']}-{aln_info['end']} "
                        f"{aln_info['strand']}]"
                    )
                else:
                    child.setHidden(True)

            # Hide reference if no children visible
            ref_item.setHidden(visible_children == 0)

            # Update count in reference label
            if visible_children > 0:
                total_children = ref_item.childCount()
                ref_item.setText(0, f"{reference_name} ({visible_children}/{total_children} reads)")

        return visible_count
