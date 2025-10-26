"""Dialog windows for Squiggy application"""

from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap

from .constants import APP_NAME, APP_VERSION, APP_DESCRIPTION
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
            "• Remora-style base coloring<br>"
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
