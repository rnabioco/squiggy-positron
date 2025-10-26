"""Main entry point for Squiggy GUI application"""

import argparse
import asyncio
import sys
from pathlib import Path

import qasync
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QMessageBox

from .constants import APP_DESCRIPTION, APP_NAME, APP_VERSION
from .utils import get_icon_path
from .viewer import SquiggleViewer


async def main_async(app, viewer, args):
    """Async initialization for the application"""
    # Auto-load file if specified
    if args.file:
        file_path = Path(args.file)
        if file_path.exists():
            try:
                viewer.pod5_file = file_path
                viewer.file_label.setText(file_path.name)
                await viewer.load_read_ids()
                viewer.statusBar().showMessage(f"Loaded {len(viewer.read_dict)} reads")
            except Exception as e:
                QMessageBox.critical(
                    viewer, "Error", f"Failed to load POD5 file:\n{str(e)}"
                )
        else:
            QMessageBox.warning(
                viewer, "File Not Found", f"File does not exist:\n{file_path}"
            )


def main():
    """Main entry point for Squiggy GUI application"""
    parser = argparse.ArgumentParser(
        description=f"{APP_NAME} - {APP_DESCRIPTION}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  squiggy                          Launch GUI
  squiggy --file data.pod5         Launch GUI with file pre-loaded
  squiggy --version                Show version information

Use File → Open Sample Data in the GUI to load bundled sample data.

Version: {APP_VERSION}
        """,
    )
    parser.add_argument("--file", "-f", type=str, help="POD5 file to open on startup")
    parser.add_argument(
        "--version", "-v", action="version", version=f"{APP_NAME} {APP_VERSION}"
    )

    args = parser.parse_args()

    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)

    # Set application icon if available
    icon_path = get_icon_path()
    if icon_path:
        app.setWindowIcon(QIcon(str(icon_path)))

    # Create main viewer window
    viewer = SquiggleViewer()

    # Set window icon (in addition to app icon)
    if icon_path:
        viewer.setWindowIcon(QIcon(str(icon_path)))

    # Show window
    viewer.show()

    # Setup qasync event loop
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)

    # Run async initialization
    with loop:
        loop.create_task(main_async(app, viewer, args))
        loop.run_forever()


if __name__ == "__main__":
    main()
