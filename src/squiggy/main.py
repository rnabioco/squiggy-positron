"""Main entry point for Squiggy GUI application"""

import argparse
import asyncio
import sys
from pathlib import Path

import qasync
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QMessageBox

from .constants import APP_DESCRIPTION, APP_NAME, APP_VERSION, PlotMode
from .utils import get_icon_path, validate_bam_reads_in_pod5
from .viewer import SquiggleViewer


def set_macos_app_name():
    """Set the macOS application name (fixes 'Python' appearing in menu bar)"""
    try:
        from Foundation import NSBundle

        bundle = NSBundle.mainBundle()
        if bundle:
            info = bundle.localizedInfoDictionary() or bundle.infoDictionary()
            if info:
                info["CFBundleName"] = APP_NAME
    except ImportError:
        # PyObjC not available (not on macOS or not installed)
        pass


async def main_async(app, viewer, args):
    """Async initialization for the application"""
    # Auto-load POD5 file if specified
    if args.pod5:
        file_path = Path(args.pod5)
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
                return
        else:
            QMessageBox.warning(
                viewer, "File Not Found", f"POD5 file does not exist:\n{file_path}"
            )
            return

    # Auto-load BAM file if specified
    if args.bam:
        bam_path = Path(args.bam)
        if bam_path.exists():
            if not args.pod5 or not viewer.pod5_file:
                QMessageBox.warning(
                    viewer,
                    "POD5 File Required",
                    "Cannot load BAM file without a POD5 file.\n"
                    "Please specify a POD5 file with --file.",
                )
                return

            try:
                # Validate that BAM reads exist in POD5
                viewer.statusBar().showMessage("Validating BAM file against POD5...")
                validation_result = await asyncio.to_thread(
                    validate_bam_reads_in_pod5, bam_path, viewer.pod5_file
                )

                if not validation_result["is_valid"]:
                    error_msg = (
                        f"BAM validation failed!\n\n"
                        f"Found {validation_result['bam_read_count']} reads in BAM file.\n"
                        f"Found {validation_result['pod5_read_count']} reads in POD5 file.\n"
                        f"{validation_result['missing_count']} BAM reads are NOT in POD5 file.\n\n"
                        f"This indicates a serious mismatch between files."
                    )
                    if validation_result["missing_reads"]:
                        # Show first few missing reads as examples
                        examples = list(validation_result["missing_reads"])[:5]
                        error_msg += "\n\nExample missing reads:\n" + "\n".join(
                            f"  - {r}" for r in examples
                        )

                    QMessageBox.critical(viewer, "BAM Validation Failed", error_msg)
                    return

                # Validation passed, load BAM file
                viewer.bam_file = bam_path
                viewer.bam_label.setText(bam_path.name)
                viewer.base_checkbox.setEnabled(True)
                viewer.base_checkbox.setChecked(True)  # Check by default
                viewer.mode_eventalign.setEnabled(True)
                viewer.mode_eventalign.setChecked(True)  # Switch to event-aligned mode
                viewer.plot_mode = PlotMode.EVENTALIGN  # Explicitly sync internal state
                viewer.dwell_time_checkbox.setEnabled(True)  # Enable dwell time option
                viewer.statusBar().showMessage(
                    f"Loaded and validated BAM file: {bam_path.name} "
                    f"({validation_result['bam_read_count']} reads)"
                )

            except Exception as e:
                QMessageBox.critical(
                    viewer, "Error", f"Failed to load BAM file:\n{str(e)}"
                )
        else:
            QMessageBox.warning(
                viewer, "File Not Found", f"BAM file does not exist:\n{bam_path}"
            )


def main():
    """Main entry point for Squiggy GUI application"""
    parser = argparse.ArgumentParser(
        description=f"{APP_NAME} - {APP_DESCRIPTION}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  squiggy                                      Launch GUI
  squiggy --pod5 data.pod5                     Launch GUI with POD5 file pre-loaded
  squiggy --pod5 data.pod5 --bam calls.bam     Launch GUI with POD5 and BAM files
  squiggy --version                            Show version information

Use File → Open Sample Data in the GUI to load bundled sample data.

Version: {APP_VERSION}
        """,
    )
    parser.add_argument("--pod5", "-p", type=str, help="POD5 file to open on startup")
    parser.add_argument(
        "--bam", "-b", type=str, help="BAM file with basecalls (optional)"
    )
    parser.add_argument(
        "--version", "-v", action="version", version=f"{APP_NAME} {APP_VERSION}"
    )

    args = parser.parse_args()

    # Set macOS app name (must be done before creating QApplication)
    set_macos_app_name()

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
