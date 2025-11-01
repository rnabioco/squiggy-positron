"""Main entry point for Squiggy GUI application"""

import argparse
import asyncio
import sys
from pathlib import Path

import qasync
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QMessageBox

from squiggy.constants import (
    APP_DESCRIPTION,
    APP_NAME,
    APP_VERSION,
    NormalizationMethod,
    PlotMode,
)
from squiggy.utils import get_icon_path


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

    # Lazy import validate_bam_reads_in_pod5 only when needed
    from squiggy.utils import validate_bam_reads_in_pod5

    # Auto-load POD5 file if specified
    if args.pod5:
        # Convert to absolute path immediately to avoid issues with CWD changes
        file_path = Path(args.pod5).resolve()
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
        # Convert to absolute path immediately to avoid issues with CWD changes
        bam_path = Path(args.bam).resolve()
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

                # Enable BAM-dependent features using panel methods
                viewer.plot_options_panel.set_bam_controls_enabled(True)
                viewer.plot_options_panel.set_plot_mode(PlotMode.EVENTALIGN)
                viewer.plot_mode = PlotMode.EVENTALIGN  # Explicitly sync internal state
                viewer.advanced_options_panel.set_dwell_time_enabled(True)
                viewer.advanced_options_panel.set_position_type_enabled(True)

                # Enable browse references button if in region search mode
                if viewer.search_panel.get_search_mode() == "region":
                    viewer.search_panel.set_browse_enabled(True)

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

    # Apply CLI settings to viewer
    if args.pod5:  # Only apply settings if files are loaded
        # Set normalization method using panel
        norm_map = {
            "none": (0, NormalizationMethod.NONE),
            "znorm": (1, NormalizationMethod.ZNORM),
            "median": (2, NormalizationMethod.MEDIAN),
            "mad": (3, NormalizationMethod.MAD),
        }
        if args.normalization in norm_map:
            idx, method = norm_map[args.normalization]
            viewer.plot_options_panel.norm_combo.setCurrentIndex(idx)
            viewer.normalization_method = method

        # Set plot mode (if specified or default based on BAM) using panel method
        if args.mode:
            mode_map = {
                "single": PlotMode.SINGLE,
                "overlay": PlotMode.OVERLAY,
                "stacked": PlotMode.STACKED,
                "eventalign": PlotMode.EVENTALIGN,
                "aggregate": PlotMode.AGGREGATE,
            }
            if args.mode in mode_map:
                mode = mode_map[args.mode]
                viewer.plot_options_panel.set_plot_mode(mode)
                viewer.plot_mode = mode
        elif args.bam:
            # Default to eventalign if BAM provided and no mode specified
            viewer.plot_options_panel.set_plot_mode(PlotMode.EVENTALIGN)
            viewer.plot_mode = PlotMode.EVENTALIGN

        # Set base annotations visibility using panel
        if args.no_show_bases:
            viewer.plot_options_panel.base_checkbox.setChecked(False)
        elif args.show_bases:
            viewer.plot_options_panel.base_checkbox.setChecked(True)
        elif args.bam and not args.mode:
            # Default to showing bases if BAM loaded
            viewer.plot_options_panel.base_checkbox.setChecked(True)

        # Set signal points visibility using panel
        if args.show_points:
            viewer.plot_options_panel.points_checkbox.setChecked(True)

        # Set dwell time scaling using advanced panel
        if args.dwell_time:
            if viewer.advanced_options_panel.dwell_time_checkbox.isEnabled():
                viewer.advanced_options_panel.dwell_time_checkbox.setChecked(True)

        # Set downsample factor using advanced panel
        viewer.advanced_options_panel.downsample_slider.setValue(args.downsample)
        viewer.advanced_options_panel.downsample_spinbox.setValue(args.downsample)

        # Set position label interval using advanced panel
        viewer.advanced_options_panel.position_interval_spinbox.setValue(
            args.position_interval
        )

        # Set reference positions using advanced panel
        if args.reference_positions:
            if viewer.advanced_options_panel.position_type_checkbox.isEnabled():
                viewer.advanced_options_panel.position_type_checkbox.setChecked(True)

    # Auto-select and display read(s) or reference if specified
    if args.read_id and viewer.pod5_file:
        # Single read ID
        items = viewer.read_list.findItems(args.read_id, Qt.MatchExactly)
        if items:
            viewer.read_list.setCurrentItem(items[0])
            await viewer.display_squiggle()
        else:
            QMessageBox.warning(
                viewer,
                "Read Not Found",
                f"Read ID '{args.read_id}' not found in POD5 file",
            )
    elif args.reads and viewer.pod5_file:
        # Multiple read IDs
        viewer.read_list.clearSelection()
        for read_id in args.reads:
            items = viewer.read_list.findItems(read_id, Qt.MatchExactly)
            if items:
                items[0].setSelected(True)
        if viewer.read_list.selectedItems():
            await viewer.display_squiggle()
        else:
            QMessageBox.warning(
                viewer,
                "Reads Not Found",
                "None of the specified read IDs were found in POD5 file",
            )
    elif args.region and viewer.pod5_file and viewer.bam_file:
        # Genomic region-based read selection
        viewer.statusBar().showMessage(f"Searching region: {args.region}...")
        try:
            # Use the viewer's search manager to filter by region
            await viewer.search_manager.filter_by_region(args.region)
            if viewer.read_list.count() > 0:
                # Auto-display if reads were found
                if viewer.plot_mode in [
                    PlotMode.OVERLAY,
                    PlotMode.STACKED,
                    PlotMode.EVENTALIGN,
                ]:
                    # For multi-read modes, select all filtered reads (up to a reasonable limit)
                    max_reads = 10 if viewer.plot_mode == PlotMode.OVERLAY else 100
                    for i in range(min(viewer.read_list.count(), max_reads)):
                        viewer.read_list.item(i).setSelected(True)
                else:
                    # For single mode, select the first read
                    viewer.read_list.setCurrentRow(0)
                await viewer.display_squiggle()
            else:
                QMessageBox.warning(
                    viewer,
                    "No Reads Found",
                    f"No reads found in region: {args.region}",
                )
        except Exception as e:
            QMessageBox.critical(
                viewer,
                "Region Search Failed",
                f"Failed to search region '{args.region}':\n{str(e)}",
            )
    elif args.reference and viewer.pod5_file and viewer.bam_file:
        # Reference sequence for aggregate mode
        viewer.statusBar().showMessage(
            f"Loading aggregate view for: {args.reference}..."
        )
        try:
            # Set selected reference and display aggregate
            viewer.selected_reference = args.reference
            await viewer.display_aggregate()
        except Exception as e:
            QMessageBox.critical(
                viewer,
                "Aggregate Display Failed",
                f"Failed to display aggregate for reference '{args.reference}':\n{str(e)}",
            )


def main():
    """Main entry point for Squiggy GUI application"""
    parser = argparse.ArgumentParser(
        description=f"{APP_NAME} - {APP_DESCRIPTION}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Basic usage
  squiggy                                             Launch GUI
  squiggy --pod5 data.pod5                            Launch GUI with POD5 file pre-loaded
  squiggy --pod5 data.pod5 --bam calls.bam            Launch GUI with POD5 and BAM files

  # With specific reads
  squiggy -p data.pod5 -b calls.bam --mode eventalign --read-id READ_ID
  squiggy -p data.pod5 --read-id READ_ID --show-points --downsample 10
  squiggy -p data.pod5 -b calls.bam --mode overlay --region "chr1:1000-2000"

  # Aggregate mode
  squiggy -p data.pod5 -b calls.bam --mode aggregate --reference "chr1"

  # Headless export (no GUI)
  squiggy -p data.pod5 --read-id READ_ID --export plot.html
  squiggy -p data.pod5 -b calls.bam --read-id READ_ID --export plot.png

  # Other
  squiggy --version                                   Show version information

Use File → Open Sample Data in the GUI to load bundled sample data.

Version: {APP_VERSION}
        """,
    )

    # File inputs
    file_group = parser.add_argument_group("File Options")
    file_group.add_argument(
        "--pod5", "-p", type=str, help="POD5 file to open on startup"
    )
    file_group.add_argument(
        "--bam", "-b", type=str, help="BAM file with basecalls (optional)"
    )

    # Plot mode and normalization
    plot_group = parser.add_argument_group("Plot Options")
    plot_group.add_argument(
        "--mode",
        "-m",
        type=str,
        choices=["single", "overlay", "stacked", "eventalign", "aggregate"],
        help="Plot mode (default: eventalign if BAM provided, else single)",
    )
    plot_group.add_argument(
        "--normalization",
        "-n",
        type=str,
        choices=["none", "znorm", "median", "mad"],
        default="median",
        help="Signal normalization method (default: median)",
    )

    # Visualization options
    viz_group = parser.add_argument_group("Visualization Options")
    viz_group.add_argument(
        "--show-bases",
        action="store_true",
        default=None,
        help="Show base annotations (when BAM file provided, default: True)",
    )
    viz_group.add_argument(
        "--no-show-bases",
        action="store_true",
        help="Hide base annotations",
    )
    viz_group.add_argument(
        "--show-points",
        action="store_true",
        help="Show individual signal data points (default: False)",
    )
    viz_group.add_argument(
        "--dwell-time",
        action="store_true",
        help="Scale x-axis by dwell time (requires eventalign mode with BAM)",
    )
    viz_group.add_argument(
        "--downsample",
        type=int,
        default=25,
        metavar="N",
        help="Downsample factor: show every Nth signal point (1-100, default: 25)",
    )
    viz_group.add_argument(
        "--position-interval",
        type=int,
        default=10,
        metavar="N",
        help="Show position label every N bases (default: 10)",
    )
    viz_group.add_argument(
        "--reference-positions",
        action="store_true",
        help="Use reference genomic coordinates (requires BAM file)",
    )

    # Read selection
    read_group = parser.add_argument_group("Read Selection")
    read_group.add_argument(
        "--read-id",
        type=str,
        help="Auto-select and display specific read ID",
    )
    read_group.add_argument(
        "--reads",
        type=str,
        nargs="+",
        help="Auto-select multiple read IDs (for overlay/stacked modes)",
    )
    read_group.add_argument(
        "--region",
        type=str,
        help="Select reads by genomic region (e.g., 'chr1:1000-2000', requires BAM file)",
    )
    read_group.add_argument(
        "--reference",
        type=str,
        help="Select reference sequence for aggregate mode (requires BAM file)",
    )

    # Export options
    export_group = parser.add_argument_group("Export Options (Headless Mode)")
    export_group.add_argument(
        "--export",
        "-e",
        type=str,
        metavar="FILE",
        help="Export plot to file and exit (no GUI). Format inferred from extension (.html, .png, .svg). Requires --pod5 and --read-id",
    )
    export_group.add_argument(
        "--export-width",
        type=int,
        default=1200,
        metavar="PX",
        help="Export width in pixels for PNG/SVG (default: 1200)",
    )
    export_group.add_argument(
        "--export-height",
        type=int,
        default=600,
        metavar="PX",
        help="Export height in pixels for PNG/SVG (default: 600)",
    )

    # Theme
    theme_group = parser.add_argument_group("Theme Options")
    theme_group.add_argument(
        "--theme",
        type=str,
        choices=["light", "dark"],
        default="dark",
        help="Application theme (default: dark)",
    )

    # Expert options
    expert_group = parser.add_argument_group("Expert Options")
    expert_group.add_argument(
        "--window-width",
        type=int,
        default=1200,
        metavar="PX",
        help="Initial window width in pixels (default: 1200)",
    )
    expert_group.add_argument(
        "--window-height",
        type=int,
        default=800,
        metavar="PX",
        help="Initial window height in pixels (default: 800)",
    )

    # Version
    parser.add_argument(
        "--version", "-v", action="version", version=f"{APP_NAME} {APP_VERSION}"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.export:
        # Export mode requires POD5 file and read ID(s)
        if not args.pod5:
            parser.error("--export requires --pod5")
        if not args.read_id and not args.reads:
            parser.error("--export requires --read-id or --reads")

    # Handle conflicting --show-bases / --no-show-bases
    if args.show_bases and args.no_show_bases:
        parser.error("Cannot use both --show-bases and --no-show-bases")

    # Validate downsample range
    if args.downsample < 1 or args.downsample > 100:
        parser.error("--downsample must be between 1 and 100")

    # Validate read selection parameters are mutually exclusive
    read_selection_args = [args.read_id, args.reads, args.region, args.reference]
    if sum(bool(arg) for arg in read_selection_args) > 1:
        parser.error(
            "Only one of --read-id, --reads, --region, or --reference can be specified"
        )

    # Validate --region requires BAM file
    if args.region and not args.bam:
        parser.error("--region requires --bam file")

    # Validate --reference requires BAM file
    if args.reference and not args.bam:
        parser.error("--reference requires --bam file")

    # Validate --reference requires aggregate mode (or set it as default)
    if args.reference and args.mode and args.mode != "aggregate":
        parser.error("--reference can only be used with --mode aggregate")
    elif args.reference and not args.mode:
        # Auto-set mode to aggregate if reference is specified
        args.mode = "aggregate"

    # Handle headless export mode (no GUI)
    if args.export:
        # Lazy import export_plot only when needed (CLI mode)
        from squiggy.cli import export_plot

        return export_plot(args)

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

    # Create main viewer window BEFORE applying theme for faster initial display
    # Lazy import SquiggleViewer to defer pod5/pysam imports
    from squiggy.viewer import SquiggleViewer

    viewer = SquiggleViewer(
        window_width=args.window_width, window_height=args.window_height
    )

    # Set window icon (in addition to app icon)
    if icon_path:
        viewer.setWindowIcon(QIcon(str(icon_path)))

    # Show window BEFORE theme application for perceived faster startup
    viewer.show()

    # Apply qt-material theme after window is shown for better perceived performance
    # This allows the window to appear faster, then theme applies
    from qt_material import apply_stylesheet

    extra = {"density_scale": "-2"}  # Maximum compactness
    theme_name = "dark_amber.xml" if args.theme == "dark" else "light_amber.xml"
    apply_stylesheet(app, theme=theme_name, extra=extra)

    # Setup qasync event loop
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)

    # Run async initialization
    with loop:
        loop.create_task(main_async(app, viewer, args))
        loop.run_forever()


if __name__ == "__main__":
    main()
