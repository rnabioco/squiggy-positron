"""Command-line interface for headless plot export"""

from pathlib import Path

import pod5
from bokeh.io import export_png, export_svgs
from rich.console import Console

from .constants import NormalizationMethod, PlotMode, Theme
from .plotter import SquigglePlotter
from .utils import get_basecall_data

# Create Rich console for styled output
console = Console()


def export_plot(args) -> int:
    """Export plot to file without launching GUI (headless mode)

    Args:
        args: Parsed command-line arguments from argparse

    Returns:
        Exit code: 0 for success, 1 for error
    """
    try:
        # Validate and load POD5 file
        pod5_path = Path(args.pod5).resolve()
        if not pod5_path.exists():
            console.print(f"[red]Error:[/red] POD5 file does not exist: {pod5_path}")
            return 1

        # Determine export format
        export_path = Path(args.export)
        if args.export_format:
            export_format = args.export_format
        else:
            # Infer from file extension
            ext = export_path.suffix.lower()
            format_map = {".html": "html", ".png": "png", ".svg": "svg"}
            export_format = format_map.get(ext, "html")

        # Parse normalization method
        norm_map = {
            "none": NormalizationMethod.NONE,
            "znorm": NormalizationMethod.ZNORM,
            "median": NormalizationMethod.MEDIAN,
            "mad": NormalizationMethod.MAD,
        }
        normalization = norm_map.get(args.normalization, NormalizationMethod.MEDIAN)

        # Parse theme
        theme = Theme.DARK if args.theme == "dark" else Theme.LIGHT

        # Load POD5 file and extract read(s)
        console.print(f"[cyan]Loading POD5 file:[/cyan] {pod5_path}")
        with pod5.Reader(pod5_path) as reader:
            # Get all read IDs
            all_read_ids = {str(read.read_id): read for read in reader.reads()}

            # Determine which read(s) to plot
            if args.read_id:
                read_ids = [args.read_id]
            elif args.reads:
                read_ids = args.reads
            else:
                console.print(
                    "[red]Error:[/red] --read-id or --reads required for export"
                )
                return 1

            # Validate read IDs exist
            missing_reads = [rid for rid in read_ids if rid not in all_read_ids]
            if missing_reads:
                console.print(
                    f"[red]Error:[/red] Read ID(s) not found in POD5 file: {missing_reads}"
                )
                return 1

            # Load BAM data if provided
            bam_data = {}
            if args.bam:
                bam_path = Path(args.bam).resolve()
                if not bam_path.exists():
                    console.print(
                        f"[red]Error:[/red] BAM file does not exist: {bam_path}"
                    )
                    return 1
                console.print(f"[cyan]Loading BAM file:[/cyan] {bam_path}")
                for read_id in read_ids:
                    sequence, seq_to_sig_map = get_basecall_data(bam_path, read_id)
                    if sequence is not None:
                        bam_data[read_id] = (sequence, seq_to_sig_map)

            # Generate plot
            console.print(
                f"[cyan]Generating plot for[/cyan] {len(read_ids)} [cyan]read(s)...[/cyan]"
            )

            if len(read_ids) == 1:
                # Single read plot
                read_id = read_ids[0]
                read = all_read_ids[read_id]
                signal = read.signal
                sample_rate = read.run_info.sample_rate

                # Get basecall data if available
                sequence = None
                seq_to_sig_map = None
                if read_id in bam_data:
                    sequence, seq_to_sig_map = bam_data[read_id]

                # Determine if we should show base labels
                show_labels = True
                if args.no_show_bases:
                    show_labels = False
                elif args.show_bases:
                    show_labels = True
                elif not args.bam:
                    # Default to not showing labels if no BAM
                    show_labels = False

                html, fig = SquigglePlotter.plot_single_read(
                    signal=signal,
                    read_id=read_id,
                    sample_rate=sample_rate,
                    sequence=sequence,
                    seq_to_sig_map=seq_to_sig_map,
                    normalization=normalization,
                    downsample=args.downsample,
                    show_dwell_time=args.dwell_time,
                    show_labels=show_labels,
                    show_signal_points=args.show_points,
                    position_label_interval=args.position_interval,
                    use_reference_positions=args.reference_positions,
                    theme=theme,
                )
            else:
                # Multiple reads plot
                reads_data = []
                for read_id in read_ids:
                    read = all_read_ids[read_id]
                    reads_data.append((read_id, read.signal, read.run_info.sample_rate))

                # Parse plot mode
                mode_map = {
                    "single": PlotMode.SINGLE,
                    "overlay": PlotMode.OVERLAY,
                    "stacked": PlotMode.STACKED,
                    "eventalign": PlotMode.EVENTALIGN,
                }
                plot_mode = (
                    mode_map.get(args.mode, PlotMode.OVERLAY)
                    if args.mode
                    else PlotMode.OVERLAY
                )

                html, fig = SquigglePlotter.plot_multiple_reads(
                    reads_data=reads_data,
                    mode=plot_mode,
                    normalization=normalization,
                    downsample=args.downsample,
                    theme=theme,
                )

            # Export to file
            console.print(f"[cyan]Exporting plot to:[/cyan] {export_path}")
            if export_format == "html":
                with open(export_path, "w") as f:
                    f.write(html)
            elif export_format == "png":
                # Set figure dimensions for export
                fig.plot_width = args.export_width
                fig.plot_height = args.export_height
                export_png(fig, filename=str(export_path))
            elif export_format == "svg":
                # Set figure dimensions for export
                fig.plot_width = args.export_width
                fig.plot_height = args.export_height
                svgs = export_svgs(fig, filename=str(export_path))
                # export_svgs returns list of SVG strings, write first one
                with open(export_path, "w") as f:
                    f.write(svgs[0])

            console.print(
                f"[green]✓[/green] Successfully exported {export_format.upper()} plot to: {export_path}"
            )
            return 0

    except Exception as e:
        console.print(f"[red]Error during export:[/red] {e}")
        import traceback

        traceback.print_exc()
        return 1
