"""Tests for CLI functionality (main.py argument parsing and cli.py export)"""

import argparse
from unittest.mock import MagicMock

import pytest


class TestArgumentParsing:
    """Tests for command-line argument parsing in main.py"""

    @pytest.fixture
    def parser(self):
        """Create argument parser for testing"""
        # Import and recreate parser setup from main.py
        from squiggy.constants import APP_DESCRIPTION, APP_NAME

        parser = argparse.ArgumentParser(
            description=f"{APP_NAME} - {APP_DESCRIPTION}",
        )

        # File inputs
        file_group = parser.add_argument_group("File Options")
        file_group.add_argument("--pod5", "-p", type=str)
        file_group.add_argument("--bam", "-b", type=str)

        # Plot mode and normalization
        plot_group = parser.add_argument_group("Plot Options")
        plot_group.add_argument(
            "--mode",
            "-m",
            type=str,
            choices=["single", "overlay", "stacked", "eventalign"],
        )
        plot_group.add_argument(
            "--normalization",
            "-n",
            type=str,
            choices=["none", "znorm", "median", "mad"],
            default="median",
        )

        # Visualization options
        viz_group = parser.add_argument_group("Visualization Options")
        viz_group.add_argument("--show-bases", action="store_true", default=None)
        viz_group.add_argument("--no-show-bases", action="store_true")
        viz_group.add_argument("--show-points", action="store_true")
        viz_group.add_argument("--dwell-time", action="store_true")
        viz_group.add_argument("--downsample", type=int, default=25)
        viz_group.add_argument("--position-interval", type=int, default=10)
        viz_group.add_argument("--reference-positions", action="store_true")

        # Read selection
        read_group = parser.add_argument_group("Read Selection")
        read_group.add_argument("--read-id", type=str)
        read_group.add_argument("--reads", type=str, nargs="+")

        # Export options
        export_group = parser.add_argument_group("Export Options")
        export_group.add_argument("--export", "-e", type=str)
        export_group.add_argument(
            "--export-format", type=str, choices=["html", "png", "svg"]
        )
        export_group.add_argument("--export-width", type=int, default=1200)
        export_group.add_argument("--export-height", type=int, default=600)

        # Theme
        theme_group = parser.add_argument_group("Theme Options")
        theme_group.add_argument(
            "--theme", type=str, choices=["light", "dark"], default="dark"
        )

        return parser

    def test_parse_pod5_file(self, parser):
        """Test parsing POD5 file argument"""
        args = parser.parse_args(["--pod5", "test.pod5"])
        assert args.pod5 == "test.pod5"

        # Test short form
        args = parser.parse_args(["-p", "test.pod5"])
        assert args.pod5 == "test.pod5"

    def test_parse_bam_file(self, parser):
        """Test parsing BAM file argument"""
        args = parser.parse_args(["--bam", "test.bam"])
        assert args.bam == "test.bam"

        # Test short form
        args = parser.parse_args(["-b", "test.bam"])
        assert args.bam == "test.bam"

    def test_parse_plot_mode(self, parser):
        """Test parsing plot mode argument"""
        for mode in ["single", "overlay", "stacked", "eventalign"]:
            args = parser.parse_args(["--mode", mode])
            assert args.mode == mode

    def test_parse_normalization(self, parser):
        """Test parsing normalization method"""
        for norm in ["none", "znorm", "median", "mad"]:
            args = parser.parse_args(["--normalization", norm])
            assert args.normalization == norm

        # Test default
        args = parser.parse_args([])
        assert args.normalization == "median"

    def test_parse_show_bases(self, parser):
        """Test parsing base annotation flags"""
        args = parser.parse_args(["--show-bases"])
        assert args.show_bases is True

        args = parser.parse_args(["--no-show-bases"])
        assert args.no_show_bases is True

    def test_parse_show_points(self, parser):
        """Test parsing show points flag"""
        args = parser.parse_args(["--show-points"])
        assert args.show_points is True

        args = parser.parse_args([])
        assert args.show_points is False

    def test_parse_dwell_time(self, parser):
        """Test parsing dwell time flag"""
        args = parser.parse_args(["--dwell-time"])
        assert args.dwell_time is True

    def test_parse_downsample(self, parser):
        """Test parsing downsample factor"""
        args = parser.parse_args(["--downsample", "10"])
        assert args.downsample == 10

        # Test default
        args = parser.parse_args([])
        assert args.downsample == 25

    def test_parse_position_interval(self, parser):
        """Test parsing position label interval"""
        args = parser.parse_args(["--position-interval", "20"])
        assert args.position_interval == 20

        # Test default
        args = parser.parse_args([])
        assert args.position_interval == 10

    def test_parse_reference_positions(self, parser):
        """Test parsing reference positions flag"""
        args = parser.parse_args(["--reference-positions"])
        assert args.reference_positions is True

    def test_parse_read_id(self, parser):
        """Test parsing single read ID"""
        args = parser.parse_args(["--read-id", "read_001"])
        assert args.read_id == "read_001"

    def test_parse_multiple_reads(self, parser):
        """Test parsing multiple read IDs"""
        args = parser.parse_args(["--reads", "read_001", "read_002", "read_003"])
        assert args.reads == ["read_001", "read_002", "read_003"]

    def test_parse_export(self, parser):
        """Test parsing export options"""
        args = parser.parse_args(["--export", "plot.html"])
        assert args.export == "plot.html"

        # Test short form
        args = parser.parse_args(["-e", "plot.png"])
        assert args.export == "plot.png"

    def test_parse_export_format(self, parser):
        """Test parsing export format"""
        for fmt in ["html", "png", "svg"]:
            args = parser.parse_args(["--export-format", fmt])
            assert args.export_format == fmt

    def test_parse_export_dimensions(self, parser):
        """Test parsing export dimensions"""
        args = parser.parse_args(["--export-width", "800", "--export-height", "400"])
        assert args.export_width == 800
        assert args.export_height == 400

        # Test defaults
        args = parser.parse_args([])
        assert args.export_width == 1200
        assert args.export_height == 600

    def test_parse_theme(self, parser):
        """Test parsing theme argument"""
        args = parser.parse_args(["--theme", "dark"])
        assert args.theme == "dark"

        args = parser.parse_args(["--theme", "light"])
        assert args.theme == "light"

        # Test default
        args = parser.parse_args([])
        assert args.theme == "dark"

    def test_parse_combined_arguments(self, parser):
        """Test parsing multiple arguments together"""
        args = parser.parse_args(
            [
                "--pod5",
                "test.pod5",
                "--bam",
                "test.bam",
                "--mode",
                "eventalign",
                "--normalization",
                "median",
                "--show-bases",
                "--read-id",
                "read_001",
            ]
        )

        assert args.pod5 == "test.pod5"
        assert args.bam == "test.bam"
        assert args.mode == "eventalign"
        assert args.normalization == "median"
        assert args.show_bases is True
        assert args.read_id == "read_001"


class TestCLIExport:
    """Tests for headless export functionality in cli.py"""

    def test_export_plot_html(self, sample_pod5_file, tmp_path):
        """Test exporting plot as HTML"""
        from squiggy.cli import export_plot

        output_file = tmp_path / "test_plot.html"

        # Create mock args
        args = MagicMock()
        args.pod5 = str(sample_pod5_file)
        args.bam = None
        args.export = str(output_file)
        args.export_format = "html"
        args.export_width = 1200
        args.export_height = 600
        args.normalization = "median"
        args.theme = "dark"
        args.downsample = 25
        args.dwell_time = False
        args.show_points = False
        args.show_bases = False
        args.no_show_bases = False
        args.position_interval = 10
        args.reference_positions = False

        # Get first read ID from POD5
        import pod5

        with pod5.Reader(sample_pod5_file) as reader:
            first_read = next(reader.reads())
            args.read_id = str(first_read.read_id)
            args.reads = None

        # Export plot
        exit_code = export_plot(args)

        assert exit_code == 0
        assert output_file.exists()
        assert output_file.stat().st_size > 0

        # Verify HTML content
        content = output_file.read_text()
        assert "<!DOCTYPE html>" in content
        assert "Bokeh" in content

    def test_export_plot_nonexistent_pod5(self, tmp_path):
        """Test export with nonexistent POD5 file"""
        from squiggy.cli import export_plot

        args = MagicMock()
        args.pod5 = str(tmp_path / "nonexistent.pod5")
        args.export = str(tmp_path / "output.html")
        args.export_format = "html"

        exit_code = export_plot(args)

        assert exit_code == 1  # Should fail

    def test_export_plot_invalid_read_id(self, sample_pod5_file, tmp_path):
        """Test export with invalid read ID"""
        from squiggy.cli import export_plot

        args = MagicMock()
        args.pod5 = str(sample_pod5_file)
        args.bam = None
        args.export = str(tmp_path / "output.html")
        args.export_format = "html"
        args.read_id = "nonexistent_read_id"
        args.reads = None

        exit_code = export_plot(args)

        assert exit_code == 1  # Should fail

    def test_export_plot_infer_format_from_extension(self, sample_pod5_file, tmp_path):
        """Test export format inference from file extension"""
        import pod5
        from squiggy.cli import export_plot

        with pod5.Reader(sample_pod5_file) as reader:
            first_read = next(reader.reads())
            read_id = str(first_read.read_id)

        for ext in [".html", ".png", ".svg"]:
            output_file = tmp_path / f"test_plot{ext}"

            args = MagicMock()
            args.pod5 = str(sample_pod5_file)
            args.bam = None
            args.export = str(output_file)
            args.export_format = None  # Let it infer
            args.export_width = 800
            args.export_height = 400
            args.normalization = "none"
            args.theme = "light"
            args.downsample = 10
            args.dwell_time = False
            args.show_points = False
            args.show_bases = False
            args.no_show_bases = False
            args.position_interval = 10
            args.reference_positions = False
            args.read_id = read_id
            args.reads = None

            exit_code = export_plot(args)

            # HTML should always work, PNG/SVG might fail if selenium/geckodriver not available
            if ext == ".html":
                assert exit_code == 0
                assert output_file.exists()

    def test_export_plot_multiple_reads(self, sample_pod5_file, tmp_path):
        """Test exporting plot with multiple reads"""
        import pod5
        from squiggy.cli import export_plot

        # Get first 3 read IDs
        with pod5.Reader(sample_pod5_file) as reader:
            read_ids = [str(read.read_id) for read in list(reader.reads())[:3]]

        output_file = tmp_path / "multi_read_plot.html"

        args = MagicMock()
        args.pod5 = str(sample_pod5_file)
        args.bam = None
        args.export = str(output_file)
        args.export_format = "html"
        args.export_width = 1200
        args.export_height = 600
        args.normalization = "znorm"
        args.theme = "dark"
        args.mode = "overlay"
        args.downsample = 25
        args.dwell_time = False
        args.show_points = False
        args.show_bases = False
        args.no_show_bases = False
        args.position_interval = 10
        args.reference_positions = False
        args.read_id = None
        args.reads = read_ids

        exit_code = export_plot(args)

        assert exit_code == 0
        assert output_file.exists()

    def test_export_plot_with_bam(self, sample_pod5_file, sample_bam_file, tmp_path):
        """Test exporting plot with BAM file"""
        import pod5
        from squiggy.cli import export_plot

        with pod5.Reader(sample_pod5_file) as reader:
            first_read = next(reader.reads())
            read_id = str(first_read.read_id)

        output_file = tmp_path / "plot_with_bases.html"

        args = MagicMock()
        args.pod5 = str(sample_pod5_file)
        args.bam = str(sample_bam_file)
        args.export = str(output_file)
        args.export_format = "html"
        args.export_width = 1200
        args.export_height = 600
        args.normalization = "median"
        args.theme = "dark"
        args.downsample = 25
        args.dwell_time = False
        args.show_points = False
        args.show_bases = True  # Show bases
        args.no_show_bases = False
        args.position_interval = 10
        args.reference_positions = False
        args.read_id = read_id
        args.reads = None

        exit_code = export_plot(args)

        # Exit code 0 means success
        assert exit_code == 0
        assert output_file.exists()

    def test_export_plot_all_normalization_methods(self, sample_pod5_file, tmp_path):
        """Test export with all normalization methods"""
        import pod5
        from squiggy.cli import export_plot

        with pod5.Reader(sample_pod5_file) as reader:
            first_read = next(reader.reads())
            read_id = str(first_read.read_id)

        for norm in ["none", "znorm", "median", "mad"]:
            output_file = tmp_path / f"plot_{norm}.html"

            args = MagicMock()
            args.pod5 = str(sample_pod5_file)
            args.bam = None
            args.export = str(output_file)
            args.export_format = "html"
            args.export_width = 800
            args.export_height = 400
            args.normalization = norm
            args.theme = "dark"
            args.downsample = 25
            args.dwell_time = False
            args.show_points = False
            args.show_bases = False
            args.no_show_bases = False
            args.position_interval = 10
            args.reference_positions = False
            args.read_id = read_id
            args.reads = None

            exit_code = export_plot(args)

            assert exit_code == 0
            assert output_file.exists()
