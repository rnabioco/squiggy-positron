"""Utility functions for Squiggy application"""

import sys
import platform
import tempfile
import shutil
from pathlib import Path
import numpy as np

try:
    import pysam

    PYSAM_AVAILABLE = True
except ImportError:
    PYSAM_AVAILABLE = False


def get_icon_path():
    """Get the path to the application icon file

    Returns the appropriate icon file for the current platform.
    Works both when running from source and when bundled with PyInstaller.

    Returns:
        Path or None: Path to icon file, or None if not found
    """
    # Determine icon file based on platform
    if platform.system() == "Windows":
        icon_name = "squiggy.ico"
    elif platform.system() == "Darwin":  # macOS
        icon_name = "squiggy.icns"
    else:  # Linux and others
        icon_name = "squiggy.png"

    # Try multiple locations
    # 1. PyInstaller bundle location (when bundled)
    if getattr(sys, "_MEIPASS", None):
        icon_path = Path(sys._MEIPASS) / icon_name
        if icon_path.exists():
            return icon_path

    # 2. Package data directory (when installed)
    try:
        if sys.version_info >= (3, 9):
            import importlib.resources as resources

            files = resources.files("squiggy")
            icon_path = files / "data" / icon_name
            if hasattr(icon_path, "as_posix"):
                path = Path(str(icon_path))
                if path.exists():
                    return path
        else:
            import pkg_resources

            icon_path = Path(
                pkg_resources.resource_filename("squiggy", f"data/{icon_name}")
            )
            if icon_path.exists():
                return icon_path
    except:
        pass

    # 3. Development location (relative to package)
    package_dir = Path(__file__).parent
    icon_path = package_dir / "data" / icon_name
    if icon_path.exists():
        return icon_path

    # 4. Build directory (during development)
    build_dir = Path(__file__).parent.parent.parent / "build" / icon_name
    if build_dir.exists():
        return build_dir

    return None


def get_logo_path():
    """Get the path to the PNG logo for display in dialogs

    Returns:
        Path or None: Path to logo file, or None if not found
    """
    # Try multiple locations for the PNG logo
    # 1. PyInstaller bundle
    if getattr(sys, "_MEIPASS", None):
        logo_path = Path(sys._MEIPASS) / "squiggy.png"
        if logo_path.exists():
            return logo_path

    # 2. Package data directory
    try:
        if sys.version_info >= (3, 9):
            import importlib.resources as resources

            files = resources.files("squiggy")
            logo_path = files / "data" / "squiggy.png"
            if hasattr(logo_path, "as_posix"):
                path = Path(str(logo_path))
                if path.exists():
                    return path
        else:
            import pkg_resources

            logo_path = Path(
                pkg_resources.resource_filename("squiggy", "data/squiggy.png")
            )
            if logo_path.exists():
                return logo_path
    except:
        pass

    # 3. Development location
    package_dir = Path(__file__).parent
    logo_path = package_dir / "data" / "squiggy.png"
    if logo_path.exists():
        return logo_path

    # 4. Build directory
    build_dir = Path(__file__).parent.parent.parent / "build" / "squiggy.png"
    if build_dir.exists():
        return build_dir

    return None


def get_sample_data_path():
    """Get the path to the bundled sample data file

    Returns:
        Path: Path to sample.pod5 file

    Raises:
        FileNotFoundError: If sample data cannot be found
    """
    try:
        # For Python 3.9+
        if sys.version_info >= (3, 9):
            import importlib.resources as resources

            files = resources.files("squiggy")
            sample_path = files / "data" / "sample.pod5"
            if hasattr(sample_path, "as_posix"):
                return Path(sample_path)
            # For traversable objects, we need to extract to temp
            temp_dir = Path(tempfile.gettempdir()) / "squiggy_data"
            temp_dir.mkdir(exist_ok=True)
            temp_file = temp_dir / "sample.pod5"
            if not temp_file.exists():
                with resources.as_file(sample_path) as f:
                    shutil.copy(f, temp_file)
            return temp_file
        else:
            # Fallback for older Python
            import pkg_resources

            sample_path = pkg_resources.resource_filename("squiggy", "data/sample.pod5")
            return Path(sample_path)
    except Exception as e:
        # Fallback: look in installed package directory
        package_dir = Path(__file__).parent
        sample_path = package_dir / "data" / "sample.pod5"
        if sample_path.exists():
            return sample_path
        raise FileNotFoundError(f"Sample data not found. Error: {e}")


def get_basecall_data(bam_file, read_id):
    """Extract basecall sequence and signal mapping from BAM file

    Args:
        bam_file: Path to BAM file
        read_id: Read identifier to search for

    Returns:
        tuple: (sequence, seq_to_sig_map) or (None, None) if not available
    """
    if not bam_file or not PYSAM_AVAILABLE:
        return None, None

    try:
        bam = pysam.AlignmentFile(str(bam_file), "rb", check_sq=False)

        # Find the read in BAM
        for read in bam.fetch(until_eof=True):
            if read.query_name == read_id:
                # Get sequence
                sequence = read.query_sequence

                # Get move table from BAM tags
                if read.has_tag("mv"):
                    move_table = np.array(read.get_tag("mv"), dtype=np.uint8)

                    # Convert move table to signal-to-sequence mapping
                    seq_to_sig_map = []
                    sig_pos = 0
                    for i, move in enumerate(move_table):
                        if move == 1:
                            seq_to_sig_map.append(sig_pos)
                        sig_pos += 1

                    bam.close()
                    return sequence, np.array(seq_to_sig_map)

        bam.close()

    except Exception as e:
        print(f"Error reading BAM file for {read_id}: {e}")

    return None, None
