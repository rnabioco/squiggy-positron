"""Path and file location utilities for Squiggy"""

import os
import platform
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def writable_working_directory():
    """Context manager to temporarily change to a writable working directory

    The pod5 library creates temporary directories in the current working directory
    during format migration. When running from a PyInstaller bundle or other read-only
    location, the CWD may not be writable. This context manager temporarily changes
    to a writable temp directory, then restores the original CWD.

    Usage:
        with writable_working_directory():
            with pod5.Reader(pod5_file) as reader:
                # Process reads...
    """
    original_cwd = os.getcwd()
    temp_dir = Path(tempfile.gettempdir()) / "squiggy_workdir"
    temp_dir.mkdir(exist_ok=True)

    try:
        os.chdir(temp_dir)
        yield temp_dir
    finally:
        os.chdir(original_cwd)


def _is_writable_dir(dir_path):
    """Check if a directory is writable

    Args:
        dir_path: Path to directory

    Returns:
        bool: True if directory is writable, False otherwise
    """
    try:
        test_file = dir_path / ".write_test"
        test_file.touch()
        test_file.unlink()
        return True
    except (OSError, PermissionError):
        return False


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
        import importlib.resources as resources

        files = resources.files("squiggy")
        icon_path = files / "data" / icon_name
        if hasattr(icon_path, "as_posix"):
            path = Path(str(icon_path))
            if path.exists():
                return path
    except Exception:
        pass

    # 3. Development location (relative to package)
    package_dir = Path(__file__).parent.parent
    icon_path = package_dir / "data" / icon_name
    if icon_path.exists():
        return icon_path

    # 4. Build directory (during development)
    build_dir = Path(__file__).parent.parent.parent.parent / "build" / icon_name
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
        import importlib.resources as resources

        files = resources.files("squiggy")
        logo_path = files / "data" / "squiggy.png"
        if hasattr(logo_path, "as_posix"):
            path = Path(str(logo_path))
            if path.exists():
                return path
    except Exception:
        pass

    # 3. Development location
    package_dir = Path(__file__).parent.parent
    logo_path = package_dir / "data" / "squiggy.png"
    if logo_path.exists():
        return logo_path

    # 4. Build directory
    build_dir = Path(__file__).parent.parent.parent.parent / "build" / "squiggy.png"
    if build_dir.exists():
        return build_dir

    return None


def get_sample_data_path():
    """Get the path to the bundled sample data file

    When running from a PyInstaller bundle, the sample data is in a read-only
    location. This function copies the sample POD5 file to a writable temporary
    directory to allow the pod5 library to perform format migration if needed.

    Returns:
        Path: Path to yeast_trna_reads.pod5 file (may be in temp directory)

    Raises:
        FileNotFoundError: If sample data cannot be found
    """
    # Create temp directory for sample data
    temp_dir = Path(tempfile.gettempdir()) / "squiggy_data"
    temp_dir.mkdir(exist_ok=True)
    temp_file = temp_dir / "yeast_trna_reads.pod5"

    try:
        # Find the bundled sample data
        source_path = None

        import importlib.resources as resources

        files = resources.files("squiggy")
        sample_path = files / "data" / "yeast_trna_reads.pod5"

        # If it's a regular file path, use it directly as source
        if hasattr(sample_path, "as_posix"):
            source_path = Path(str(sample_path))
        else:
            # For traversable objects, extract to temp
            if not temp_file.exists():
                with resources.as_file(sample_path) as f:
                    shutil.copy(f, temp_file)
            return temp_file

        # If we got a source path, check if it's in a read-only location
        # (PyInstaller bundle) and copy to temp
        if source_path and source_path.exists():
            # Always copy when running from PyInstaller bundle
            # Check sys._MEIPASS (PyInstaller) or if parent dir is not writable
            is_pyinstaller = getattr(sys, "_MEIPASS", None) is not None
            is_readonly = not _is_writable_dir(source_path.parent)

            if is_pyinstaller or is_readonly:
                # Copy to temp directory to ensure pod5 library can create temp files
                # Always copy if source is newer or temp doesn't exist
                should_copy = (
                    not temp_file.exists()
                    or temp_file.stat().st_size == 0
                    or temp_file.stat().st_size != source_path.stat().st_size
                )
                if should_copy:
                    shutil.copy(source_path, temp_file)
                return temp_file
            else:
                # Source is in writable location, return as-is
                return source_path

        # Last resort: look in package directory
        package_dir = Path(__file__).parent.parent
        sample_path = package_dir / "data" / "yeast_trna_reads.pod5"

        if sample_path.exists():
            # Always copy when running from PyInstaller bundle
            is_pyinstaller = getattr(sys, "_MEIPASS", None) is not None
            is_readonly = not _is_writable_dir(sample_path.parent)

            if is_pyinstaller or is_readonly:
                should_copy = (
                    not temp_file.exists()
                    or temp_file.stat().st_size == 0
                    or temp_file.stat().st_size != sample_path.stat().st_size
                )
                if should_copy:
                    shutil.copy(sample_path, temp_file)
                return temp_file
            return sample_path

        raise FileNotFoundError("Sample data file not found in any location")

    except Exception as e:
        raise FileNotFoundError(f"Sample data not found. Error: {e}") from None


def get_test_data_path():
    """Get the path to the bundled test data directory

    Returns the path to the squiggy/data directory which contains test/demo files:
    - yeast_trna_reads.pod5
    - yeast_trna_mappings.bam
    - yeast_trna_mappings.bam.bai
    - yeast_trna.fa
    - yeast_trna.fa.fai

    Returns:
        str: Path to the squiggy/data directory

    Raises:
        FileNotFoundError: If data directory cannot be found

    Examples:
        >>> from pathlib import Path
        >>> data_dir = Path(get_test_data_path())
        >>> pod5_file = data_dir / 'yeast_trna_reads.pod5'
    """
    import importlib.util

    try:
        # Find the squiggy package location
        spec = importlib.util.find_spec("squiggy")
        if spec is None or spec.origin is None:
            raise FileNotFoundError("Could not locate squiggy package")

        package_dir = os.path.dirname(spec.origin)
        data_dir = os.path.join(package_dir, "data")

        # Verify the directory exists
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Data directory not found at {data_dir}")

        return data_dir

    except Exception as e:
        raise FileNotFoundError(f"Test data directory not found. Error: {e}") from None


def get_sample_bam_path():
    """Get the path to the bundled sample BAM file

    When running from a PyInstaller bundle, the sample data is in a read-only
    location. This function copies the sample BAM file (and its index) to a
    writable temporary directory.

    Returns:
        Path: Path to yeast_trna_mappings.bam file (may be in temp directory)
        Returns None if BAM file not found

    """
    # Create temp directory for sample data
    temp_dir = Path(tempfile.gettempdir()) / "squiggy_data"
    temp_dir.mkdir(exist_ok=True)
    temp_bam = temp_dir / "yeast_trna_mappings.bam"
    temp_bai = temp_dir / "yeast_trna_mappings.bam.bai"

    try:
        # Find the bundled sample BAM file
        source_bam = None
        source_bai = None

        import importlib.resources as resources

        files = resources.files("squiggy")
        sample_bam_path = files / "data" / "yeast_trna_mappings.bam"
        sample_bai_path = files / "data" / "yeast_trna_mappings.bam.bai"

        # If it's a regular file path, use it directly as source
        if hasattr(sample_bam_path, "as_posix"):
            source_bam = Path(str(sample_bam_path))
            source_bai = Path(str(sample_bai_path))
        else:
            # For traversable objects, extract to temp
            if not temp_bam.exists():
                with resources.as_file(sample_bam_path) as f:
                    shutil.copy(f, temp_bam)
            if not temp_bai.exists() and sample_bai_path:
                try:
                    with resources.as_file(sample_bai_path) as f:
                        shutil.copy(f, temp_bai)
                except Exception:
                    pass  # BAI file optional
            return temp_bam

        # If we got a source path, check if it's in a read-only location
        if source_bam and source_bam.exists():
            # Always copy when running from PyInstaller bundle
            is_pyinstaller = getattr(sys, "_MEIPASS", None) is not None
            is_readonly = not _is_writable_dir(source_bam.parent)

            if is_pyinstaller or is_readonly:
                # Copy to temp directory
                should_copy_bam = (
                    not temp_bam.exists()
                    or temp_bam.stat().st_size == 0
                    or temp_bam.stat().st_size != source_bam.stat().st_size
                )
                if should_copy_bam:
                    shutil.copy(source_bam, temp_bam)

                # Copy index file if it exists
                if source_bai and source_bai.exists():
                    should_copy_bai = (
                        not temp_bai.exists()
                        or temp_bai.stat().st_size == 0
                        or temp_bai.stat().st_size != source_bai.stat().st_size
                    )
                    if should_copy_bai:
                        shutil.copy(source_bai, temp_bai)
                return temp_bam
            else:
                # Source is in writable location, return as-is
                return source_bam

        # Last resort: look in package directory
        package_dir = Path(__file__).parent.parent
        sample_bam = package_dir / "data" / "yeast_trna_mappings.bam"
        sample_bai = package_dir / "data" / "yeast_trna_mappings.bam.bai"

        if sample_bam.exists():
            # Always copy when running from PyInstaller bundle
            is_pyinstaller = getattr(sys, "_MEIPASS", None) is not None
            is_readonly = not _is_writable_dir(sample_bam.parent)

            if is_pyinstaller or is_readonly:
                should_copy_bam = (
                    not temp_bam.exists()
                    or temp_bam.stat().st_size == 0
                    or temp_bam.stat().st_size != sample_bam.stat().st_size
                )
                if should_copy_bam:
                    shutil.copy(sample_bam, temp_bam)

                if sample_bai.exists():
                    should_copy_bai = (
                        not temp_bai.exists()
                        or temp_bai.stat().st_size == 0
                        or temp_bai.stat().st_size != sample_bai.stat().st_size
                    )
                    if should_copy_bai:
                        shutil.copy(sample_bai, temp_bai)
                return temp_bam
            return sample_bam

        return None  # BAM file is optional

    except Exception:
        return None  # BAM file is optional
