#!/usr/bin/env python3
"""Check that the development environment is properly configured"""

import sys
from pathlib import Path


def check_python_version():
    """Check Python version is 3.8+"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required, found", f"{version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_dependencies():
    """Check that required dependencies are installed"""
    required = [
        ("numpy", "NumPy"),
        ("pandas", "pandas"),
        ("pod5", "pod5"),
        ("PySide6", "PySide6"),
        ("qasync", "qasync"),
        ("plotnine", "plotnine"),
        ("pysam", "pysam"),
    ]

    optional = [
        ("pytest", "pytest (dev)"),
        ("ruff", "ruff (dev)"),
    ]

    all_ok = True
    for module_name, display_name in required:
        try:
            __import__(module_name)
            print(f"✅ {display_name}")
        except ImportError:
            print(f"❌ {display_name} - not installed")
            all_ok = False

    for module_name, display_name in optional:
        try:
            __import__(module_name)
            print(f"✅ {display_name}")
        except ImportError:
            print(f"⚠️  {display_name} - optional, not installed")

    # Check macOS-specific
    if sys.platform == "darwin":
        try:
            __import__("Foundation")
            print("✅ PyObjC (macOS)")
        except ImportError:
            print("⚠️  PyObjC (macOS) - recommended for proper app name")
            print('   Install with: pip install -e ".[macos]"')

    return all_ok


def check_sample_data():
    """Check that sample data exists"""
    test_data_dir = Path("tests/data")
    if not test_data_dir.exists():
        print(f"❌ Test data directory not found: {test_data_dir}")
        return False

    pod5_files = list(test_data_dir.glob("*.pod5"))
    bam_files = list(test_data_dir.glob("*.bam"))

    if pod5_files:
        print(f"✅ Sample POD5 files found: {len(pod5_files)}")
    else:
        print(f"⚠️  No POD5 files in {test_data_dir}")

    if bam_files:
        print(f"✅ Sample BAM files found: {len(bam_files)}")
    else:
        print(f"⚠️  No BAM files in {test_data_dir}")

    return True


def check_package_structure():
    """Check that package structure is correct"""
    src_dir = Path("src/squiggy")
    if not src_dir.exists():
        print(f"❌ Source directory not found: {src_dir}")
        return False

    required_files = [
        "__init__.py",
        "main.py",
        "viewer.py",
        "plotter.py",
        "dialogs.py",
        "utils.py",
        "constants.py",
        "alignment.py",
        "normalization.py",
    ]

    all_ok = True
    for filename in required_files:
        filepath = src_dir / filename
        if filepath.exists():
            print(f"✅ {filepath}")
        else:
            print(f"❌ Missing: {filepath}")
            all_ok = False

    return all_ok


def main():
    """Run all checks"""
    print("=" * 60)
    print("Squiggy Development Environment Check")
    print("=" * 60)

    print("\n🔍 Checking Python version...")
    python_ok = check_python_version()

    print("\n🔍 Checking dependencies...")
    deps_ok = check_dependencies()

    print("\n🔍 Checking sample data...")
    data_ok = check_sample_data()

    print("\n🔍 Checking package structure...")
    structure_ok = check_package_structure()

    print("\n" + "=" * 60)
    if python_ok and deps_ok and structure_ok:
        print("✅ Development environment is ready!")
        print("\nRun the app with:")
        print("  squiggy -p tests/data/simplex_reads.pod5")
        return 0
    else:
        print("❌ Development environment has issues")
        print("\nTo fix:")
        print('  pip install -e ".[dev]"')
        if sys.platform == "darwin":
            print('  pip install -e ".[macos]"  # macOS only')
        return 1


if __name__ == "__main__":
    sys.exit(main())
