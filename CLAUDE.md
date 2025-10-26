# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Squiggy is a desktop GUI application for visualizing Oxford Nanopore sequencing data ("squiggle plots") from POD5 files. It's built as a standalone application that can be distributed as native executables for Windows, macOS, and Linux.

The application supports optional base annotations overlaid on signal data when a BAM file with basecalls is provided, inspired by ONT's remora visualization approach.

## Architecture

### Application Stack
- **GUI Framework**: PySide6 (Qt for Python) - provides cross-platform desktop UI
- **Data Processing**: pod5 library for reading Oxford Nanopore POD5 files
- **BAM Support**: pysam library (optional) for reading basecall data from BAM files
- **Visualization**: plotnine (ggplot2-style) for generating squiggle plots from signal data
- **Distribution**: PyInstaller packages the app into standalone executables/bundles

### Key Components

**squiggy/squiggy/main.py** - Single-file application containing:
- `SquigglePlotter`: Static plotting logic that converts raw signal data into time-series plots with optional base annotations
- `SquiggleViewer`: Main QMainWindow with file selectors (POD5 + optional BAM), read list with search, plot display, and base annotation toggle
- Plot rendering workflow: POD5 → signal data → (optional: BAM → basecalls + move table) → plotnine plot → PNG buffer → QPixmap → display
- Base annotation visualization inspired by ONT's remora: colored rectangles (A=green, C=blue, G=orange, T=red) with base labels

**squiggy.spec** - PyInstaller configuration for building native apps:
- Specifies entry point as `squiggy/main.py`
- Declares hidden imports (pod5, plotnine, PySide6 modules) that PyInstaller can't auto-detect
- Creates macOS .app bundle with proper Info.plist configuration

### Data Flow

**Basic (POD5 only):**
1. User selects POD5 file → `pod5.Reader` loads file
2. All read IDs extracted and displayed in searchable list
3. User selects read → signal data retrieved with sample rate
4. Signal converted to time-series DataFrame
5. Plotnine generates plot → saved to BytesIO buffer as PNG
6. PNG loaded as QPixmap and displayed in Qt widget

**Enhanced (POD5 + BAM):**
1. User additionally selects BAM file → `pysam.AlignmentFile` opens file
2. User enables "Show base annotations" checkbox
3. When read is selected, BAM is queried for matching read_id
4. Move table extracted from BAM 'mv' tag → converted to seq_to_sig_map array
5. Base sequence and mapping passed to plotter
6. Plotnine adds colored `geom_rect` and `geom_text` layers for bases
7. Combined plot rendered and displayed

## Development Commands

### Environment Setup
```bash
# Install dependencies (use uv or pip)
pip install -r requirements.txt

# Install with dev dependencies
pip install -e ".[dev]"
```

### Running the Application
```bash
# Run from source
python squiggy/squiggy/main.py

# Or using the installed console script
squiggy
```

### Code Formatting
```bash
# Format code with black
black squiggy/
```

### Testing
```bash
# Run tests (when test suite exists)
pytest tests/
```

### Building Standalone Executables

**Local build (any platform):**
```bash
pyinstaller squiggy.spec
# Output: dist/Squiggy (Linux), dist/Squiggy.exe (Windows), dist/Squiggy.app (macOS)
```

**Platform-specific packaging:**
```bash
# macOS: Create DMG installer (requires create-dmg)
brew install create-dmg
pyinstaller squiggy.spec
create-dmg --volname "Squiggy Installer" --app-drop-link 600 185 Squiggy-macos.dmg dist/Squiggy.app

# Windows: Create ZIP archive
pyinstaller squiggy.spec
Compress-Archive -Path dist/Squiggy.exe -DestinationPath Squiggy-windows-x64.zip

# Linux: Create tarball
pyinstaller squiggy.spec
tar -czf Squiggy-linux-x64.tar.gz -C dist Squiggy
```

## Release Process

Releases are automated via GitHub Actions (`.github/workflows/build.yml`):
- Triggered on version tags (e.g., `v0.1.0`) or manual workflow dispatch
- Builds executables for Windows, macOS, and Linux in parallel
- Creates GitHub release with platform-specific installers attached
- Release artifacts: `.zip` (Windows), `.dmg` (macOS), `.tar.gz` (Linux)

To create a release:
```bash
git tag v0.1.0
git push origin v0.1.0
```

## Important Constraints

### PyInstaller Considerations
- All non-standard-library imports must be listed in `squiggy.spec` `hiddenimports` if PyInstaller fails to detect them
- Resource files (icons, data files) must be declared in `datas` list
- The spec file uses `console=False` to hide the terminal window on launch

### POD5 File Handling
- POD5 files use HDF5 format with VBZ compression (requires `vbz_h5py_plugin`)
- Always use context managers (`with pod5.Reader(...)`) to ensure proper file cleanup
- Read objects contain: `read_id`, `signal` (numpy array), `sample_rate`, and metadata

### BAM File Handling (Optional Feature)
- BAM support requires pysam library (installed via requirements.txt)
- The app gracefully degrades if pysam is not available (BAM button disabled)
- BAM files must contain move tables in 'mv' tag (standard in modern Dorado basecaller output)
- Move table format: uint8 array where 1 indicates new base, 0 indicates same base (stride)
- Seq-to-sig mapping computed by accumulating signal positions for each base transition

### Qt/PySide6 Notes
- Plot display uses QLabel with QPixmap (not matplotlib embedding) for simplicity
- Plotnine plots are rendered to PNG buffers rather than displayed directly
- Window geometry: 1200x800 default, with 1:3 split between read list and plot area

### Source Code Location
The actual source code is in `squiggy/squiggy/` (nested directory structure):
- `squiggy/` is the virtual environment directory (created by venv/virtualenv)
- `squiggy/squiggy/` contains the installable package source code
- Note: There is also `.venv/` at the project root (alternate venv)

When editing source code, always modify files in `squiggy/squiggy/`, not the venv site-packages.
