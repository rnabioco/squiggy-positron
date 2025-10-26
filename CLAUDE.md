# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Squiggy is a desktop GUI application for visualizing Oxford Nanopore sequencing data ("squiggle plots") from POD5 files. It's built as a standalone application that can be distributed as native executables for Windows, macOS, and Linux.

The application supports optional base annotations overlaid on signal data when a BAM file with basecalls is provided, inspired by ONT's remora visualization approach.

## Architecture

### Application Stack
- **GUI Framework**: PySide6 (Qt for Python) - provides cross-platform desktop UI
- **Data Processing**: pod5 library for reading Oxford Nanopore POD5 files
- **Signal Processing**: ont-remora library for advanced signal analysis
- **Visualization**: plotnine (ggplot2-style) for generating squiggle plots from signal data
- **Distribution**: PyInstaller packages the app into standalone executables/bundles
- **Documentation**: MkDocs with Material theme for user documentation
- **Testing**: pytest for unit and integration tests

### Project Structure

The codebase is organized as a standard Python package in `src/squiggy/`:

```
src/squiggy/
├── __init__.py         # Package initialization with version info
├── main.py            # Entry point and CLI argument parsing
├── viewer.py          # SquiggleViewer - Main QMainWindow GUI
├── plotter.py         # SquigglePlotter - Plotting logic and visualization
├── dialogs.py         # Custom dialog windows (About, etc.)
├── utils.py           # Utility functions (file I/O, data processing)
└── constants.py       # Application constants and configuration

build/
├── squiggy.spec       # PyInstaller build specification
├── squiggy.png        # Application logo (PNG)
├── squiggy.ico        # Windows icon
└── squiggy.icns       # macOS icon

tests/data/
├── README.md          # Sample data documentation
├── *.pod5             # Sample POD5 files for testing
└── *.bam              # Sample BAM alignment files
```

### Key Components

**src/squiggy/main.py** - Application entry point:
- Parses command-line arguments (`--file` to pre-load POD5 files)
- Initializes Qt application and launches SquiggleViewer
- Defines `main()` function as console script entry point

**src/squiggy/viewer.py** - Main GUI window (SquiggleViewer):
- File menu with "Open POD5 File" and "Open Sample Data" options
- Read list widget with search/filter functionality
- Plot display area showing squiggle plots
- Status bar with file information

**src/squiggy/plotter.py** - Plotting logic (SquigglePlotter):
- Converts raw signal data into time-series DataFrames
- Generates plotnine plots with customizable styling
- Renders plots to PNG buffers for Qt display
- Handles plot caching and performance optimization

**src/squiggy/dialogs.py** - Custom dialog windows:
- About dialog with version and license information
- Potentially other dialogs for settings/preferences

**src/squiggy/utils.py** - Utility functions:
- POD5 file reading and validation
- Signal data extraction and processing
- Sample data location and loading

**src/squiggy/constants.py** - Application constants:
- Color schemes for plots
- Default window sizes and UI settings
- File format specifications

### Data Flow

**Application Startup:**
1. `main.py` parses CLI arguments (optional `--file` parameter)
2. Qt application initialized
3. SquiggleViewer window created and displayed
4. If `--file` provided, POD5 file automatically loaded

**POD5 File Loading:**
1. User selects "Open POD5 File" or "Open Sample Data" from File menu
2. `utils.py` validates and opens file with `pod5.Reader`
3. All read IDs extracted and loaded into searchable QListWidget
4. File path displayed in status bar

**Read Visualization:**
1. User selects read from list (or searches by ID)
2. `utils.py` extracts signal data and metadata for selected read
3. Signal data passed to `plotter.py` SquigglePlotter
4. Plotnine generates time-series plot → saved to BytesIO buffer as PNG
5. PNG loaded as QPixmap and displayed in QLabel widget
6. Read information displayed in status bar

## Development Commands

### Environment Setup
```bash
# Clone the repository
git clone https://github.com/rnabioco/squiggy.git
cd squiggy

# Install dependencies (using pip)
pip install -r requirements.txt

# Or install as editable package with dev dependencies
pip install -e ".[dev]"
```

### Running the Application
```bash
# Run using installed console script (recommended)
squiggy

# Run with a specific POD5 file pre-loaded
squiggy --file /path/to/file.pod5

# Run directly from source
python -m squiggy.main

# Or run the module directly
python src/squiggy/main.py
```

### Code Formatting and Linting
```bash
# Format code with ruff
ruff format src/ tests/

# Lint and auto-fix issues with ruff
ruff check --fix src/ tests/

# Check without auto-fixing
ruff check src/ tests/
```

### Testing
```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest -v tests/

# Run specific test file
pytest tests/test_plotting.py

# Run with coverage report
pytest --cov=squiggy tests/
```

### Documentation

```bash
# Serve documentation locally with live reload
mkdocs serve

# Build documentation
mkdocs build

# Deploy documentation to GitHub Pages
mkdocs gh-deploy
```

### Building Standalone Executables

**Prerequisites:**
```bash
# Install PyInstaller
pip install pyinstaller
```

**Local build (any platform):**
```bash
# Build using spec file (recommended - run from build/ directory)
cd build
pyinstaller squiggy.spec

# Or build directly from source (run from project root)
pyinstaller --name Squiggy \
    --windowed \
    --icon build/squiggy.icns \
    --add-data "tests/data/*.pod5:squiggy/data" \
    --add-data "tests/data/README.md:squiggy/data" \
    --add-data "build/squiggy.png:squiggy/data" \
    --add-data "build/squiggy.ico:squiggy/data" \
    --add-data "build/squiggy.icns:squiggy/data" \
    src/squiggy/main.py

# Output: dist/Squiggy (Linux), dist/Squiggy.exe (Windows), dist/Squiggy.app (macOS)
```

**Platform-specific packaging:**
```bash
# macOS: Create DMG installer (requires create-dmg)
brew install create-dmg
cd build
pyinstaller squiggy.spec
create-dmg \
  --volname "Squiggy Installer" \
  --window-pos 200 120 \
  --window-size 800 400 \
  --icon-size 100 \
  --app-drop-link 600 185 \
  Squiggy-macos.dmg \
  dist/Squiggy.app

# Windows: Create ZIP archive
cd build
pyinstaller squiggy.spec
Compress-Archive -Path dist/Squiggy.exe -DestinationPath Squiggy-windows-x64.zip

# Linux: Create tarball
cd build
pyinstaller squiggy.spec
tar -czf Squiggy-linux-x64.tar.gz -C dist Squiggy
```

**Note:** The build process is automated via GitHub Actions for releases. See `.github/workflows/build.yml` for the CI/CD configuration.

## Release Process

Releases are automated via GitHub Actions (`.github/workflows/build.yml`):
- **Version Releases**: Triggered on version tags (e.g., `v0.1.0`)
- **Development Builds**: Triggered on pushes to `main` branch (macOS only)
- Builds executables for Windows, macOS, and Linux in parallel
- Creates GitHub release with platform-specific installers attached
- Release artifacts: `.zip` (Windows), `.dmg` (macOS), `.tar.gz` (Linux)

**To create a version release:**
```bash
# Update version in src/squiggy/__init__.py and pyproject.toml
# Commit changes
git add src/squiggy/__init__.py pyproject.toml
git commit -m "Bump version to 0.1.0"

# Create and push tag
git tag v0.1.0
git push origin v0.1.0
```

**Development builds:**
- Automatically created on every push to `main`
- Available as "latest" pre-release on GitHub
- macOS .dmg only (for faster CI builds)

## Important Constraints

### Project Structure
- **Source code location**: `src/squiggy/` (standard Python package layout)
- **Virtual environment**: `.venv/` at project root (do NOT edit files here)
- **Tests**: `tests/` directory with pytest fixtures in `conftest.py`
- **Documentation**: `docs/` directory with MkDocs markdown files
- When editing source code, always modify files in `src/squiggy/`, not `.venv/`

### PyInstaller Considerations
- All non-standard-library imports must be listed in `build/squiggy.spec` `hiddenimports` if PyInstaller fails to detect them
- Resource files are in two locations:
  - Icons: `build/` directory (squiggy.png, squiggy.ico, squiggy.icns)
  - Sample data: `tests/data/` directory (*.pod5 files and README.md)
- Application icons available in multiple formats: `.png`, `.ico` (Windows), `.icns` (macOS)
- The spec file uses `--windowed` to hide the terminal window on launch
- Sample POD5 files from `tests/data/` are bundled with the application for "Open Sample Data" feature
- All data files are bundled into `squiggy/data` in the final distribution

### POD5 File Handling
- POD5 files use HDF5 format with VBZ compression (requires `vbz_h5py_plugin`)
- Always use context managers (`with pod5.Reader(...)`) to ensure proper file cleanup
- Read objects contain: `read_id`, `signal` (numpy array), `sample_rate`, and metadata
- Signal data is float16 (half-precision) to save memory
- Large POD5 files (>10,000 reads) may require lazy loading strategies

### ONT Remora Integration
- `ont-remora` library is used for advanced signal analysis
- Remora provides signal normalization and preprocessing utilities
- May be used for future features like base modification detection

### Qt/PySide6 Notes
- Plot display uses QLabel with QPixmap (not matplotlib embedding) for simplicity
- Plotnine plots are rendered to PNG buffers (BytesIO) rather than displayed directly
- Window geometry: 1200x800 default, with optimal split between read list and plot area
- File dialogs use native OS dialogs via Qt
- Search functionality is case-insensitive and filters the read list in real-time

### Testing Considerations
- Tests require sample POD5 files in `tests/data/` directory
- If sample data is missing, tests are skipped (not failed) via `pytest.skip()`
- Use `conftest.py` fixtures for shared test resources
- Mock POD5 file I/O for unit tests to avoid large test data files

## Coding Style and Conventions

### Code Formatting
- **Always** use `ruff` for formatting and linting (configured in `pyproject.toml`)
- Run `ruff format src/ tests/` before committing
- Run `ruff check --fix src/ tests/` to auto-fix linting issues
- Follow PEP 8 style guidelines (enforced by ruff)
- Line length: 88 characters (Black-compatible)
- Python compatibility: 3.8+ (specified in `target-version`)

### Code Organization
- Keep modules focused and single-purpose
- Separate GUI code (`viewer.py`, `dialogs.py`) from logic (`plotter.py`, `utils.py`)
- Put constants and configuration in `constants.py`
- Use type hints for function signatures (Python 3.8+ compatible)

### Documentation
- Add docstrings to all public functions and classes
- Use Google-style docstrings for consistency
- Update MkDocs documentation when adding user-facing features
- Keep CLAUDE.md updated with architectural changes

### Imports
- Use absolute imports: `from squiggy.utils import load_pod5`
- Group imports: stdlib, third-party, local (separated by blank lines)
- Avoid wildcard imports (`from module import *`)

### Error Handling
- Use specific exception types (not bare `except:`)
- Show user-friendly error messages via Qt dialogs
- Log errors for debugging (consider adding logging module)

### Git Commit Messages
- Use conventional commits format: `type(scope): description`
- Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`
- Examples:
  - `feat(viewer): add export plot functionality`
  - `fix(plotter): correct signal normalization`
  - `docs: update installation instructions`
