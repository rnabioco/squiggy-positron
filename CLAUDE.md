# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Squiggy is a desktop GUI application for visualizing Oxford Nanopore sequencing data ("squiggle plots") from POD5 files. It's built as a standalone application that can be distributed as native executables for Windows, macOS, and Linux.

The application supports optional base annotations overlaid on signal data when a BAM file with basecalls is provided.

## Architecture

### Application Stack
- **GUI Framework**: PySide6 (Qt for Python) - provides cross-platform desktop UI
- **Async Framework**: qasync - integrates Python's asyncio with Qt event loop for non-blocking I/O
- **Data Processing**: pod5 library for reading Oxford Nanopore POD5 files
- **Visualization**: Bokeh for interactive squiggle plots with zoom, pan, and hover tools
- **Web Rendering**: QWebEngineView for displaying interactive Bokeh plots in Qt
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
├── plotter_bokeh.py   # BokehSquigglePlotter - Interactive bokeh plotting
├── alignment.py       # Base annotation data structures (BaseAnnotation, AlignedRead)
├── normalization.py   # Signal normalization functions (z-norm, median, MAD)
├── dialogs.py         # Custom dialog windows (About, Reference Browser, etc.)
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
- Initializes Qt application with qasync event loop for async/await support
- Launches SquiggleViewer with async file loading if `--file` provided
- Defines `main()` function as console script entry point

**src/squiggy/viewer.py** - Main GUI window (SquiggleViewer):
- File menu with "Open POD5 File" and "Open Sample Data" options
- Three-panel layout: read list (left), plot display (center), controls (right)
- Bottom search panel with three modes: Read ID, Reference Region, Sequence
- CollapsibleBox widgets for organizing file info and plot settings
- QWebEngineView for displaying interactive Bokeh plots
- Uses async methods (@qasync.asyncSlot) for non-blocking file I/O and plot generation

**src/squiggy/plotter_bokeh.py** - Bokeh plotting engine (BokehSquigglePlotter):
- Converts raw signal data into interactive Bokeh figures
- Four plot modes: SINGLE, OVERLAY, STACKED, EVENTALIGN
- Interactive features: zoom, pan, reset, hover tooltips, base annotation toggle
- Generates HTML with embedded JavaScript for Qt WebEngine display
- Handles signal normalization and downsampling for performance

**src/squiggy/alignment.py** - Base annotation data structures:
- BaseAnnotation dataclass: maps bases to signal positions
- AlignedRead dataclass: stores read sequence with genomic alignment info
- extract_alignment_from_bam(): parses BAM files for event-aligned data

**src/squiggy/normalization.py** - Signal normalization:
- normalize_signal(): dispatcher for normalization methods
- z_normalize(): Z-score normalization (mean=0, std=1)
- median_normalize(): Median-centered normalization
- mad_normalize(): Median absolute deviation (robust to outliers)

**src/squiggy/dialogs.py** - Custom dialog windows:
- AboutDialog: version and license information
- ReferenceBrowserDialog: browsable table of reference sequences from BAM files
- ExportDialog: format selection (HTML/PNG/SVG), dimension controls with aspect ratio lock, zoom-level export option
- CollapsibleBox widget: reusable UI component for expandable sections

**src/squiggy/utils.py** - Utility functions:
- POD5 file reading and validation
- Signal data extraction and processing with downsampling
- BAM file operations: indexing, reference extraction, region queries
- Sample data location and loading
- reverse_complement(): DNA sequence utilities
- get_reference_sequence_for_read(): extract reference sequence for alignment

**src/squiggy/constants.py** - Application constants:
- PlotMode enum: SINGLE, OVERLAY, STACKED, EVENTALIGN
- NormalizationMethod enum: NONE, ZNORM, MEDIAN, MAD
- Okabe-Ito colorblind-friendly base colors (purines=orange, pyrimidines=blue)
- Default window sizes (1200x800) and UI settings
- Signal downsampling thresholds for performance

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
1. User selects read(s) from list (supports multi-select with Ctrl/Cmd/Shift)
2. `utils.py` extracts signal data and metadata for selected read(s)
3. Signal data passed to `plotter_bokeh.py` BokehSquigglePlotter
4. Signal normalized using selected method (NONE, ZNORM, MEDIAN, or MAD)
5. Bokeh generates interactive figure with zoom/pan tools and hover tooltips
6. HTML with embedded JavaScript generated via `file_html()` with CDN resources
7. HTML loaded into QWebEngineView for display
8. Read information and statistics displayed in status bar

**Event-Aligned Visualization (with BAM file):**
1. User loads BAM file with base-to-signal alignment ("mv" tag required)
2. Select read from list and switch to EVENTALIGN plot mode
3. `alignment.py` extracts base annotations from BAM alignment
4. Bokeh renders bases as colored rectangles over signal trace
5. Interactive toggle button allows showing/hiding base letters
6. Base colors use Okabe-Ito palette for colorblind accessibility

**Plot Export:**
1. User selects "Export Plot" from File menu (Ctrl/Cmd+E)
2. `ExportDialog` displays with format options (HTML/PNG/SVG), dimension controls, and zoom-level checkbox
3. If "Export current zoom level" is checked:
   - JavaScript bridge (`_get_current_view_ranges()`) executes JS in QWebEngineView
   - Extracts current x_range and y_range from Bokeh plot's interactive state
   - Returns tuples: `((x_start, x_end), (y_start, y_end))`
4. Export methods (`_export_html()`, `_export_png()`, `_export_svg()`) called based on format
5. For PNG/SVG exports:
   - Store original figure dimensions, sizing_mode, and ranges
   - Temporarily set `sizing_mode = None` to enable explicit dimensions
   - Apply custom width/height and optional zoom ranges to figure
   - Call Bokeh's `export_png()` or `export_svgs()` (requires selenium + geckodriver)
   - Restore original figure state in finally block
6. For HTML exports:
   - Write stored `current_plot_html` directly to file (no modification needed)
7. Success message displayed with export details

## Development Commands

### Environment Setup
```bash
# Clone the repository
git clone https://github.com/rnabioco/squiggy.git
cd squiggy

# Install dependencies (using uv - recommended)
uv pip install -e ".[dev]"

# Or using pip
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
# Run all tests (using uv - recommended)
uv run pytest tests/

# Run with verbose output
uv run pytest -v tests/

# Run specific test file
uv run pytest tests/test_bokeh_plotting.py

# Run with coverage report
uv run pytest --cov=squiggy tests/

# Or using pytest directly (if installed globally)
pytest -v tests/
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
# Install PyInstaller (using uv - recommended)
uv pip install pyinstaller

# Or using pip
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
- **Export dependencies** for PNG/SVG:
  - `selenium`, `PIL`, `pillow` added to `hiddenimports`
  - Selenium 4.6+ includes Selenium Manager which auto-downloads geckodriver when first needed
  - No manual geckodriver installation or bundling required
  - Driver is cached in user's home directory (`~/.cache/selenium/`) for reuse
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

### Signal Analysis
- Signal normalization handled by `normalization.py` module with four methods:
  - NONE: Raw signal values (picoamperes)
  - ZNORM: Z-score normalization (mean=0, std=1) - best for comparing across reads
  - MEDIAN: Median-centered (median=0) - simple baseline correction
  - MAD: Median absolute deviation - robust to outliers
- Signal downsampling uses LTTB algorithm for >100K samples to maintain visual fidelity
- Base annotation visualization uses Okabe-Ito colorblind-friendly palette
- Base-to-signal alignment extracted from BAM "mv" (move table) tag
  - Move table format: `[stride, move_0, move_1, ...]` where stride is neural network downsampling factor
  - Stride values: 5 for DNA models (R9.4.1, R10.4.1), 10-12 for RNA models
  - Each move table position represents `stride` signal samples
  - Dwell time calculation accounts for stride to produce realistic values (1-20 ms per base)
- Future features may include base modification detection and methylation calling

### Qt/PySide6 Notes
- Plot display uses QWebEngineView to render interactive Bokeh HTML plots
- Bokeh plots are generated as HTML with embedded JavaScript and loaded via `setHtml()`
- Window geometry: 1200x800 default, with three-panel splitter layout
- File dialogs use native OS dialogs via Qt
- Search functionality has three modes:
  - Read ID: case-insensitive filtering of read list in real-time
  - Reference Region: queries BAM file for reads in genomic regions (chr1:1000-2000)
  - Sequence: searches reference sequences for DNA motifs (with reverse complement option)
- CollapsibleBox widgets provide expandable sections for file info and plot controls
- JavaScript interaction: viewer.py can execute JavaScript on Bokeh plots for programmatic zoom

### Async Programming with qasync
- **qasync** integrates Python's `asyncio` with Qt's event loop for non-blocking operations
- All blocking I/O operations (POD5 file reading, plot generation) run in thread pools via `asyncio.to_thread()`
- UI methods that perform blocking operations are decorated with `@qasync.asyncSlot()` for async execution
- Status messages are shown during async operations to inform users of progress
- **Best practices:**
  - Use `async def` for any method that performs file I/O, network requests, or heavy computation
  - Decorate async slot handlers (connected to Qt signals) with `@qasync.asyncSlot()`
  - Move blocking operations to separate `_blocking()` methods and call them with `asyncio.to_thread()`
  - Always update UI elements on the main thread after awaiting async operations
  - Provide user feedback (status bar messages, loading indicators) for long-running async tasks
- **Key async methods:**
  - `load_read_ids()` - Loads POD5 read IDs without blocking the UI
  - `update_file_info()` - Calculates file statistics asynchronously
  - `display_squiggle()` - Generates Bokeh HTML plots in background thread
  - `open_pod5_file()` / `open_sample_data()` - File opening with async loading
  - `filter_reads_by_region()` - Queries BAM file for reads in genomic regions
  - `search_sequence()` - Searches reference sequences for DNA motifs
  - `_generate_plot_blocking()` - Blocking function for Bokeh HTML generation (called via `asyncio.to_thread()`)

### Testing Considerations
- Tests require sample POD5 files in `tests/data/` directory
- If sample data is missing, tests are skipped (not failed) via `pytest.skip()`
- Use `conftest.py` fixtures for shared test resources
- Mock POD5 file I/O for unit tests to avoid large test data files

## Common Development Tasks

### Adding a New Feature to the UI
1. **Add UI elements** in `viewer.py` `init_ui()` method or related layout methods
2. **Connect signals** using Qt's signal/slot mechanism (e.g., `button.clicked.connect(handler)`)
3. **Create async handler** using `@qasync.asyncSlot()` decorator for any blocking operations
4. **Update constants** in `constants.py` if adding new configuration values
5. **Write tests** in `tests/` directory using pytest fixtures from `conftest.py`
6. **Update documentation** in `docs/` if user-facing feature

### Testing with Sample Data
The repository includes sample data in `tests/data/`:
- `mod_reads.pod5` - Small POD5 file with reads for quick testing
- `mod_mappings.bam` - Corresponding BAM file with basecalls and alignments
- Use these for development: `squiggy -p tests/data/mod_reads.pod5 -b tests/data/mod_mappings.bam`
- To test event-aligned mode with base annotations: load files with `-p` and `-b` flags, select read, switch to EVENTALIGN mode

### Debugging Tips
- **Qt issues**: Check Qt Designer documentation and PySide6 examples
- **Async issues**: Ensure all blocking I/O is wrapped in `asyncio.to_thread()`
- **Plot issues**: Test plotting functions independently in `plotter_bokeh.py`
- **BAM parsing**: Use `pysam.AlignmentFile(..., check_sq=False)` to avoid header issues
- **POD5 files**: Always use context managers to ensure proper cleanup
- **Bokeh JavaScript**: Use browser dev tools to debug Bokeh interactions - QWebEngineView uses Chromium
- **WebEngine debugging**: Enable remote debugging with `--remote-debugging-port=9222` environment variable

### UI Design Patterns

**CollapsibleBox Widget (`viewer.py:63-107`)**:
The CollapsibleBox is a reusable expandable/collapsible section widget:
- Uses QToolButton with arrow icon that rotates on toggle
- QScrollArea with animated height transition (QPropertyAnimation)
- Height animates from 0 to content height over 300ms
- Usage: `box = CollapsibleBox("Title")` then `box.set_content_layout(your_layout)`
- Default state: collapsed (height=0)

**Three-Panel Splitter Layout**:
Main window uses QSplitter with three sections:
- Left panel (20%): Collapsible file info and plot controls
- Center panel (60%): QWebEngineView for Bokeh plots
- Right panel (20%): Read list with multi-select support
- Bottom: Search panel with mode-specific controls

**Color Scheme (Okabe-Ito Palette)**:
Base colors changed from standard to colorblind-friendly palette:
- Purines (A, G): Orange/warm colors (#E69F00, #D55E00)
- Pyrimidines (C, T): Blue/cool colors (#0072B2, #56B4E9)
- Signal line: Gray (#A9A9A9) for better contrast with base annotations
- Unknown (N): Gray (#808080)
- See `constants.py:33-43` for full color definitions

### Working with the Reference Browser
The reference browser dialog (`dialogs.py:102-263`) is instantiated from `viewer.py:834-868`:
- To test: Load a BAM file, switch to "Reference Region" search mode, click "Browse References"
- Data comes from `utils.py:get_bam_references()` which returns list of dict with keys: name, length, read_count
- Selection populates the search field and triggers automatic search

### Using Sequence Search
The sequence search feature (`viewer.py:1208-1440`) allows finding DNA motifs in reference sequences:
- Requires: BAM file loaded, read selected, EVENTALIGN plot mode
- Search modes: forward strand only, or include reverse complement
- Search function (`_search_sequence_in_reference()`) runs in background thread
- Results displayed in CollapsibleBox with clickable items
- Clicking result executes JavaScript to zoom Bokeh plot to matched region
- Uses `utils.py:get_reference_sequence_for_read()` to extract reference sequence
- Uses `utils.py:reverse_complement()` for reverse complement searches

### Making Changes to Plotting
- Main plotting logic is in `plotter_bokeh.py` using Bokeh for interactive plots
- Four plot modes: SINGLE, OVERLAY, STACKED, EVENTALIGN (defined in `constants.py`)
- Plots are generated as Bokeh figure objects, converted to HTML with `file_html()`
- HTML loaded into QWebEngineView via `setHtml()` for interactive display
- Signal normalization handled by `normalization.py`: NONE, ZNORM, MEDIAN, MAD
- Interactive tools: zoom (wheel/box), pan, reset, hover tooltips
- Base annotations use LabelSet and Rect glyphs with CustomJS toggle callbacks
- Color scheme uses Okabe-Ito palette for colorblind accessibility

### Adding New File Format Support
1. Add parsing logic to `utils.py`
2. Update file dialogs in `viewer.py` to accept new extensions
3. Ensure async loading with `asyncio.to_thread()` for blocking I/O
4. Add validation similar to `validate_bam_reads_in_pod5()`

## Known Issues and Gotchas

### macOS-Specific
- **"Python" in menu bar**: Requires PyObjC (`uv pip install -e ".[macos]"`). Fixed via `set_macos_app_name()` in `main.py:17-29`
- **App icon**: Works in .app bundle but may not show when running from CLI
- **File dialogs**: Use native macOS dialogs automatically via Qt

### Qt/PySide6 Issues
- **Signal/slot connections**: Must connect before showing widget, or connection won't fire
- **Main thread requirement**: All UI updates must happen on main thread (safe in `@qasync.asyncSlot()`)
- **QWebEngineView loading**: HTML content loads asynchronously - use `loadFinished` signal if timing matters
- **Layout updates**: Call `layout.update()` or `widget.adjustSize()` after dynamic changes
- **CollapsibleBox animation**: Use `QPropertyAnimation` on `maximumHeight` for smooth expand/collapse
- **Splitter sizes**: Set proportional sizes with `setSizes([left_ratio, right_ratio])` based on total width

### Async/qasync Gotchas
- **Decorator required**: Qt slots that use `await` MUST have `@qasync.asyncSlot()` decorator
- **Blocking calls**: Any I/O or CPU-intensive work must be wrapped in `asyncio.to_thread()`
- **Background threads**: Never call Qt methods from background threads (use signals or await back to main thread)
- **Event loop**: Only one event loop per app - initialized in `main()` via `qasync.QEventLoop(app)`

### POD5 File Handling
- **Read objects are temporary**: Don't store Read objects outside context manager
- **Store IDs only**: Store `str(read.read_id)` not the Read object itself
- **Large files**: Files with >10,000 reads may require batched loading
- **VBZ compression**: Requires `vbz_h5py_plugin` (installed automatically with pod5)

### BAM/pysam Issues
- **check_sq=False required**: Use `pysam.AlignmentFile(..., check_sq=False)` to avoid SQ line issues
- **Index required for region queries**: BAM must be indexed (.bai) for `fetch(chrom, start, end)`
- **Move table tag**: Base-to-signal mapping requires "mv" tag in BAM (created by dorado/guppy)
  - Format: `mv:B:c,stride,move_0,move_1,...` (first element is stride, rest are moves)
  - Extract stride: `stride = int(move_table[0])`
  - Extract moves: `moves = move_table[1:]`
  - Signal position calculation: `signal_pos += stride` (not `+= 1`)
  - Reference: https://github.com/hiruna72/squigualiser/blob/main/docs/move_table.md
- **Read iteration**: Use `fetch(until_eof=True)` to iterate all reads without index

### Bokeh/Plotting
- **Memory usage**: Large plots with many data points can slow rendering - automatic downsampling applied for >100K samples
- **HTML generation**: Use `file_html(figure, CDN)` to generate standalone HTML with CDN resources
- **Interactive tools**: Configure tools in figure creation: `tools="pan,wheel_zoom,box_zoom,reset,save"`
- **JavaScript callbacks**: Use `CustomJS` for client-side interactions (e.g., toggle base annotations)
- **CDN resources**: Bokeh uses CDN for JavaScript libraries - requires internet connection on first load
- **Data sources**: Use `ColumnDataSource` for all glyphs - enables hover tooltips and dynamic updates
- **Color mapping**: Use `LinearColorMapper` with `transform()` for continuous color scales
- **Performance**: For event-aligned plots with many bases, consider limiting label rendering to visible zoom range

### QWebEngineView/Web Rendering
- **setHtml() vs setUrl()**: Use `setHtml(html_content, baseUrl)` for embedding generated HTML
- **Base URL required**: Set `baseUrl=QUrl("http://localhost/")` to enable CDN resource loading
- **JavaScript execution**: Use `page().runJavaScript(js_code)` to execute JS on loaded Bokeh plots
- **Async rendering**: Page loads asynchronously - wait for `loadFinished` signal if needed
- **Context isolation**: JavaScript runs in separate context from Python - no direct object sharing
- **Debugging**: Enable web inspector with `settings().setAttribute(QWebEngineSettings.JavascriptEnabled, True)`
- **Memory**: Each QWebEngineView instance runs a separate Chromium process - can be resource intensive

### Testing Gotchas
- **Sample data required**: Tests skip (not fail) if `tests/data/*.pod5` missing
- **Relative paths**: Run `uv run pytest` from project root, not `tests/` directory
- **Qt application**: Can't create multiple QApplication instances in same process
- **Async tests**: Use pytest-asyncio or run async functions via `asyncio.run()`

### PyInstaller Build Issues
- **Hidden imports**: Some imports aren't auto-detected - add to `hiddenimports` in spec file
  - Bokeh may require: `bokeh.models`, `bokeh.palettes`, `bokeh.transform`
  - QWebEngine requires: `PySide6.QtWebEngineCore`, `PySide6.QtWebEngineWidgets`
  - Export features require: `selenium`, `PIL`, `pillow`
- **Selenium Manager**: Selenium 4.6+ automatically downloads geckodriver when needed
  - No manual geckodriver installation required on build machine
  - No geckodriver bundling in PyInstaller spec required
  - Driver cached in user's `~/.cache/selenium/` directory on first use
  - Requires internet connection on first PNG/SVG export to download driver
- **Data files**: Must explicitly list in `datas` parameter
- **Platform differences**: Test builds on target platform (macOS != Linux != Windows)
- **Permissions**: macOS requires code signing for distribution outside App Store
- **Size**: Bundled Qt + QWebEngine adds ~150MB to executable size due to Chromium
  - Geckodriver adds ~6MB per platform
- **WebEngine resources**: QWebEngine requires additional resources - ensure Qt plugins are bundled correctly

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
- Separate GUI code (`viewer.py`, `dialogs.py`) from logic (`plotter_bokeh.py`, `utils.py`)
- Separate data structures (`alignment.py`) from algorithms (`normalization.py`)
- Put constants and configuration in `constants.py`
- Use type hints for function signatures (Python 3.8+ compatible)
- New modules should be added to the Project Structure documentation in CLAUDE.md

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
