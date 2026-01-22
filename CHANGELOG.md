# Squiggy Release Notes

## v0.1.24-alpha (2026-01-22)

Pipeline session integration for external workflow support.

### Features

- **Pipeline Session Auto-Detection**: Automatically detects `squiggy-session.json` files in workspace folders and prompts to load them (#174)
- **Load Pipeline Session Command**: New command to manually load pipeline sessions via file picker
- **Relative Path Resolution**: Pipeline sessions support paths relative to the session file location, enabling portable session files
- **Checksum Validation**: Warns users when session files reference data that has changed since export

### Improvements

- **User Preferences**: "Don't Ask Again" option for pipeline session auto-detection prompts

## v0.1.23-alpha (2026-01-22)

Performance optimizations for POD5 and BAM file operations, plus RNA basecalling auto-detection.

### Features

- **RNA Basecalling Auto-Detection**: Automatically detects RNA basecalling models and enables RNA mode for correct signal orientation

### Performance

- **POD5 Read Access Optimization**: Use POD5's native indexed selection for O(1) read lookups instead of O(n) iteration through all reads
- **BAM Reference Lookup Optimization**: Use indexed `bam.fetch()` for O(log n) reference queries instead of O(n) full file scans
- **Batch Read Loading**: Optimized batch read operations using selection parameter with `missing_ok=True`

### Fixes

- **Read Explorer Table**: Improved column alignment and fixed reference collapse bug
- **Loading Timeouts**: Increased sample loading timeouts from 30s to 120s for larger files

## v0.1.22-alpha (2025-12-05)

New visualization features for BAM files without move tables and improved aggregate plot context.

### Features

- **Pileup-Only Visualization**: Added support for visualizing BAM files without move tables, displaying base pileup and quality tracks when signal alignment is not available (#171)
- **Reference Header Panel**: Aggregate plots now include a reference header panel showing reference name, position range, and optional sample name for better context (#170)
- **Genomic DNA Test Data Script**: Added script for extracting genomic DNA test data to support expanded testing scenarios (#172)

### Improvements

- **Kernel Restart**: Improved kernel restart reliability and user feedback
- **Status Bar Version Display**: Moved version display from sidebar title to status bar for cleaner UI
- **Startup Progress Messages**: Clarified startup progress messages for better user feedback

## v0.1.21-alpha (2025-12-05)

UX improvements with simplified kernel architecture and cleaner status notifications.

### Features

- **Status Bar Notifications**: Replaced notification pop-ups with non-intrusive status bar messages for a cleaner user experience (#167)

### Improvements

- **Simplified Kernel Architecture**: Streamlined to single dedicated kernel, eliminating complexity and improving reliability (#168)
- **Documentation**: Synced API reference documentation and updated example notebook (#169)

## v0.1.20-alpha (2025-12-03)

Pre-release polish with Python codebase modularization and improved reliability.

### Improvements

- **Python Codebase Refactoring**: Major modularization of Python codebase with dead code removal for better maintainability (#164)
- **Logging Improvements**: Enhanced logging throughout the extension for better debugging and diagnostics (#165)

### Fixes

- **Virtual Environment Discovery**: Fixed issue where squiggy venv was not being discovered correctly in some configurations (#165)
- **Documentation Cleanup**: Removed outdated manual install requirement from extension description

## v0.1.19-alpha (2025-12-02)

Critical fix for Python interpreter selection when installed from OpenVSX.

### Fixes

- **Python Interpreter Selection**: Fixed issue where extension would use system Python instead of the squiggy venv when installed from OpenVSX. The extension now explicitly finds and selects the squiggy venv runtime rather than relying on Positron's auto-discovery preference.

## v0.1.18-alpha (2025-12-02)

Major simplification of installation: Squiggy now bundles its Python package and automatically creates a virtual environment on first use. No external Python package installation required.

### Features

- **Bundled Python Package**: The squiggy Python library is now bundled directly with the extension, eliminating the need for manual `pip install squiggy-positron` or PyPI installation
- **Automatic Virtual Environment Setup**: Extension automatically creates and configures a dedicated virtual environment on first activation, ensuring isolated and reproducible environments (#163)

### Fixes

- **README Formatting**: Fixed markdown formatting issues in README documentation

## v0.1.17-alpha (2025-12-02)

Bug fixes, UX improvements, and new data filtering features for POD5/BAM workflows.

### Features

- **Adaptive Downsampling**: Extreme-scale optimization for visualizing large signal datasets (#159)
- **POD5/BAM Filtering Script**: New `filter_reads.py` script supporting POD5 directory input, BAM output, and end_reason filtering
- **Modification Heatmap Filters**: Added frequency and count filters for modification heatmap visualization (#157)

### Improvements

- **React 18 Migration**: Critical UX fixes including error boundaries and memory leak prevention (#151)
- **Type Safety**: Comprehensive type safety and performance improvements across the codebase (#153)
- **Accessibility**: Accessibility improvements for React panels following WCAG guidelines (#152)
- **CSS Refactoring**: Extracted inline CSS to dedicated stylesheets for Samples and Plot Options panels (#158)
- **Aggregate Plot Zoom**: Fixed zoom behavior in aggregate plots for better navigation
- **Documentation**: Reorganized documentation files following CLAUDE.md conventions

### Fixes

- **Demo Session Loading**: Fixed read count population when loading demo/session samples (#162)
- **Sample Unloading**: Resolved "sample not found" error after unloading demo data (#161)
- **Modal Dialogs**: Removed duplicate Cancel buttons from modal dialogs
- **POD5 Error Handling**: Added error handling to skip corrupted POD5 files gracefully
- **UUID Conversion**: Fixed UUID to string conversion for POD5 read IDs
- **Console Pollution**: Cleaned up print statements polluting console output

### Internal

- Coverage-based region identification refactoring in filter_reads.py
- Updated npm dependencies and security audit

## v0.1.16-alpha (2025-11-14)

Comprehensive release with improved UI organization, reference display standardization, and background kernel architecture for cleaner Variables pane.

### Features

- **Background Kernel Architecture**: Experimental dual-mode operation isolates extension UI from user's foreground kernel, eliminating Variables pane clutter (#143)
- **PyPI Publication**: Python package now published to production PyPI as `squiggy-positron` (#147)
- **Sample Management**: Delete samples from samples panel for better workflow control (#128)

### Improvements

- **Reference Display**: Standardized FASTA reference display with Signal/Sequence toggle across all panels (#141)
- **Reference Navigation**: Enhanced reference sequence navigation and exploration UI (#134)
- **Package Installation**: Redesigned Python package installation interface with better progress feedback (#138)
- **Delayed Activation**: Extension now activates only after Python package installation completes (#126)
- **Console Logging**: Added Python console logging functionality for better debugging (#130)

### Fixes

- **Variables Pane**: Cleaned up Positron Variables pane by hiding internal state variables (#145)
- **Composite Plots**: Fixed handling of mixed-reference samples in composite read plots (#133)
- **Composite Selection**: All selected samples now properly included in composite plots (#127)
- **Read Explorer**: Fixed empty state and auto-load issues (#115, #122)

### Documentation

- **Startup Documentation**: Updated to reflect modern startup procedure (#139)
- **Quickstart Guide**: Improved README quickstart section (#136)

## v0.1.15-alpha (2025-11-07)

Python packaging modernization and PyPI publication enhancements.

### Improvements

- **Python Packaging**: Modernized packaging configuration with updated pyproject.toml and migrated build workflow from pip+build to uv

### Fixes

- **Twine Publishing**: Fixed Twine command to use `uv tool run twine` instead of `uv run twine` for proper isolated execution

### Internal

- Published Python package to PyPI as squiggy-positron (#68)

## v0.1.14-alpha (2025-11-06)

Maintenance release with bug fixes for Read Explorer UI.

### Fixes

- **Read Explorer Empty State**: Fixed auto-loading of reads when POD5+BAM files are loaded, ensuring proper empty state handling (#115, #122)
- **Code Quality**: Fixed linting issues across the codebase

## v0.1.13-alpha (2025-11-06)

Major release with unified multi-sample workflow, Python library refactoring, performance optimizations, and comprehensive bug fixes.

### Features

- **Unified Multi-Sample Workflow**: Complete UI consolidation for loading and managing multiple samples with proper visualization selection (#97)
  - New Samples Panel with expandable cards, visualization checkboxes (👁️), color pickers, and inline renaming
  - Okabe-Ito colorblind-friendly palette auto-assigned to samples
  - Multi-sample registry loading via `squiggy.load_sample()` API
  - Read Explorer integration with sample selector dropdown
  - Auto-matching of POD5/BAM files by filename stem
  - Session-level and per-sample FASTA file support
- **Manual File Association**: Change BAM/FASTA files for samples via [Change] buttons with validation (#107)
  - POD5/BAM validation checks read ID overlap (blocks if 0%, warns if <20%)
  - BAM/FASTA validation checks reference name overlap (blocks if 0%, warns if <50%)
  - Clear error messages for mismatched files

### Performance

- **BAM Loading Optimization**: 3-4x faster initial loads, 74-400x faster cached loads (#96)
  - Consolidated 4 separate file scans into 1 efficient pass
  - Full metadata caching (references, modifications, event alignment, read mapping)
  - 100K reads: ~8s initial vs ~30s previously, <20ms cached vs ~10s
  - 1M reads: ~60s initial vs ~3min previously, <50ms cached vs minutes

### Improvements

- **Python Library Refactoring**: 84% reduction in `__init__.py` size, improved code organization (#109)
  - Created `squiggy/plotting.py` (1,059 lines) - all 6 plotting functions moved from `__init__.py`
  - Created `squiggy/logging_config.py` - centralized logging with `SQUIGGY_LOG_LEVEL` env var support
  - Reduced `__init__.py` from 1,268 → 198 lines
  - Replaced print statements with proper logging throughout codebase
- **Test Data Consolidation**: Eliminated ~1.9 MB repository bloat (#99)
  - Single source of truth: `squiggy/data/` (removed duplicate `tests/data/`)
  - Added `get_test_data_path()` utility function
  - Updated all references and Git LFS patterns
- **Test Coverage**: Python coverage improved from 69% → 72% (#108)
  - Added 44 new tests for `squiggy/utils.py` (47% → 56% coverage)
  - Tests for ModelProvenance, sequence utilities, file validation, BAM analysis, comparison utilities

### Fixes

- **Aggregate Plot Improvements**: Fixed multiple critical bugs (#95)
  - Fixed modification coverage calculation (was always 0)
  - Fixed modification probability statistics bias
  - Fixed base pileup position mapping using `get_aligned_pairs()` for correct indel handling
  - Added relative coordinate transformation (x-axis renumbering starting at position 1)
  - Improved Modifications Explorer panel labels ("Modification Confidence Filter")
- **Soft-Clip Boundary Handling**: Plots no longer extend beyond aligned regions (#93)
  - Added `get_base_annotation_boundaries()` and `get_base_annotation_range()` utilities
  - Aggregate plots use base annotation boundaries instead of full signal length
  - X-axis clipping feature with consensus-based thresholding (>50% max coverage)
  - New UI toggle: "Clip x-axis to consensus region" (default: enabled)

### Internal

- **Version Management**: Auto-update package-lock.json in version sync script
- **CI/CD**: Remove duplicated data from vsix bundles

## v0.1.12-alpha (2025-11-04)

Aggregate plot enhancements with modification heatmaps, dwell time visualization, and improved UI clarity.

### Features

- **Modification Heatmap Panel**: Visualize base modification patterns in aggregate plots with frequency-based opacity showing both modification prevalence and confidence (#76)
- **Dwell Time Track**: Display mean dwell time with confidence bands across reference positions in aggregate plots (#76)
- **Dynamic Panel Visibility**: Toggle individual panels (modifications, pileup, dwell time, signal, quality) in aggregate plots via UI checkboxes (#76)
- **Modifications Explorer Integration**: Aggregate plots now respect modification filters (probability thresholds and enabled types) from the Modifications Explorer panel (#76)

### Improvements

- **UI Clarity**: Renamed "Plot Type" → "Analysis Type" and "Plot Mode" → "View Mode" with "Single Read" → "Standard" to eliminate confusing duplication (#76)
- **Modification Visualization**: Heatmap opacity based on `frequency × probability` with automatic normalization for visibility across different filtering thresholds (#76)
- **Dwell Time Auto-scaling**: Y-axis automatically adapts to data range after zoom operations (#76)
- **Panel Naming**: "Plot Options" renamed to "Advanced Plotting" to reflect expanded functionality (#76)

### Fixes

- **Type Coercion**: Fixed modification filtering by converting integer ChEBI codes to strings for consistent comparison with UI filters (#76)
- **Coverage Calculation**: Modification frequency now correctly calculated from total coverage (all reads) not just modified reads (#76)

### Documentation

- Updated README.md with aggregate plot features
- Added comprehensive "Aggregate Plots" section to User Guide with interpretation tips
- Created issue #75 to revisit modification heatmap with real-world data

## v0.1.11-alpha (2025-11-04)

Session management improvements with demo session support and enhanced code quality.

### Features

- **Session Management Panel**: Added new "Session Manager" panel with demo session support for quick onboarding and testing (#81)

### Documentation

- **React-First UI Guidelines**: Added comprehensive React-first UI development guidance to CLAUDE.md for consistent panel implementation

### Internal

- **CI Configuration**: Disabled Codecov PR comments to reduce notification noise
- **Release Process**: Added quality checks (linting and formatting) to release workflow

## v0.1.10-alpha (2025-11-04)

UX improvement with search mode toggle for better reference navigation.

### Improvements

- **Search Mode Toggle**: Added toggle button in reads explorer search bar to switch between reference name search (default) and read ID search. This fixes an issue where searching for reference names would clear the panel in lazy-loading mode, and provides a cleaner UX by separating the two search use cases.

## v0.1.9-alpha (2025-11-04)

Performance and UX improvements with optimized data loading and enhanced reads explorer.

### Improvements

- **Reads Explorer UX**: Added sortable "Reads" column, sticky reference headers when scrolling, and proper refresh functionality that queries backend state
- **POD5/BAM Loading Performance**: 30-40x speedup in data loading through optimized reference mapping and batch processing (#73)
- **Aggregate Plot Annotations**: Added reference base annotations to aggregate plots for better context (#74)
- **Documentation**: Renamed all documentation files to lowercase convention and fixed internal cross-references

### Fixes

- **Downsampling Default**: Changed default downsampling factor to 5 for better signal quality preservation

## v0.1.8-alpha (2025-11-03)

Multi-sample comparison feature with delta visualization and improved documentation.

### Features

- **Multi-Sample Comparison**: Load multiple samples (POD5 + BAM pairs) and visualize signal differences with delta plots showing deviation from reference baseline (#71)
- **Sample Management UI**: New "Sample Comparison Manager" panel for loading, viewing, and managing comparison sessions with interactive controls

### Improvements

- **Visual Identity**: New Squiggy logo with googly eyes and green EKG trace for better brand recognition
- **Documentation**: Added screenshot to documentation and comprehensive multi-sample comparison guide

### Fixes

- **Motif Aggregate Coordinates**: Fixed coordinate system using proper BAM alignment for accurate motif positioning (#71)
- **Delta Comparison**: Improved plot layout and delta calculation implementation for clearer visualization

## v0.1.7-alpha (2025-11-02)

State management improvements with FASTA integration and cleaner Python namespace.

### Improvements

- **Unified Session State**: Integrated FASTA file support into the `_squiggy_session` object, consolidating all file state into a single Variables pane entry
- **Cleaner Namespace**: Eliminated legacy global variables (`_squiggy_reader`, `_squiggy_read_ids`, etc.) from Python namespace by migrating to session-based approach
- **UI Panel Naming**: Renamed panels to "Modifications Explorer" and "Motif Explorer" for clarity

### Fixes

- **Kernel Restart**: Fixed FASTA files and reads explorer not clearing properly after kernel restart
- **Path Objects**: Converted Path objects to strings in API classes to prevent namespace pollution

## v0.1.6-alpha (2025-11-02)

Major refactoring release with improved code organization, new motif search features, and comprehensive documentation updates.

### Features

- **Motif Search and Visualization**: Search for IUPAC sequence motifs in reference genomes and visualize aggregate signal patterns centered on motif matches (#43, #51)
- **API Documentation**: Auto-generated API reference using mkdocstrings with code examples (#50)

### Improvements

- **Strategy Pattern Architecture**: Complete refactoring of plotting system using Strategy Pattern with 5 plot strategies (SINGLE, OVERLAY, STACKED, EVENTALIGN, AGGREGATE) for better maintainability and extensibility (#62)
- **Rendering Package**: Created `squiggy/rendering/` package with reusable components (ThemeManager, BaseAnnotationRenderer, ModificationTrackBuilder)
- **Session State**: Consolidated kernel state into single `_squiggy_session` object for cleaner Variables pane (#59)
- **Launch Configuration**: Template-based `.vscode/launch.json` system for consistent test workspace setup (#63)

### Fixes

- **CI/CD**: Fixed shell command injection vulnerability in release workflow that caused jobs to hang
- **Documentation**: Added `docs-build` task for CI to prevent indefinite server runs
- **Read Explorer**: Fixed reference dropdown expansion bug

### Documentation

- Updated all documentation to reflect new rendering/ package structure and Strategy Pattern architecture
- Removed outdated references to deleted `plotter.py` module (2,049 lines removed)
- Added comprehensive Copilot/AI agent instructions
- Fixed API documentation code example rendering

### Internal

- Removed legacy `plotter.py` and `SquigglePlotter` class in favor of Strategy Pattern
- Improved test workspace directory naming with branch and commit info
- Updated `.gitignore` and VSIX package exclusions
- Changed email notification setting to 'off'

## v0.1.5-alpha (2025-11-01)

Maintenance release focused on code quality and internal architecture improvements.

### Improvements

- **Modularized Extension Architecture**: Major refactoring of TypeScript codebase reducing main extension.ts by 88% (993 → 116 lines). Logic now organized into specialized modules for better maintainability and type safety.

### Internal

- Fixed /release command to avoid pushing all tags

## v0.1.4-alpha (2025-11-01)

Maintenance release with version management improvements and UI enhancements.

### Improvements

- **Automated Version Synchronization**: Added sync script to maintain version consistency across package.json, Python package, and sidebar title. Version in package.json is now single source of truth.
- **Files Panel Upgrade**: Migrated to React webview with sortable table layout for better performance and user experience

### Documentation

- Updated initialization files to reflect recent structural changes

## v0.1.0 (2025-10-31)

### 🎉 Initial Release

First production release of Squiggy as a Positron IDE extension for Oxford Nanopore signal visualization.

### ✨ Features

- **Positron IDE Integration**: Works seamlessly with active Python kernel
- **Interactive Visualizations**: Bokeh-powered plots with zoom, pan, and hover tooltips
- **POD5 Support**: Load and visualize Oxford Nanopore signal data
- **BAM Integration**: Overlay base annotations and modifications on signal plots
- **Event-Aligned Plotting**: Base-level resolution with move table support
- **Modification Analysis**: Filter and visualize base modifications (5mC, 6mA, etc.) with probability thresholds
- **Advanced Filtering**: Search reads by ID, reference sequence, or modification type
- **Plot Export**: Save visualizations as HTML (interactive), PNG, or SVG
- **Test Data Bundled**: Includes yeast tRNA example data (~2MB) for quick experimentation

### 🔧 Installation & Environment

- **PEP 668 Compliance**: Automatically detects externally-managed Python environments (Homebrew, system Python)
- **Virtual Environment Detection**: Identifies venv and conda environments
- **Smart Installation**:
  - Automatic installation in safe environments (venv/conda)
  - Clear guidance for system Python users to create virtual environments
  - Manual installation guide with copy-able commands
- **Zero Console Pollution**: Silent environment checks using Positron's variable access API
- **Environment Re-checking**: Detects manual installations between operations

### 🐍 Python Package

- **Dependencies**: Includes pod5, pysam, bokeh, numpy
- **Python Version**: Requires 3.12+
- **Signal Processing**:
  - Normalization methods: Z-score, Median, MAD
  - Smart downsampling for large signals (100k+ samples)
  - Dwell time scaling options

### 📚 Documentation

- Comprehensive README with virtual environment setup instructions
- User guide with detailed Python environment configuration
- Developer guide for extension development
- Clear PEP 668 error handling and guidance

### 🔨 Technical Improvements

- Proper error handling for BAM file loading
- Fixed missing squiggy availability check in `openBAMFile()`
- Improved installation check logic to detect manual installations
- Enhanced error messages for PEP 668 scenarios
- Test data properly bundled in packaged extension

### ⚙️ Requirements

- **Positron IDE**: 2025.6.0+
- **Python**: 3.12+ in virtual environment (venv or conda recommended)
- **squiggy package**: Installed automatically or manually via pip

### ⚠️ Known Limitations

- Requires active Python console in Positron
- BAM files must be indexed (.bai file required)
- Event-aligned plots require 'mv' tag in BAM (from dorado/guppy basecallers)

### 👥 Contributors

- Jay Hesselberth (@jayhesselberth)
- With assistance from Claude Code
