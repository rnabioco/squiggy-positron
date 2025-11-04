# Squiggy Release Notes

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
