# Squiggy Release Notes

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
