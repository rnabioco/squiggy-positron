# Squiggy

A desktop application for visualizing Oxford Nanopore sequencing data from POD5 files.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)

## Overview

Squiggy is a cross-platform GUI application that allows you to visualize raw nanopore signal data (squiggle plots) from POD5 files. It provides an intuitive interface for browsing reads, searching by read ID, and displaying time-series plots of the electrical current signal.

## Features

- **POD5 File Support**: Native support for Oxford Nanopore's POD5 file format
- **Bundled Sample Data**: Get started immediately with included example POD5 files
- **Interactive Read Browser**: Browse and search through all reads in a POD5 file
- **High-Quality Plots**: Generate publication-ready squiggle plots with customizable styling
- **Cross-Platform**: Available as standalone executables for Windows, macOS, and Linux
- **Fast Performance**: Efficient handling of large POD5 files with thousands of reads

## Installation

### Option 1: Download Pre-built Executables (Recommended)

Download the latest release for your platform:

- **macOS**: [Squiggy-macos.dmg](https://github.com/rnabioco/squiggler/releases/latest)
- **Windows**: [Squiggy-windows-x64.zip](https://github.com/rnabioco/squiggler/releases/latest)
- **Linux**: [Squiggy-linux-x64.tar.gz](https://github.com/rnabioco/squiggler/releases/latest)

For the latest development build (macOS only), download from the ["latest" release](https://github.com/rnabioco/squiggler/releases/tag/latest).

### Option 2: Install from Source

```bash
# Clone the repository
git clone https://github.com/rnabioco/squiggler.git
cd squiggler

# Install dependencies
pip install -r requirements.txt

# Run the application
python squiggy/squiggy/main.py
```

Or install as a package:

```bash
pip install -e .
squiggy
```

## Usage

### Quick Start with Sample Data

Squiggy comes with bundled sample data to help you get started:

1. **Launch Squiggy**: Open the application
2. **Open Sample Data**: Go to **File → Open Sample Data** (or press `Ctrl+Shift+O` / `Cmd+Shift+O`)
3. **Explore**: Browse the sample reads and click any read to view its squiggle plot

### Working with Your Own Data

1. **Launch Squiggy**: Open the application
2. **Open POD5 File**: Click "Open POD5 File" or go to **File → Open POD5 File...** (`Ctrl+O` / `Cmd+O`)
3. **Browse Reads**: All read IDs will be displayed in the left panel
4. **Search**: Use the search box to filter reads by ID
5. **View Squiggle**: Click any read to display its squiggle plot

### Command Line Options

```bash
# Launch GUI
squiggy

# Launch with a specific file pre-loaded
squiggy --file data.pod5

# Run from source
python squiggy/squiggy/main.py

# Run from source with file
python squiggy/squiggy/main.py --file data.pod5
```

## Development

### Setup Development Environment

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Format code
black squiggy/

# Run tests
pytest tests/
```

### Building Executables

Build standalone executables for distribution:

```bash
# Install PyInstaller
pip install pyinstaller

# Build for your current platform
pyinstaller squiggy.spec

# Output will be in dist/ directory
```

Platform-specific packaging:

```bash
# macOS: Create DMG
brew install create-dmg
pyinstaller squiggy.spec
create-dmg --volname "Squiggy Installer" --app-drop-link 600 185 Squiggy-macos.dmg dist/Squiggy.app

# Windows: Create ZIP
pyinstaller squiggy.spec
Compress-Archive -Path dist/Squiggy.exe -DestinationPath Squiggy-windows-x64.zip

# Linux: Create tarball
pyinstaller squiggy.spec
tar -czf Squiggy-linux-x64.tar.gz -C dist Squiggy
```

## Requirements

### Runtime Requirements
- Python 3.8 or higher (for source installation)
- For POD5 files with VBZ compression: `vbz_h5py_plugin` (automatically installed)

### Dependencies
- `numpy>=1.20.0` - Array operations
- `pandas>=1.3.0` - Data manipulation
- `pod5>=0.3.0` - POD5 file reading
- `PySide6>=6.5.0` - GUI framework
- `plotnine>=0.12.0` - Plot generation

## Architecture

Squiggy is built with:
- **GUI**: PySide6 (Qt for Python) for cross-platform desktop interface
- **Data Processing**: POD5 library for reading Oxford Nanopore files
- **Visualization**: plotnine (ggplot2-style) for creating squiggle plots
- **Distribution**: PyInstaller for packaging as standalone executables

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Format code with black (`black squiggy/`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Release Process

Releases are automated via GitHub Actions:

1. Tag a version: `git tag v0.1.0`
2. Push the tag: `git push origin v0.1.0`
3. GitHub Actions will automatically build executables for all platforms
4. A new release will be created with all platform binaries attached

Development builds (macOS only) are automatically created on every push to `main` and available under the ["latest" release tag](https://github.com/rnabioco/squiggler/releases/tag/latest).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Oxford Nanopore Technologies for the POD5 file format and libraries
- The Python scientific computing community (NumPy, Pandas, Matplotlib ecosystems)

### Note on Implementation

Squiggy's plotting and I/O functionality is independently implemented using standard scientific Python libraries (plotnine, pod5, pysam). While inspired by common nanopore visualization conventions, the implementation does not use or depend on ont-remora. This design choice was made to:
- Avoid heavy dependencies (particularly PyTorch)
- Maintain MIT license compatibility
- Keep the application lightweight and focused on visualization

## Citation

If you use Squiggy in your research, please cite:

```
[Citation information to be added]
```

## Support

- **Issues**: [GitHub Issues](https://github.com/rnabioco/squiggler/issues)
- **Discussions**: [GitHub Discussions](https://github.com/rnabioco/squiggler/discussions)
