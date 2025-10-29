# Squiggy

🚧 **squiggy is under active development.** *Caveat emptor*. 🚧

A desktop application for visualizing Oxford Nanopore sequencing data from POD5 files.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
[![codecov](https://codecov.io/gh/rnabioco/squiggy/branch/main/graph/badge.svg)](https://codecov.io/gh/rnabioco/squiggy)

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

- **macOS**: [Squiggy-macos.dmg](https://github.com/rnabioco/squiggy/releases/latest)
- **Windows**: [Squiggy-windows-x64.zip](https://github.com/rnabioco/squiggy/releases/latest)
- **Linux**: [Squiggy-linux-x64.tar.gz](https://github.com/rnabioco/squiggy/releases/latest)

For the latest development build (macOS only), download from the ["latest" release](https://github.com/rnabioco/squiggy/releases/tag/latest).

### Option 2: Install from Source

```bash
# Install Git LFS (if not already installed)
# macOS
brew install git-lfs

# Ubuntu/Debian
sudo apt-get install git-lfs

# Windows (using chocolatey)
choco install git-lfs

# Initialize Git LFS
git lfs install

# Clone the repository
git clone https://github.com/rnabioco/squiggy.git
cd squiggy

# Pull LFS files (test data)
git lfs pull

# Install dependencies (using uv - recommended)
uv pip install -e .

# Or using pip
pip install -e .

# Run the application
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

## Contributing

Contributions are welcome! For development setup, workflow, and contribution guidelines, please see:

- [Development & Contributing Guide](https://rnabioco.github.io/squiggy/development/) - Complete guide for contributors

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Oxford Nanopore Technologies for the POD5 file format and libraries
- The Python scientific computing community (NumPy, Pandas, Matplotlib ecosystems)
- **[Remora](https://github.com/nanoporetech/remora)** - Oxford Nanopore's modified base calling and visualization toolkit
- **[Squigualiser](https://github.com/hiruna72/squigualiser)** - Efficient nanopore signal visualization tool by Hiruna Samarakoon

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

- **Issues**: [GitHub Issues](https://github.com/rnabioco/squiggy/issues)
- **Discussions**: [GitHub Discussions](https://github.com/rnabioco/squiggy/discussions)
