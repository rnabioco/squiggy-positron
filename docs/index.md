# Squiggy

Squiggy is a desktop GUI application for visualizing Oxford Nanopore sequencing data ("squiggle plots") from POD5 files.

## Features

- 📂 Load and browse POD5 files containing Oxford Nanopore sequencing data
- 🔍 Search and filter reads by ID
- 📊 Generate high-quality squiggle plots showing raw signal data
- 💻 Cross-platform support (Windows, macOS, Linux)
- 🚀 Standalone executable - no Python installation required

## Quick Start

1. Download the latest release for your platform
2. Launch the Squiggy application
3. Click "Select POD5 File" to load your data
4. Browse or search for reads in the list
5. Click on a read to visualize its signal data

## Architecture

Squiggy is built with:

- **PySide6** (Qt for Python) - Cross-platform desktop UI framework
- **pod5** - Library for reading Oxford Nanopore POD5 files
- **plotnine** - ggplot2-style plotting for generating squiggle plots
- **PyInstaller** - Packages the app into standalone executables

## Source Code

The source code is available on [GitHub](https://github.com/rnabioco/squiggler).

## License

See the repository for license information.
