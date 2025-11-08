# Squiggy - Positron Extension

Squiggy is a Positron IDE extension for visualizing Oxford Nanopore sequencing data from POD5 files directly in your workspace.

![Squiggy Screenshot](screenshot.png)

## Documentation

- **[User Guide](user_guide.md)** - Complete guide to using the extension
- **[Multi-Sample Comparison Guide](multi_sample_comparison.md)** - Compare 2+ POD5 datasets with delta tracks
- **[Quick Reference](quick_reference.md)** - Commands, shortcuts, and common workflows
- **[Developer Guide](developer.md)** - Setup and contribution guide
- **[API Reference](api.md)** - Python package documentation

## Features

- **Positron Integration**: Works with your active Python kernel
- **Interactive Visualization**: Bokeh-powered plots with zoom, pan, and hover
- **Base Annotations**: Overlay base calls on signal data (requires BAM file)
- **Read Filtering**: Search by read ID, reference region, or sequence motif
- **Modification Analysis**: Filter and visualize base modifications with probability thresholds
- **Multi-Sample Comparison**: Load 2-6+ samples and compare with delta tracks showing differences

## System Requirements

- **Positron IDE**: Version 2024.09.0 or later
- **Operating Systems**: macOS (Intel/Apple Silicon), Linux, Windows
- **Python**: 3.12 or later
- **Memory**: 4GB RAM minimum (8GB recommended for large POD5 files)
- **Disk Space**: Varies by dataset size (POD5 files can be several GB)

## Installation

### Install from OpenVSX (Recommended)

1. Open Positron IDE
2. Open Extensions panel (`Cmd+Shift+X` / `Ctrl+Shift+X`)
3. Search for "Squiggy"
4. Click **Install**

Or visit the [OpenVSX marketplace page](https://open-vsx.org/extension/rnabioco/squiggy-positron).

### First Time Setup

After installing the extension, a **Setup** panel will appear in the sidebar if the Python package is not detected. The panel provides:

- Step-by-step instructions to create a virtual environment
- Copy-paste commands for installation
- Direct link to select your Python interpreter
- Automatic verification of setup completion

No command-line experience needed—just follow the guided setup!

### Alternative: Install from VSIX

Download the latest `.vsix` file from [GitHub Releases](https://github.com/rnabioco/squiggy-positron/releases) and install via `Extensions` → `...` → `Install from VSIX...`
