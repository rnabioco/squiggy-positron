# Squiggy - Positron Extension

Squiggy is a Positron IDE extension for visualizing Oxford Nanopore sequencing data from POD5 files directly in your workspace.

![Squiggy Screenshot](screenshot.png)

## Documentation

- **[User Guide](USER_GUIDE.md)** - Complete guide to using the extension
- **[Multi-Sample Comparison Guide](MULTI_SAMPLE_COMPARISON.md)** - Compare 2+ POD5 datasets with delta tracks (NEW!)
- **[Quick Reference](QUICK_REFERENCE.md)** - Commands, shortcuts, and common workflows
- **[Developer Guide](DEVELOPER.md)** - Setup and contribution guide
- **[API Reference](api.md)** - Python package documentation

## Features

- **Positron Integration**: Works with your active Python kernel
- **Interactive Visualization**: Bokeh-powered plots with zoom, pan, and hover
- **Base Annotations**: Overlay base calls on signal data (requires BAM file)
- **Read Filtering**: Search by read ID, reference region, or sequence motif
- **Modification Analysis**: Filter and visualize base modifications with probability thresholds
- **Multi-Sample Comparison** (NEW!): Load 2-6+ samples and compare with delta tracks showing differences

## System Requirements

- **Positron IDE**: Version 2024.09.0 or later
- **Operating Systems**: macOS (Intel/Apple Silicon), Linux, Windows
- **Python**: 3.12 or later
- **Memory**: 4GB RAM minimum (8GB recommended for large POD5 files)
- **Disk Space**: Varies by dataset size (POD5 files can be several GB)

## Installation

### Recommended: Install from OpenVSX

Search for "Squiggy" in the Positron Extensions marketplace and click Install, or visit the [OpenVSX marketplace page](https://open-vsx.org/extension/rnabioco/squiggy-positron).

### Alternative: Install from VSIX

Download the latest `.vsix` file from [GitHub Releases](https://github.com/rnabioco/squiggy-positron/releases) and install in Positron via `Extensions` → `...` → `Install from VSIX...`
