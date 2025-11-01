# Squiggy - Positron Extension

Squiggy is a Positron IDE extension for visualizing Oxford Nanopore sequencing data from POD5 files directly in your workspace.

## Documentation

- **[User Guide](USER_GUIDE.md)** - Complete guide to using the extension
- **[Developer Guide](DEVELOPER.md)** - Setup and contribution guide
- **[README](../README.md)** - Quick start and overview

## Features

- **Positron Integration**: Works with your active Python kernel
- **Interactive Visualization**: Bokeh-powered plots with zoom, pan, and hover
- **Base Annotations**: Overlay base calls on signal data (requires BAM file)
- **Read Filtering**: Search by read ID, reference region, or sequence motif
- **Modification Analysis**: Filter and visualize base modifications with probability thresholds

## Installation

Download the latest `.vsix` file from [GitHub Releases](https://github.com/rnabioco/squiggy/releases) and install in Positron via `Extensions` → `...` → `Install from VSIX...`
