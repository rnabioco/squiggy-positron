# Squiggy - Positron Extension

🚧 **squiggy is under active development.** *Caveat emptor*. 🚧

A Positron IDE extension for visualizing Oxford Nanopore sequencing data from POD5 files directly in your workspace.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)
[![codecov](https://codecov.io/gh/rnabioco/squiggy/branch/main/graph/badge.svg)](https://codecov.io/gh/rnabioco/squiggy)

## Overview

Squiggy is a Positron extension that integrates nanopore signal visualization into your data science workflow. Work with POD5 and BAM files directly in Positron, leveraging the active Python kernel for seamless data exploration.

**Key Features:**
- **Positron Integration**: Works with your active Python kernel - no separate environment needed
- **Interactive Visualization**: Bokeh-powered plots with zoom, pan, and hover tooltips
- **Base Annotations**: Overlay base calls and modifications on signal data when using BAM files
- **Read Filtering**: Search by read ID, reference region, or sequence motif
- **Modification Analysis**: Filter and visualize base modifications (5mC, 6mA, etc.) with probability thresholds

## Screenshots

*Coming soon*

## Installation

### From VSIX (Recommended)

1. Download the latest `.vsix` file from [Releases](https://github.com/rnabioco/squiggy/releases)
2. In Positron: `Extensions` → `...` → `Install from VSIX...`
3. Select the downloaded `.vsix` file

### From Source (Development)

```bash
git clone https://github.com/rnabioco/squiggy.git
cd squiggy

# Install dependencies (Python + Node.js + npm packages)
pixi install && pixi run setup

# Build extension
pixi run build
```

## Quick Start

### 1. Load Data Files

Open the Squiggy sidebar (click the Squiggy icon in the activity bar):

- **Open POD5 File**: Load your nanopore signal data
- **Open BAM File** (optional): Add alignments for base annotations and advanced filtering

### 2. Browse Reads

The **Reads** panel shows all reads in the POD5 file:
- Grouped by reference if BAM file is loaded
- Use the search bar to filter by read ID or reference name
- Click any read to visualize

### 3. Customize Plots

Use the **Plot Options** panel to configure:
- **Plot mode**: Raw signal vs event-aligned with bases
- **Normalization**: None, Z-score, Median, or MAD
- **X-axis scaling**: Base positions vs cumulative dwell time
- **Downsample threshold**: For large signals (default: 100,000 samples)

### 4. Explore Modifications (BAM with MM/ML tags)

If your BAM file contains base modifications:
- The **Base Modifications** panel appears automatically
- Filter by modification type (5mC, 6mA, etc.)
- Set probability threshold to focus on high-confidence calls
- Toggle coloring by dwell time vs modification probability

### 5. Export Plots

- **File → Export Plot** (Ctrl/Cmd+E)
- Formats: HTML (interactive), PNG, SVG
- Option to export at current zoom level

## Requirements

- **Positron IDE** (version 2025.6.0+)
- **Python 3.12+** with the `squiggy` package:
  ```bash
  pip install squiggy  # Includes: pod5, bokeh, numpy, pysam
  # OR for development: pixi install
  ```
- **Optional**: BAM file with basecalls for advanced features

## Extension Development

See [DEVELOPER.md](docs/DEVELOPER.md) for detailed development setup and contribution guidelines.

### Quick Development Setup

```bash
# Install dependencies
pixi install

# Main development commands
pixi run dev      # Watch mode (auto-compile TypeScript)
pixi run test     # Run all tests (Python + TypeScript)
pixi run build    # Build extension (.vsix)
pixi run docs     # Serve documentation locally

# Code quality
pixi run lint     # Lint Python + TypeScript
pixi run format   # Format Python + TypeScript
```

## Architecture

The extension has two main components:

1. **TypeScript Extension** (`src/`): Positron IDE integration
   - Communicates with Python via Positron's runtime API
   - Manages UI panels (file info, read list, plot options, modifications)
   - Displays Bokeh plots in webview panels

2. **Python Package** (`squiggy/`): Signal processing and plotting
   - Reads POD5 and BAM files
   - Generates interactive Bokeh visualizations
   - Handles signal normalization and downsampling

## Documentation

- [User Guide](docs/USER_GUIDE.md) - Complete usage guide
- [Developer Guide](docs/DEVELOPER.md) - Extension development setup
- [Online Docs](https://rnabioco.github.io/squiggy/) - MkDocs documentation site

## Contributing

Contributions are welcome! Please see [DEVELOPER.md](docs/DEVELOPER.md) for:
- Development setup
- Coding standards
- Testing guidelines
- Pull request process

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Positron Team** at Posit for the excellent IDE and extension API
- **Oxford Nanopore Technologies** for POD5 format and libraries
- **[Remora](https://github.com/nanoporetech/remora)** - Modified base calling toolkit
- **[Squigualiser](https://github.com/hiruna72/squigualiser)** - Signal visualization inspiration

## Citation

If you use Squiggy in your research, please cite:

```
[Citation information to be added]
```

## Support

- **Issues**: [GitHub Issues](https://github.com/rnabioco/squiggy/issues)
- **Discussions**: [GitHub Discussions](https://github.com/rnabioco/squiggy/discussions)
