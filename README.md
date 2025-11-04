# Squiggy - Positron Extension

🚧 **squiggy is under active development.** *Caveat emptor*. 🚧

A Positron IDE extension for visualizing Oxford Nanopore sequencing data from POD5 files directly in your workspace.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)
[![OpenVSX](https://img.shields.io/open-vsx/v/rnabioco/squiggy-positron)](https://open-vsx.org/extension/rnabioco/squiggy-positron)
[![codecov](https://codecov.io/gh/rnabioco/squiggy-positron/branch/main/graph/badge.svg)](https://codecov.io/gh/rnabioco/squiggy-positron)

## Overview

Squiggy is a Positron extension that integrates nanopore signal visualization into your data science workflow. Work with POD5 and BAM files directly in Positron, leveraging the active Python kernel for seamless data exploration.

![Squiggy in action](resources/screenshot.png)

**Key Features:**
- **Positron Integration**: Works with your active Python kernel - no separate environment needed
- **Interactive Visualization**: Bokeh-powered plots with zoom, pan, and hover tooltips
- **Base Annotations**: Overlay base calls and modifications on signal data when using BAM files
- **Read Filtering**: Search by read ID, reference region, or sequence motif
- **Modification Analysis**: Filter and visualize base modifications (5mC, 6mA, etc.) with probability thresholds
- **Aggregate Plots**: Multi-read visualizations with modification heatmaps, dwell time, and quality tracks

## Installation

1. Download the latest `.vsix` file from [Releases](https://github.com/rnabioco/squiggy-positron/releases)
2. In Positron: `Extensions` → `...` → `Install from VSIX...`
3. Select the downloaded `.vsix` file

> For development installation, see the [Developer Guide](https://rnabioco.github.io/squiggy-positron/developer-guide/).

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

Use the **Advanced Plotting** panel to configure:
- **Analysis Type**: Single Read or Aggregate (multi-read statistics)
- **View Mode**: Standard or Event-Aligned (with base annotations)
- **Normalization**: None, Z-score, Median, or MAD
- **X-axis scaling**: Base positions vs cumulative dwell time
- **Downsample threshold**: For large signals (default: 100,000 samples)

For **Aggregate Plots** (requires BAM):
- Select reference sequence and maximum reads to include
- Toggle individual panels: Modifications, Pileup, Dwell Time, Signal, Quality
- View modification heatmaps showing frequency and confidence
- Explore dwell time patterns with confidence bands

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

## Python API for Notebooks

Squiggy provides an object-oriented Python API for use in Jupyter notebooks and Python scripts. See [`examples/notebook_api_demo.ipynb`](examples/notebook_api_demo.ipynb) for a complete tutorial.

## Architecture

Squiggy uses the **Strategy Pattern** to make adding new plot types easy and maintainable:

- **PlotFactory** - Creates the appropriate plotting strategy based on plot mode
- **5 Plot Strategies** - Each plot type (SINGLE, OVERLAY, STACKED, EVENTALIGN, AGGREGATE) is a separate strategy class
- **Reusable Components** - ThemeManager, BaseAnnotationRenderer, ModificationTrackBuilder shared across strategies
- **Easy Extension** - Adding new plot types requires only creating a new strategy class

This design makes it straightforward to add new visualization types (like A/B comparison plots) without modifying existing code.

See the [Developer Guide](https://rnabioco.github.io/squiggy-positron/developer-guide/#strategy-pattern-architecture) for detailed architecture documentation.

## Requirements

- **Positron IDE** (version 2025.6.0+)
- **Python 3.12+** with an active Python console
- **squiggy Python package**: The extension will prompt to install automatically on first use

### Python Environment Setup

**Recommended**: Use `uv` for fast, reliable Python environment management:

```bash
# Install uv (if not already installed)
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Create virtual environment and install squiggy
uv venv
source .venv/bin/activate  # macOS/Linux
# OR
.venv\Scripts\activate     # Windows

uv pip install squiggy  # Includes: pod5, bokeh, numpy, pysam
```

**Alternative (using venv)**: Standard Python virtual environment:

```bash
# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate  # macOS/Linux
# OR
.venv\Scripts\activate     # Windows

# Install squiggy (automatic via extension or manual)
pip install squiggy
```

> **Important**: If you're using Homebrew Python or system Python, you **must** use a virtual environment. Modern Python installations follow [PEP 668](https://peps.python.org/pep-0668/) and prevent direct package installation to system Python.
>
> When you first open a POD5 file, Squiggy will:
> 1. Check if the Python package is installed
> 2. Detect if you're using a virtual environment
> 3. Prompt to install automatically (if in venv/mamba) or show manual setup instructions (if system Python)

**Alternative (using mamba)**: If you need mamba environments:

```bash
# Install mamba (much faster than conda)
# See: https://mamba.readthedocs.io/en/latest/installation.html

mamba create -n squiggy python=3.12
mamba activate squiggy
pip install squiggy
```

### Optional Requirements

- **BAM file** with basecalls for advanced features (event-aligned plots, modifications)

## Documentation

📚 **Full documentation available at [rnabioco.github.io/squiggy-positron](https://rnabioco.github.io/squiggy-positron/)**

- [User Guide](https://rnabioco.github.io/squiggy-positron/user-guide/) - Complete usage guide
- [Developer Guide](https://rnabioco.github.io/squiggy-positron/developer-guide/) - Extension development setup
- [API Reference](https://rnabioco.github.io/squiggy-positron/api/) - Python API documentation

## Contributing

Contributions are welcome! Please see the [Developer Guide](https://rnabioco.github.io/squiggy-positron/developer-guide/) for:
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

- **Issues**: [GitHub Issues](https://github.com/rnabioco/squiggy-positron/issues)
- **Discussions**: [GitHub Discussions](https://github.com/rnabioco/squiggy-positron/discussions)
