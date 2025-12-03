# Squiggy - Positron Extension

🚧 **squiggy is under active development.** *Caveat emptor*. 🚧

A Positron IDE extension for visualizing Oxford Nanopore sequencing data from POD5 files directly in your workspace.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)
[![OpenVSX](https://img.shields.io/open-vsx/v/rnabioco/squiggy-positron)](https://open-vsx.org/extension/rnabioco/squiggy-positron)
[![codecov](https://codecov.io/gh/rnabioco/squiggy-positron/branch/main/graph/badge.svg)](https://codecov.io/gh/rnabioco/squiggy-positron)

## Overview

Squiggy is a Positron extension that integrates nanopore signal visualization into your data science workflow. Work with POD5 and BAM files directly in Positron, leveraging interactive Bokeh plots for data exploration.

![Squiggy in action](resources/screenshot.png)

**Key Features:**
- **One-Step Installation**: Just install the extension - Python environment is set up automatically
- **Interactive Visualization**: Bokeh-powered plots with zoom, pan, and hover tooltips
- **Base Annotations**: Overlay base calls and modifications on signal data when using BAM files
- **Read Filtering**: Search by read ID, reference region, or sequence motif
- **Modification Analysis**: Filter and visualize base modifications (5mC, 6mA, etc.) with probability thresholds
- **Aggregate Plots**: Multi-read visualizations with modification heatmaps, dwell time, and quality tracks

## Installation

### Prerequisites

- **Positron IDE** (version 2025.6.0+)
- **Python 3.12+** available on your system
- **uv** - Fast Python package manager ([install uv](https://docs.astral.sh/uv/getting-started/installation/))

### Install the Extension

1. Download the latest `.vsix` file from [Releases](https://github.com/rnabioco/squiggy-positron/releases)
2. In Positron: `Extensions` → `...` → `Install from VSIX...`
3. Select the downloaded `.vsix` file

Or install from the Open VSX Registry:
- Search for "Squiggy" in Positron's Extensions panel

### Automatic Python Setup

When you first use Squiggy, it automatically:
1. Creates a dedicated virtual environment at `~/.venvs/squiggy`
2. Installs the bundled `squiggy` Python package using `uv`
3. Configures a background kernel for extension operations

**No manual Python package installation required!** The extension handles everything.

> **Note**: The automatic setup requires `uv` to be installed. If `uv` is not found, you'll be prompted to install it.

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

The **Plotting** panel provides three analysis workflows:

#### Per-Read Plots
View individual reads with overlay or stacked layouts:
- **Overlay**: Alpha-blended signals on shared axes (good for pattern comparison)
- **Stacked**: Vertically offset signals (squigualiser-style, best for ≤20 reads)
- Configure max reads per sample (2-100)
- Requires: POD5 file

#### Composite Read Plots (Aggregate)
Multi-read statistics aligned to a reference sequence:
- Select reference sequence and max reads (10-500)
- **View Style** (for 2+ samples):
  - *Overlay*: Mean signals from all samples on one plot
  - *Multi-Track*: Detailed 5-track view for each sample
- **Visible Panels** (toggle individually):
  - Base modifications - Heatmaps showing modification frequency and confidence
  - Base pileup - Coverage and base composition
  - Dwell time - Mean dwell with confidence bands
  - Signal - Mean normalized signal with confidence bands
  - Quality scores - Mean quality with confidence bands
- **X-Axis Display**:
  - Clip to consensus region (focus on high-coverage areas)
  - Transform to relative coordinates (anchor position 1 to first reference base)
- Requires: BAM file with alignments

#### 2-Sample Comparisons
Compare signal differences between exactly two samples:
- Select exactly 2 samples from Sample Manager
- Choose reference sequence
- Generates delta plots showing signal differences (B - A)
- Configure max reads per sample (10-500)
- Requires: 2 samples with BAM files

#### Common Options (All Plot Types)
- **Normalization**: None, Z-score, Median, or MAD
- **Sample Manager Integration**: Use eye icons in Sample Manager to select which samples to visualize

> **Tip**: For multi-sample workflows, enable the samples you want to visualize using the eye icons in the Sample Manager panel, then choose your analysis type in the Plotting panel.

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

To use the API in your own environment, activate the Squiggy venv:

```bash
source ~/.venvs/squiggy/bin/activate
python -c "import squiggy; print(squiggy.__version__)"
```

## Architecture

Squiggy uses the **Strategy Pattern** to make adding new plot types easy and maintainable:

- **PlotFactory** - Creates the appropriate plotting strategy based on plot mode
- **7 Plot Strategies** - Each plot type (SINGLE, OVERLAY, STACKED, EVENTALIGN, AGGREGATE, DELTA, SIGNAL_OVERLAY_COMPARISON) is a separate strategy class
- **Reusable Components** - ThemeManager, BaseAnnotationRenderer, ModificationTrackBuilder shared across strategies
- **Easy Extension** - Adding new plot types requires only creating a new strategy class

This design makes it straightforward to add new visualization types (like A/B comparison plots) without modifying existing code.

See the [Developer Guide](https://rnabioco.github.io/squiggy-positron/developer-guide/#strategy-pattern-architecture) for detailed architecture documentation.

## Requirements

- **Positron IDE** (version 2025.6.0+)
- **Python 3.12+** available on your system
- **uv** for automatic Python environment setup

### Optional Requirements

- **BAM file** with basecalls for advanced features (event-aligned plots, modifications)

## Troubleshooting

### Reset the Squiggy Environment

If you encounter issues with the Python environment, you can reset it:

1. Open Command Palette (`Cmd+Shift+P` / `Ctrl+Shift+P`)
2. Run: `Squiggy: Reset Virtual Environment`

This will delete `~/.venvs/squiggy` and recreate it on next use.

### Manual Environment Setup

If automatic setup fails, you can manually create the environment:

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv with Python 3.12
uv venv ~/.venvs/squiggy --python 3.12

# Activate and install squiggy from the extension
source ~/.venvs/squiggy/bin/activate
# The extension will install the package on first use
```

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
