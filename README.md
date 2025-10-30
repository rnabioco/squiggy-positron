# Squiggy

🚧 **squiggy is under active development.** *Caveat emptor*. 🚧

A desktop application for visualizing Oxford Nanopore sequencing data from POD5 files.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)
[![codecov](https://codecov.io/gh/rnabioco/squiggy/branch/main/graph/badge.svg)](https://codecov.io/gh/rnabioco/squiggy)

## Overview

Squiggy is a cross-platform GUI application for visualizing raw nanopore signal data (squiggle plots) from POD5 files. It provides an intuitive interface for browsing reads, searching by read ID or genomic region, and displaying interactive time-series plots of electrical current signals with optional base annotations.

## Features

- 📂 Native POD5 file support with bundled sample data
- 🔍 Search by read ID, genomic region, or DNA sequence motif
- 📊 Five visualization modes: Single, Overlay, Stacked, Event-aligned, Aggregate
- 📐 Signal normalization (Z-score, Median, MAD) for cross-read comparison
- 🧬 Optional base annotations from BAM files with colorblind-friendly palette
- 💾 Export plots to HTML, PNG, or SVG (publication quality)
- 🚀 Standalone executable - no Python installation required

## Installation

### Pre-built Executables (Recommended)

Download the latest release for your platform:

- **macOS**: [Squiggy-macos.dmg](https://github.com/rnabioco/squiggy/releases/latest)

For development builds, see the [latest release](https://github.com/rnabioco/squiggy/releases/tag/latest).

### From Source

```bash
git clone https://github.com/rnabioco/squiggy.git
cd squiggy
git lfs pull  # Requires git-lfs: brew install git-lfs
uv pip install -e .  # Requires uv: brew install uv
uv run squiggy
```

## Quick Start

1. **Launch Squiggy** and go to **File → Open Sample Data** (or `Cmd+Shift+O`)
2. **Browse reads** in the left panel
3. **Click a read** to view its signal plot
4. **Try different modes** in the Plot Options panel (Overlay, Stacked, Event-aligned)

For detailed instructions, see the [Quick Start Guide](https://rnabioco.github.io/squiggy/quickstart/).

## Documentation

- 📚 [Quick Start Guide](https://rnabioco.github.io/squiggy/quickstart/) - Get up and running quickly
- 📖 [Usage Guide](https://rnabioco.github.io/squiggy/usage/) - Complete feature documentation
- 💻 [Development Guide](https://rnabioco.github.io/squiggy/development/) - Contributing and building from source
- 🏠 [Documentation Home](https://rnabioco.github.io/squiggy/)

## Contributing

Contributions are welcome! See the [Development & Contributing Guide](https://rnabioco.github.io/squiggy/development/) for setup instructions and guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Oxford Nanopore Technologies for the POD5 file format and libraries
- [Remora](https://github.com/nanoporetech/remora) - Modified base calling and visualization toolkit
- [Squigualiser](https://github.com/hiruna72/squigualiser) - Efficient signal visualization by Hiruna Samarakoon
- The Python scientific computing community

## Citation

```
[Citation information to be added]
```

## Support

- **Issues**: [GitHub Issues](https://github.com/rnabioco/squiggy/issues)
- **Discussions**: [GitHub Discussions](https://github.com/rnabioco/squiggy/discussions)
