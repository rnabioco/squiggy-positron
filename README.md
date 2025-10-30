# Squiggy

🚧 **squiggy is under active development.** *Caveat emptor*. 🚧

A desktop application for visualizing Oxford Nanopore sequencing data from POD5 files.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)
[![codecov](https://codecov.io/gh/rnabioco/squiggy/branch/main/graph/badge.svg)](https://codecov.io/gh/rnabioco/squiggy)

## Overview

Squiggy is a cross-platform GUI application that allows you to visualize raw nanopore signal data (squiggle plots) from POD5 files. It provides an intuitive interface for browsing reads, searching by read ID, and displaying time-series plots of the electrical current signal.

## Features

- **POD5 File Support**: Native support for Oxford Nanopore's POD5 file format
- **Bundled Sample Data**: Get started immediately with included example POD5 files
- **Interactive Read Browser**: Browse and search through all reads in a POD5 file
- **High-Quality Plots**: Generate publication-ready squiggle plots with customizable styling
- **macOS Native**: Available as a standalone macOS application
- **Fast Performance**: Efficient handling of large POD5 files with thousands of reads

## Installation

### Option 1: Download Pre-built Executables (Recommended)

Download the latest release for your platform:

- **macOS**: [Squiggy-macos.dmg](https://github.com/rnabioco/squiggy/releases/latest)

For the latest development build (macOS only), download from the ["latest" release](https://github.com/rnabioco/squiggy/releases/tag/latest).

### Option 2: Install from Source

```bash
git clone https://github.com/rnabioco/squiggy.git
cd squiggy

# brew install git-lfs
# git lfs install
git lfs pull

# brew install uv
# uv venv
uv pip install -e .

# Run the application
uv run squiggy
```

## Usage

### Quick Start with Sample Data

Squiggy comes with bundled sample data (yeast [aa-tRNA-seq](https://pubmed.ncbi.nlm.nih.gov/40835813/)) to help you get started:

1. **Launch Squiggy**: Open the application
2. **Open Sample Data**: Go to **File → Open Sample Data** (or press `Ctrl+Shift+O` / `Cmd+Shift+O`)
3. **Explore**: Browse the sample reads and click any read to view its squiggle plot

### Working with Your Own Data

**For the best experience, load both POD5 and BAM files together** to enable base annotations, genomic region filtering, and sequence search:

1. **Launch Squiggy with both files** (recommended):
   ```bash
   squiggy --pod5 data.pod5 --bam alignments.bam
   # Or use short form
   squiggy -p data.pod5 -b alignments.bam
   ```

2. **Alternatively, load files via GUI**:
   - Open the application
   - Go to **File → Open POD5 File...** (`Ctrl+O` / `Cmd+O`)
   - Go to **File → Open BAM File...** to add alignments
   - Browse reads and click any read to view its squiggle plot with base annotations

**Why load both files?** Loading only the POD5 file shows raw signal without sequence context. Adding the BAM file enables:
- Event-aligned visualization with base annotations overlaid on signal
- Genomic region-based read filtering (e.g., chr1:1000-2000)
- DNA sequence motif search within reads

### Command Line Options

```bash
# Launch with both POD5 and BAM files (recommended)
squiggy --pod5 data.pod5 --bam alignments.bam
squiggy -p data.pod5 -b alignments.bam

# Launch with just POD5 file (limited functionality)
squiggy --pod5 data.pod5
squiggy -p data.pod5

# Launch GUI without pre-loading files
squiggy

# Run from source with both files
python -m squiggy.main --pod5 data.pod5 --bam alignments.bam
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

## Citation

If you use Squiggy in your research, please cite:

```
[Citation information to be added]
```

## Support

- **Issues**: [GitHub Issues](https://github.com/rnabioco/squiggy/issues)
- **Discussions**: [GitHub Discussions](https://github.com/rnabioco/squiggy/discussions)
