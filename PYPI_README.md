# squiggy-positron

> ⚠️ **IMPORTANT**: This package is the Python backend for the [Squiggy Positron extension](https://github.com/rnabioco/squiggy-positron) and is **not designed for standalone use**. It is optimized for integration with the Positron IDE and may not function correctly outside of that environment.

## What is this?

`squiggy-positron` provides the Python plotting and data processing backend for the Squiggy extension in Positron IDE. Squiggy enables interactive visualization of Oxford Nanopore sequencing data (POD5 files) directly within the Positron IDE.

## Installation

This package is intended to be installed as a dependency of the Squiggy Positron extension:

```bash
uv pip install squiggy-positron
```

After installing the package, install the Squiggy extension in Positron IDE from the Extensions marketplace.

## Usage

This package is designed to work with the Positron extension and requires:
- An active Python kernel in Positron
- POD5 files from Oxford Nanopore sequencing
- Optional: BAM files for alignment annotations

**For proper usage instructions, please refer to the [Squiggy extension documentation](https://github.com/rnabioco/squiggy-positron).**

## Import Name

While the PyPI package is named `squiggy-positron`, you import it as:

```python
import squiggy
```

This maintains compatibility with the extension's expected interface.

## Features

- Load and visualize POD5 files (Oxford Nanopore raw signal data)
- Multiple plot modes: single read, overlay, stacked, event-aligned, aggregate
- Signal normalization (Z-score, median, MAD)
- Base modification visualization (5mC, 6mA, etc.)
- BAM alignment integration for base annotations
- Interactive Bokeh plots with zoom/pan

## Requirements

- Python ≥ 3.12
- Positron IDE (for full functionality)
- POD5 files from Oxford Nanopore sequencing

## Why is this separate?

This package is published to PyPI to:
1. Allow standard Python package management with `uv`/`pip`
2. Enable project-based virtual environment workflows
3. Provide version control and updates independent of the extension
4. Support reproducible environments

## Support

For issues, questions, or contributions:
- **Extension Issues**: https://github.com/rnabioco/squiggy-positron/issues
- **Documentation**: https://github.com/rnabioco/squiggy-positron
- **Repository**: https://github.com/rnabioco/squiggy-positron

## License

MIT License - see [LICENSE](https://github.com/rnabioco/squiggy-positron/blob/main/LICENSE) for details.

## Citation

If you use Squiggy in your research, please cite:

```
Hesselberth Lab, RNA Bioscience Initiative
University of Colorado Anschutz Medical Campus
https://github.com/rnabioco/squiggy-positron
```
