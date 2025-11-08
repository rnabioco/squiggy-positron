# squiggy-positron

> ⚠️ **IMPORTANT**: This package is the Python backend for the [Squiggy Positron extension](https://github.com/rnabioco/squiggy-positron) and is **not designed for standalone use**. It is optimized for integration with the Positron IDE and may not function correctly outside of that environment.

## What is this?

`squiggy-positron` provides the Python plotting and data processing backend for the Squiggy extension in Positron IDE. Squiggy enables interactive visualization of Oxford Nanopore sequencing data (POD5 files) directly within the Positron IDE.

## Installation

> ⚠️ **Temporary**: Until the next minor release on PyPI, please install from TestPyPI using the command below.

This package is intended to be installed as a dependency of the Squiggy Positron extension:

```bash
uv pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    squiggy-positron
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

## Testing the Installation (TestPyPI)

If you're testing the package from TestPyPI before the official release, follow these steps:

### 1. Set Up Project Directory

```bash
# Create a new project directory
mkdir squiggy-test
cd squiggy-test
```

### 2. Open in Positron

1. Open the `squiggy-test` directory in Positron IDE
2. Start a Python kernel (Python 3.12+)

### 3. Install from TestPyPI

In the Positron terminal, run:

```bash
uv pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    squiggy-positron
```

### 4. Verify Installation

The package includes bundled test data - no need to download anything separately!

In the Positron Python console:

```python
# Test 1: Basic import
import squiggy
print(f"Squiggy version: {squiggy.__version__}")

# Test 2: Get bundled test data
pod5_path = squiggy.get_test_data_path('yeast_trna_reads.pod5')
print(f"Test data: {pod5_path}")

# Test 3: Load POD5 file
pod5 = squiggy.Pod5File(pod5_path)
print(f"Loaded {len(pod5)} reads")

# Test 4: Create a plot
read = pod5.get_read(pod5.read_ids[0])
fig = read.plot(mode='SINGLE', normalization='ZNORM')
print("Plot created successfully!")

pod5.close()
```

### 5. Interactive Testing (Optional)

For a comprehensive test suite, download the testing notebook:

```bash
# Download the test notebook from GitHub
curl -O https://raw.githubusercontent.com/rnabioco/squiggy-positron/main/examples/testing_testpypi.ipynb
```

Then open `testing_testpypi.ipynb` in Positron. This notebook tests:
- Package import and dependencies
- Bundled test data access
- POD5 and BAM file loading
- All plot modes and normalization methods
- Error handling

**All test data is included in the package** - no Git LFS or manual downloads required!

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
