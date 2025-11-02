# API Reference

Squiggy provides both a simple functional API and an object-oriented API for working with Oxford Nanopore sequencing data.

## Quick Start

```python
from squiggy import load_pod5, load_bam, plot_read

# Load POD5 file
reader, read_ids = load_pod5("data.pod5")

# Optionally load BAM for base annotations
load_bam("alignments.bam")

# Generate plot
html = plot_read(read_ids[0])
```

## API Organization

- **[Core Functions](core.md)** - Simple functional API for loading files and plotting
- **[Object-Oriented API](api.md)** - Class-based API for notebook workflows
- **[Plotting](plotting.md)** - SquigglePlotter class for customized plots
- **[Alignment](alignment.md)** - Base annotation extraction from BAM files
- **[Modifications](modifications.md)** - Base modification parsing and filtering
- **[Normalization](normalization.md)** - Signal normalization methods
- **[Utilities](utils.md)** - Helper functions for signal processing
