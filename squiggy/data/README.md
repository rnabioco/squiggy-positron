# Squiggy Demo Data

This directory contains demo data files bundled with the squiggy Python package.

## Files

- **yeast_trna_reads.pod5** (1.7 MB) - 180 Oxford Nanopore reads from yeast tRNA
- **yeast_trna_mappings.bam** (113 KB) - Alignments of the reads to reference
- **yeast_trna_mappings.bam.bai** (1.5 KB) - BAM index file
- **yeast_trna.fa** (507 bytes) - Reference sequence (yeast tRNA)
- **yeast_trna.fa.fai** (129 bytes) - FASTA index file

## Usage

### From Positron Extension

The Squiggy extension provides a **"Load Demo Session"** button that automatically loads these files for quick exploration.

### From Python

```python
import squiggy
import importlib.util
import os

# Find the package location
spec = importlib.util.find_spec('squiggy')
package_dir = os.path.dirname(spec.origin)
data_dir = os.path.join(package_dir, 'data')

# Load demo files
pod5_path = os.path.join(data_dir, 'yeast_trna_reads.pod5')
bam_path = os.path.join(data_dir, 'yeast_trna_mappings.bam')

squiggy.load_pod5(pod5_path)
squiggy.load_bam(bam_path)

# Plot a read
plot = squiggy.plot_read('read_001', plot_mode='EVENTALIGN')
```

## Data Source

These files are used for both demos and the squiggy test suite.

## Size

Total size: ~1.9 MB (includes POD5, BAM, and FASTA files)

These files are included in the Python package to provide users with an instant way to explore squiggy's capabilities without needing to download or provide their own nanopore data.
