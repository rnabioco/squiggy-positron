# API Reference

Squiggy provides both a simple functional API and an object-oriented API for working with Oxford Nanopore sequencing data.

## Quick Start

```python
from squiggy import load_pod5, load_bam, plot_read, get_read_ids

# Load POD5 file (populates global kernel state)
load_pod5("data.pod5")

# Get read IDs from loaded file
read_ids = get_read_ids()

# Optionally load BAM for base annotations
load_bam("alignments.bam")

# Generate plot
html = plot_read(read_ids[0])
```

## Core Functions

::: squiggy.load_pod5

::: squiggy.load_bam

::: squiggy.plot_read

::: squiggy.plot_reads

## Object-Oriented API

::: squiggy.api.Pod5File

::: squiggy.api.BamFile

::: squiggy.api.Read

## Alignment

::: squiggy.alignment.BaseAnnotation

::: squiggy.alignment.AlignedRead

::: squiggy.alignment.extract_alignment_from_bam

::: squiggy.alignment.get_base_to_signal_mapping

## Modifications

::: squiggy.modifications.ModificationAnnotation

::: squiggy.modifications.extract_modifications_from_alignment

::: squiggy.modifications.detect_modification_provenance

## Normalization

::: squiggy.normalization.normalize_signal

::: squiggy.normalization.NormalizationMethod

## Utilities

::: squiggy.utils.get_basecall_data

::: squiggy.utils.downsample_signal

::: squiggy.utils.get_bam_references

::: squiggy.utils.get_reads_in_region

::: squiggy.utils.parse_region

::: squiggy.utils.reverse_complement
