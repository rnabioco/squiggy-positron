# API Reference

Squiggy provides both a simple functional API and an object-oriented API for working with Oxford Nanopore sequencing data.

## Quick Start

```python
from squiggy import load_pod5, load_bam, plot_read, get_read_ids

# Load POD5 file (populates global kernel state)
reader, read_ids = load_pod5("data.pod5")

# Optionally load BAM for base annotations
load_bam("alignments.bam")

# Generate plot (routes to Positron Plots pane automatically)
html = plot_read(read_ids[0])
```

## File I/O

::: squiggy.load_pod5

::: squiggy.load_bam

::: squiggy.load_fasta

::: squiggy.close_pod5

::: squiggy.close_bam

::: squiggy.close_fasta

::: squiggy.get_current_files

::: squiggy.get_read_ids

::: squiggy.get_bam_modification_info

::: squiggy.get_bam_event_alignment_status

::: squiggy.get_read_to_reference_mapping

## Plotting Functions

### Single File Plotting

::: squiggy.plot_read

::: squiggy.plot_reads

::: squiggy.plot_aggregate

### Multi-Sample Plotting

::: squiggy.plot_motif_aggregate_all

::: squiggy.plot_signal_overlay_comparison

::: squiggy.plot_delta_comparison

## Multi-Sample Management

::: squiggy.load_sample

::: squiggy.get_sample

::: squiggy.list_samples

::: squiggy.remove_sample

::: squiggy.close_all_samples

::: squiggy.get_common_reads

::: squiggy.get_unique_reads

::: squiggy.compare_samples

## Session Management

::: squiggy.io.SquiggyKernel

::: squiggy.io.Sample

::: squiggy.io.LazyReadList

::: squiggy.io.Pod5Index

::: squiggy.get_reads_batch

::: squiggy.get_read_by_id

::: squiggy.get_reads_for_reference_paginated

## Object-Oriented API

::: squiggy.api.Pod5File

::: squiggy.api.BamFile

::: squiggy.api.FastaFile

::: squiggy.api.Read

::: squiggy.api.figure_to_html

## Alignment

::: squiggy.alignment.BaseAnnotation

::: squiggy.alignment.AlignedRead

::: squiggy.alignment.extract_alignment_from_bam

::: squiggy.alignment.get_base_to_signal_mapping

## Modifications

::: squiggy.modifications.ModificationAnnotation

::: squiggy.modifications.extract_modifications_from_alignment

::: squiggy.modifications.detect_modification_provenance

## Motif Search

::: squiggy.motif.MotifMatch

::: squiggy.iupac_to_regex

::: squiggy.search_motif

::: squiggy.count_motifs

::: squiggy.motif.IUPAC_CODES

## Normalization

::: squiggy.normalize_signal

::: squiggy.constants.NormalizationMethod

## Constants

::: squiggy.constants.PlotMode

::: squiggy.constants.Theme

::: squiggy.constants.BASE_COLORS

::: squiggy.constants.BASE_COLORS_DARK

## Plot Strategies

::: squiggy.create_plot_strategy

## Utilities

### Signal Processing

::: squiggy.utils.get_basecall_data

::: squiggy.downsample_signal

### BAM/Alignment Utilities

::: squiggy.get_bam_references

::: squiggy.get_reads_in_region

::: squiggy.get_reference_sequence_for_read

::: squiggy.utils.open_bam_safe

::: squiggy.utils.validate_sq_headers

::: squiggy.utils.index_bam_file

### General Utilities

::: squiggy.parse_region

::: squiggy.reverse_complement

::: squiggy.get_test_data_path

::: squiggy.extract_model_provenance

::: squiggy.utils.ModelProvenance

::: squiggy.parse_plot_parameters
