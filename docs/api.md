# API Reference

Squiggy ships a Python package with two entry points: a **functional API** (global
kernel state, used by the Positron extension) and an **object-oriented API**
(`squiggy.api`, no global state — preferred for notebooks).

!!! note "Hand-maintained reference"
    This page is written by hand. Zensical does not yet support the `mkdocstrings`
    plugin, so signatures here are mirrored from the package. The docstrings remain the
    source of truth — use `help(squiggy.plot_read)` in a kernel for full parameter docs.

## Quick Start

### Object-oriented API (recommended for notebooks)

```python
from squiggy.api import Pod5File, BamFile

# Context manager handles cleanup; no global state is touched
with Pod5File("data.pod5") as pod5:
    for read in pod5.iter_reads(limit=5):
        print(read.read_id, len(read.signal))

    read = pod5.get_read("read_001")
    fig = read.plot(mode="EVENTALIGN", normalization="ZNORM")  # returns a Bokeh figure
```

### Functional API (used by the extension)

```python
import squiggy

squiggy.load_pod5("data.pod5")          # populates the global squiggy_kernel
read_ids = squiggy.get_read_ids()
squiggy.load_bam("alignments.bam")      # optional: base annotations / modifications

# Routes to Positron's Plots pane via bokeh.io.show(); also returns the HTML string
html = squiggy.plot_read(read_ids[0], mode="EVENTALIGN")
```

---

## File I/O

```python
load_pod5(file_path: str) -> None
```
Load a POD5 file into the global kernel session.

```python
load_bam(file_path: str, build_ref_mapping: bool = True, use_cache: bool = True) -> None
```
Load a BAM file into the global kernel session (builds the reference→read mapping).

```python
load_fasta(file_path: str) -> None
```
Load a FASTA reference file into the global kernel session.

```python
close_pod5() -> None
close_bam() -> None
close_fasta() -> None
```
Close/clear the POD5 reader, BAM state, or FASTA state respectively.

```python
get_current_files() -> dict[str, str | None]
```
Return the paths of the currently loaded POD5/BAM/FASTA files.

```python
get_read_ids() -> list[str]
```
Return the list of read IDs from the currently loaded POD5 file.

```python
get_bam_modification_info(file_path: str) -> dict
```
Check whether a BAM file contains base-modification tags (`MM`/`ML`).

```python
get_bam_event_alignment_status(file_path: str) -> bool
```
Check whether a BAM file contains event-alignment data (the `mv` move-table tag).

```python
get_read_to_reference_mapping() -> dict[str, list[str]]
```
Return a mapping of reference name → read IDs from the currently loaded BAM.

---

## Plotting

All `plot_*` functions return a Bokeh HTML string and, inside the extension, route the
figure to Positron's Plots pane via `bokeh.io.show()`.

### Single-file plots

```python
plot_read(read_id: str, mode: str = "SINGLE", normalization: str = "ZNORM",
          theme: str = "LIGHT", ...) -> str
```
Plot a single read. `mode` accepts `SINGLE` or `EVENTALIGN`.

```python
plot_reads(read_ids: list, mode: str = "OVERLAY", normalization: str = "ZNORM",
           theme: str = "LIGHT", ...) -> str
```
Plot multiple reads together. `mode` accepts `OVERLAY`, `STACKED`, or `REFERENCE_OVERLAY`.

```python
plot_aggregate(reference_name: str, max_reads: int = 100, normalization: str = "ZNORM",
               theme: str = "LIGHT", show_modifications: bool = True, ...) -> str
```
Aggregate multi-read visualization (signal/dwell/quality/coverage tracks + pileup) for a
reference sequence.

```python
plot_pileup(reference_name: str, max_reads: int = 100, theme: str = "LIGHT",
            show_modifications: bool = True, ...) -> str
```
Pileup-only visualization for BAM files that lack move tables (no per-base signal).

```python
plot_motif_aggregate_all(fasta_file: str, motif: str, upstream: int = None,
                         downstream: int = None, max_reads_per_motif: int = 100,
                         normalization: str = "ZNORM", theme: str = "LIGHT",
                         strand: str = "both") -> str
```
Aggregate visualization across **all** matches of an IUPAC `motif` in the reference.

### Multi-sample comparison plots

```python
plot_delta_comparison(sample_names: list[str], reference_name: str = "Default",
                      normalization: str = "NONE", theme: str = "LIGHT",
                      max_reads: int | None = None) -> str
```
Delta track comparing two or more samples (`DELTA` mode).

```python
plot_signal_overlay_comparison(sample_names: list[str], reference_name: str | None = None,
                               normalization: str = "ZNORM", theme: str = "LIGHT",
                               max_reads: int | None = None) -> str
```
Overlay raw signal from multiple samples (`SIGNAL_OVERLAY_COMPARISON` mode).

```python
plot_aggregate_comparison(sample_names: list[str], reference_name: str,
                          metrics: list[str] | None = None, max_reads: int | None = None,
                          normalization: str = "ZNORM", theme: str = "LIGHT",
                          view_style: str = "overlay", ...) -> str
```
Per-sample aggregate statistics shown side by side (`AGGREGATE_COMPARISON` mode).
`view_style` is `"overlay"` or `"stacked"`.

See [Multi-Sample Comparison](multi_sample_comparison.md) for a full walkthrough.

---

## Multi-sample management

```python
load_sample(name: str, pod5_path: str, bam_path: str | None = None,
            fasta_path: str | None = None) -> Sample
get_sample(name: str) -> Sample | None
list_samples() -> list[str]
remove_sample(name: str) -> None
close_all_samples() -> None
```
Load, look up, list, unload, and clear named samples in the global session.

```python
get_common_reads(sample_names: list[str]) -> set[str]
get_unique_reads(sample_name: str, exclude_samples: list[str] | None = None) -> set[str]
compare_samples(sample_names: list[str]) -> dict
```
Set operations and summary comparison across samples.

---

## Performance / batched read access

```python
get_reads_batch(read_ids: list[str], sample_name: str | None = None) -> dict[str, ReadRecord]
get_read_by_id(read_id: str, sample_name: str | None = None) -> ReadRecord | None
get_reads_for_reference_paginated(reference_name: str, offset: int = 0,
                                  limit: int | None = None) -> list[str]
```
Index-backed batch/single fetch and paginated per-reference read listing for large datasets.

---

## Signal, motif, and reference utilities

```python
normalize_signal(signal: np.ndarray, method: NormalizationMethod) -> np.ndarray
downsample_signal(signal: np.ndarray, downsample_factor: int = None) -> np.ndarray
```
Normalize (`NONE`/`ZNORM`/`MEDIAN`/`MAD`) or downsample a signal array.

```python
search_motif(fasta_file, motif: str, region: str | None = None,
             strand: Literal["+", "-", "both"] = "both") -> Iterator[MotifMatch]
count_motifs(fasta_file, motif: str, region: str | None = None,
             strand: Literal["+", "-", "both"] = "both") -> int
iupac_to_regex(pattern: str) -> str
```
Search/count IUPAC motifs in a FASTA reference, or convert a motif to a regex.

```python
get_bam_references(bam_file) -> list[dict]
get_reads_in_region(bam_file, chromosome, start=None, end=None) -> dict
get_reference_sequence_for_read(bam_file, read_id) -> tuple[str | None, int | None, object | None]
parse_region(region_str: str) -> tuple[str | None, int | None, int | None]
reverse_complement(seq: str) -> str
get_test_data_path() -> Path
```
BAM/region helpers, region-string parsing, reverse complement, and the bundled test-data path.

---

## Object-oriented API (`squiggy.api`)

All classes support the context-manager protocol (`with ... as`).

### `Pod5File(path)`
POD5 reader with lazy loading.

- `iter_reads(limit=None)` — yield `Read` objects.
- `get_read(read_id) -> Read` — fetch one read by ID.
- `close()` — release the reader.

### `BamFile(path)`
BAM alignment reader.

- `get_alignment(read_id)` — alignment for a read.
- `get_modifications_info()` — MM/ML modification summary.
- `get_reads_overlapping_motif(...)`, `iter_region(...)`, `close()`.

### `FastaFile(path)`
FASTA reference reader with motif search.

- `fetch(...)`, `search_motif(...)`, `count_motifs(...)`, `close()`.

### `Read`
A single POD5 read with signal data (yielded by `Pod5File`).

- `plot(mode="SINGLE", normalization="ZNORM", theme="LIGHT", bam_file=None, ...) -> figure`
  — return a Bokeh figure for this read.
- `get_normalized(...)`, `get_alignment(...)`.

### `figure_to_html(fig) -> str`
Embed a Bokeh figure as a standalone HTML string (for notebook display / file export).

---

## Session state

```python
from squiggy.io import squiggy_kernel   # global SquiggyKernel instance
```

`SquiggyKernel` consolidates all loaded state into a single object that appears in the
Positron Variables pane. Key members:

- `samples`, `list_samples()`, `get_sample(name)`, `load_sample(...)`, `remove_sample(name)`
- `close_pod5()`, `close_bam()`, `close_fasta()`, `close_all()`

`Sample` represents one POD5/BAM/FASTA set; `LazyReadList` is a virtual list of read IDs
backed by the POD5 reader's read-ID metadata (`materialize()` to realize it).

---

## Constants (`squiggy.constants`)

**`PlotMode`** — `SINGLE`, `OVERLAY`, `STACKED`, `EVENTALIGN`, `AGGREGATE`, `DELTA`,
`SIGNAL_OVERLAY_COMPARISON`, `AGGREGATE_COMPARISON`, `REFERENCE_OVERLAY`.

**`NormalizationMethod`** — `NONE`, `ZNORM`, `MEDIAN`, `MAD`.

**`Theme`** — `LIGHT`, `DARK`. Color maps: `BASE_COLORS`, `BASE_COLORS_DARK`.
