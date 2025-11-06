# Quick Reference Guide

Fast lookup for common tasks and commands.

## Commands

### File Operations

| Command | Description |
|---------|-------------|
| `squiggy.openPOD5` | Open a POD5 file (single file mode) |
| `squiggy.openBAM` | Open a BAM file for alignments |
| `squiggy.openFASTA` | Open a FASTA file for reference |
| `squiggy.closePOD5` | Close POD5 file |
| `squiggy.closeBAM` | Close BAM file |
| `squiggy.closeFASTA` | Close FASTA file |
| `squiggy.loadTestData` | Load example test files |
| `squiggy.loadDemoSession` | Load pre-configured demo session |

### Plotting

| Command | Description |
|---------|-------------|
| `squiggy.plotRead` | Plot selected read(s) |
| `squiggy.plotAggregate` | Plot aggregate for reference |
| `squiggy.plotMotifAggregate` | Plot motif-centered aggregate |
| `squiggy.plotDeltaComparison` | Plot delta between samples |

### Multi-Sample

| Command | Description |
|---------|-------------|
| `squiggy.loadSample` | Load sample for comparison |
| `squiggy.plotDeltaComparison` | Compare loaded samples |
| `squiggy.loadTestMultiReadDataset` | Load test multi-read dataset |

### UI & Session Management

| Command | Description |
|---------|-------------|
| `squiggy.refreshReads` | Refresh read list |
| `squiggy.clearState` | Clear all extension state |
| `squiggy.saveSession` | Save current session |
| `squiggy.restoreSession` | Restore saved session |
| `squiggy.exportSession` | Export session to file |
| `squiggy.importSession` | Import session from file |

## Keyboard Shortcuts

### Default Shortcuts

| Action | Shortcut | Platform |
|--------|----------|----------|
| Command Palette | `Cmd+Shift+P` | macOS |
| Command Palette | `Ctrl+Shift+P` | Linux/Windows |
| Find | `Cmd+F` / `Ctrl+F` | All |

### Suggested Custom Shortcuts

Add to `keybindings.json`:

```json
[
  {
    "key": "cmd+shift+o",
    "command": "squiggy.openPOD5"
  },
  {
    "key": "cmd+shift+b",
    "command": "squiggy.openBAM"
  },
  {
    "key": "cmd+shift+p",
    "command": "squiggy.plotRead"
  },
  {
    "key": "cmd+shift+l",
    "command": "squiggy.loadSample"
  },
  {
    "key": "cmd+shift+d",
    "command": "squiggy.plotDeltaComparison"
  }
]
```

## Common Workflows

### Single Read Analysis

```
1. Load POD5 file
2. Find read in list
3. Click [Plot]
4. View signal in Plots pane
```

### Aggregate Analysis

```
1. Load POD5 + BAM file
2. Right-click reference in Read Explorer
3. Select [Plot Aggregate]
4. View signal distribution
```

### Base Modification Analysis

```
1. Load POD5 + BAM (with MM/ML tags)
2. BAM loaded → Modifications panel appears
3. Set probability threshold
4. Enable/disable modification types
5. Plot read → Mods overlay shown
```

### Multi-Sample Comparison

```
1. Load Sample (v5.0)
2. Load Sample (v6.0)
3. Sample Comparison Manager shows both
4. Check both samples
5. Click [Start Comparison]
6. Delta plot in Plots pane
```

### Motif Analysis

```
1. Load POD5 + BAM + FASTA
2. Click [Motif Explorer] in sidebar
3. Enter motif pattern
4. Click [Search]
5. Results show matches
6. Click match → Motif plot appears
```

## Settings

### Configuration Options

Access via: `Preferences` → `Settings` → search "squiggy"

| Setting | Type | Default | Notes |
|---------|------|---------|-------|
| `squiggy.defaultPlotMode` | enum | SINGLE | SINGLE or EVENTALIGN |
| `squiggy.defaultNormalization` | enum | ZNORM | ZNORM, MAD, MEDIAN, NONE |
| `squiggy.downsampleThreshold` | number | 100000 | Signal points before downsampling |
| `squiggy.theme` | enum | LIGHT | LIGHT or DARK (auto-detects VSCode) |
| `squiggy.aggregateSampleSize` | number | 100 | Max reads for aggregate (10-10000) |

### Example settings.json

```json
{
  "squiggy.defaultPlotMode": "EVENTALIGN",
  "squiggy.defaultNormalization": "MAD",
  "squiggy.downsampleThreshold": 50000,
  "squiggy.aggregateSampleSize": 200
}
```

## Python API Cheat Sheet

### Basic Operations

```python
from squiggy import load_pod5, load_bam, close_pod5, close_bam

# Load files
reader, read_ids = load_pod5("data.pod5")
load_bam("alignments.bam")

# Get info
print(len(read_ids))  # Number of reads

# Close
close_pod5()
close_bam()
```

### Single Read Plotting

```python
from squiggy import plot_read

# Plot with defaults
html = plot_read("read_001")

# Plot with options
html = plot_read(
    "read_001",
    plot_mode="EVENTALIGN",
    normalization="ZNORM",
    scale_x_by_dwell=True,
    show_mods=True,
    mod_filter={"5mC": 0.8}
)
```

### Aggregate Plotting

```python
from squiggy import plot_reads

# Plot aggregate for reads
html = plot_reads(
    ["read_001", "read_002", "read_003"],
    plot_mode="AGGREGATE",
    normalization="ZNORM"
)
```

### Multi-Sample Comparison

```python
from squiggy import (
    load_sample,
    compare_samples,
    get_common_reads,
    plot_delta_comparison
)

# Load samples
load_sample("v5.0", "/data/v5.0.pod5")
load_sample("v6.0", "/data/v6.0.pod5")

# Get read overlap
common = get_common_reads("v5.0", "v6.0")
print(f"Common reads: {len(common)}")

# Compare statistics
comp = compare_samples(["v5.0", "v6.0"])
print(f"Reads only in v5.0: {len(comp['unique_to_a'])}")

# Generate delta plot
html = plot_delta_comparison(
    sample_names=["v5.0", "v6.0"],
    normalization="ZNORM",
    theme="LIGHT"
)
```

### Object-Oriented API

```python
from squiggy import Pod5File, BamFile

# Create objects
pod5 = Pod5File("/data/file.pod5")
bam = BamFile("/data/alignments.bam")

# Get reads
reads = pod5.reads()
for read in reads[:5]:
    print(read.read_id, read.median_signal)

# Plot via object
html = read.plot(mode="EVENTALIGN", normalization="ZNORM")
```

## File Formats

### POD5

Oxford Nanopore signal data format.

```bash
# Inspect POD5
pod5 inspect summary file.pod5

# Extract reads
pod5 extract reads file.pod5 -o extracted/
```

### BAM

Aligned read format. **Must be indexed.**

```bash
# Index BAM
samtools index file.bam          # Creates file.bam.bai

# Check contents
samtools view file.bam | head -1

# Verify tags (mv for events, MM/ML for mods)
samtools view file.bam | head -1 | tr '\t' '\n'
```

### FASTA

Reference sequences.

```bash
# Check FASTA
head file.fasta
wc -l file.fasta
```

## Troubleshooting Checklist

### Extension Won't Load

- [ ] Positron 2024.09.0 or later?
- [ ] Python 3.12+?
- [ ] Virtual environment activated?
- [ ] Reload extension: View → Extensions → Reload

### POD5 Won't Open

- [ ] File exists and readable?
- [ ] VBZ codec available? (`pip install pod5`)
- [ ] Not corrupted? (`pod5 inspect summary file.pod5`)

### BAM Won't Load

- [ ] File indexed? (`.bai` present)
- [ ] Compatible reference? (Same genome version)
- [ ] Readable? (`samtools view file.bam | head`)

### Plot Blank

- [ ] Read ID exists? (Check Read Explorer)
- [ ] POD5 loaded? (Files panel should show file)
- [ ] Try different plot mode (SINGLE vs EVENTALIGN)

### Modifications Not Showing

- [ ] BAM has MM/ML tags? (`samtools view file.bam | grep MM`)
- [ ] Modifications panel visible?
- [ ] Try enabling/disabling filters

### Multi-Sample Comparison Fails

- [ ] At least 2 samples loaded?
- [ ] Samples have BAM files?
- [ ] BAM files aligned to same reference?
- [ ] Common reads exist? (`compare_samples()`)

## Performance Tips

| Task | Recommendation |
|------|-----------------|
| Large POD5 (>1GB) | Load in batches |
| Many reads (>10K) | Use aggregate mode |
| Slow zoom/pan | Reduce signal points (downsample) |
| Memory usage | Close unused samples |
| Motif search | Limit to 1-2 samples |

## Getting Help

- **Documentation**: [GitHub Docs](https://github.com/rnabioco/squiggy-positron/tree/main/docs)
- **Issues**: [Report bugs](https://github.com/rnabioco/squiggy-positron/issues)
- **Discussions**: [Ask questions](https://github.com/rnabioco/squiggy-positron/discussions)
- **Email**: Open an issue with [question] tag

## Version Info

- **Extension**: Check `About` in extensions panel
- **Python Package**: `pip show squiggy`
- **Positron**: Help → About
