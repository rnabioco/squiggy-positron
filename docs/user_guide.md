# Squiggy Extension - User Guide

Complete guide to using the Squiggy Positron extension for nanopore signal visualization.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Sample Data](#sample-data)
- [Loading Data](#loading-data)
- [Browsing Reads](#browsing-reads)
- [Plotting Reads](#plotting-reads)
- [Plot Customization](#plot-customization)
- [Base Modifications](#base-modifications)
- [Exporting Plots](#exporting-plots)
- [Multi-Sample Comparison](#multi-sample-comparison) (NEW!)
- [Common Errors](#common-errors)
- [Keyboard Shortcuts](#keyboard-shortcuts)

## Prerequisites

Before using Squiggy, ensure you have:

- **Positron IDE** 2024.09.0 or later installed
- **Python 3.12+** with a virtual environment (venv, conda, or uv)
- **POD5 files** from Oxford Nanopore sequencing
- **BAM files** (optional, required for base annotations and modifications)
  - Must be indexed (`.bai` file present)
  - Should contain move tables (`mv` tag) for event-aligned visualization
  - May contain modification tags (`MM`/`ML`) for modification analysis

## Getting Started

### Installation

1. Download the latest `.vsix` file from [GitHub Releases](https://github.com/rnabioco/squiggy-positron/releases)
2. In Positron: `Extensions` → `...` → `Install from VSIX...`
3. Select the downloaded `.vsix` file
4. Reload Positron when prompted

### Python Requirements

**Important**: Use a virtual environment for Python package management.

#### Setting Up a Virtual Environment

**Option 1: venv (Recommended)**

```bash
# Create virtual environment in your project directory
python3 -m venv .venv

# Activate it
source .venv/bin/activate  # macOS/Linux
# OR
.venv\Scripts\activate     # Windows

# Install squiggy package
pip install squiggy
```

**Option 2: conda**

```bash
conda create -n squiggy python=3.12
conda activate squiggy
pip install squiggy
```

**Option 3: Automatic Installation via Extension**

When you first open a POD5 file, Squiggy will:
1. Check if the Python package is installed in your active kernel
2. Detect your Python environment type (venv, conda, or system Python)
3. Prompt you to install automatically (if safe) or show manual instructions

> **Note**: If you're using Homebrew Python or system Python, the extension will refuse automatic installation and guide you to create a virtual environment first. This follows [PEP 668](https://peps.python.org/pep-0668/) guidelines for externally-managed Python environments.

#### Python Environment in Positron

After creating your virtual environment:
1. Use Positron's **Interpreter selector** to choose your environment
2. Start a new Python console (it will use the selected interpreter)
3. The extension will work with this active kernel

**Dependencies installed with squiggy**: `pod5`, `pysam`, `bokeh`, `numpy`

### Opening the Extension

Click the Squiggy icon in the Activity Bar (left sidebar) to open the extension panels:
- **Files** - POD5/BAM file information
- **Search** - Filter reads by ID or reference
- **Reads** - Hierarchical read list
- **Plot Options** - Visualization settings
- **Base Modifications** - Modification filtering (when BAM loaded)

## Sample Data

Squiggy includes sample data for testing and learning. The test data is available in the repository:

- **Location**: `tests/data/` directory
- **POD5 file**: `yeast_trna_reads.pod5` (180 reads from yeast tRNA sequencing)
- **BAM file**: `yeast_trna_mappings.bam` (corresponding alignments with base calls)

To use the sample data:

```bash
# Clone the repository (includes sample data via Git LFS)
git clone https://github.com/rnabioco/squiggy-positron.git
cd squiggy-positron
git lfs pull  # Download POD5/BAM files

# Sample files are now in tests/data/
ls tests/data/
```

Alternatively, download individual files from the [GitHub repository](https://github.com/rnabioco/squiggy-positron/tree/main/tests/data).

## Loading Data

### Loading a POD5 File

**Method 1: Command Palette**
1. Open Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P`)
2. Type `Squiggy: Open POD5 File`
3. Select your `.pod5` file

**Method 2: File Panel**
1. Click "Open POD5 File" in the Files panel
2. Browse and select your file

**What you'll see:**
- File path and size in Files panel
- All read IDs in the Reads panel (flat list)

### Loading a BAM File (Optional but Recommended)

BAM files enable advanced features:
- Event-aligned visualization with base annotations
- Read grouping by reference sequence
- Genomic region filtering
- Base modification analysis

**To load:**
1. Open Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P`)
2. Type `Squiggy: Open BAM File`
3. Select your `.bam` file (must have index `.bai` in same directory)

**What you'll see:**
- Reads grouped by reference in Reads panel
- Event-aligned plot mode available
- Base Modifications panel (if BAM contains MM/ML tags)

### Requirements for BAM Files

- **Indexed**: Must have `.bai` index file
- **Aligned**: Reads must be aligned to reference sequences
- **Move table**: For event-aligned mode, BAM must contain `mv` tag (created by dorado/guppy)
- **Modifications**: For modification analysis, BAM must contain `MM`/`ML` tags

## Browsing Reads

### Read List Views

**Without BAM** (flat list):
```
read_001
read_002
read_003
...
```

**With BAM** (grouped by reference):
```
▼ chr1 (150 reads)
  ├─ read_001
  ├─ read_002
  └─ ...
▼ chr2 (85 reads)
  ├─ read_050
  └─ ...
```

### Searching Reads

Use the Search panel to filter reads:

**Search by Read ID:**
- Type: `read_001` to find exact match
- Partial match: `001` finds all reads containing "001"
- Case-insensitive

**Search by Reference** (BAM loaded):
- Type: `chr1` to show only chr1 reads
- Partial match: `chr` shows all chromosome reads

**Clear search**: Delete text to show all reads

## Plotting Reads

### Single Read

1. Click any read in the Reads panel
2. Plot appears in the main panel

### Multiple Reads (coming soon)

Currently, plotting multiple reads simultaneously is not supported but is planned for future releases.

## Plot Customization

### Plot Mode

**SINGLE** (default):
- Shows raw signal trace
- X-axis: sample number
- Y-axis: picoamperes (pA)

**EVENTALIGN** (requires BAM with `mv` tag):
- Shows signal with base annotations overlaid
- Colored rectangles represent bases (A/C/G/T)
- Base letters displayed on rectangles
- X-axis: base position or cumulative dwell time

### Normalization Methods

Choose from Plot Options panel:

- **NONE**: Raw signal in picoamperes
- **ZNORM** (default): Z-score normalization (mean=0, std=1)
  - Best for comparing across reads
- **MEDIAN**: Median-centered (median=0)
  - Simple baseline correction
- **MAD**: Median absolute deviation
  - Robust to outliers

### X-Axis Scaling (EVENTALIGN mode only)

- **Base positions** (default): X-axis shows base number (1, 2, 3, ...)
- **Cumulative dwell time**: X-axis shows time in milliseconds
  - Useful for analyzing kinetics

### Downsample Threshold

For large signals (>100,000 samples by default):
- Automatically downsamples using LTTB algorithm
- Preserves visual fidelity while improving performance
- Adjust threshold in Plot Options if needed

### Interactive Features

All plots support:
- **Zoom**: Mouse wheel or box select
- **Pan**: Click and drag
- **Reset**: Reset view button
- **Hover**: Tooltips show values at cursor
- **Save**: Built-in Bokeh save button (downloads PNG)

## Base Modifications

### When Available

The Base Modifications panel appears when:
1. BAM file is loaded
2. BAM contains MM/ML tags (modification calls)

### Modification Types

Common modification types:
- **5mC**: 5-methylcytosine
- **6mA**: 6-methyladenine
- **m5C**: Alternative 5mC notation

### Filtering

**By Modification Type:**
- Check/uncheck modification types to show/hide
- Useful when multiple modification types present

**By Probability:**
- Slider sets minimum probability threshold (0.0 to 1.0)
- Only modifications above threshold are displayed
- Default: 0.0 (show all)

### Visualization Options

**Color by Dwell Time** (default):
- Bases colored by time signal spent at that base
- Longer dwell = warmer color
- Useful for identifying slow regions

**Color by Modification Probability**:
- Bases colored by modification confidence
- Higher probability = warmer color
- Useful for identifying high-confidence modifications

## Exporting Plots

### Export Options

**Command Palette** → `Squiggy: Export Plot` (or `Ctrl+E` / `Cmd+E`)

**Formats:**
- **HTML**: Interactive Bokeh plot (with zoom/pan)
- **PNG**: Static raster image
- **SVG**: Static vector image (best for publications)

**Dimensions:**
- Width and height in pixels
- Lock aspect ratio option (linked icon)

**Zoom Level:**
- **Full plot** (default): Export entire signal
- **Current zoom**: Export only visible region
  - Useful for focusing on specific sections

### Export Quality

**For Publications:**
- Use SVG format for vector graphics
- Set high dimensions (e.g., 1200x800)
- Export at full zoom for complete data

**For Presentations:**
- PNG works well for slides
- Can export specific zoom regions for detail views

**For Sharing:**
- HTML format preserves interactivity
- Recipients can zoom/pan themselves

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+Shift+P` / `Cmd+Shift+P` | Command Palette |
| `Ctrl+E` / `Cmd+E` | Export Plot |
| Click read | Plot read |
| Mouse wheel | Zoom plot |
| Click + drag | Pan plot |

## Tips & Tricks

### Performance

- **Large POD5 files**: Reads load incrementally
- **High-coverage regions**: Use downsampling to improve render speed
- **Many modifications**: Filter by probability to reduce visual clutter

### Workflow

1. **Always load BAM with POD5** for full functionality
2. **Start with default settings** then customize
3. **Use search** to quickly find reads of interest
4. **Export HTML** for sharing interactive plots
5. **Adjust normalization** based on your analysis goals

### Common Issues

**Plot doesn't show:**
- Check that read is selected in Reads panel
- Verify Python kernel is running
- Check Output panel for errors

**No base annotations:**
- Ensure BAM file is loaded
- Verify BAM has `mv` tag (check with `samtools view`)
- Switch to EVENTALIGN plot mode

**No modifications panel:**
- Ensure BAM file contains MM/ML tags
- Check that BAM was basecalled with modification detection

**Export fails:**
- PNG/SVG export requires selenium (auto-downloads geckodriver)
- First export may take longer (driver download)
- Check Output panel for errors

## Example Workflows

### Workflow 1: Basic Signal Visualization

1. Load POD5 file
2. Browse reads in list
3. Click read to plot
4. Adjust normalization if needed
5. Export as PNG for report

### Workflow 2: Modification Analysis

1. Load POD5 and BAM (with MM/ML tags)
2. Switch to EVENTALIGN mode
3. Select read from list
4. Open Base Modifications panel
5. Set probability threshold (e.g., 0.8)
6. Filter to specific modification type
7. Color by modification probability
8. Export zoom region as SVG

### Workflow 3: Kinetic Analysis

1. Load POD5 and BAM (with `mv` tag)
2. Switch to EVENTALIGN mode
3. Enable "Scale x-axis by cumulative dwell time"
4. Enable "Color by dwell time"
5. Identify slow-translocation regions
6. Export specific regions for analysis

## Multi-Sample Comparison

The multi-sample comparison feature allows you to load 2-6+ POD5 datasets simultaneously and compare them with delta tracks showing differences in signal and statistics.

### Quick Start

**Step 1: Load Multiple Samples**

```
Command Palette → "Load Sample (Multi-Sample Comparison)"
├─ Sample name: "v5.0"
├─ Select POD5: data/v5.0.pod5
└─ (Optional) Select BAM: data/v5.0.bam
```

Repeat for each sample you want to load.

**Step 2: View Samples**

In the sidebar, find the **"Sample Comparison Manager"** panel showing all loaded samples with:
- Read counts
- File paths
- Status badges (BAM/FASTA loaded)
- Unload buttons

**Step 3: Run Comparison**

1. Check the checkboxes next to 2+ samples
2. Click **"Start Comparison"** button
3. Delta plot appears in the **Plots** pane

### Use Cases

- **Basecaller Comparison**: Compare guppy v3.0 vs v5.0 vs v6.0
- **Model Evaluation**: Test different pre-trained models on same data
- **QC Between Runs**: Compare signal quality across sequencing runs
- **Protocol Optimization**: Evaluate different library prep methods

### Understanding Delta Plots

The delta plot shows: **Signal B - Signal A**

**Color Coding**:
- 🔴 **Red**: Sample B has higher signal
- 🔵 **Blue**: Sample A has higher signal
- ⚫ **Gray**: No significant difference
- **Bands**: Show confidence/variability

### Python API

For notebook-based analysis:

```python
from squiggy import load_sample, compare_samples, plot_delta_comparison

# Load samples
load_sample("v5.0", "/data/v5.0.pod5")
load_sample("v6.0", "/data/v6.0.pod5")

# Compare read overlap
comparison = compare_samples(["v5.0", "v6.0"])
print(f"Common reads: {len(comparison['common_reads'])}")

# Generate delta plot
plot_delta_comparison(
    sample_names=["v5.0", "v6.0"],
    normalization="ZNORM",
    theme="LIGHT"
)
```

### Advanced Topics

For comprehensive documentation on multi-sample comparison including:
- Sample management and requirements
- Interpreting results
- Troubleshooting
- Performance tips

See the **[Multi-Sample Comparison Guide](multi_sample_comparison.md)**.

## Common Errors

### "BAM index not found"

**Problem**: BAM file is not indexed (missing `.bai` file)

**Solution**: Create an index using samtools:
```bash
samtools index your_file.bam
```

### "No move table found in BAM alignment"

**Problem**: BAM file doesn't contain move tables (`mv` tag), which are required for event-aligned visualization

**Solution**:
- Ensure your BAM was generated with dorado or guppy basecaller (modern versions)
- Older basecalling methods may not include move tables
- Re-basecall your POD5 files with a recent version of dorado

### "Cannot read POD5 file" or "File format error"

**Problem**: POD5 file is corrupted or uses an incompatible format

**Solution**:
- Verify file integrity: `pod5 inspect summary your_file.pod5`
- Ensure you have the latest version of the pod5 Python package
- Check that the file was completely downloaded (not truncated)

### "Python package not installed"

**Problem**: The `squiggy` Python package is not available in the active Python kernel

**Solution**:
- Ensure you're using a virtual environment (not system Python)
- Install the package: `pip install squiggy`
- Restart the Python console in Positron
- Check the selected Python interpreter in Positron matches your virtual environment

### "Plots not rendering" or "Blank plot panel"

**Problem**: Webview or browser rendering issues

**Solution**:
- Reload Positron window (`Ctrl+R` / `Cmd+R`)
- Try a different plot mode (SINGLE vs EVENTALIGN)
- Check the Output panel (Squiggy) for error messages
- Ensure the read ID exists in the POD5 file

### "Modifications not showing"

**Problem**: BAM file doesn't contain modification tags (`MM`/`ML`)

**Solution**:
- Verify your BAM has modification tags: `samtools view your_file.bam | head -1`
- Look for `MM:Z:` and `ML:B:C` tags in the output
- If missing, re-basecall with modification calling enabled (dorado with modification model)

## Getting Help

- **GitHub Issues**: [Report bugs](https://github.com/rnabioco/squiggy-positron/issues)
- **GitHub Discussions**: [Ask questions](https://github.com/rnabioco/squiggy-positron/discussions)
- **Documentation**: [Online docs](https://rnabioco.github.io/squiggy-positron/)
