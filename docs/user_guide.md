# Squiggy Extension - User Guide

Complete guide to using the Squiggy Positron extension for nanopore signal visualization.

## Installation

### Recommended: Install from OpenVSX

1. Open Positron IDE
2. Open the Extensions panel (sidebar icon or `Cmd+Shift+X` / `Ctrl+Shift+X`)
3. Search for "Squiggy"
4. Click **Install**

Or visit the [OpenVSX marketplace page](https://open-vsx.org/extension/rnabioco/squiggy-positron).

### Alternative: Install from VSIX

1. Download the latest `.vsix` file from [GitHub Releases](https://github.com/rnabioco/squiggy-positron/releases)
2. In Positron: `Extensions` → `...` → `Install from VSIX...`
3. Select the downloaded file
4. Reload when prompted

## Getting Started

### First Time Setup

When you first activate Squiggy, if the Python package is not installed, you'll see a **Setup** panel in the sidebar with step-by-step instructions:

1. **Create a virtual environment** - The panel provides commands you can copy to create a `.venv` folder in your workspace
2. **Install the Python package** - Copy the provided `uv pip install` command to install `squiggy-positron`
3. **Select Python interpreter** - Click the "Select Python Interpreter" button and choose your new virtual environment
4. **Verify** - Click "Check Again" to confirm the setup is complete

The setup panel will automatically disappear once Squiggy detects the Python package in your active environment.

> **Why a virtual environment?** Squiggy requires the `squiggy-positron` Python package along with dependencies like `pod5`, `pysam`, and `bokeh`. Virtual environments keep these isolated from your system Python, following Python best practices.

### Opening the Extension

Once setup is complete, click the **Squiggy icon** in the Activity Bar (left sidebar) to reveal:

- **Setup** - Installation instructions (only visible when package not installed)
- **Files** - POD5/BAM/FASTA file information
- **Search** - Filter reads by ID or reference
- **Reads** - Hierarchical read list with Plot buttons
- **Advanced Plotting** - Visualization settings and analysis type
- **Base Modifications** - Modification filtering (appears when BAM with mods loaded)
- **Samples** - Multi-sample comparison manager

## Sample Data

Squiggy includes sample data for testing and learning:

- **POD5 file**: `yeast_trna_reads.pod5` (180 reads from yeast tRNA sequencing)
- **BAM file**: `yeast_trna_mappings.bam` (corresponding alignments with base calls)

**To load sample data:**
1. Open Command Palette (`Cmd+Shift+P` / `Ctrl+Shift+P`)
2. Type `Squiggy: Load Test Data`
3. Sample files automatically load into the extension

Alternatively, download files directly from the [GitHub repository](https://github.com/rnabioco/squiggy-positron/tree/main/squiggy/data).

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

### Multiple Reads

See the [Aggregate Plots](#aggregate-plots) section for multi-read visualization.

## Plot Customization

### Analysis Type

**Single Read** (default):
- Visualize individual read signal traces
- Choose between Standard or Event-Aligned view modes

**Aggregate** (requires BAM):
- Multi-read statistics across a reference region
- Dynamic panels for modifications, pileup, dwell time, signal, and quality
- See [Aggregate Plots](#aggregate-plots) section for details

### View Mode (Single Read only)

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

Signal downsampling reduces data points for faster rendering while preserving visual quality:

- **Adaptive mode** (default): Automatically calculates optimal downsampling
  - Small signals: 5x downsampling (every 5th point)
  - Large signals: Adaptive (targets ~50,000 points for smooth rendering)
- **Manual override**: Set explicit downsample factor in Plot Options
  - `downsample=1`: No downsampling (all points, may be slow for large signals)
  - `downsample=10`: Every 10th point (more aggressive)
- **Stride-based sampling**: Uses every Nth point (simple and fast)

**Note**: Downsampling is for visualization only and may miss signal spikes between sampled points. For quantitative analysis, work with the original data via the Python API.

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

## Aggregate Plots

Aggregate plots visualize statistics across multiple reads aligned to the same reference region. This is useful for identifying patterns, modifications, and signal characteristics across a population of reads.

### Requirements

- **BAM file** with alignments must be loaded
- Reads must be aligned to reference sequences

### Creating Aggregate Plots

1. In the **Advanced Plotting** panel, select **Analysis Type: Aggregate**
2. Choose a **Reference** sequence from the dropdown
3. Set **Maximum Reads** to include (default: 100)
4. Select which panels to display (see below)
5. Click **Generate Aggregate Plot** button

### Available Panels

You can toggle individual panels on/off before generating the plot:

#### Base Modifications Panel
- **Heatmap visualization** of modification patterns
- Y-axis: Modification types (5mC, 6mA, pseudouridine, etc.)
- X-axis: Reference positions
- **Opacity**: Represents `frequency × probability`
  - Frequency: Fraction of reads with this modification at this position
  - Probability: Mean confidence when modification is called
  - Normalized to [0.2, 1.0] range for visibility
- **Hover tooltips** show:
  - Frequency (fraction of reads modified)
  - Mean probability (confidence)
  - Read counts (modified / total coverage)
  - Standard deviation
- **Filtering**: Integrates with Base Modifications panel
  - Set probability thresholds
  - Enable/disable specific modification types

#### Pileup Panel
- **Base coverage** at each reference position
- Bar chart showing read depth
- Reference sequence labels on bars (if motif search used, motif bases highlighted)

#### Dwell Time Panel
- **Mean dwell time** (milliseconds) at each base position
- Confidence bands showing ±1 standard deviation
- **Auto-scaling y-axis**: Adapts to data range after zoom
- Only available when BAM contains move tables (`mv` tag)

#### Signal Panel (default)
- **Mean signal** (picoamperes) at each reference position
- Confidence bands showing ±1 standard deviation
- Normalization applied (Z-score, Median, MAD, or None)

#### Quality Panel (default)
- **Mean quality scores** at each base position
- Confidence bands showing ±1 standard deviation

### Interpretation

**Modification Heatmap:**
- Dark/opaque rectangles: High frequency AND high confidence
- Faint rectangles: Low frequency OR low confidence
- No rectangle: Modification not detected or filtered out

**Dwell Time:**
- Peaks indicate positions where polymerase pauses
- May correlate with modifications or difficult-to-sequence regions

**Signal Patterns:**
- Consistent signal = uniform base calling
- High variance (wide bands) = heterogeneous reads or difficult regions

### Tips

- Start with fewer reads (50-100) for faster rendering
- Use motif search to focus on specific sequences of interest
- Combine modification filters with aggregate plots to focus on high-confidence calls
- Export as HTML to preserve interactivity for presentations

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

Squiggy is optimized for typical nanopore datasets with automatic performance tuning:

**Adaptive Downsampling** (New in v0.3.3)
- Automatically optimizes plot rendering for large signals
- Targets ~50,000 data points per plot for smooth interactivity
- Small signals (< 250K samples): uses default downsampling (5x)
- Large signals (> 250K samples): increases downsampling automatically
- You can always override with manual downsample setting

**Plot Type Performance**
- **Single read plots**: Efficiently handles signals with 1M+ samples
- **Overlay plots**:
  - Optimal: ≤ 20 reads
  - Acceptable: up to 50 reads
  - For more reads: use Aggregate plot mode instead
- **Aggregate plots**: Efficiently handles 100s of reads via pre-aggregation

**Performance Tips**
- **Large POD5 files**: Reads load incrementally (lazy loading)
- **High-coverage regions**: Increase downsample setting if plots feel sluggish
- **Many reads**: Use Aggregate mode for population-level analysis instead of overlaying 50+ reads
- **Many modifications**: Filter by probability to reduce visual clutter
- **Export performance**: HTML export captures current zoom level for faster loading

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

## Advanced: Python API

While most users interact with Squiggy through the extension's graphical interface, a Python API is available for programmatic access and notebook-based analysis.

**When to use the Python API:**
- Custom automation workflows
- Batch processing multiple files
- Integration with existing analysis pipelines
- Jupyter/IPython notebook analysis

**Quick example:**

```python
from squiggy import load_pod5, load_bam, plot_read

# Load files
load_pod5("data.pod5")
load_bam("alignments.bam")

# Generate plot
html = plot_read("read_001", plot_mode="EVENTALIGN", normalization="ZNORM")
```

For complete API documentation, see the **[API Reference](api.md)**.

> **Note:** 99% of users will not need the Python API. The extension's UI provides all common functionality through an intuitive interface.

## System Requirements

**Software:**
- **Positron IDE** 2024.09.0 or later
- **Python** 3.12 or later
- **Operating Systems**: macOS (Intel/Apple Silicon), Linux, Windows

**Hardware:**
- **Memory**: 4GB RAM minimum (8GB+ recommended for large POD5 files)
- **Disk Space**: Varies by dataset (POD5 files can be several GB)

**Data Files:**
- **POD5 files** - Oxford Nanopore signal data (required)
- **BAM files** - Aligned reads (optional, enables advanced features)
  - Must be indexed (`.bai` file in same directory)
  - Should contain move tables (`mv` tag) for event-aligned visualization
  - May contain modification tags (`MM`/`ML`) for base modification analysis

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
- Install the package: `pip install squiggy-positron`
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

## Logging and Debugging

Squiggy uses a centralized logging system to help you troubleshoot issues without cluttering your Python console.

### Where Logs Appear

**Extension Logs** - All Squiggy operations are logged to the **Output Channel**:

1. Open the Output panel: `Cmd+Shift+U` (macOS) / `Ctrl+Shift+U` (Windows/Linux)
2. Select **"Squiggy"** from the dropdown menu
3. All file loading, plotting, and error messages appear here

**Python Console** - Your Python console stays clean! Squiggy does NOT print logging messages to the Python console. Only your own Python code and standard Python exceptions appear there.

### Quick Access to Logs

When an error occurs, Squiggy shows a notification with a **"Show Logs"** button - click it to instantly open the Output Channel with all relevant error details.

### What Gets Logged

The Squiggy output channel shows:

- File loading operations (POD5, BAM, FASTA)
- Plot generation requests and results
- Python exceptions with full stack traces
- Extension activation/deactivation
- Session state management
- Warnings and validation errors

### Example Log Output

```
[14:23:45.123] [INFO] Squiggy extension activated
[14:23:47.456] [INFO] Loading POD5 file: reads.pod5
[14:23:50.789] [INFO] Loaded POD5: reads.pod5 (1,234 reads)
[14:24:12.234] [ERROR] Failed to load BAM file
FileNotFoundError: BAM file not found at path: /data/missing.bam
```

### Sharing Logs for Bug Reports

When reporting issues:

1. Reproduce the problem
2. Open Output panel → Squiggy (`Cmd+Shift+U` / `Ctrl+Shift+U`)
3. Copy all logs
4. Include them in your [GitHub issue](https://github.com/rnabioco/squiggy-positron/issues) along with:
   - Steps to reproduce
   - File sizes and types (POD5, BAM, FASTA)
   - Python version and OS

For more details on the logging architecture, see the [Logging Guide](guides/LOGGING.md).

## Getting Help

- **GitHub Issues**: [Report bugs](https://github.com/rnabioco/squiggy-positron/issues)
- **GitHub Discussions**: [Ask questions](https://github.com/rnabioco/squiggy-positron/discussions)
- **Documentation**: [Online docs](https://rnabioco.github.io/squiggy-positron/)
