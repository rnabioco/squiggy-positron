# Squiggy Extension - User Guide

Complete guide to using the Squiggy Positron extension for nanopore signal visualization.

## Table of Contents

- [Getting Started](#getting-started)
- [Loading Data](#loading-data)
- [Browsing Reads](#browsing-reads)
- [Plotting Reads](#plotting-reads)
- [Plot Customization](#plot-customization)
- [Base Modifications](#base-modifications)
- [Exporting Plots](#exporting-plots)
- [Keyboard Shortcuts](#keyboard-shortcuts)

## Getting Started

### Installation

1. Download the latest `.vsix` file from [GitHub Releases](https://github.com/rnabioco/squiggy/releases)
2. In Positron: `Extensions` → `...` → `Install from VSIX...`
3. Select the downloaded `.vsix` file
4. Reload Positron when prompted

### Python Requirements

Ensure your Python environment has the squiggy package:

```bash
pip install squiggy
```

This installs dependencies: `pod5`, `pysam`, `bokeh`, `numpy`

### Opening the Extension

Click the Squiggy icon in the Activity Bar (left sidebar) to open the extension panels:
- **Files** - POD5/BAM file information
- **Search** - Filter reads by ID or reference
- **Reads** - Hierarchical read list
- **Plot Options** - Visualization settings
- **Base Modifications** - Modification filtering (when BAM loaded)

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

## Getting Help

- **GitHub Issues**: [Report bugs](https://github.com/rnabioco/squiggy/issues)
- **GitHub Discussions**: [Ask questions](https://github.com/rnabioco/squiggy/discussions)
- **Documentation**: [Online docs](https://rnabioco.github.io/squiggy/)
