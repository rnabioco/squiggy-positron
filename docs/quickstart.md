# Quick Start Guide

This guide will help you get started with Squiggy quickly, from installation to your first visualizations.

## Installation

### Option 1: Pre-built Executable (Recommended)

The easiest way to get started is to download a pre-built executable for your platform:

**macOS:**
1. Download [Squiggy-macos.dmg](https://github.com/rnabioco/squiggy/releases/latest)
2. Open the DMG file
3. Drag Squiggy.app to your Applications folder
4. Right-click the app and select "Open" (first launch only)

!!! note "macOS Security"
    On first launch, macOS may block the app. Go to System Preferences → Security & Privacy and click "Open Anyway".

**Development Builds:**
For the latest development version (macOS only), download from the ["latest" release](https://github.com/rnabioco/squiggy/releases/tag/latest).

### Option 2: Install from Source (Development)

**Note**: This quickstart is for the old standalone Qt application. For the **Positron extension** (recommended), see the main [README](../README.md#installation).

If you prefer to run the legacy standalone app from source:

```bash
# Clone the repository
git clone https://github.com/rnabioco/squiggy-positron.git
cd squiggy-positron

# Install git-lfs (if not already installed)
# macOS: brew install git-lfs
# Ubuntu: sudo apt-get install git-lfs
git lfs install
git lfs pull

# Option A: Using pixi (manages Python + Node.js)
pixi install && pixi run setup
pixi run build

# Option B: Using uv (Python only, requires Node.js separately)
# Install uv: https://github.com/astral-sh/uv
uv venv
source .venv/bin/activate  # macOS/Linux
uv pip install -e ".[dev,export]"
npm install
npm run package

# The extension is now in build/squiggy-positron-*.vsix
# Install in Positron: Extensions → ... → Install from VSIX
```

## First Steps: Sample Data

Squiggy comes with bundled sample data to help you explore features immediately.

### Opening Sample Data

1. **Launch Squiggy**
2. Go to **File → Open Sample Data** (or press `Ctrl+Shift+O` / `Cmd+Shift+O`)
3. The sample data loads automatically:
   - POD5 file: Yeast tRNA sequencing reads ([aa-tRNA-seq](https://pubmed.ncbi.nlm.nih.gov/40835813/))
   - BAM file: Corresponding alignments with basecalls
   - 180 reads mapping to yeast tRNA genes

### Exploring a Read

1. **Browse reads** in the left panel - the read list shows all read IDs
2. **Click any read** - The signal plot displays in the center panel
3. **Examine the plot:**
   - X-axis: Time in seconds
   - Y-axis: Raw signal in picoamperes (pA)
   - Interactive: Use mouse wheel to zoom, drag to pan
   - Hover over the signal to see tooltips with exact values

### Trying Different Modes

Expand the **Plot Options** panel on the right side and try different visualization modes:

**Single Read Mode** (default):
- Shows one read at a time
- Best for examining individual read quality

**Overlay Mode:**
1. Select multiple reads (Ctrl+Click or Cmd+Click)
2. Switch to **Overlay** mode
3. Change normalization to **Z-score** for better comparison
4. Each read appears in a different color

**Event-Aligned Mode:**
1. Select a single read
2. Switch to **Event-aligned** mode
3. See bases overlaid on the signal as colored bands
4. Toggle "Show base letters" to display/hide labels

## Working with Your Own Data

### Loading POD5 Files Only

For basic signal visualization without base annotations:

**Via GUI:**
1. Launch Squiggy
2. Go to **File → Open POD5 File...** (`Ctrl+O` / `Cmd+O`)
3. Select your POD5 file
4. Browse and click reads to visualize

**Via Command Line:**
```bash
squiggy --pod5 /path/to/data.pod5
# Or use short form
squiggy -p /path/to/data.pod5
```

### Loading POD5 + BAM Files (Recommended)

For the full experience with base annotations, region filtering, and sequence search:

**Via Command Line (Recommended):**
```bash
# Load both files together
squiggy --pod5 data.pod5 --bam alignments.bam
# Or use short form
squiggy -p data.pod5 -b alignments.bam

# Auto-select specific read in eventalign mode
squiggy -p data.pod5 -b alignments.bam --mode eventalign --read-id READ_ID

# Auto-select reads from genomic region
squiggy -p data.pod5 -b alignments.bam --mode overlay --region "chr1:1000-2000"

# Launch aggregate mode for a reference sequence
squiggy -p data.pod5 -b alignments.bam --mode aggregate --reference "chr1"
```

**Via GUI:**
1. Open the POD5 file first (File → Open POD5 File)
2. Then open the BAM file (File → Open BAM File)
3. Event-aligned, genomic region search, and sequence search features are now enabled

!!! tip "Why Load Both Files?"
    Loading only the POD5 file shows raw signal without sequence context. Adding the BAM file enables:

    - **Event-aligned visualization** with base annotations overlaid on signal
    - **Genomic region filtering** (e.g., chr1:1000-2000)
    - **DNA sequence motif search** within reads
    - **Aggregate mode** for multi-read pileup analysis

### BAM File Requirements

Your BAM file must contain:
- Matching read IDs with the POD5 file
- **Move tables** (generated by modern basecallers like Dorado or Guppy)
- Move tables stored in the "mv" BAM tag

## Basic Features

### Read Selection

**Single Selection:**
- Click any read to visualize it

**Multiple Selection:**
- Hold `Ctrl` (Windows/Linux) or `Cmd` (macOS) and click to select multiple reads
- Hold `Shift` and click to select a range
- Use with Overlay or Stacked modes to compare reads

### Search Features

The Search panel (bottom right) offers three modes:

**1. Read ID Search:**
- Type in the search box to filter reads in real-time
- Case-insensitive matching
- Great for large files with thousands of reads

**2. Reference Region Search** (requires BAM):
- Switch search mode to "Reference Region"
- Enter a region like `chr1:1000-2000`
- Click "Browse References" to see available references
- Filters read list to show only reads from that region

**3. Sequence Search** (requires BAM):
- Switch search mode to "Sequence"
- Enter a DNA motif (e.g., `GGACT`)
- Check "Include reverse complement" to search both strands
- Results show all matches in the current read's reference sequence
- Click a result to zoom the plot to that position

### Signal Normalization

Normalization is crucial for comparing signals across reads:

- **None (raw)**: Original picoampere values - best for single reads
- **Z-score**: Mean=0, Std=1 - best for comparing reads (recommended for Overlay/Stacked)
- **Median**: Median-centered - robust to outliers
- **MAD**: Median absolute deviation - most robust to extreme outliers

### Plot Modes Overview

Squiggy offers five visualization modes (see the [Usage Guide](usage.md) for details):

1. **Single**: Traditional one-read-at-a-time view
2. **Overlay**: Compare multiple reads on same axes (up to 10 reads)
3. **Stacked**: Vertically offset reads (Squigualiser-style)
4. **Event-aligned**: Base annotations with per-base signal averaging
5. **Aggregate**: Multi-read pileup with signal, base, and quality tracks

## Next Steps

Now that you've got the basics:

- **Explore plot modes**: Try Overlay with 3-5 reads and Z-score normalization
- **Experiment with search**: Use genomic regions or sequence motifs
- **Export plots**: File → Export Plot (`Ctrl+E` / `Cmd+E`) - supports HTML, PNG, SVG
- **Read the full docs**: Check out the [Usage Guide](usage.md) for complete feature documentation

## Keyboard Shortcuts

- `Ctrl/Cmd + O`: Open POD5 file
- `Ctrl/Cmd + Shift + O`: Open sample data
- `Ctrl/Cmd + E`: Export current plot
- `Ctrl/Cmd + D`: Toggle dark/light theme
- `Ctrl/Cmd + Q`: Quit application

## Troubleshooting

**Application won't start (macOS):**
- Right-click the app and select "Open" (first launch)
- Check System Preferences → Security & Privacy
- Allow the app to run

**Can't open POD5 file:**
- Verify the file is valid POD5 format (not corrupted)
- Check file permissions

**Event-aligned mode not available:**
- Ensure you've loaded a BAM file (File → Open BAM File)
- Verify BAM contains matching read IDs
- Check that BAM has move tables ("mv" tag)

**Plots not displaying:**
- Try selecting a different read
- Check the status bar for error messages
- Ensure the read has valid signal data

## Getting Help

- **Documentation**: [Complete usage guide](usage.md)
- **Issues**: [GitHub Issues](https://github.com/rnabioco/squiggy/issues)
- **Discussions**: [GitHub Discussions](https://github.com/rnabioco/squiggy/discussions)
