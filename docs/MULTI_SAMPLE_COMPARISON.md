# Multi-Sample Comparison Guide

Complete guide to comparing multiple POD5 datasets using the Squiggy multi-sample comparison feature.

## Table of Contents

- [Overview](#overview)
- [Key Concepts](#key-concepts)
- [Quick Start](#quick-start)
- [Sample Management](#sample-management)
- [Running Comparisons](#running-comparisons)
- [Interpreting Results](#interpreting-results)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

## Overview

The multi-sample comparison feature allows you to:

- **Load 2-6+ datasets** simultaneously in a single session
- **Compare signal characteristics** between samples
- **Visualize delta tracks** showing differences in aggregate statistics
- **Analyze read overlaps** to find common and unique reads
- **Explore signal distributions** across multiple basecallers or conditions

### Use Cases

- **Basecaller Comparison**: Compare output from different guppy versions (v3.0, v5.0, v6.0)
- **Model Evaluation**: Test multiple pre-trained models on the same flowcell data
- **QC Between Runs**: Compare signal quality across different sequencing runs
- **Protocol Optimization**: Evaluate performance of different library prep methods
- **Multi-Condition Analysis**: Compare treated vs control samples

## Key Concepts

### Sample

A **sample** is a complete analysis unit consisting of:

- **POD5 file** (required): Raw nanopore signal data
- **BAM file** (optional): Aligned reads with base annotations
- **FASTA file** (optional): Reference sequences for motif analysis
- **Sample name** (user-defined): Unique identifier (e.g., "guppy_v5.0", "model_a")

### Session

The **session** is a container managing all loaded samples. You can have one session with multiple samples, or close everything and start fresh.

### Delta Track

A **delta track** visualizes the difference between two samples:

- **Delta Signal**: B - A (sample B minus sample A)
- **Color Coding**:
  - 🔴 **Red**: Sample B has higher signal (B > A)
  - 🔵 **Blue**: Sample A has higher signal (A > B)
  - ⚫ **Gray**: No significant difference (≈0)
- **Confidence Bands**: Show uncertainty range (±1 std dev)

## Quick Start

### 1. Load First Sample

```
Command Palette → "Load Sample (Multi-Sample Comparison)"
├─ Enter sample name: "v5.0"
├─ Select POD5 file: data/run1.pod5
├─ Select BAM file (optional): data/run1.bam
└─ Select FASTA file (optional): skip
```

### 2. Load Second Sample

```
Command Palette → "Load Sample (Multi-Sample Comparison)"
├─ Enter sample name: "v6.0"
├─ Select POD5 file: data/run2.pod5
├─ Select BAM file (optional): data/run2.bam
└─ Confirm
```

### 3. View Samples

In the **Squiggy sidebar**, find the **Sample Comparison Manager** panel showing:

```
Loaded Samples (2)

☑ v5.0
  POD5: /data/run1.pod5
  Reads: 1,234
  BAM FASTA
  [Unload]

☑ v6.0
  POD5: /data/run2.pod5
  Reads: 1,234
  BAM FASTA
  [Unload]

Selected: 2 sample(s)
[Start Comparison]
```

### 4. Run Comparison

Click **"Start Comparison"** button → Delta plot appears in **Plots** pane

## Sample Management

### Loading Samples

**Method 1: Command Palette**

```
Command Palette (Cmd/Ctrl+Shift+P)
→ Search: "Load Sample"
→ Enter name: e.g., "basecaller_v5"
→ Select POD5 file
→ (Optional) Select BAM file
→ (Optional) Select FASTA file
```

**Method 2: Keyboard Shortcut**

(If configured in your keybindings.json)

```json
{
  "key": "cmd+shift+l",
  "command": "squiggy.loadSample"
}
```

### Sample Requirements

| File | Required | Format | Notes |
|------|----------|--------|-------|
| POD5 | ✅ Yes | `.pod5` | Raw signal data from Oxford Nanopore |
| BAM | ❌ No | `.bam` + `.bai` | Must be indexed; contains alignments |
| FASTA | ❌ No | `.fa`, `.fasta`, `.fna` | Reference for motif analysis |

### Viewing Sample Details

The **Sample Comparison Manager** panel shows:

- **Sample name**: User-defined identifier
- **POD5 path**: Location of signal file
- **Read count**: Number of reads in the POD5
- **BAM badge**: Present if BAM file loaded
- **FASTA badge**: Present if FASTA file loaded

### Removing Samples

```
Sample Comparison Manager panel
→ Click [Unload] button next to sample name
→ Confirm in dialog
→ Sample removed from session
```

**Note**: Removing a sample doesn't delete the files, only closes them in the session.

## Running Comparisons

### Selecting Samples

In the **Sample Comparison Manager** panel:

1. Check the checkboxes next to samples to compare
2. Need **minimum 2 samples** for comparison
3. Can select all samples (2, 3, 4, etc.)

```
☐ v5.0
☑ v6.0     ← Selected
☑ v7.0     ← Selected
☐ v8.0
```

### Generating Delta Plot

**Option 1: Via Panel**

```
Sample Comparison Manager
→ Check 2+ samples
→ Click [Start Comparison]
→ Plot appears in Plots pane
```

**Option 2: Via Command Palette**

```
Command Palette → "Plot Delta Comparison"
→ Multi-select samples in quickpick
→ Confirm selection
→ Plot appears
```

### Plot Customization

Access **Plot Options** panel to adjust:

- **Normalization**: ZNORM (default), MAD, MEDIAN, NONE
- **Theme**: Auto-detect light/dark from VSCode
- **Export Format**: PNG, SVG, or HTML

## Interpreting Results

### Delta Signal Track

**What it shows**: Difference in aggregate signal between samples

```
Δ Signal = Signal_B - Signal_A
```

**Reading the plot**:

- **Red region** (positive): Sample B has higher signal at this position
  - Indicates higher amplitude, cleaner signal, or different pore characteristic
- **Blue region** (negative): Sample A has higher signal
  - Indicates Sample A's basecaller or model is more sensitive
- **Gray region** (near zero): Similar signal between samples
  - Good agreement or similar basecaller characteristics

**Confidence bands**: Shaded area around the delta line

- Shows variability in the difference
- Wider bands = more inconsistent differences across reads
- Narrow bands = consistent differences (likely systematic)

### Delta Statistics Track

Shows coverage comparison:

- **Coverage A**: Number of reads mapped at each position in sample A
- **Coverage B**: Number of reads mapped at each position in sample B
- **Difference**: Indicates alignment differences or sample quality variations

### Example Interpretation

```
Comparing: v5.0 vs v6.0

Signal:   v6.0 shows +0.2 to +0.5 pA higher signal (red region)
          → v6.0 basecaller produces stronger signals

Coverage: Both samples have similar coverage
          → Good read overlap, comparable sequencing depth

Conclusion: v6.0 improved signal quality without losing reads
```

## Advanced Usage

### Python API

For notebook-based analysis:

```python
from squiggy import load_sample, compare_samples, plot_delta_comparison

# Load samples
load_sample("model_a", "/data/model_a.pod5", bam_path="/data/model_a.bam")
load_sample("model_b", "/data/model_b.pod5", bam_path="/data/model_b.bam")

# Get comparison statistics
comparison = compare_samples(["model_a", "model_b"])
print(f"Common reads: {len(comparison['common_reads'])}")
print(f"Unique to A: {len(comparison['unique_to_a'])}")
print(f"Unique to B: {len(comparison['unique_to_b'])}")

# Generate delta plot
html = plot_delta_comparison(
    sample_names=["model_a", "model_b"],
    normalization="ZNORM",
    theme="LIGHT"
)
```

### Reading Overlap Analysis

Check which reads are present in both samples:

```python
from squiggy import get_common_reads, get_unique_reads

# Reads in both samples
common = get_common_reads("model_a", "model_b")
print(f"{len(common)} reads in both samples")

# Reads only in sample A
unique_a = get_unique_reads("model_a", "model_b")
print(f"{len(unique_a)} reads only in A")

# Reads only in sample B
unique_b = get_unique_reads("model_b", "model_a")
print(f"{len(unique_b)} reads only in B")
```

### Signal Distribution Comparison

Compare statistical properties:

```python
from squiggy import compare_signal_distributions

dist = compare_signal_distributions(signal_a, signal_b)
print(f"Mean A: {dist['mean_a']:.2f} pA")
print(f"Mean B: {dist['mean_b']:.2f} pA")
print(f"Difference: {dist['mean_a'] - dist['mean_b']:.2f} pA")
```

### Batch Comparison Workflow

```python
from squiggy import load_sample, plot_delta_comparison

# Load multiple basecaller versions
versions = {
    "v5.0": "/data/guppy_v5.0.pod5",
    "v6.0": "/data/guppy_v6.0.pod5",
    "v7.0": "/data/guppy_v7.0.pod5"
}

for name, pod5_path in versions.items():
    load_sample(name, pod5_path)

# Compare v6.0 vs v5.0
plot_delta_comparison(["v5.0", "v6.0"], theme="LIGHT")

# Compare v7.0 vs v6.0
plot_delta_comparison(["v6.0", "v7.0"], theme="LIGHT")
```

## Troubleshooting

### "Delta comparison requires at least 2 loaded samples"

**Problem**: Trying to start comparison with <2 samples

**Solution**:
1. Load at least one more sample
2. Both samples need different POD5 files (same file loaded twice is allowed but unusual)

### Plot appears empty or shows no data

**Problem**: Delta plot shows but lacks visualization

**Causes & Solutions**:

| Cause | Solution |
|-------|----------|
| No common reads between samples | Verify BAM files are mapped to same reference |
| Samples too different in depth | Try samples with similar read counts |
| Very short region | Pan/zoom the plot to see detail |

**Check sample compatibility**:

```python
from squiggy import compare_samples

comparison = compare_samples(["sample_a", "sample_b"])

# Should have reasonable overlap
if len(comparison['common_reads']) < 10:
    print("⚠️ Warning: Very few common reads!")
    print(f"Sample A: {comparison['total_reads_a']} reads")
    print(f"Sample B: {comparison['total_reads_b']} reads")
```

### Red outline around delta values means alignment issues

**Check**:

1. BAM files must be indexed (`.bai` file present)
2. BAM files should be aligned to the same reference
3. CIGAR strings must be valid

**Verify BAM files**:

```bash
# Check if indexed
samtools index sample.bam  # Creates sample.bam.bai

# Check reference compatibility
samtools view -H sample.bam | grep @SQ
```

### Unload not working or gives error

**Solution**:

1. Close the sample in the panel
2. If stuck, clear all state:
   ```
   Command Palette → "Clear All State (After Kernel Restart)"
   → Restart Python kernel in Positron
   ```

### Extension crashes when loading large files

**Symptom**: Positron hangs or extension becomes unresponsive

**Solutions**:

1. **For POD5 files > 1GB**: Load in batches
   - Close other samples before loading new ones
   - Use a subset of reads if possible

2. **For BAM files**: Ensure they're indexed
   ```bash
   samtools index large.bam
   ```

3. **Restart**: Kill kernel and reload extension
   ```
   Python kernel restart button (🔄) in Positron
   ```

## Performance Tips

### Best Practices

1. **Start with 2-3 samples** before loading more
2. **Use BAM files from same reference** for consistency
3. **Keep file sizes under 500MB** when possible
4. **Close unused samples** to free memory

### Memory Usage

Approximate memory per sample:

| File Type | Typical Size | Memory Impact |
|-----------|--------------|---------------|
| POD5 | 100-500 MB | ~200 MB in session |
| BAM (indexed) | 50-200 MB | ~100 MB in session |
| FASTA | <10 MB | Negligible |

**Total for 3 samples**: ~900 MB RAM (varies by complexity)

### For Large Datasets

If working with very large POD5 files (>500 MB):

```python
# Load full file
load_sample("full", "/data/large.pod5")

# Or load subset of reads (if framework allows)
# This would require custom Python code to sample reads
```

## Related Topics

- [User Guide](USER_GUIDE.md) - Basic Squiggy usage
- [API Reference](api.md) - Python API documentation
- [Developer Guide](DEVELOPER.md) - Architecture and extension development
- [Issue #61](https://github.com/rnabioco/squiggy-positron/issues/61) - Feature implementation details
