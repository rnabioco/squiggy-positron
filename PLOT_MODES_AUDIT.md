# Squiggy Plot Modes Audit

## Current Plot Modes (As Implemented)

### 1. **SINGLE / EVENTALIGN** (Read Explorer only, not in Plotting panel)
**Python Strategy**: `single_read.py` / `eventalign.py`
**When Used**: Click on individual read in Read Explorer
**Requirements**: POD5 file
**Features**:
- Single read raw signal visualization
- Optional base annotations (EVENTALIGN mode requires BAM)
- Optional modification tracks (requires BAM with MM/ML tags)
- Dwell time visualization
- Signal point markers

**Samples**: Single read (N=1)

---

### 2. **MULTI_READ_OVERLAY**
**Python Strategy**: `overlay.py`
**UI Label**: "Multi-Read Overlay"
**Requirements**: POD5 file
**Features**:
- Multiple individual reads overlaid with alpha blending
- **NEW**: Multi-sample support with sample-based coloring
- Each sample uses one color for all its reads
- Tiered alpha (0.3-0.8 based on read count)
- Legend shows read IDs (truncated to 12 chars)
- HoverTool shows read ID, sample, position, signal value

**Samples**: 1+ samples
**Reads per sample**: User-configurable (default: 50)
**Alignment**: Not required (works with or without BAM)

---

### 3. **MULTI_READ_STACKED**
**Python Strategy**: `stacked.py`
**UI Label**: "Multi-Read Stacked"
**Requirements**: POD5 file
**Features**:
- Multiple individual reads stacked vertically with offsets
- **NEW**: Multi-sample support with sample-based coloring
- Each sample uses one color for all its reads
- No alpha blending (full opacity since reads don't overlap)
- Auto-calculates vertical offset based on signal range
- Legend shows read IDs (truncated)

**Samples**: 1+ samples
**Reads per sample**: User-configurable (default: 50)
**Alignment**: Not required

---

### 4. **AGGREGATE** (Single Sample)
**Python Strategy**: `aggregate.py`
**UI Label**: "Aggregate (Single Sample)"
**Requirements**: POD5 + BAM (requires alignment to reference)
**Features**:
- **Up to 5 synchronized tracks**:
  1. **Modifications heatmap** (optional, if MM/ML tags present)
  2. **Base pileup** (stacked proportions: A/C/G/T/N)
  3. **Dwell time per base** with confidence bands (optional)
  4. **Mean signal** with confidence bands (mean ± std)
  5. **Quality scores** by position
- Shows aggregate statistics across N reads aligned to a reference
- Reference sequence annotation

**Samples**: Single sample only
**Reads**: User-configurable (default: 100)
**Alignment**: **Required** (must have BAM)

---

### 5. **COMPARE_SIGNAL_OVERLAY** (Multi-Sample Aggregate Overlay)
**Python Strategy**: `signal_overlay_comparison.py`
**UI Label**: "Compare Aggregate Overlay"
**Requirements**: POD5 + BAM for each sample
**Features**:
- Overlays **mean signals** from 2+ samples
- Each sample = one line (aggregate, not individual reads)
- Sample-based coloring (Okabe-Ito palette)
- Coverage track showing aligned read count per position per sample
- Reference sequence annotation
- **Clips to aligned regions** (respects soft-clipping)

**Samples**: 2+ samples
**Reads per sample**: User-configurable (default: 100)
**Alignment**: **Required**

---

### 6. **COMPARE_AGGREGATE** (Multi-Sample Aggregate Comparison)
**Python Strategy**: `aggregate_comparison.py`
**UI Label**: "Compare Aggregate (Multi-Track)"
**Requirements**: POD5 + BAM for each sample
**Features**:
- **4-5 synchronized tracks per sample**:
  1. **Modifications heatmap** (optional)
  2. **Base pileup** (stacked proportions)
  3. **Dwell time** (optional)
  4. **Signal** (mean ± std)
  5. **Quality scores**
- Each sample gets its own set of tracks
- Samples stacked vertically for side-by-side comparison
- Reference sequence annotation

**Samples**: 2+ samples
**Reads per sample**: User-configurable (default: 100)
**Alignment**: **Required**

---

### 7. **COMPARE_SIGNAL_DELTA** (2-Sample Difference)
**Python Strategy**: `delta.py` (exists but not yet wired up in UI)
**UI Label**: "2-Sample Delta (Signal Difference)"
**Requirements**: POD5 + BAM for 2 samples
**Features**:
- Shows **difference** between Sample A and Sample B aggregate signals
- Deviation from zero line
- Highlights regions where samples differ

**Samples**: Exactly 2 samples
**Reads per sample**: User-configurable
**Alignment**: **Required**

---

## Current Issues / Redundancies

### 1. **AGGREGATE vs COMPARE_SIGNAL_OVERLAY**
Both show aggregate statistics, but:
- **AGGREGATE**: Single sample, multi-track (5 tracks)
- **COMPARE_SIGNAL_OVERLAY**: Multi-sample, single track (just signal)

**Potential Consolidation**: Could merge into a unified "Aggregate" mode that:
- Works with 1+ samples
- When 1 sample → show all 5 tracks (current AGGREGATE behavior)
- When 2+ samples → show overlaid signals OR side-by-side multi-track (user toggle?)

---

### 2. **MULTI_READ_OVERLAY vs AGGREGATE**
Both can visualize multiple reads, but:
- **MULTI_READ_OVERLAY**: Individual reads (raw signals)
- **AGGREGATE**: Statistical summary (mean + bands)

**Key Difference**: AGGREGATE requires BAM alignment, MULTI_READ_OVERLAY doesn't

**Potential Consolidation**: Could add a toggle to MULTI_READ_OVERLAY:
- "Show individual reads" (current behavior)
- "Show aggregate statistics" (compute mean/std on the fly)
- This would allow aggregate-style visualization WITHOUT requiring BAM alignment

---

### 3. **COMPARE_AGGREGATE vs COMPARE_SIGNAL_OVERLAY**
Both compare multiple samples with aggregates:
- **COMPARE_AGGREGATE**: Each sample gets 5 tracks (vertical stacking)
- **COMPARE_SIGNAL_OVERLAY**: All samples overlay on 1 signal track

**Potential Consolidation**: These could be views of the same mode:
- Default: Overlay (easier to compare signal directly)
- Optional: Expand to multi-track per sample (more detail)

---

## Proposed Simplified Taxonomy

### Option A: **Merge by Granularity**

1. **Single Read** (Read Explorer only)
   - Individual read visualization
   - Modes: SINGLE, EVENTALIGN

2. **Multi-Read (Raw)**
   - Individual reads overlaid or stacked
   - Works with 1+ samples
   - Modes: OVERLAY, STACKED
   - **No BAM required**

3. **Aggregate (Statistical)**
   - Statistical summary (mean/std/coverage)
   - Works with 1+ samples
   - **NEW**: Toggle between:
     - Simple (signal only)
     - Detailed (5-track view)
   - When 1 sample: Show all tracks
   - When 2+ samples: Overlay OR side-by-side
   - **Requires BAM for alignment**

4. **Comparison (Delta/Difference)**
   - Difference between 2 samples
   - **Requires BAM**

---

### Option B: **Merge by Alignment Requirement**

1. **Single Read** (Read Explorer only)
   - SINGLE, EVENTALIGN

2. **Multi-Read (No Alignment Required)**
   - OVERLAY, STACKED
   - Works with raw POD5 data
   - Can be single or multi-sample
   - **NEW**: Optional aggregate view (compute stats on selected reads)

3. **Aligned Multi-Read (Requires BAM)**
   - AGGREGATE (1+ samples)
   - Modes: Overlay, Multi-track, Delta
   - Shows: Signal, Pileup, Quality, Dwell, Mods

---

## Recommendation

**Merge AGGREGATE and COMPARE_SIGNAL_OVERLAY** into a unified **"Aggregate View"** that:

### **New "Aggregate" Mode (Unified)**
- **Samples**: 1+ samples
- **Requirements**: BAM required
- **View Options** (user toggle):
  - **Simple Overlay**: Mean signal + confidence bands overlaid (current COMPARE_SIGNAL_OVERLAY)
  - **Detailed Multi-Track**: Full 5-track view per sample (current AGGREGATE + COMPARE_AGGREGATE)

### **Keep Separate**:
- **MULTI_READ_OVERLAY** and **MULTI_READ_STACKED**: Different enough (individual reads vs aggregates) and work WITHOUT BAM
- **COMPARE_SIGNAL_DELTA**: Specialized difference plot, not a general-purpose view

### **Benefits**:
- Reduces from 5 main modes to 3: Multi-Read Overlay, Multi-Read Stacked, Aggregate
- Aggregate naturally scales from 1 to N samples
- User can choose detail level (overlay vs multi-track) without switching modes
- Clearer mental model: "Do you want individual reads or aggregate statistics?"

---

## Questions for Discussion

1. Should MULTI_READ_OVERLAY support an "aggregate" toggle that computes mean/std WITHOUT requiring BAM alignment?
2. Should AGGREGATE have a view toggle (simple overlay vs detailed multi-track)?
3. Should COMPARE_SIGNAL_DELTA remain as a separate mode or become a view option within Aggregate?
4. Should we rename modes for clarity?
   - "Multi-Read Overlay" → "Individual Reads (Overlaid)"
   - "Multi-Read Stacked" → "Individual Reads (Stacked)"
   - "Aggregate" → "Aggregate Statistics"
