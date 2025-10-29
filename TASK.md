# Task: Add Dwell Time Display and Visualization

## Description
Implement functionality to compute, store, and visualize per-base dwell time derived from move tables in aligned nanopore reads (Issue #17).

## Objective
Extract dwell time from move tables and integrate it into Squiggy's data model and visualization system. This feature will expose dwell time as a signal feature for visualization and analysis.

## Background
- **Dwell time** = duration a base remains in the nanopore before translocation
- Computed from move tables: `dwell_ms = (mv × stride / sample_rate_hz) × 1000`
- Essential for modification detection, signal clarity, and co-variation analyses

## Implementation Phases

### Phase 1: Data Model Enhancement (Priority: HIGH)
**Files to Modify:**
- `src/squiggy/alignment.py` - Add `dwell_time_ms` field to `BaseAnnotation` dataclass
- `src/squiggy/alignment.py` - Compute dwell time in `_parse_alignment()` function
- `src/squiggy/utils.py` - Pass `sample_rate` from POD5 to BAM parsing functions

**Changes:**
1. Add `dwell_time_ms: float | None = None` to `BaseAnnotation` (line 23)
2. Modify `extract_alignment_from_bam()` to accept `sample_rate` parameter
3. Calculate dwell time when creating each `BaseAnnotation`:
   ```python
   dwell_samples = signal_end - signal_start
   dwell_time_ms = (dwell_samples / sample_rate) * 1000
   ```

### Phase 2: Enhanced Tooltips (Priority: HIGH)
**Files to Modify:**
- `src/squiggy/plotter.py` - Add dwell time to hover tooltips

**Changes:**
1. Add dwell time to `ColumnDataSource` data dictionaries
2. Update tooltip definitions to include:
   ```python
   ("Dwell Time", "@dwell{0.2f} ms")
   ```
3. Update tooltips in:
   - EVENTALIGN mode (line ~1150-1162)
   - OVERLAY mode
   - SINGLE mode

### Phase 3: Dwell Time Track in Aggregate Mode (Priority: MEDIUM)
**Files to Modify:**
- `src/squiggy/utils.py` - Add `calculate_dwell_statistics()` function
- `src/squiggy/plotter.py` - Add fourth track to `plot_aggregate()`

**Changes:**
1. Create `calculate_dwell_statistics()` similar to `calculate_quality_by_position()`
2. Return: `{"positions": [...], "mean_dwell": [...], "std_dwell": [...], "median_dwell": [...]}`
3. Add synchronized track showing:
   - Mean dwell time line
   - Confidence bands (mean ± 1 std dev)
   - Tooltip: Position, Mean, Std dev, Median, Coverage

### Phase 4: Color-Mapped Overlay (Priority: LOW)
**Files to Modify:**
- `src/squiggy/plotter.py` - Enable dwell-based coloring in overlay mode

**Changes:**
1. Calculate mean dwell time per read
2. Use `LinearColorMapper` with colorblind-friendly palette
3. Add colorbar showing dwell time scale
4. Note: Partial implementation already exists (line 476-511)

## Success Criteria
- [ ] `BaseAnnotation` dataclass stores pre-computed dwell time
- [ ] Dwell time appears in all relevant hover tooltips
- [ ] Aggregate mode displays fourth track with dwell time statistics
- [ ] No performance degradation (dwell calculated once, not repeatedly)
- [ ] Tests pass with sample data (yeast_trna_reads.pod5 + yeast_trna_mappings.bam)
- [ ] Documentation updated in CLAUDE.md

## Testing Strategy
```bash
# Test with sample data
uv run squiggy -p tests/data/yeast_trna_reads.pod5 -b tests/data/yeast_trna_mappings.bam

# Verify:
1. Switch to EVENTALIGN mode - hover should show dwell time
2. Switch to AGGREGATE mode - should see dwell time track
3. Check tooltip includes "Dwell Time: X.XX ms"
4. Validate dwell times are realistic (1-20 ms typical range)
```

## Integration Points
- **Issue #11 (Modification-Aware)**: Dwell time + mod probability in tooltips
- **Issue #13 (Alignment Clarity)**: Dwell outliers indicate basecaller errors
- **Issue #15 (Co-Variation Maps)**: Dwell as quantitative feature for correlation
- **Issue #16 (Motif Explorer)**: Aggregate dwell profiles across motif instances

## Notes
- Branch: add-dwell-time-display
- Base: main
- Created: 2025-10-29
- Worktree: /Users/jayhesselberth/devel/rnabioco/squiggy-add-dwell-time-display
- Difficulty: 2/5 (Easy-Moderate)
- Estimated time: 3-4 hours for Phase 1+2, 10-15 hours total

## Key Technical Considerations
1. **Sample rate propagation**: Need to thread through POD5 → BAM parsing
2. **Stride handling**: Already correct in existing code (alignment.py:86, 131)
3. **Performance**: Pre-compute in data model to avoid repeated calculations
4. **Units**: Always display as "ms" (milliseconds) for clarity

## Related Files
- `src/squiggy/alignment.py` - Data structures for base annotations
- `src/squiggy/plotter.py` - Visualization and plotting
- `src/squiggy/utils.py` - POD5/BAM file operations
- `src/squiggy/constants.py` - Application constants
- `tests/data/` - Sample POD5 and BAM files for testing
