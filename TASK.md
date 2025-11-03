# Task: Multi-File A/B Comparison with Δ Tracks (Issue #61)

## Description
Implementation of multi-file A/B comparison feature for Squiggy, allowing side-by-side comparison of two POD5/BAM datasets with delta tracks showing differences between them.

## Objective
Implement the complete comparison feature across 7 phases:
1. Session infrastructure (ComparisonSession class)
2. Aggregate functions for parallel computation
3. Plotting with delta tracks
4. TypeScript/extension integration
5. Export functionality
6. Testing
7. Documentation

## Phase 1: Multi-Sample Session Infrastructure ✅ COMPLETE

### Architecture: One Session, Multiple Samples
Instead of separate ComparisonSession with session_a/b, unified SquiggySession now manages multiple named POD5/BAM pairs (samples) simultaneously.

### Files Modified
- `squiggy/io.py` - **Sample class** (NEW), enhanced **SquiggySession** with multi-sample support
- `squiggy/utils.py` - ModelProvenance dataclass, header parsing functions (kept from previous design)
- `squiggy/__init__.py` - Export Sample class and multi-sample API functions
- `tests/conftest.py` - Updated cleanup for multi-sample sessions
- `tests/test_multi_sample.py` - 22 comprehensive multi-sample tests (NEW)

### Completed Tasks
- [x] Create Sample class to encapsulate POD5/BAM pairs
- [x] Enhance SquiggySession with samples dict (name -> Sample)
- [x] Implement load_sample(name, pod5_path, bam_path, fasta_path) method
- [x] Implement get_sample(name), list_samples(), remove_sample(name) methods
- [x] Add public API convenience functions
- [x] Export Sample class and new functions in squiggy/__init__.py
- [x] Keep ModelProvenance, extract_model_provenance, validate_sq_headers from Phase 1a
- [x] Update test fixtures and cleanup logic
- [x] Write 22 comprehensive multi-sample tests

## Phase 1 Success Criteria ✅ MET
- [x] One unified session manages 2-6+ POD5/BAM pairs (flexible naming)
- [x] Sample class cleanly encapsulates each file pair
- [x] Backward compatible with existing single-sample API
- [x] Model provenance detection available (ModelProvenance class)
- [x] Reference validation available (validate_sq_headers function)
- [x] All 22 new tests pass + 532 total tests pass (no regressions)
- [x] Code follows project style guidelines

## Phase 2: Aggregate Functions ✅ COMPLETE

### Objectives
Implement aggregate computation functions for parallel statistics calculation across both datasets.

### Files Modified
- `squiggy/utils.py` - Added 3 comparison utility functions (NEW)
- `squiggy/io.py` - Added 3 read comparison functions (NEW)
- `squiggy/__init__.py` - Exported Phase 2 functions with proper categorization
- `tests/test_phase2_comparison.py` - 21 comprehensive comparison tests (NEW)

### Completed Tasks
- [x] Implement compare_read_sets() in squiggy/utils.py
- [x] Implement calculate_delta_stats() in squiggy/utils.py
- [x] Implement compare_signal_distributions() in squiggy/utils.py
- [x] Add get_common_reads() in squiggy/io.py
- [x] Add get_unique_reads() in squiggy/io.py
- [x] Add compare_samples() in squiggy/io.py
- [x] Export new functions in squiggy/__init__.py
- [x] Write tests in tests/test_phase2_comparison.py

### Phase 2 Success Criteria ✅ MET
- [x] Read comparison functions work correctly (get_common_reads, get_unique_reads)
- [x] Delta calculations implemented for statistics (calculate_delta_stats)
- [x] Signal distribution comparisons implemented (compare_signal_distributions)
- [x] Comprehensive sample comparison with reference validation (compare_samples)
- [x] All 21 Phase 2 tests pass + 553 total tests pass (no regressions)
- [x] Code follows project style guidelines

## Phase 3: Plotting with Delta Tracks ✅ COMPLETE

### Objectives
Implement plotting strategies to visualize comparisons with delta tracks showing differences between datasets.

### Files Modified
- `squiggy/plot_strategies/delta.py` - New DeltaPlotStrategy (NEW)
- `squiggy/constants.py` - Added DELTA plot mode and delta-specific constants
- `squiggy/plot_factory.py` - Updated to support DeltaPlotStrategy
- `squiggy/__init__.py` - Added plot_delta_comparison() public API function
- `tests/test_phase3_delta.py` - 19 comprehensive delta plotting tests (NEW)

### Completed Tasks
- [x] Create DeltaPlotStrategy in squiggy/plot_strategies/delta.py
- [x] Implement two-track delta visualization (signal + stats)
- [x] Add delta track rendering with confidence bands
- [x] Add DELTA mode to PlotMode enum in constants.py
- [x] Extend plot_factory.py to support delta plot mode
- [x] Create delta visualization with color-coding (positive/negative/neutral)
- [x] Add plot_delta_comparison() convenience function
- [x] Write 19 comprehensive tests for delta plotting
- [x] Ensure all 572 tests pass (19 Phase 3 + 21 Phase 2 + 22 Phase 1 + 510 existing)

### Phase 3 Success Criteria ✅ MET
- [x] Delta tracks display correctly on plots with confidence bands
- [x] Color-coded visualization: red for positive, blue for negative, gray for neutral
- [x] Works with 2+ samples for comparison (scalable architecture)
- [x] All existing plot modes remain functional (572 tests pass, zero regressions)
- [x] Comprehensive test coverage for delta plotting scenarios
- [x] Code follows project style guidelines and Strategy Pattern
- [x] Support for both light and dark themes
- [x] Proper error handling and data validation

---

## Notes
- Branch: feature/comparison-session
- Base: main
- Related: Issue #33, PR #59 (SquiggySession foundation)
- Implementation plan in Issue #61

## Architecture: One Session, Multiple Samples

From redesign feedback:
- Single SquiggySession manages multiple named POD5/BAM pairs (samples)
- Each pair is a Sample object (name, pod5_path, pod5_reader, read_ids, bam_path, bam_info, model_provenance, fasta_path, fasta_info)
- Scalable to 2-6+ samples (not limited to A/B)
- Flexible user-defined naming (vs hard-coded session_a/session_b)

Key Classes:
```python
class Sample:
    """Represents one POD5/BAM file pair"""
    name: str
    pod5_path: str
    pod5_reader: pod5.Reader
    read_ids: list[str]
    bam_path: str | None
    bam_info: dict | None
    model_provenance: dict | None
    fasta_path: str | None
    fasta_info: dict | None

class SquiggySession:
    """Manages multiple samples"""
    samples: dict[str, Sample]  # name -> Sample

    # Multi-sample methods
    load_sample(name, pod5_path, bam_path=None, fasta_path=None) -> Sample
    get_sample(name) -> Sample | None
    list_samples() -> list[str]
    remove_sample(name) -> None
```

## Summary

This task implements a complete multi-sample comparison framework for Squiggy:

**Phase 1**: One unified SquiggySession manages multiple named POD5/BAM pairs (Sample objects). Scalable to 2-6+ samples with flexible user-defined naming.

**Phase 2**: Comparison utility functions compute read overlaps, delta statistics (B - A), and signal distribution differences between samples.

**Phase 3**: Delta plotting visualizes differences via synchronized tracks showing delta signal with confidence bands and coverage comparisons.

All three phases are complete with 572 passing tests and zero regressions.

## Progress
- [x] Worktree created
- [x] Phase 1 complete (Session infrastructure) - 22 tests
- [x] Phase 2 complete (Aggregate functions and comparison) - 21 tests
- [x] Phase 3 complete (Plotting with delta tracks) - 19 tests
- [ ] Phase 4-7 pending (TypeScript integration, Export, Testing, Documentation)
