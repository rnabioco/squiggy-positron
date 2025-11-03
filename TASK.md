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

## Phase 2: Aggregate Functions (Current Focus)

### Objectives
Implement aggregate computation functions for parallel statistics calculation across both datasets.

### Tasks
- [ ] Implement calculate_aggregate_stats_comparison() in squiggy/utils.py
- [ ] Implement calculate_delta_stats() in squiggy/utils.py
- [ ] Implement compare_read_sets() in squiggy/utils.py
- [ ] Add load_motif_dataset_a() and load_motif_dataset_b() in squiggy/io.py
- [ ] Add get_common_reads() in squiggy/io.py
- [ ] Add get_unique_reads_a() and get_unique_reads_b() in squiggy/io.py
- [ ] Export new functions in squiggy/__init__.py
- [ ] Write tests in tests/test_comparison.py (expand existing)

### Success Criteria
- [ ] Aggregate statistics computed correctly for both datasets
- [ ] Delta tracks calculated for visualization
- [ ] Read comparison functions work correctly
- [ ] All tests pass
- [ ] Code follows project style guidelines

## Notes
- Branch: feature/comparison-session
- Base: main
- Related: Issue #33, PR #59 (SquiggySession foundation)
- Implementation plan in Issue #61

## Architecture Notes

From Issue #61:
- ComparisonSession manages two SquiggySession instances (A and B)
- Detects model mismatches via @PG header parsing
- Validates matching SQ headers between datasets
- Creates two global instances: _squiggy_session and _squiggy_comparison

Key Classes:
```python
class ComparisonSession:
    session_a: SquiggySession
    session_b: SquiggySession
    has_model_mismatch: bool
    model_provenance_a: ModelProvenance
    model_provenance_b: ModelProvenance
```

## Progress
- [x] Worktree created
- [x] Phase 1 complete (Session infrastructure)
- [ ] Phase 2 in progress (Aggregate functions)
- [ ] Phase 3-7 pending (Plotting, Integration, Export, Testing, Documentation)
