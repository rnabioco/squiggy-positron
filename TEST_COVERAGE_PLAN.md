# Test Coverage Improvement Plan

**Branch**: `feature/improve-test-coverage`
**Created**: 2025-11-05
**Context**: Working around active development in #102 (Advanced Plotting integration)

---

## Current Test Status

### Overall Coverage
- **Python**: 587 tests passing, 2 skipped - **69% coverage**
- **TypeScript**: 125 tests passing - No coverage metrics yet

### Test Distribution
- **Python**: 23 test files covering core functionality
- **TypeScript**: 9 test suites covering backend, state, utils, services

---

## Coverage by Module

### Strong Coverage (>90%) ✅
- **100%**: `constants.py`, `normalization.py`, `plot_factory.py`, `theme_manager.py`
- **98%**: `single_read.py`, `modification_track_builder.py`, `eventalign.py`
- **97%**: `alignment.py`, `motif.py`, `stacked.py`, `overlay.py`
- **96%**: `delta.py`
- **95%**: `api.py`
- **94%**: `base_annotation_renderer.py`
- **91%**: `base.py` (plot strategies base class)

### Moderate Coverage (60-90%) ⚠️
- **75%** - `io.py` (134 lines uncovered)
- **69%** - `aggregate.py` (75 lines uncovered)
- **61%** - `modifications.py` (28 lines uncovered)

### Low Coverage (<60%) 🔴
- **47%** - `utils.py` (478 lines uncovered) - **LARGEST GAP**
- **42%** - `__init__.py` (181 lines uncovered) - Legacy API
- **33%** - `cache.py` (103 lines uncovered) - New feature
- **17%** - `signal_overlay_comparison.py` (91 lines uncovered) - Being replaced

---

## Priority Plan: Avoid #102 Conflicts

### 🚫 Areas Under Active Development (AVOID)

These files are being modified in #102 and should NOT be tested until after merge:

1. **`__init__.py`** - +186 lines of changes
2. **`signal_overlay_comparison.py`** - Being replaced by `aggregate_comparison.py`
3. **`plot_factory.py`** - Adding new plot modes
4. **`squiggy-plot-options-*`** - Massive UI refactor (982 line changes)
5. **`squiggy-runtime-api.ts`** - +81 lines of new API methods
6. **`src/types/messages.ts`** - Changing for comparison workflows
7. **Sample-related functions in `io.py`** - Multi-sample registry changes

---

## ✅ Safe-to-Test Priorities (No Conflicts)

### Priority 1: `utils.py` - 47% Coverage 🔴

**Uncovered**: 478 lines
**Status**: NOT touched by #102
**Impact**: High - core utilities used throughout extension

#### Areas Needing Tests

**File Validation Utilities**
- `validate_pod5_file()`
- `validate_bam_file()`
- `validate_fasta_file()`
- File existence checks
- File format validation
- Corrupted file handling

**BAM Analysis Functions**
- `get_bam_references()`
- `get_bam_statistics()`
- `parse_bam_header()`
- Reference sequence extraction
- Alignment statistics
- Missing index handling

**Read Filtering**
- `filter_reads_by_quality()`
- `filter_reads_by_length()`
- `filter_reads_by_reference()`
- Complex filter combinations
- Empty result handling

**Quality Score Utilities**
- `calculate_mean_qscore()`
- `phred_to_probability()`
- `probability_to_phred()`
- Edge cases (Q0, Q60+)

**Sequence Utilities**
- `reverse_complement()`
- `translate_sequence()`
- `find_motifs()`
- Invalid base handling
- Empty sequence handling

**Signal Processing Helpers**
- `detect_signal_outliers()`
- `smooth_signal()`
- `calculate_signal_stats()`
- Edge cases (constant signal, single point)

**Error Handling**
- File permission errors
- Malformed data
- Resource exhaustion
- Concurrent access

#### Suggested Test Structure

```python
# tests/test_utils_file_validation.py
class TestPOD5Validation:
    def test_valid_pod5_file(self):
        ...
    def test_nonexistent_pod5_file(self):
        ...
    def test_corrupted_pod5_file(self):
        ...
    def test_pod5_file_permissions(self):
        ...

# tests/test_utils_bam_analysis.py
class TestBAMReferences:
    ...

# tests/test_utils_read_filtering.py
class TestQualityFiltering:
    ...

# tests/test_utils_signal_processing.py
class TestSignalOutlierDetection:
    ...
```

---

### Priority 2: `cache.py` - 33% Coverage 🔴

**Uncovered**: 103 lines
**Status**: NOT touched by #102
**Impact**: High - performance-critical, new feature needs validation

#### Areas Needing Tests

**Cache Operations**
- `CacheManager.get()`
- `CacheManager.set()`
- `CacheManager.invalidate()`
- `CacheManager.clear()`
- Cache hit scenarios
- Cache miss scenarios

**BAM Caching**
- `cache_bam_header()`
- `cache_bam_index()`
- `cache_reference_sequences()`
- Multi-file caching
- Cache invalidation on file change

**Performance & Memory**
- Cache size limits
- LRU eviction
- Memory usage under load
- Large file handling
- Concurrent access

**Error Handling**
- Disk space exhaustion
- Cache corruption
- File deletion while cached
- Partial cache states

#### Suggested Test Structure

```python
# tests/test_cache.py
class TestCacheManager:
    def test_cache_hit(self):
        ...
    def test_cache_miss(self):
        ...
    def test_cache_invalidation(self):
        ...
    def test_cache_size_limit(self):
        ...

class TestBAMCaching:
    def test_cache_bam_header(self):
        ...
    def test_cache_invalidation_on_file_change(self):
        ...
    def test_concurrent_access(self):
        ...

class TestCachePerformance:
    def test_large_file_caching(self):
        ...
    def test_memory_limits(self):
        ...
```

---

### Priority 3: `modifications.py` - 61% Coverage ⚠️

**Uncovered**: 28 lines
**Status**: NOT touched by #102
**Impact**: Medium - base modification features

#### Areas Needing Tests

**Edge Cases**
- Unknown modification types
- Missing MM/ML tags
- Malformed probability values
- Empty modification lists
- Probability values outside [0, 1]

**ChEBI Code Handling**
- Valid ChEBI codes
- Invalid ChEBI codes
- Mixed code/name formats
- Case sensitivity

**Modification Filtering**
- Probability thresholding edge cases
- Multiple modification types
- Type-specific filtering
- Empty results after filtering

**Integration**
- Modifications without alignment
- Alignment without modifications
- Multiple reads with different mods
- Performance with many modifications

#### Suggested Test Structure

```python
# tests/test_modifications_edge_cases.py
class TestModificationEdgeCases:
    def test_unknown_modification_type(self):
        ...
    def test_malformed_probability(self):
        ...
    def test_empty_mm_ml_tags(self):
        ...

class TestChEBICodeHandling:
    def test_valid_chebi_codes(self):
        ...
    def test_invalid_chebi_codes(self):
        ...

class TestModificationFiltering:
    def test_threshold_edge_cases(self):
        ...
    def test_multiple_types_filtering(self):
        ...
```

---

### Priority 4: `aggregate.py` - 69% Coverage ⚠️

**Uncovered**: 75 lines
**Status**: NOT touched by #102
**Impact**: Medium - aggregate plot edge cases

#### Areas Needing Tests

**Edge Cases**
- Sparse coverage regions
- Single-read aggregates
- Empty reference regions
- Extremely long reads
- Very short reads

**Quality Track Variants**
- Missing quality scores
- Quality = 0 (unassigned)
- Quality > 60 (edge of scale)
- Mixed quality ranges

**Signal Extremes**
- Very high signal values
- Very low/negative signal values
- Constant signal across region
- High variance regions

**Reference Handling**
- Unknown reference names
- Empty reference sequence
- Reference with no coverage

#### Suggested Test Structure

```python
# tests/test_aggregate_edge_cases.py
class TestAggregateEdgeCases:
    def test_sparse_coverage(self):
        ...
    def test_single_read_aggregate(self):
        ...
    def test_empty_reference_region(self):
        ...

class TestQualityTrackEdgeCases:
    def test_missing_quality_scores(self):
        ...
    def test_extreme_quality_values(self):
        ...

class TestSignalExtremes:
    def test_very_high_signal(self):
        ...
    def test_constant_signal(self):
        ...
```

---

### Priority 5: `io.py` (Non-Sample Functions) - 75% Coverage ⚠️

**Uncovered**: 134 lines (but some are sample-related, avoid those)
**Status**: Partially modified by #102 (sample functions)
**Impact**: Medium - core file I/O

#### Safe to Test (Not Sample-Related)

**POD5 Reader Error Handling**
- Reader crashes on corrupted files
- VBZ decompression errors
- Incomplete POD5 files
- Missing required fields

**BAM Index Validation**
- Missing `.bai` file
- Mismatched index
- Corrupted index
- Index older than BAM

**File Path Resolution**
- Relative paths
- Symlinks
- Special characters in paths
- Very long paths
- Network paths

**Resource Cleanup**
- File handle cleanup on errors
- Memory leaks with repeated loads
- Cleanup on process termination

**Concurrent File Access**
- Multiple readers on same file
- Read while writing
- Lock handling

#### Avoid Testing (Sample-Related)

- `load_sample()`
- `_squiggy_session.samples`
- Multi-sample registry functions
- Sample comparison utilities

#### Suggested Test Structure

```python
# tests/test_io_error_handling.py
class TestPOD5ErrorHandling:
    def test_corrupted_pod5(self):
        ...
    def test_vbz_decompression_error(self):
        ...

class TestBAMIndexValidation:
    def test_missing_bai_file(self):
        ...
    def test_corrupted_index(self):
        ...

class TestFilePathResolution:
    def test_relative_paths(self):
        ...
    def test_symlinks(self):
        ...

class TestResourceCleanup:
    def test_cleanup_on_error(self):
        ...
    def test_memory_leak_prevention(self):
        ...
```

---

## TypeScript Testing Gaps

### Areas Without Tests

**React Components** (Not in #102, but complex to test)
- `squiggy-reads-core.tsx`
- `squiggy-samples-core.tsx` (avoid - related to #102)
- `squiggy-read-item.tsx`
- `squiggy-reference-group.tsx`
- `column-resizer.tsx`

**Panel Providers** (Some in #102)
- `squiggy-file-panel.ts` (safe)
- `squiggy-reads-view-pane.ts` (safe)
- `squiggy-modifications-panel.ts` (safe)
- `squiggy-samples-panel.ts` (avoid - #102 related)

**Command Handlers**
- `file-commands.ts` (partially in #102)
- `plot-commands.ts` (in #102 - avoid)

**Webview Communication**
- Message passing protocols
- Error handling in webviews
- State synchronization

### Suggested TypeScript Test Additions

Focus on areas NOT in #102:

```typescript
// src/views/__tests__/squiggy-file-panel.test.ts
describe('FilePanelProvider', () => {
    test('displays POD5 metadata', ...);
    test('displays BAM metadata', ...);
    test('handles missing files', ...);
});

// src/views/__tests__/squiggy-reads-view-pane.test.ts
describe('ReadsViewPane', () => {
    test('virtualizes large read lists', ...);
    test('filters reads by search', ...);
    test('groups by reference', ...);
});

// src/views/__tests__/squiggy-modifications-panel.test.ts
describe('ModificationsPanelProvider', () => {
    test('displays modification types', ...);
    test('filters by probability', ...);
    test('handles no modifications', ...);
});
```

---

## Implementation Strategy

### Phase 1: Core Utilities (Week 1)
1. **Day 1-2**: `utils.py` file validation tests
2. **Day 3**: `utils.py` BAM analysis tests
3. **Day 4**: `utils.py` read filtering and signal processing tests
4. **Day 5**: Review and refine

### Phase 2: Performance & Caching (Week 2)
1. **Day 1-2**: `cache.py` basic operations
2. **Day 3**: `cache.py` performance and memory tests
3. **Day 4**: `cache.py` error handling
4. **Day 5**: Integration testing

### Phase 3: Edge Cases (Week 3)
1. **Day 1**: `modifications.py` edge cases
2. **Day 2**: `aggregate.py` edge cases
3. **Day 3**: `io.py` non-sample error handling
4. **Day 4-5**: TypeScript panel tests

### Phase 4: Post-#102 Merge
1. Test new `aggregate_comparison.py` strategy
2. Test Advanced Plotting UI integration
3. Test multi-sample API methods
4. Consider removing/deprecating `signal_overlay_comparison.py` tests
5. Update `__init__.py` tests for new API methods

---

## Success Metrics

### Target Coverage Goals
- **Python overall**: 69% → **85%+**
- **`utils.py`**: 47% → **90%+**
- **`cache.py`**: 33% → **90%+**
- **`modifications.py`**: 61% → **95%+**
- **`aggregate.py`**: 69% → **90%+**
- **`io.py`** (non-sample): 75% → **90%+**

### Quality Metrics
- All new tests must pass
- No flaky tests (should pass consistently)
- Fast execution (<15 seconds for new test suite)
- Clear test names and documentation
- Good edge case coverage

---

## Notes

- Focus on **safe areas** that won't conflict with #102
- Defer testing of comparison/advanced plotting until after #102 merge
- Prioritize high-impact, low-coverage modules
- Use existing test data in `squiggy/data/` or `tests/data/`
- Follow existing test patterns (pytest classes, descriptive names)
- Run full test suite before committing: `pixi run test`

---

## Test Data Requirements

### Existing Test Data
- `squiggy/data/yeast_trna_reads.pod5` (180 reads)
- `squiggy/data/yeast_trna_mappings.bam` (alignments)
- Additional test data in `tests/data/`

### May Need to Create
- Corrupted POD5 file (for error testing)
- BAM without index (for validation testing)
- POD5 with edge case signal values
- Small test files for performance benchmarks
- Files with special characters in paths

Use `scripts/create_test_data.py` if new test data needed.

---

## Related Issues

- #102 - Advanced Plotting integration (active development)
- Future: Consider setting up automated coverage reporting in CI
- Future: Add TypeScript coverage tracking

---

**Last Updated**: 2025-11-05
