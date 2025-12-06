# TSV Import Implementation Progress

**Status**: Phase 1 Complete - Foundation Implemented
**Last Updated**: 2025-11-15
**Session Branch**: `claude/review-repo-status-01Nr8A7Yeb4sHzxSzBJ8FFsu`

## Overview

Implementing TSV (tab-separated values) import functionality for bulk sample loading in Squiggy. Users can define samples in a spreadsheet and import 2-24 samples at once.

**Target Use Cases**:
- Bulk loading (load 10+ samples at once)
- Workflow organization (track file associations in spreadsheet)
- Easy metadata association
- Reproducibility (share manifests with collaborators)

**Strategy**: Lazy loading by default (register samples, load to kernel only when plotting)

---

## Phase 1: Parsing & Validation ✅ COMPLETE

### 1.1 TSV Parser ✅ COMPLETE
**File**: `src/services/tsv-parser.ts`

- [x] Interface definitions (`TSVSampleSpec`, `TSVParseResult`)
- [x] Full implementation of `TSVParser.parse()`
- [x] Auto-detect delimiter (tab vs comma)
- [x] Header validation (require `sample_name` and `pod5` columns)
- [x] Detect duplicate sample names
- [x] Handle optional columns (bam, fasta with `-` or empty = missing)
- [x] Skip empty lines and comments (`#`)
- [x] Comprehensive error and warning reporting
- [x] **Tests**: 28 test cases covering valid input, errors, warnings, edge cases
  - Test file: `src/services/__tests__/tsv-parser.test.ts`

**Features**:
- ✅ Tab and comma delimiter support
- ✅ Case-insensitive column names
- ✅ Line number tracking for errors
- ✅ Windows (CRLF) line ending support
- ✅ Handles paths with spaces
- ✅ Warnings for missing optional files

### 1.2 Path Resolver ✅ COMPLETE
**File**: `src/services/tsv-path-resolver.ts`

- [x] Enum definitions (`PathResolutionStrategy`)
- [x] Full implementation of `TSVPathResolver` class
- [x] Absolute path resolution
- [x] TSV-relative resolution (relative to TSV file directory)
- [x] Workspace-relative resolution (relative to workspace root)
- [x] Auto strategy (tries multiple approaches)
- [x] File existence checking
- [ ] **Tests**: Need to add unit tests

**Strategies**:
1. **Absolute**: Use path as-is (`/data/sample.pod5`)
2. **TSV-relative**: Relative to TSV file location (`data/sample.pod5` → `<tsv-dir>/data/sample.pod5`)
3. **Workspace-relative**: Relative to workspace root
4. **Auto**: Try absolute → TSV-relative → workspace-relative (first success wins)

### 1.3 TSV Validator ✅ COMPLETE
**File**: `src/services/tsv-validator.ts`

- [x] Interface definitions (`ValidationResult`)
- [x] Full implementation of `TSVValidator` class
- [x] File existence validation (POD5 required, BAM/FASTA optional)
- [x] Sample name conflict detection
- [x] Batch validation support
- [x] Summary statistics helper
- [ ] **Tests**: Need to add unit tests
- [ ] **Future**: POD5/BAM overlap validation (deferred to load time)

**Validation Rules**:
- ✅ POD5 missing → BLOCK (error)
- ✅ BAM missing → WARN (optional)
- ✅ FASTA missing → WARN (optional)
- ✅ Sample name conflict → BLOCK (error)
- 🔜 POD5/BAM overlap check (expensive, deferred to load time)

---

## Phase 2: UI & Commands ✅ COMPLETE

### 2.1 TSV Import Command ✅ COMPLETE
**File**: `src/commands/tsv-commands.ts`

- [x] Command registration (`squiggy.importSamplesFromTSV`)
- [x] File picker UI (prioritize .tsv file selection)
- [x] TSV parsing integration
- [x] Validation workflow
- [x] Validation results preview (QuickPick UI)
- [x] Import preview with confirmation dialog
- [x] Smart loading strategy (eager ≤10 samples, lazy ≥20 samples)
- [x] Sample registration in extension state
- [x] Unified state integration
- [ ] **Actual kernel loading**: Currently stubbed (registers metadata only)

**UI Flow**:
1. File picker → Select TSV
2. Parse TSV → Show errors if any
3. Validate samples → Show errors if any
4. Preview import → Show sample count, files, warnings
5. Confirm → Import samples
6. Progress notification → Load samples (TODO: actual kernel loading)
7. Success message → Refresh Samples panel

### 2.2 Command Registration ✅ COMPLETE
- [x] Added to `package.json` commands list
- [x] Imported in `extension.ts`
- [x] Registered in activation function
- [x] Icon: `$(table)` (table icon)
- [x] Enablement: `squiggy.packageInstalled`

**Command Palette**:
- Title: "Squiggy: Import Samples from TSV"
- Category: Squiggy
- Available when package installed

---

## Phase 3: Loading Integration ⚠️ PARTIAL

### 3.1 Sample Loading ⚠️ STUBBED
**File**: `src/commands/tsv-commands.ts` (function `loadSamplesFromTSV`)

- [x] Progress notification
- [x] Iterate through validated samples
- [x] Create `SampleInfo` objects
- [x] Add to extension state (`state.addSample()`)
- [x] Add to unified state (`state.addLoadedItem()`)
- [x] TSV metadata tracking (`sourceType: 'tsv'`, `tsvGroup: 'tsv_<timestamp>'`)
- [ ] **TODO**: Actual kernel loading via `FileLoadingService.loadSampleIntoRegistry()`
- [ ] **TODO**: Eager vs lazy loading implementation
- [ ] **TODO**: Error handling for kernel load failures

**Current Behavior**: Samples are registered in TypeScript state but NOT loaded to Python kernel.

**Next Steps**:
1. Implement eager loading: Call `service.loadSampleIntoRegistry()` for each sample
2. Implement lazy loading: Defer kernel load until plotting
3. Add lazy load trigger in plot commands (check `sample.isLoaded`, load if false)

### 3.2 Lazy Loading Trigger ⬜ NOT STARTED
**File**: `src/commands/plot-commands.ts`

- [ ] Add `ensureSamplesLoaded()` function
- [ ] Call before plotting to load TSV samples on-demand
- [ ] Update `sample.isLoaded` flag after loading

---

## Phase 4: UI Integration ⬜ NOT STARTED

### 4.1 Samples Panel Grouping ⬜ NOT STARTED
**File**: `src/views/squiggy-samples-panel.ts`

- [ ] Group samples by `sourceType` (TSV vs manual)
- [ ] Show TSV batch ID (`tsvGroup`)
- [ ] Visual indicator for lazy-loaded samples (⚠️ not loaded yet)
- [ ] Batch operations (delete all from TSV group)

### 4.2 Session Persistence ⬜ NOT STARTED
**File**: `src/types/squiggy-session-types.ts`

- [ ] Add `tsvMetadata` to `SessionState` interface
- [ ] Track TSV batch import metadata
- [ ] Save/restore TSV-imported samples
- [ ] Preserve `sourceType` and `tsvGroup` in session

### 4.3 Read Explorer Integration ⬜ NOT STARTED
**File**: `src/views/squiggy-reads-view-pane.ts`

- [ ] Trigger lazy load when selecting TSV sample
- [ ] Transparent UX (no user intervention needed)

---

## Testing Status

### Unit Tests
- ✅ **TSVParser**: 28 tests, all passing
- ⬜ **TSVPathResolver**: Not yet implemented
- ⬜ **TSVValidator**: Not yet implemented
- ⬜ **TSV Commands**: Not yet implemented

### Integration Tests
- ⬜ Full import workflow
- ⬜ Eager vs lazy loading
- ⬜ Session persistence
- ⬜ UI integration

### Manual Testing
- ⬜ Import small TSV (5 samples, eager loading)
- ⬜ Import large TSV (20 samples, lazy loading)
- ⬜ Path resolution (absolute, relative, TSV-relative)
- ⬜ Validation errors (missing files, duplicates)
- ⬜ Plotting with TSV samples
- ⬜ Session save/restore

---

## Files Created/Modified

### New Files (Phase 1 & 2)
- `src/services/tsv-parser.ts` - Parser implementation (153 lines)
- `src/services/tsv-path-resolver.ts` - Path resolver (172 lines)
- `src/services/tsv-validator.ts` - Validator (144 lines)
- `src/commands/tsv-commands.ts` - Import commands (367 lines)
- `src/services/__tests__/tsv-parser.test.ts` - Parser tests (344 lines)
- `docs/TSV_IMPORT_PROGRESS.md` - This file

### Modified Files
- `package.json` - Added `squiggy.importSamplesFromTSV` command
- `src/extension.ts` - Imported and registered TSV commands

**Total new code**: ~1180 lines (including tests and docs)

---

## Next Session Tasks

### Priority 1: Complete Phase 3 (Loading Integration)
**Estimated time**: 45-60 minutes

**Tasks**:
1. Implement actual kernel loading in `loadSamplesFromTSV()`
   - Call `service.loadSampleIntoRegistry()` for eager mode
   - Skip kernel load for lazy mode
   - Handle errors gracefully
2. Add lazy loading trigger in plot commands
   - Create `ensureSamplesLoaded()` helper
   - Check `sample.isLoaded` before plotting
   - Load on-demand if needed
3. Test with real data
   - Create test TSV files (5 samples, 20 samples)
   - Verify eager loading works
   - Verify lazy loading triggers on plot

### Priority 2: Add Tests
**Estimated time**: 30-45 minutes

**Tasks**:
1. Path resolver tests (`tsv-path-resolver.test.ts`)
   - Test absolute paths
   - Test relative paths
   - Test auto strategy
   - Test file existence checks
2. Validator tests (`tsv-validator.test.ts`)
   - Test validation success/failure
   - Test sample name conflicts
   - Test missing file handling
3. Integration test
   - Test full import workflow
   - Mock file system and kernel API

### Priority 3: UI Integration
**Estimated time**: 60-90 minutes

**Tasks**:
1. Samples panel grouping
2. Session persistence
3. Read Explorer integration

---

## Design Decisions

### Loading Strategy
- **≤5 samples**: Always eager (fast, small overhead)
- **6-10 samples**: Eager (default threshold)
- **11-19 samples**: Lazy (avoid kernel overload)
- **≥20 samples**: Always lazy

**Rationale**: Eager loading provides better UX for small batches (immediate availability), while lazy loading scales to large datasets without overwhelming the kernel.

### Path Resolution
- **Auto strategy by default**: Try multiple approaches, use first success
- **Priority**: Absolute → TSV-relative → workspace-relative
- **Error reporting**: Show which strategy succeeded for transparency

### Validation Strictness
- **POD5 missing**: Block import (required file)
- **BAM/FASTA missing**: Warn but allow (optional files)
- **Duplicate names**: Block import (uniqueness required)
- **Sample name conflicts**: Block import (avoid overwriting existing samples)

### State Management
- **TSV metadata**: Track import source (`sourceType: 'tsv'`)
- **Batch grouping**: All samples from same TSV share `tsvGroup` ID
- **Lazy load flag**: `isLoaded: boolean` tracks kernel state
- **Unified state**: Sync with existing `LoadedItem` system for cross-panel integration

---

## Known Limitations

1. **Single POD5 per sample**: Currently supports one POD5 file per sample. Future: Support comma-separated lists for technical replicates.
2. **No TSV editing UI**: Import only. Future: Allow editing sample associations before loading.
3. **No export to TSV**: Cannot export current samples to TSV format. Future: Reverse operation.
4. **No POD5/BAM overlap validation**: Deferred to load time (expensive operation). Future: Optional quick validation.
5. **Kernel loading stubbed**: Actual kernel integration pending (Priority 1 for next session).

---

## References

- Original design doc: `docs/TSV_IMPORT_FUTURE_DESIGN.md`
- Existing sample loading: `src/services/file-loading-service.ts` (method `loadSampleIntoRegistry()`)
- Existing file commands: `src/commands/file-commands.ts` (see `loadSamplesFromDropped()` for similar pattern)
- Extension state: `src/state/extension-state.ts` (interface `SampleInfo`)
