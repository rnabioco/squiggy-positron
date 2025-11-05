# Phase 3 - UI Consolidation & Workflow Refinement

**Status: PLANNED**
**Depends On:** Phase 2 (Unified State Foundation) ✅ COMPLETE
**Estimated Effort:** 8-12 hours
**Priority:** High - Completes the unified workflow vision

## Overview

Phase 3 consolidates the loading and management workflows by making **File Explorer** the single interface for *loading* data, while **Sample Comparison Manager** becomes purely a *management and configuration* interface.

This eliminates UI duplication and creates a cleaner mental model:
- **Loading** → File Explorer
- **Managing** → Sample Comparison Manager
- **Comparison** → Sample Comparison Manager

## Current State (End of Phase 2)

### File Explorer ("Squiggy File Explorer")
- Shows loaded POD5 files individually
- No sample bundle view
- No sample loading UI

### Sample Comparison Manager ("Sample Comparison Manager")
- Shows loaded samples with full paths
- Has "+ Load More Samples" button (duplicates loading UI)
- Has unload buttons
- Shows BAM/FASTA associations (read-only)
- Has checkboxes for comparison selection

## Desired State (End of Phase 3)

### File Explorer ("Squiggy File Explorer")
**Shows both:**
1. **Individual POD5 files** (loaded via command)
   - Filename, type, size, read count

2. **Sample bundles** (loaded via "+ Sample" button)
   - Sample name
   - POD5 path
   - BAM path (if associated)
   - FASTA path (if associated)
   - Visual distinction from raw POD5s

**Single loading interface:**
- "+ Sample" button (moved from Sample Comparison Manager)
- Open dialog to select POD5, optionally BAM and FASTA
- Auto-match BAM if naming follows conventions
- Allow custom sample name

### Sample Comparison Manager
**Becomes pure management interface:**
- Sample name (editable)
- BAM/FASTA association viewer
- Checkboxes for comparison selection
- Color assignment (NEW)
- Unload button
- **NO loading UI** - that's in File Explorer now

---

## Phase 3 Tasks

### Task 3.1: Display Sample Bundles in File Explorer

**Files to Modify:**
- `src/views/squiggy-file-panel.ts`
- `src/types/messages.ts` (if needed for new message types)

**Changes:**
1. Update `_files` array to include both:
   - Raw POD5 files from `_loadedItems` where `type === 'pod5'`
   - Sample bundles from `_loadedItems` where `type === 'sample'`

2. Create new `FileItem` variant for samples showing:
   ```typescript
   {
     name: "sample_name",
     type: "SAMPLE",  // vs "POD5"
     pod5Path: "...",
     bamPath: "...",
     fastaPath: "...",
     hasBam: boolean,
     hasFasta: boolean
   }
   ```

3. Update webview to display samples differently:
   - Maybe a collapsible section "Samples (N)"
   - Show sample name prominently
   - Show BAM/FASTA indicators

**Testing:**
- Load sample from Samples Panel
- Verify it appears in File Explorer as a "Sample" type
- Verify it distinguishes from raw POD5s

---

### Task 3.2: Move Sample Loading to File Explorer

**Files to Modify:**
- `src/views/squiggy-file-panel.ts` (webview side)
- `src/views/components/squiggy-sample-loader.tsx` (NEW - if React needed)
- `src/commands/file-commands.ts` (backend)

**Changes:**
1. Add "+ Sample" button to File Explorer UI
2. When clicked, open sample loading dialog:
   - Select POD5 file
   - Optionally select BAM file (with auto-match by name)
   - Optionally select FASTA file
   - Enter sample name (with suggestions from file name)
   - Confirm loading

3. Button handler calls new command:
   - `squiggy.loadSampleFromUI` or similar
   - Same logic as current Sample Comparison Manager's "Load More Samples"
   - Uses FileLoadingService (from Phase 2)
   - Adds to unified state

**Code to Reuse:**
- `SamplesPanelProvider.confirmSampleNames()` - dialog logic
- `SamplesPanelProvider.matchFilePairs()` - file matching
- `FileLoadingService.loadSample()` - actual loading

**Testing:**
- Click "+ Sample" in File Explorer
- Load a sample with BAM and FASTA
- Verify sample appears in both File Explorer and Sample Comparison Manager
- Verify cross-panel sync works

---

### Task 3.3: Remove Duplicate Loading UI from Sample Comparison Manager

**Files to Modify:**
- `src/views/squiggy-samples-panel.ts`
- `src/views/squiggy-samples-panel.html` or React components

**Changes:**
1. Remove "Load More Samples" button
2. Remove "Load Demo Session" button (optional - could keep for quick testing)
3. Remove sample loading dialog logic
4. Keep only:
   - Sample display (name, POD5, BAM, FASTA)
   - Checkboxes for comparison
   - Unload button

**Result:**
- Sample Comparison Manager focused solely on management
- Cleaner, less cluttered UI
- Single source of truth for loading (File Explorer)

**Testing:**
- Verify "+ Sample" button gone from Samples Panel
- Verify "+ Sample" button works in File Explorer
- Verify existing unload functionality still works

---

### Task 3.4: Add Sample Management Features

**Files to Modify:**
- `src/views/squiggy-samples-panel.ts`
- `src/state/extension-state.ts` (if adding color/name to LoadedItem)

**Enhancement 1: Editable Sample Names**
- Click on sample name → becomes editable
- Update state when name changes
- Persist across sessions

**Enhancement 2: Color Assignment**
- Color picker next to each sample
- Save color in unified state
- Use color in plots when rendering

**Enhancement 3: Visual BAM/FASTA Indicators**
- Show badges: "BAM" "FASTA" near sample name
- Clicking badge could show path
- Or expandable section showing all metadata

**Changes to LoadedItem Interface** (in Phase 2, extend now):
```typescript
interface LoadedItem {
  // ... existing fields ...
  displayName?: string;      // For renames
  displayColor?: string;     // For plot coloring (hex or CSS color)
}
```

**Testing:**
- Edit sample name in Comparison Manager
- Verify name updates in File Explorer too
- Assign color to sample
- Generate plot, verify uses assigned color

---

## Implementation Order

1. **Task 3.1** first (display samples in File Explorer)
   - Simpler, validates cross-panel updates still working

2. **Task 3.2** next (add loading button to File Explorer)
   - Most impactful for workflow
   - Depends on 3.1 working

3. **Task 3.3** then (remove from Samples Panel)
   - Clean up UI once loading moved
   - Lowest risk, mostly deletions

4. **Task 3.4** last (add management features)
   - Enhances Samples Panel for its new role
   - Can be partially deferred if needed

---

## Testing Strategy

### Unit Tests to Add
- File Explorer correctly displays sample bundles alongside POD5s
- Sample loading dialog validates inputs
- Sample name/color updates flow to unified state
- Samples Panel properly displays management controls

### Manual Testing
1. Load sample from File Explorer
   - Appears in File Explorer ✅
   - Appears in Samples Panel ✅
   - Cross-panel sync ✅

2. Edit sample name in Samples Panel
   - Name updates in File Explorer ✅
   - Persists in session ✅

3. Assign color to sample
   - Color displayed in Samples Panel ✅
   - Used when generating plots ✅

4. Session save/restore
   - All samples restored ✅
   - Names preserved ✅
   - Colors preserved ✅
   - Associations preserved ✅

---

## Risk Assessment

### Low Risk
- ✅ Phase 2 foundation solid
- ✅ Unified state handles updates
- ✅ Reusing existing logic (FileLoadingService, confirmSampleNames)

### Medium Risk
- ⚠️ UI changes in File Explorer (needs webview work)
- ⚠️ Sample bundle display (new UI pattern)

### Mitigation
- Keep "+ Sample" button simple initially
- Test thoroughly in Positron before considering complete
- Can always defer color assignment to follow-up

---

## Success Criteria

- ✅ File Explorer shows sample bundles alongside POD5 files
- ✅ "+ Sample" button in File Explorer loads samples
- ✅ Sample loading dialog works (select POD5, BAM, FASTA)
- ✅ "Load More Samples" removed from Sample Comparison Manager
- ✅ Sample name editing works in Comparison Manager
- ✅ Color assignment works (optional v1)
- ✅ Cross-panel sync maintained
- ✅ Session save/restore works
- ✅ All 109+ tests still passing

---

## Files Summary

### To Modify
- `src/views/squiggy-file-panel.ts` (main changes)
- `src/views/squiggy-samples-panel.ts` (removals + enhancements)
- `src/commands/file-commands.ts` (new commands if needed)
- `src/state/extension-state.ts` (extend LoadedItem interface)
- `src/types/messages.ts` (if webview message types change)

### To Create
- `src/views/components/squiggy-sample-loader.tsx` (optional - if React needed)
- `src/__tests__/phase3-ui-consolidation.test.ts` (new tests)

### To Update Docs
- `docs/UNIFIED_STATE_ARCHITECTURE.md` - reference File Explorer as loading point
- `docs/MANUAL_TESTING_GUIDE.md` - update test scenarios for new UI
- `CHANGELOG.md` - document Phase 3 changes

---

## Timeline

- **Design Review**: 1 hour
- **Task 3.1 Implementation**: 2-3 hours
- **Task 3.2 Implementation**: 3-4 hours
- **Task 3.3 Implementation**: 1-2 hours
- **Task 3.4 Implementation**: 2-3 hours (optional/extensible)
- **Testing & Bug Fixes**: 2-3 hours

**Total: 11-16 hours** (8-12 without 3.4 enhancements)

---

## Next Steps

1. Review this plan ✓
2. Confirm Phase 3 approach with team
3. Begin Task 3.1 implementation
4. Progressive testing after each task
5. Document changes as we go
