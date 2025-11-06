# Known Issues - Phase 3 Refactoring

## Issues Found During Testing

### 1. ⚠️ CRITICAL: Sample Sync with Python Backend During Comparisons
**Status**: Identified during testing, needs investigation

**Description**: When samples are loaded through the UI ("Load Sample Data" button in File Explorer), they appear correctly in the Sample Manager UI. However, when attempting to run "Start Comparison", the Python backend reports `ValueError: Sample 'xxx' not found`.

**Symptoms**:
- Samples load successfully and display in Sample Manager
- User selects samples and clicks "Start Comparison"
- Error: "Sample 'sample_name' not found in Python session"
- Workaround: Re-load the same samples together, or use "Load Test Data"

**Root Cause**: Likely architectural issue with how samples loaded via UI file picker are registered in the Python squiggy session. Samples are properly added to:
- Unified state (ExtensionState)
- Sample Manager UI display
- Legacy state (ExtensionState.addSample)

But may not be properly synchronized with Python's `squiggy.io._squiggy_session` for comparison operations.

**Workaround**:
1. Use "Load Test Data" command which properly registers samples
2. Re-load samples together instead of one at a time

**Next Steps**:
- Investigate FileLoadingService and how samples are loaded into Python session
- Check squiggy.io._squiggy_session registration during loadSample operations
- Ensure Python backend receives proper sample metadata

---

## Fixed Issues (Phase 3)

### ✅ FASTA Not Showing After "Set FASTA for All"
**Fixed in**: commit c9079ac

When user clicked "Set FASTA for All" button to assign a session-level FASTA file, the expanded sample cards still showed "FASTA (not set)" instead of displaying the session FASTA.

**Root Cause**: FASTA display logic only checked `sample.fastaPath`, not `state.sessionFastaPath`.

**Solution**: Updated FASTA display to show either per-sample FASTA or session FASTA, labeled as "(session)" when appropriate.

---

### ✅ Pane Title Name
**Fixed in**: commit c9079ac

Renamed "Sample Comparison Manager" to "Sample Manager" for clarity.

---

### ✅ Unclear Error for BAM-Only Loading
**Fixed in**: commit 61eb4e3, improved in commit c9079ac

User would see generic error "No POD5 selected" when trying to load only BAM files. Updated to provide clearer message explaining that POD5 is required.

---

### ✅ Sample Name Prompts During Loading
**Fixed in**: commit 61eb4e3

Removed modal dialogs that appeared during file loading workflow. Sample names are now auto-generated from filenames and can be edited in Sample Manager.

---

## Testing Notes

All changes tested with:
- Single POD5 file loading ✅
- Multiple POD5 files loading ✅
- POD5 + BAM together ✅
- "Load Test Data" command ✅
- Sample Manager UI expand/collapse ✅
- "Set FASTA for All" button ✅
- Checkbox selection ✅
- Sample name editing (double-click) ✅
- Color picker ✅
- Empty state message ✅

**Issue reproduced**: Multi-sample comparison plot generation
- Load 3 POD5 files individually
- Set FASTA for all
- Select samples and click "Start Comparison"
- Error: "Sample not found in Python session"

---

## Architecture Notes

### Current Flow for UI-Loaded Samples
1. User clicks "Load Sample Data" in File Explorer
2. File picker opens, user selects POD5/BAM files
3. `loadSamplesFromFilePicker()` categorizes files by extension
4. `loadSamplesFromFilePicker()` auto-matches BAM to POD5 by filename stem
5. `loadSamplesFromDropped()` is called for each sample
6. Each sample:
   - Gets LoadedItem created with `type: 'sample'`
   - Gets added to unified state via `state.addLoadedItem()`
   - Gets added to legacy state via `state.addSample()`
   - Gets loaded to Python via `FileLoadingService.loadSample()`

### Current Flow for Test Data
1. User runs "Load Test Data" command
2. Test files are loaded via `FileLoadingService` and converted to LoadedItem
3. Same unified + legacy state updates happen
4. Additional Python session setup (not clear if different)

### Potential Issue
The Python `squiggy.plot_signal_overlay_comparison()` function expects samples to be in the registry. May need to:
- Check how test data registration differs from UI file loading
- Ensure proper Python squiggy session state updates during FileLoadingService operations
- Verify sample metadata is preserved through the entire pipeline

