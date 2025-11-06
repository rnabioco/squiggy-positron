# Investigation Report: Sample Sync Issue with Python Backend

**Status**: ROOT CAUSE IDENTIFIED ✅
**Date**: Phase 3 Testing
**Issue**: Multi-sample comparison fails with "Sample 'xxx' not found" error

---

## Executive Summary

The multi-sample comparison feature fails because the UI file loading system uses the **single-file global session** while the comparison function expects samples to be registered in a **multi-sample session registry**. These are two different Python APIs that are incompatible.

---

## Root Cause Analysis

### Architecture Overview

The squiggy Python package has **two different file loading systems**:

#### System A: Global Session (Single File)
```python
# Used by current UI file loading
import squiggy
squiggy.load_pod5('file1.pod5')      # Loads into _squiggy_session.reader
squiggy.load_bam('file1.bam')        # Loads into _squiggy_session.bam_path

# Result: Can only have ONE POD5 and ONE BAM loaded at a time
# State: _squiggy_session.reader, _squiggy_session.bam_path, _squiggy_session.read_ids
```

#### System B: Multi-Sample Registry (Multiple Files)
```python
# Used by comparison plotting
import squiggy
squiggy.load_sample('sample_1', 'ala_subset.pod5', 'ala_subset.bam')
squiggy.load_sample('sample_2', 'arg_subset.pod5', 'arg_subset.bam')

# Result: Can have MULTIPLE named samples loaded simultaneously
# State: _squiggy_session.samples = {
#   'sample_1': Sample(name='sample_1', pod5_reader, bam_path, ...),
#   'sample_2': Sample(name='sample_2', pod5_reader, bam_path, ...),
# }
```

### The Mismatch

When user clicks "Load Sample Data" to load samples for comparison:

**What Happens (Current):**
1. User selects: `ala_subset.pod5`, `arg_subset.pod5`, `asn_subset.pod5`
2. FileLoadingService calls:
   - `squiggyAPI.loadPOD5('ala_subset.pod5')`  → Updates global `_squiggy_session.reader`
   - `squiggyAPI.loadPOD5('arg_subset.pod5')`  → REPLACES `_squiggy_session.reader`
   - `squiggyAPI.loadPOD5('asn_subset.pod5')`  → REPLACES `_squiggy_session.reader` again
3. User clicks "Start Comparison" with these 3 sample names
4. Python calls `_squiggy_session.get_sample('ala_subset')` → NOT FOUND
   - Reason: Only the last file is in the global session

**What Should Happen:**
1. User selects: `ala_subset.pod5`, `arg_subset.pod5`, `asn_subset.pod5`
2. FileLoadingService should call:
   - `squiggyAPI.loadSample('ala_subset', 'ala_subset.pod5', 'ala_subset.bam')`
   - `squiggyAPI.loadSample('arg_subset', 'arg_subset.pod5', 'arg_subset.bam')`
   - `squiggyAPI.loadSample('asn_subset', 'asn_subset.pod5', 'asn_subset.bam')`
3. Python stores all three in `_squiggy_session.samples` dict
4. User clicks "Start Comparison" with these 3 sample names
5. Python calls `_squiggy_session.get_sample('ala_subset')` → FOUND ✅

---

## Code Locations

### System A: Global Session (Current)
**TypeScript:**
- `src/backend/squiggy-runtime-api.ts:52-68` - `loadPOD5()` method
- `src/backend/squiggy-runtime-api.ts:91-125` - `loadBAM()` method

**Python:**
- `squiggy/io.py:1044+` - `load_pod5()` function
- `squiggy/io.py:1098+` - `load_bam()` function
- `squiggy/io.py:250-450` - SquiggySession class (global `_squiggy_session`)

### System B: Multi-Sample Registry (Missing from TS API)
**Python:**
- `squiggy/io.py:453-564` - `SquiggySession.load_sample()` method
- `squiggy/io.py:1218-1243` - `load_sample()` function
- `squiggy/io.py:1246-1260` - `get_sample()` function
- `squiggy/io.py:1263-1275` - `list_samples()` function

**TypeScript:**
- **MISSING**: No TypeScript wrapper for `squiggy.load_sample()`!

---

## Why "Load Test Data" Works

The "Load Test Data" command (`squiggy.loadTestData`) in file-commands.ts:117:
1. Loads single POD5 and BAM sequentially
2. Never attempts multi-sample comparison
3. Only tests single-file workflows
4. Therefore, it never exposes this architectural gap

---

## Solution Options

### Option 1: Add Missing TypeScript API Method (RECOMMENDED)
**Effort**: Low (30 minutes)
**Risk**: Low

**Steps**:
1. Add `loadSample(name: string, pod5Path: string, bamPath?: string, fastaPath?: string)` method to SquiggyRuntimeAPI
2. Update FileLoadingService to use this method when loading samples for comparison
3. Store loaded samples with their names so comparison can find them

**Implementation**:
```typescript
// Add to src/backend/squiggy-runtime-api.ts
async loadSample(
    name: string,
    pod5Path: string,
    bamPath?: string,
    fastaPath?: string
): Promise<SampleLoadResult> {
    const escapedName = name.replace(/'/g, "\\'");
    const escapedPod5 = pod5Path.replace(/'/g, "\\'");
    const escapedBam = bamPath?.replace(/'/g, "\\'");
    const escapedFasta = fastaPath?.replace(/'/g, "\\'");

    const bamArg = escapedBam ? `, '${escapedBam}'` : '';
    const fastaArg = escapedFasta ? `, '${escapedFasta}'` : '';

    await this._client.executeSilent(`
import squiggy
from squiggy.io import _squiggy_session
squiggy.load_sample('${escapedName}', '${escapedPod5}'${bamArg}${fastaArg})
`);

    // Return metadata about loaded sample
    // ... extract read counts, BAM info, etc.
}
```

### Option 2: Merge Both Systems in Python (NOT RECOMMENDED)
**Effort**: High (complex refactoring)
**Risk**: High

Would require changing Python squiggy API to make the two systems compatible. Not worth it.

---

## Immediate Workarounds

Users can work around this by:
1. Using "Load Test Data" command (works for test data only)
2. Reloading all samples at once in a single operation
3. Using Python console directly: `squiggy.load_sample('name1', 'file1.pod5')`

---

## Impact Assessment

**Affected Features**:
- ✅ Single-file workflows (plot_read, plot_aggregate, etc.)
- ✅ File loading and display
- ✅ Sample Manager UI
- ❌ Multi-sample comparison plotting
- ❌ Start Comparison button

**Affected Users**:
- Anyone trying to use "Start Comparison" with files loaded via UI
- Does NOT affect "Load Test Data" workflow

---

## Recommended Fix

Implement Option 1 above (add `loadSample()` method to SquiggyRuntimeAPI):

1. **Phase 3.5**: Add TypeScript API method
2. **Phase 3.5**: Update FileLoadingService to use new method
3. **Phase 3.5**: Test multi-sample comparison
4. **Total Time**: ~1 hour

This is a minimal, low-risk fix that bridges the architectural gap between UI file loading and Python sample registry.

---

## Files to Modify

1. `src/backend/squiggy-runtime-api.ts` - Add `loadSample()` method
2. `src/services/file-loading-service.ts` - Add conditional logic to use `loadSample()` for multi-file scenarios
3. `src/commands/file-commands.ts` - Update `loadSamplesFromDropped()` to track sample names properly

---

## Testing Plan

After implementing the fix:

1. Load 3 POD5 files with matching BAM files via "Load Sample Data"
2. Go to Sample Manager and select all 3 samples
3. Click "Start Comparison"
4. Verify: Plot generates successfully (no "sample not found" error)
5. Verify: All 3 samples show in the overlay visualization with proper colors

---

## Related Issues

- Issue #79: TSV import feature (depends on this being fixed first)
- May need to update error messages further after fix

