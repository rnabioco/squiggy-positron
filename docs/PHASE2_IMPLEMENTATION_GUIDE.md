# Phase 2 Implementation Guide - Concrete Refactoring Steps

**Status**: Tasks 2.1 and 2.2 Complete
**Completed**:
- ✅ 2.1: Created FileLoadingService with all load methods
- ✅ 2.2: Added unified state infrastructure to ExtensionState

**Next Steps**: Tasks 2.3-2.7 (This document)

---

## Task 2.3: Integrate File Panel with FileLoadingService & Unified State

### Overview
Refactor `openPOD5File()`, `openBAMFile()`, and `openFASTAFile()` commands to use FileLoadingService and add unified state support.

### Current Implementation Pattern (file-commands.ts:391-687)

Each file-opening function currently:
1. Calls `ensureSquiggyAvailable(state)`
2. Loads file via `state.squiggyAPI.loadXXX(filePath)` or subprocess backend
3. Extracts metadata with `fs.stat(filePath)`
4. Updates `state.currentPod5File`, etc.
5. Updates UI: `filePanelProvider.setPOD5()`, `plotOptionsProvider.updatePod5Status()`, etc.

### Refactoring Steps

#### Step 1: Update openPOD5File()

**Before**:
```typescript
async function openPOD5File(filePath: string, state: ExtensionState): Promise<void> {
    // ensureSquiggyAvailable check...
    // Positron or subprocess loading...
    const result = await state.squiggyAPI.loadPOD5(filePath);
    const stats = await fs.stat(filePath);
    state.currentPod5File = filePath;
    state.filePanelProvider?.setPOD5({...});
    state.plotOptionsProvider?.updatePod5Status(true);
}
```

**After**:
```typescript
async function openPOD5File(filePath: string, state: ExtensionState): Promise<void> {
    const squiggyAvailable = await ensureSquiggyAvailable(state);
    if (!squiggyAvailable) { /* error */ }

    await safeExecuteWithProgress(
        async () => {
            const service = new FileLoadingService(state);
            const result = await service.loadFile(filePath, 'pod5');

            if (!result.success) {
                throw new Error(result.error);
            }

            // Create LoadedItem
            const item: LoadedItem = {
                id: `pod5:${filePath}`,
                type: 'pod5',
                pod5Path: filePath,
                readCount: (result as POD5LoadResult).readCount,
                fileSize: result.fileSize,
                fileSizeFormatted: result.fileSizeFormatted,
                hasAlignments: false,
                hasReference: false,
                hasMods: false,
                hasEvents: false,
            };

            // Add to unified state (triggers events)
            state.addLoadedItem(item);

            // Maintain legacy state for backward compat
            state.currentPod5File = filePath;

            // Get reads and update UI
            const readIds = await state.squiggyAPI!.getReadIds(0, 1000);
            state.readsViewPane?.setReads(readIds);
            state.plotOptionsProvider?.updatePod5Status(true);
        },
        ErrorContext.POD5_LOAD,
        'Opening POD5 file...'
    );
}
```

#### Step 2: Update openBAMFile()

Similar pattern to openPOD5File:
- Use `FileLoadingService.loadFile(filePath, 'bam')`
- Extract metadata from result
- Create LoadedItem with BAM metadata (numReferences, hasModifications, hasEventAlignment)
- Call `state.addLoadedItem(item)`
- Update legacy state: `state.currentBamFile = filePath`
- Update UI: modifications panel, plot options, references

#### Step 3: Update openFASTAFile()

- Use `FileLoadingService.loadFile(filePath, 'fasta')`
- Create minimal LoadedItem (fewer properties for FASTA)
- Call `state.addLoadedItem(item)`
- Update legacy state: `state.currentFastaFile = filePath`

### File Panel Subscription

Update `FilePanelProvider` constructor to subscribe to `onLoadedItemsChanged`:

```typescript
constructor(extensionUri: vscode.Uri, state?: ExtensionState) {
    // ... existing code ...

    if (state) {
        state.onLoadedItemsChanged((items) => {
            // Convert LoadedItem[] to FileItem[] for display
            const fileItems = items.map(item => ({
                path: item.pod5Path,
                filename: path.basename(item.pod5Path),
                type: (item.type === 'pod5' ? 'POD5' : 'SAMPLE') as 'POD5' | 'BAM' | 'FASTA',
                size: item.fileSize,
                sizeFormatted: item.fileSizeFormatted,
                numReads: item.readCount,
                hasMods: item.hasMods,
                hasEvents: item.hasEvents,
            }));

            this._files = fileItems;
            this.updateView();
        });
    }
}
```

### Verification Checklist
- [ ] `openPOD5File` uses FileLoadingService and adds to unified state
- [ ] `openBAMFile` uses FileLoadingService and adds to unified state
- [ ] `openFASTAFile` uses FileLoadingService and adds to unified state
- [ ] File Panel subscribes to `onLoadedItemsChanged` and updates view
- [ ] Legacy state vars (`currentPod5File`, etc.) still updated for backward compat
- [ ] All UI updates still work (reads view, plot options, modifications panel)
- [ ] Test: Load POD5 → File Panel shows file → Load BAM → File Panel updates

---

## Task 2.4: Integrate Samples Panel with FileLoadingService & Unified State

### Overview
Refactor sample loading in `loadSampleForComparison()` and `loadSamplesFromDropped()` to use FileLoadingService and unified state.

### Current Implementation Pattern

`loadSampleForComparison()` (lines 718-815) and similar functions:
1. Prompt for sample name, POD5 path, BAM path, FASTA path
2. Call `state.squiggyAPI.loadSample(name, pod5Path, bamPath, fastaPath)`
3. Create SampleInfo and call `state.addSample(sampleInfo)` (legacy)
4. Call `samplesProvider?.refresh()` to update UI

### Refactoring Steps

#### Step 1: Create Helper Method in FileLoadingService

The `loadSample()` method already exists (private in current design), make it public or create a wrapper:

```typescript
// In FileLoadingService
async loadSample(
    pod5Path: string,
    bamPath?: string,
    fastaPath?: string
): Promise<{pod5Result, bamResult?, fastaResult?}> {
    // Already implemented - just use it
}
```

#### Step 2: Update loadSampleForComparison()

**Before**:
```typescript
const result = await state.squiggyAPI.loadSample(sampleName, pod5Path, bamPath, fastaPath);
state.addSample({
    name: sampleName,
    pod5Path,
    bamPath,
    fastaPath,
    readCount: result.numReads,
    hasBam: !!bamPath,
    hasFasta: !!fastaPath,
});
samplesProvider?.refresh();
```

**After**:
```typescript
const service = new FileLoadingService(state);
const result = await service.loadSample(pod5Path, bamPath, fastaPath);

if (!result.pod5Result.success) {
    throw new Error(result.pod5Result.error);
}

// Create LoadedItem for sample
const item: LoadedItem = {
    id: `sample:${sampleName}`,
    type: 'sample',
    sampleName,
    pod5Path,
    bamPath,
    fastaPath,
    readCount: result.pod5Result.readCount,
    fileSize: result.pod5Result.fileSize,
    fileSizeFormatted: result.pod5Result.fileSizeFormatted,
    hasAlignments: result.bamResult?.success ?? false,
    hasReference: result.fastaResult?.success ?? false,
    hasMods: (result.bamResult as BAMLoadResult)?.hasModifications ?? false,
    hasEvents: (result.bamResult as BAMLoadResult)?.hasEventAlignment ?? false,
};

// Add to unified state (triggers onLoadedItemsChanged)
state.addLoadedItem(item);

// Maintain legacy state for backward compat
state.addSample({
    name: sampleName,
    pod5Path,
    bamPath,
    fastaPath,
    readCount: result.pod5Result.readCount,
    hasBam: !!bamPath,
    hasFasta: !!fastaPath,
});

// Samples Panel subscribes to onLoadedItemsChanged, so no manual refresh needed
// But for now, still call refresh() for safety during transition
samplesProvider?.refresh();
```

#### Step 3: Update loadSamplesFromDropped()

Apply same pattern as loadSampleForComparison() for each sample in the queue.

### Samples Panel Subscription

Update `SamplesPanelProvider` constructor:

```typescript
constructor(extensionUri: vscode.Uri, state: ExtensionState) {
    // ... existing code ...

    // Subscribe to loaded items changes
    state.onLoadedItemsChanged((items) => {
        // Filter for samples only
        const sampleItems = items
            .filter(item => item.type === 'sample')
            .map(item => ({
                name: item.sampleName!,
                pod5Path: item.pod5Path,
                bamPath: item.bamPath,
                fastaPath: item.fastaPath,
                readCount: item.readCount,
                hasBam: item.hasAlignments,
                hasFasta: item.hasReference,
            }));

        this._samples = sampleItems;
        this.updateView();
    });

    // Subscribe to comparison selection changes
    state.onComparisonChanged((ids) => {
        // Filter for samples
        this._selectedSamples = new Set(
            ids.filter(id => id.startsWith('sample:')).map(id => id.substring(7))
        );
        this.updateView();
    });
}
```

### Verification Checklist
- [ ] `loadSampleForComparison()` uses FileLoadingService
- [ ] Sample loading creates LoadedItem with type: 'sample'
- [ ] `state.addLoadedItem()` called (triggers onLoadedItemsChanged)
- [ ] Samples Panel subscribes to onLoadedItemsChanged
- [ ] Samples Panel subscribes to onComparisonChanged
- [ ] `loadSamplesFromDropped()` uses FileLoadingService
- [ ] `loadTestMultiReadDataset()` updated to use unified state
- [ ] Test: Load sample → Samples Panel shows it → Load second sample → Both visible

---

## Task 2.5: Cross-Panel Synchronization Testing

### Test 1: File Panel → Samples Panel Awareness
1. Open POD5 via File Explorer command
2. Verify File Panel displays file
3. Verify Samples Panel could theoretically see the file (check unified state)
4. *Future*: Samples Panel UI changes when File Panel loads items

### Test 2: Samples Panel → File Panel Awareness
1. Load sample via Samples Panel
2. Verify Samples Panel displays sample
3. Verify File Panel could theoretically see it (check unified state)
4. *Future*: File Panel UI changes when Samples Panel loads items

### Test 3: Event Firing
1. Subscribe to onLoadedItemsChanged in test
2. Load a file/sample
3. Verify event fires with correct LoadedItem
4. Verify all panels receive event

### Test 4: Session Save/Restore
1. Load POD5 and sample
2. Save session
3. Restore session
4. Verify both items appear in unified state
5. Verify File Panel and Samples Panel both updated

---

## Task 2.6: Session Management Update

### Update toSessionState()

Change line 375-392 to use unified state:

```typescript
toSessionState(): SessionState {
    const samples: { [sampleName: string]: SampleSessionState } = {};

    // Iterate through unified state instead of legacy _loadedSamples
    for (const item of this.getLoadedItems()) {
        if (item.type === 'sample') {
            samples[item.sampleName!] = {
                pod5Paths: [item.pod5Path],
                bamPath: item.bamPath,
                fastaPath: item.fastaPath,
            };
        } else if (item.type === 'pod5') {
            // Single-file mode
            samples['Default'] = {
                pod5Paths: [item.pod5Path],
                bamPath: item.bamPath,
                fastaPath: item.fastaPath,
            };
        }
    }

    // ... rest of method unchanged ...
}
```

### Update fromSessionState()

The deserialization should populate unified state (in addition to legacy state):

```typescript
// After loading each sample, also add to unified state:
const item: LoadedItem = {
    id: `sample:${sampleName}`,
    type: 'sample',
    sampleName,
    pod5Path: resolvedPod5Paths[0],
    bamPath: resolvedBamPath,
    fastaPath: resolvedFastaPath,
    readCount: pod5Result.numReads,
    fileSize: pod5Stats.size,
    fileSizeFormatted: formatFileSize(pod5Stats.size),
    hasAlignments: !!resolvedBamPath,
    hasReference: !!resolvedFastaPath,
    hasMods: bamResult.hasModifications,
    hasEvents: bamResult.hasEventAlignment,
};
this.addLoadedItem(item);
```

---

## Task 2.7: Documentation & Cleanup

### Update CLAUDE.md

Add new section documenting unified state pattern:

```markdown
## Unified Extension State (Issue #92)

The extension maintains a single unified registry of loaded items (files/samples)
in ExtensionState to ensure File Panel and Samples Panel stay synchronized.

### LoadedItem Structure

```

### Code Comments

Add JSDoc to FileLoadingService:
- Explain deduplication benefits
- Link to Phase 1 design doc
- Show before/after examples

### Remove or Deprecate Legacy Methods

Mark as `@deprecated` but keep for now:
- ExtensionState.setPOD5() on FilePanelProvider
- ExtensionState.setBAM() on FilePanelProvider

### Commit Message

```
feat: implement unified extension state (#92)

- Create FileLoadingService to eliminate file loading duplication
- Add LoadedItem interface for unified item registry
- Add event emitters to ExtensionState for cross-panel sync
- Integrate File Panel with unified state
- Integrate Samples Panel with unified state
- Update session management for unified state

Benefits:
- File Panel and Samples Panel now stay synchronized
- 40%+ reduction in duplicated file-loading logic
- Single source of truth for loaded files/samples
- Event-driven architecture for loosely coupled panels
```

---

## Implementation Order

1. **2.3 (File Panel)**: Safer refactor, fewer dependencies
2. **2.4 (Samples Panel)**: Follows same pattern as File Panel
3. **2.5 (Testing)**: Verify synchronization works
4. **2.6 (Sessions)**: Update for unified state
5. **2.7 (Docs)**: Final cleanup

---

## Rollback Plan

If issues arise:
1. Both legacy state vars and unified state are maintained
2. Can disable event listeners if causing problems
3. Tests catch regressions immediately
4. Can revert to previous commit if needed

---

**Next Step**: Implement Task 2.3 using this guide
