# Phase 1: Analysis & Design - Unified Extension State

**Issue**: #92 - Unified extension state: File Explorer and Sample Comparison Manager should coordinate while serving distinct roles

**Status**: Design Phase Complete (Ready for Phase 2 Implementation)

**Date**: November 4, 2025

---

## Executive Summary

The File Explorer and Sample Comparison Manager currently suffer from **two critical issues**:

### Issue 1: State Synchronization (Fragmentation)
- Loading files in File Explorer doesn't update Sample Comparison Manager
- Loading files in Sample Comparison Manager doesn't update File Explorer
- Changes to kernel state aren't reflected across UI panels
- Both panels maintain separate, uncoordinated state

### Issue 2: Functional Duplication
- File loading logic duplicated between File Panel and Samples Panel
- Metadata extraction (file size, read count) done in two places
- Similar error handling patterns in both panels
- When fixing bugs, changes must be made in multiple locations

This document presents the Phase 1 analysis and proposes a **comprehensive solution**:
1. **Unified state architecture** (Option A): Consolidates state management into `ExtensionState` with event emitters for cross-panel synchronization
2. **FileLoadingService**: Centralized file loading operations eliminating duplication between panels

Together, these address both synchronization and maintainability goals.

---

## Part 1: Current Architecture Analysis

### 1.1 File Explorer (Legacy Single-File Mode)

**Location**: `src/views/squiggy-file-panel.ts`, `src/views/components/squiggy-files-core.tsx`

**State Storage**:
```typescript
// FilePanelProvider (_files property)
_files: FileItem[] = [];

// Each FileItem
{
  path: string;
  filename: string;
  type: 'POD5' | 'BAM' | 'FASTA';
  size: number;
  sizeFormatted: string;
  numReads?: number;
  numRefs?: number;
  hasMods?: boolean;
  hasEvents?: boolean;
}

// ExtensionState (legacy props)
_currentPod5File?: string;
_currentBamFile?: string;
_currentFastaFile?: string;
```

**State Change Triggers**:
- `openPOD5File()` command → loads file → calls `filePanelProvider.setPOD5(fileInfo)`
- `openBAMFile()` command → loads file → calls `filePanelProvider.setBAM(fileInfo)`
- `openFASTAFile()` command → loads file → calls `filePanelProvider.setFASTA(fileInfo)`
- `clearAllFiles()` command → clears state → calls `clearPOD5()`, `clearBAM()`, `clearFASTA()`

**Data Flow**:
```
Extension Command (e.g., squiggy.openPOD5)
  ↓
file-commands.ts: openPOD5File(filePath, state)
  ↓
state.squiggyAPI.loadPOD5(filePath)  [Kernel execution]
  ↓
state.filePanelProvider.setPOD5({path, numReads, size})
  ↓
FilePanelProvider._files array updated
  ↓
updateView() → postMessage to React webview
  ↓
React component renders file list
```

**Current Limitations**:
- Single-file mode: only tracks one POD5, one BAM, one FASTA at a time
- No awareness of multi-sample context (Samples Panel)
- No event notifications to other panels
- State updates are **pull-based** (panels must call `refresh()` to sync)

---

### 1.2 Sample Comparison Manager (Multi-Sample Mode)

**Location**: `src/views/squiggy-samples-panel.ts`, `src/views/components/squiggy-samples-core.tsx`

**State Storage**:
```typescript
// SamplesPanelProvider
_samples: SampleItem[] = [];
_selectedSamples: Set<string> = new Set();
_pendingMaxReads: number | null = null;

// ExtensionState (multi-sample foundation)
_loadedSamples: Map<string, SampleInfo> = new Map();
_selectedSamplesForComparison: string[] = [];
_sessionFastaPath: string | null = null;

// SampleItem interface (from messages.ts)
{
  name: string;
  pod5Path: string;
  bamPath?: string;
  fastaPath?: string;
  readCount: number;
  hasBam: boolean;
  hasFasta: boolean;
}
```

**State Change Triggers**:
- File drag/drop → `handleFilesDropped(filePaths)`
- `loadSample` command → `squiggy.loadSamplesFromUI()`
- User selects sample → `selectSample(sampleName, selected)` message
- User clicks "Compare" → `startComparison(sampleNames)` message
- User unloads sample → `unloadSample(sampleName)` message
- `loadTestMultiReadDataset` command → loads demo samples

**Data Flow**:
```
User Action (drop files / select "Compare")
  ↓
React component sends message → samplesPanel.handleMessage()
  ↓
samplesPanel processes message
  ↓
If new samples: squiggy.loadSample(name, pod5Path, bamPath, fastaPath)
  ↓
state.squiggyAPI.loadSample() [Kernel execution]
  ↓
state.addSample(sampleInfo) [ExtensionState._loadedSamples]
  ↓
samplesProvider.refresh()
  ↓
samplesProvider.updateView() → postMessage to React
  ↓
React component renders sample list
  ↓
If comparison requested: samplesProvider.onDidRequestComparison event
  ↓
extension.ts listener → squiggy.plotSignalOverlayComparison command
```

**Current Limitations**:
- Multi-sample mode operates independently from File Explorer
- Selection state (`_selectedSamples`) separate from comparison selection (`_selectedSamplesForComparison`)
- No notification when File Panel loads files
- No knowledge of what File Panel is showing

---

### 1.3 ExtensionState: Bridging Layer (But Not Coordinating)

**Location**: `src/state/extension-state.ts`

**Current Role**:
```typescript
export class ExtensionState {
  // Backend references
  _positronClient?: PositronRuntimeClient;
  _squiggyAPI?: SquiggyRuntimeAPI;
  _pythonBackend?: PythonBackend | null;

  // UI panel references
  _readsViewPane?: ReadsViewPane;
  _plotOptionsProvider?: PlotOptionsViewProvider;
  _filePanelProvider?: FilePanelProvider;
  _modificationsProvider?: ModificationsPanelProvider;
  _samplesProvider?: SamplesPanelProvider;

  // LEGACY single-file state
  _currentPod5File?: string;
  _currentBamFile?: string;
  _currentFastaFile?: string;

  // MULTI-SAMPLE foundation (Phase 4)
  _loadedSamples: Map<string, SampleInfo> = new Map();
  _selectedSamplesForComparison: string[] = [];
  _sessionFastaPath: string | null = null;

  // Session management (NEW - from main)
  toSessionState(): SessionState { }
  fromSessionState(session: SessionState): Promise<void> { }
  loadDemoSession(context: ExtensionContext): Promise<void> { }

  // Multi-sample methods (NEW - from main)
  addSample(sample: SampleInfo): void { }
  removeSample(name: string): void { }
  addSampleToComparison(sampleName: string): void { }
  removeSampleFromComparison(sampleName: string): void { }
}
```

**Current Problem**:
- Has references to both File Panel and Samples Panel
- Holds both legacy single-file and multi-sample state
- **No event emitters** - changes don't notify panels
- **No coordination mechanism** - panels don't know about each other's state changes
- Session serialization depends on understanding what samples are "loaded" (ambiguous)

---

## Part 2: State Synchronization Analysis

### 2.1 Identified Synchronization Points

**Point 1: File Loading**
```
Scenario A: User loads POD5 via File Explorer
  [CURRENT] File Panel updates → ExtensionState.currentPod5File set
  [MISSING] Samples Panel doesn't know about this

Scenario B: User loads sample via Samples Panel
  [CURRENT] Samples Panel updates → ExtensionState._loadedSamples
  [MISSING] File Panel doesn't know about this
```

**Point 2: Selection State**
```
[ISSUE] File Panel has no selection state (legacy)
[ISSUE] Samples Panel has _selectedSamples (local)
[ISSUE] ExtensionState has _selectedSamplesForComparison (different concept)
[MISSING] No unified view of "what's selected for what"
```

**Point 3: File Clearing**
```
Scenario: User clears files in one panel
  [MISSING] Other panel doesn't update
  [MISSING] Session state may become inconsistent
```

**Point 4: Session Serialization**
```
toSessionState() method must determine:
  - Are we in single-file mode or multi-sample mode?
  - Which samples are "the current ones"?
  - What should be saved/restored?
[MISSING] This is ambiguous with current dual-state approach
```

### 2.2 Communication Gaps

| Aspect | File Panel | Samples Panel | ExtensionState | Gap |
|--------|-----------|--------------|-----------------|-----|
| Tracks files/samples | ✓ (_files) | ✓ (_samples) | ✓ (_loadedSamples) | **Three separate registries** |
| Notifies on change | ✗ | ✗ | ✗ | **No push notifications** |
| Aware of other panel | ✗ | ✗ | ✓ (has refs) | **One-way knowledge only** |
| Syncs selections | ✗ | ✓ (local) | Partial | **Multiple selection sources** |
| Can deserialize session | ✗ | ✗ | ✓ | **Only ExtensionState understands context** |

---

## Part 2.5: Functional Duplication Analysis

### Current Duplicated Operations

Both File Panel and Samples Panel perform similar operations with duplicated code:

| Operation | File Panel | Samples Panel | Duplication |
|-----------|-----------|---------------|-------------|
| **Load POD5** | `openPOD5File()` → `state.squiggyAPI.loadPOD5()` | `loadSample()` → `state.squiggyAPI.loadPOD5()` | ✗ Both load, different paths |
| **Load BAM** | `openBAMFile()` → `state.squiggyAPI.loadBAM()` | `loadSample()` → `state.squiggyAPI.loadBAM()` | ✗ Both load, different paths |
| **Load FASTA** | `openFASTAFile()` → `state.squiggyAPI.loadFASTA()` | Session-level only | ~ Partial |
| **Extract Metadata** | File size, read count in command | File size, read count in API | ✗ Two places doing same work |
| **Error Handling** | Try/catch + user message | Try/catch + user message | ✗ Similar patterns |
| **Panel Update** | `updateView()` → `postMessage()` | `updateView()` → `postMessage()` | ✗ Same pattern repeated |
| **Event Triggering** | Manual in commands | Manual in handlers | ✗ Could be centralized |

### Root Causes

1. **Separate Load Paths**:
   ```typescript
   // File Panel uses commands directly
   async function openPOD5File(filePath: string, state: ExtensionState) {
     const result = await state.squiggyAPI.loadPOD5(filePath);
     // Extract metadata manually
     const stats = await fs.stat(filePath);
     filePanelProvider.setPOD5({path, numReads: result.numReads, size: stats.size});
   }

   // Samples Panel uses API indirectly
   async loadSample(name, pod5Path, bamPath) {
     const result = await squiggyAPI.loadPOD5(pod5Path);
     // Extract metadata manually (again!)
     // ... build SampleInfo object
     addSample(sampleInfo);
   }
   ```

2. **No Shared Service Layer**:
   - File loading logic lives in commands (File Panel)
   - File loading logic lives in handlers (Samples Panel)
   - No centralized "LoadFile" service

3. **Duplicate Metadata Extraction**:
   - Both need: file size, read count, BAM info, etc.
   - Both use `fs.stat()`, `squiggyAPI.getVariable()`
   - Both format results differently

4. **Parallel Update Patterns**:
   - Both call `updateView()`
   - Both send messages to React
   - Could use shared base class or mixin

### Deduplication Strategy

Create a **FileLoadingService** to centralize operations:

```typescript
// NEW: src/services/file-loading-service.ts

export class FileLoadingService {
  constructor(private state: ExtensionState) {}

  /**
   * Load a single file and return normalized metadata
   * Shared entry point for both File Panel and Samples Panel
   */
  async loadFile(
    filePath: string,
    fileType: 'pod5' | 'bam' | 'fasta'
  ): Promise<FileLoadResult> {
    switch (fileType) {
      case 'pod5':
        return this.loadPOD5(filePath);
      case 'bam':
        return this.loadBAM(filePath);
      case 'fasta':
        return this.loadFASTA(filePath);
    }
  }

  /**
   * Load POD5 file with full metadata extraction
   * Returns normalized result usable by both panels
   */
  private async loadPOD5(filePath: string): Promise<POD5LoadResult> {
    try {
      const result = await this.state.squiggyAPI!.loadPOD5(filePath);
      const metadata = await this.extractFileMetadata(filePath, 'pod5');

      return {
        success: true,
        filePath,
        fileType: 'pod5',
        fileSize: metadata.fileSize,
        fileSizeFormatted: metadata.fileSizeFormatted,
        readCount: result.numReads,
        error: null,
      };
    } catch (error) {
      return {
        success: false,
        filePath,
        fileType: 'pod5',
        error: `Failed to load POD5: ${error}`,
        fileSize: 0,
        fileSizeFormatted: '0 B',
        readCount: 0,
      };
    }
  }

  /**
   * Shared metadata extraction (file size, read counts, etc.)
   */
  private async extractFileMetadata(
    filePath: string,
    type: 'pod5' | 'bam' | 'fasta'
  ): Promise<FileMetadata> {
    const fs = await import('fs/promises');
    const stats = await fs.stat(filePath);

    return {
      fileSize: stats.size,
      fileSizeFormatted: this.formatFileSize(stats.size),
      lastModified: stats.mtime,
      isReadable: (stats.mode & 0o400) !== 0,
    };
  }

  /**
   * Shared utility: format bytes to human-readable
   */
  private formatFileSize(bytes: number): string {
    const units = ['B', 'KB', 'MB', 'GB'];
    let size = bytes;
    let unitIdx = 0;
    while (size >= 1024 && unitIdx < units.length - 1) {
      size /= 1024;
      unitIdx++;
    }
    return `${size.toFixed(1)} ${units[unitIdx]}`;
  }
}

// Usage in File Panel:
async function openPOD5File(filePath: string, state: ExtensionState) {
  const service = new FileLoadingService(state);
  const result = await service.loadFile(filePath, 'pod5');

  if (result.success) {
    // Single unified update
    state.addLoadedItem({
      id: `pod5:${filePath}`,
      type: 'pod5',
      pod5Path: filePath,
      readCount: result.readCount,
      fileSize: result.fileSize,
      fileSizeFormatted: result.fileSizeFormatted,
      // ... other properties
    });
  } else {
    vscode.window.showErrorMessage(result.error!);
  }
}

// Usage in Samples Panel:
async function loadSample(name, pod5Path, bamPath, fastaPath) {
  const service = new FileLoadingService(state);

  // Load POD5
  const pod5Result = await service.loadFile(pod5Path, 'pod5');
  if (!pod5Result.success) throw new Error(pod5Result.error);

  // Load BAM if provided
  let bamResult = null;
  if (bamPath) {
    bamResult = await service.loadFile(bamPath, 'bam');
  }

  // Create unified item from results
  state.addLoadedItem({
    id: `sample:${name}`,
    type: 'sample',
    sampleName: name,
    pod5Path,
    bamPath,
    fastaPath,
    readCount: pod5Result.readCount,
    hasAlignments: bamResult?.success ?? false,
    fileSize: pod5Result.fileSize,
    fileSizeFormatted: pod5Result.fileSizeFormatted,
    // ... other properties
  });
}
```

### Benefits of FileLoadingService

| Aspect | Before | After |
|--------|--------|-------|
| **Code Duplication** | File/BAM/FASTA loading in 2 places | Single source of truth |
| **Metadata Extraction** | Duplicated in commands and API | One `extractFileMetadata()` method |
| **Error Handling** | Similar try/catch blocks × 2 | Centralized error handling |
| **Consistency** | File Panel and Samples have different result formats | Normalized `FileLoadResult` for both |
| **Testing** | Test file loading in both panels | Test `FileLoadingService` once |
| **Maintenance** | Change loading logic → update 2 places | Change loading logic → update 1 place |

---

## Part 3: Unified State Design (Option A)

### 3.1 Architecture Overview

```
BEFORE (Current - Fragmented):
┌──────────────────────────────────────────────────────────┐
│                  ExtensionState                          │
│  _currentPod5File     _loadedSamples                     │
│  _currentBamFile      _selectedSamplesForComparison     │
│  (legacy)             (multi-sample)                     │
└──────────────────────────────────────────────────────────┘
       ↓                    ↓
   [File Panel]         [Samples Panel]
   _files array         _samples array
   (local copy)         (local copy)
   ✗ Not sync'd         ✗ Not sync'd


AFTER (Unified - Coordinated):
┌──────────────────────────────────────────────────────────┐
│              ExtensionState (UNIFIED)                    │
│                                                          │
│  _loadedItems: Map<id, LoadedItem>     [UNIFIED]        │
│  _selectedForUI: Set<id>               [UNIFIED]        │
│  _selectedForComparison: Set<id>       [UNIFIED]        │
│                                                          │
│  Event Emitters:                                         │
│  - onItemsChanged(LoadedItem[])                          │
│  - onSelectionChanged(id[])                              │
│  - onComparisonChanged(id[])                            │
└──────────────────────────────────────────────────────────┘
       ↓                    ↓
   [File Panel]         [Samples Panel]
   subscribes to        subscribes to
   onItemsChanged       onItemsChanged
   refreshes when       refreshes when
   notified ✓ SYNC'd    notified ✓ SYNC'd
```

### 3.2 Unified State Structure

Add to `ExtensionState`:

```typescript
// ========== UNIFIED STATE (replaces fragmented approach) ==========

/**
 * Represents a loaded item (POD5, sample, etc.)
 * Unified registry replacing legacy _currentPod5File + multi-sample _loadedSamples
 */
export interface LoadedItem {
  id: string;                    // Unique identifier (e.g., "pod5:/path/to/file", "sample:sampleName")
  type: 'pod5' | 'sample';

  // Core file info
  pod5Path: string;              // POD5 file path
  bamPath?: string;              // Optional BAM alignment file
  fastaPath?: string;            // Optional reference sequence

  // Metadata
  readCount: number;             // Number of reads in POD5
  hasAlignments: boolean;        // Whether BAM is loaded
  hasReference: boolean;         // Whether FASTA is loaded
  hasMods: boolean;              // Whether BAM has modifications
  hasEvents: boolean;            // Whether BAM has event alignment

  // For samples
  sampleName?: string;           // Human-readable name (e.g., "Sample1", "Yeast tRNA")

  // File metadata
  fileSize: number;              // POD5 file size in bytes
  fileSizeFormatted: string;     // Human-readable size (e.g., "2.5 MB")
}

/**
 * Central unified state manager
 */
private _loadedItems: Map<string, LoadedItem> = new Map();          // All loaded files/samples
private _selectedItemIds: Set<string> = new Set();                  // Selected in UI
private _itemsForComparison: Set<string> = new Set();               // Selected for comparison mode

/**
 * Session-level files (apply to all items)
 */
private _sessionFastaPath: string | null = null;                    // Global FASTA (already exists)

/**
 * Event emitters for cross-panel synchronization
 */
private _onLoadedItemsChanged: vscode.EventEmitter<LoadedItem[]> = new vscode.EventEmitter();
private _onSelectionChanged: vscode.EventEmitter<string[]> = new vscode.EventEmitter();
private _onComparisonChanged: vscode.EventEmitter<string[]> = new vscode.EventEmitter();

// Public event accessors
get onLoadedItemsChanged(): vscode.Event<LoadedItem[]> {
  return this._onLoadedItemsChanged.event;
}

get onSelectionChanged(): vscode.Event<string[]> {
  return this._onSelectionChanged.event;
}

get onComparisonChanged(): vscode.Event<string[]> {
  return this._onComparisonChanged.event;
}
```

### 3.3 State Management Methods

Add to `ExtensionState`:

```typescript
// ========== UNIFIED ITEM MANAGEMENT ==========

/**
 * Add or update a loaded item
 */
addLoadedItem(item: LoadedItem): void {
  this._loadedItems.set(item.id, item);
  this._notifyLoadedItemsChanged();
}

/**
 * Remove a loaded item by ID
 */
removeLoadedItem(id: string): void {
  this._loadedItems.delete(id);
  // Also remove from selections
  this._selectedItemIds.delete(id);
  this._itemsForComparison.delete(id);
  this._notifyLoadedItemsChanged();
  this._notifySelectionChanged();
  this._notifyComparisonChanged();
}

/**
 * Get all loaded items
 */
getLoadedItems(): LoadedItem[] {
  return Array.from(this._loadedItems.values());
}

/**
 * Get a specific item by ID
 */
getLoadedItem(id: string): LoadedItem | undefined {
  return this._loadedItems.get(id);
}

// ========== SELECTION MANAGEMENT ==========

/**
 * Update selection in UI (e.g., checkbox clicked)
 */
setSelectedItems(ids: string[]): void {
  this._selectedItemIds = new Set(ids);
  this._notifySelectionChanged();
}

/**
 * Get currently selected items
 */
getSelectedItems(): string[] {
  return Array.from(this._selectedItemIds);
}

/**
 * Toggle selection of an item
 */
toggleItemSelection(id: string): void {
  if (this._selectedItemIds.has(id)) {
    this._selectedItemIds.delete(id);
  } else {
    this._selectedItemIds.add(id);
  }
  this._notifySelectionChanged();
}

// ========== COMPARISON MANAGEMENT ==========

/**
 * Update items selected for comparison
 */
setComparisonItems(ids: string[]): void {
  this._itemsForComparison = new Set(ids);
  this._notifyComparisonChanged();
}

/**
 * Get items selected for comparison
 */
getComparisonItems(): string[] {
  return Array.from(this._itemsForComparison);
}

// ========== LEGACY BRIDGE (Backward Compatibility) ==========

/**
 * Bridge methods maintain API compatibility during migration
 * These map legacy single-file methods to unified state
 */
addSample(sample: SampleInfo): void {
  const id = `sample:${sample.name}`;
  const item: LoadedItem = {
    id,
    type: 'sample',
    sampleName: sample.name,
    pod5Path: sample.pod5Path,
    bamPath: sample.bamPath,
    fastaPath: sample.fastaPath,
    readCount: sample.readCount,
    hasAlignments: sample.hasBam,
    hasReference: sample.hasFasta,
    hasMods: false,  // Will be set by caller
    hasEvents: false, // Will be set by caller
    fileSize: 0,      // Will be set by caller
    fileSizeFormatted: '0 B',
  };
  this.addLoadedItem(item);
  // Also maintain legacy _loadedSamples for now
  this._loadedSamples.set(sample.name, sample);
}

// ========== NOTIFICATION SYSTEM ==========

private _notifyLoadedItemsChanged(): void {
  this._onLoadedItemsChanged.fire(this.getLoadedItems());
}

private _notifySelectionChanged(): void {
  this._onSelectionChanged.fire(this.getSelectedItems());
}

private _notifyComparisonChanged(): void {
  this._onComparisonChanged.fire(this.getComparisonItems());
}
```

### 3.4 Panel Integration Pattern

**File Panel Integration**:

```typescript
// In FilePanelProvider constructor
constructor(extensionUri: vscode.Uri, state?: ExtensionState) {
  // ... existing code ...

  // Subscribe to unified state changes
  if (state) {
    state.onLoadedItemsChanged((items) => {
      // Convert LoadedItem[] to FileItem[] for display
      const fileItems = items.map(item => ({
        path: item.pod5Path,
        filename: path.basename(item.pod5Path),
        type: 'POD5',
        size: item.fileSize,
        sizeFormatted: item.fileSizeFormatted,
        numReads: item.readCount,
        hasMods: item.hasMods,
        hasEvents: item.hasEvents,
      }));

      this._files = fileItems;
      this.updateView(); // Refresh React component
    });
  }
}

// When loading a file via command
async function openPOD5File(filePath: string, state: ExtensionState) {
  // ... load via API ...

  const item: LoadedItem = {
    id: `pod5:${filePath}`,
    type: 'pod5',
    pod5Path: filePath,
    readCount: numReads,
    // ... other properties ...
  };

  // Add to unified state (triggers onLoadedItemsChanged event)
  state.addLoadedItem(item);

  // File Panel automatically updates via subscription
}
```

**Samples Panel Integration**:

```typescript
// In SamplesPanelProvider constructor
constructor(extensionUri: vscode.Uri, state: ExtensionState) {
  // ... existing code ...

  // Subscribe to unified state changes
  state.onLoadedItemsChanged((items) => {
    // Filter for samples (type: 'sample')
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
    this.updateView(); // Refresh React component
  });

  state.onComparisonChanged((ids) => {
    // Update comparison selections
    this._selectedSamples = new Set(ids);
    this.updateView();
  });
}
```

---

## Part 4: Event-Driven Synchronization

### 4.1 Event Flow Diagram

```
File Panel User Action (Open POD5)
  ↓
"squiggy.openPOD5" command executed
  ↓
state.addLoadedItem(pod5Item)
  ↓
state._onLoadedItemsChanged.fire(items)
  ↓
┌─────────────────────────────────────┐
│ [File Panel] listens and            │
│ calls updateView()                  │
│                                     │ (propagates)
│ [Samples Panel] listens and         │
│ calls updateView() (filters samples)│
│                                     │
│ [Reads View] listens for changes    │
│ [Plot Options] aware of file change │
└─────────────────────────────────────┘
  ↓
Both panels reflect current state ✓
```

### 4.2 Event Types

1. **onLoadedItemsChanged**: Fire when items loaded/unloaded
   - Used by: File Panel, Samples Panel, Reads View, Plot Options
   - Payload: `LoadedItem[]` (all current items)

2. **onSelectionChanged**: Fire when UI selection changes
   - Used by: File Panel, Samples Panel (for UI state)
   - Payload: `string[]` (selected item IDs)

3. **onComparisonChanged**: Fire when comparison selection changes
   - Used by: Samples Panel (for comparison mode)
   - Payload: `string[]` (IDs selected for comparison)

---

## Part 5: Session Serialization Update

### 5.1 Updated toSessionState()

With unified state, session serialization becomes unambiguous:

```typescript
toSessionState(): SessionState {
  const samples: { [sampleName: string]: SampleSessionState } = {};

  // Iterate through all loaded items
  for (const item of this._loadedItems.values()) {
    if (item.type === 'sample') {
      // Multi-sample mode
      samples[item.sampleName!] = {
        pod5Paths: [item.pod5Path],
        bamPath: item.bamPath,
        fastaPath: item.fastaPath,
      };
    } else if (item.type === 'pod5' && !item.sampleName) {
      // Single-file mode (legacy)
      samples['Default'] = {
        pod5Paths: [item.pod5Path],
        bamPath: item.bamPath,
        fastaPath: item.fastaPath,
      };
    }
  }

  return {
    version: '1.0.0',
    timestamp: new Date().toISOString(),
    samples,
    plotOptions: this._plotOptionsProvider?.getOptions() || {},
    modificationFilters: this._modificationsProvider?.getFilters?.(),
    ui: {
      expandedSamples: this.getAllSampleNames(),
      selectedSamplesForComparison: Array.from(this._itemsForComparison),
    },
  };
}
```

---

## Part 6: Implementation Roadmap (Phase 2)

### Phase 2 Tasks (Proposed)

**2.1: Create FileLoadingService (Deduplication)**
- [ ] Create `src/services/file-loading-service.ts`
- [ ] Implement `loadFile()` method for POD5, BAM, FASTA
- [ ] Implement shared `extractFileMetadata()` method
- [ ] Implement shared `formatFileSize()` utility
- [ ] Define normalized `FileLoadResult` interface
- [ ] Test: Service correctly loads files and extracts metadata

**2.2: Implement Unified State Foundation**
- [ ] Add LoadedItem interface to `src/types/`
- [ ] Add event emitters to ExtensionState
- [ ] Add unified state methods (addLoadedItem, removeLoadedItem, etc.)
- [ ] Add notification methods (private)
- [ ] Test: Events fire correctly when items added/removed

**2.3: Integrate File Panel with FileLoadingService & Unified State**
- [ ] Update openPOD5File command to use FileLoadingService
- [ ] Update openBAMFile command to use FileLoadingService
- [ ] Update openFASTAFile command to use FileLoadingService
- [ ] Subscribe to onLoadedItemsChanged in FilePanelProvider constructor
- [ ] Remove duplicated file loading logic
- [ ] Test: File Panel correctly displays items from unified state

**2.4: Integrate Samples Panel with FileLoadingService & Unified State**
- [ ] Update loadSample handler to use FileLoadingService
- [ ] Subscribe to onLoadedItemsChanged (filter for type: 'sample')
- [ ] Subscribe to onComparisonChanged for comparison selections
- [ ] Remove duplicated file loading logic
- [ ] Test: Samples Panel correctly displays items from unified state

**2.5: Cross-Panel Synchronization Testing**
- [ ] Load POD5 via File Panel → verify Samples Panel aware
- [ ] Load sample via Samples Panel → verify File Panel aware
- [ ] Clear files → verify both panels sync
- [ ] Session save/restore → verify round-trip integrity

**2.6: Session Management Update**
- [ ] Update toSessionState() to use unified state
- [ ] Update fromSessionState() to populate unified state
- [ ] Test demo session loading with unified state
- [ ] Test session save/restore cycle

**2.7: Documentation & Cleanup**
- [ ] Update CLAUDE.md with unified state pattern
- [ ] Add JSDoc comments to FileLoadingService
- [ ] Add JSDoc comments to new ExtensionState methods
- [ ] Document deduplication benefits in code comments
- [ ] Remove or deprecate legacy methods gradually
- [ ] Update README.md if needed

---

## Part 7: Key Design Decisions

### Decision 1: Event-Based vs Direct Updates
**Chosen**: Event-based with emitters
- **Why**: Decouples panels from each other, maintains single source of truth in ExtensionState
- **Alternative**: Direct method calls (tightly coupled, harder to test)

### Decision 2: Unified ID Scheme
**Chosen**: `"type:identifier"` format (e.g., `"pod5:/path"`, `"sample:name"`)
- **Why**: Unambiguous, supports both single files and samples in same structure
- **Alternative**: Single namespace (harder to distinguish types)

### Decision 3: Where to Store Selection State
**Chosen**: ExtensionState (not individual panels)
- **Why**: Single source of truth, serializable for sessions
- **Alternative**: Store in panels (harder to sync, lost on panel close)

### Decision 4: Backward Compatibility
**Chosen**: Bridge methods (`addSample()`) maintain old API during migration
- **Why**: Gradual migration, existing commands continue working
- **Alternative**: Immediate full refactor (risky, harder to test incrementally)

### Decision 5: Deduplication via FileLoadingService
**Chosen**: Create shared `FileLoadingService` for file loading operations
- **Why**:
  - Eliminates duplicated file loading logic in File Panel and Samples Panel
  - Single source of truth for metadata extraction
  - Consistent error handling across both panels
  - Easier to test and maintain
  - Reduces code review burden (no duplicated patterns to check)
- **Alternative 1**: Keep duplication (harder to maintain, bugs fixed in one place may miss the other)
- **Alternative 2**: Inherit from common base class (adds complexity, less flexible than composition)
- **Pattern**: Service layer composition (preferred over inheritance for extensibility)

---

## Part 8: Success Criteria

### Phase 1 Success
✅ Comprehensive analysis of current state management
✅ Identified all synchronization gaps
✅ Designed unified state structure with TypeScript interfaces
✅ Documented event-driven pattern for cross-panel sync
✅ Created Phase 2 implementation roadmap
✅ All design decisions documented and justified

### Phase 2 Success
✅ FileLoadingService eliminates file loading duplication between panels
✅ File metadata extraction centralized in one place
✅ Error handling unified and consistent across both panels
✅ File Panel and Samples Panel subscribe to unified state events
✅ Loading file in File Panel updates Samples Panel immediately
✅ Loading sample in Samples Panel updates File Panel immediately
✅ Selection state synchronized across panels
✅ Session save/restore maintains unified state
✅ No console pollution from sync mechanism
✅ All existing tests pass
✅ New tests for FileLoadingService pass
✅ Demo session loads seamlessly with both panels visible
✅ Code metrics: 40%+ reduction in duplicated file-loading logic

---

## Part 9: Open Questions & Future Considerations

### Q1: Multiple POD5 Files per Sample
**Current**: One POD5 per item
**Future**: Support concatenated reads from multiple POD5s
**Impact**: LoadedItem.pod5Path should become pod5Paths: string[]

### Q2: File Panel Selection State
**Current**: File Panel has no selection (just listing)
**Future**: Should users be able to select files?
**Impact**: May need different UI affordances for File Panel vs Samples Panel

### Q3: Async State Operations
**Current**: Events fire synchronously
**Future**: Handle slow operations (loading large POD5s)
**Impact**: May need loading states and progress indicators

### Q4: Backward Compatibility Duration
**Current**: Bridge methods maintain old API
**Plan**: Remove legacy methods in v1.0
**Timeline**: Deprecate in Phase 2, remove in Phase 3

---

## Appendix: File Locations Reference

| Component | File | Key Class/Interface |
|-----------|------|-------------------|
| State | `src/state/extension-state.ts` | `ExtensionState` |
| File Panel | `src/views/squiggy-file-panel.ts` | `FilePanelProvider` |
| File Panel UI | `src/views/components/squiggy-files-core.tsx` | `FilesCore` |
| Samples Panel | `src/views/squiggy-samples-panel.ts` | `SamplesPanelProvider` |
| Samples UI | `src/views/components/squiggy-samples-core.tsx` | `SamplesCore` |
| Types | `src/types/messages.ts` | `FileItem`, `SampleItem` |
| Commands | `src/commands/file-commands.ts` | `registerFileCommands()` |
| Extension | `src/extension.ts` | `activate()` |
| Session | `src/state/session-state-manager.ts` | `SessionStateManager` |

---

## Appendix: Related Issues

- **#86**: Replace File Explorer (deferred - build on unified state)
- **#79**: TSV import & redesign (deferred - separate work stream)
- **#92**: This issue - Unified extension state

---

**Document Status**: Ready for Phase 2 Implementation
**Next Step**: Review design with team, then proceed to Phase 2 (Implementation)
