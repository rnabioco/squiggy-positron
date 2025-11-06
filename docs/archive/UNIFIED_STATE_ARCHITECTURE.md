# Unified State Architecture Guide

## Overview

The Squiggy extension uses a **unified state architecture** to manage files, samples, and selections across multiple UI panels. This guide explains the pattern for developers maintaining or extending the code.

## Core Concept: Single Source of Truth

Instead of each panel maintaining its own state copy, a central `ExtensionState` registry holds all loaded items, with panels subscribing to change events.

```
┌─────────────────────────────────────────┐
│      Unified ExtensionState             │
│  - _loadedItems (Map)                   │
│  - _selectedItemIds (Set)               │
│  - _itemsForComparison (Set)            │
│  - Event Emitters                       │
└────────────┬──────────────┬─────────────┘
             │              │
        ┌────▼────┐    ┌────▼────┐
        │File     │    │Samples  │
        │Panel    │    │Panel    │
        └─────────┘    └─────────┘
```

## Key Data Structures

### LoadedItem Interface

Represents a single loaded file or sample in the unified registry:

```typescript
interface LoadedItem {
    // Unique identifier with type prefix
    id: string;  // "pod5:/path/file.pod5" | "sample:name" | "fasta:/path/file.fasta"

    // Item type
    type: 'pod5' | 'sample' | 'fasta';

    // File paths
    pod5Path: string;      // Required
    bamPath?: string;      // Optional alignment file
    fastaPath?: string;    // Optional reference sequence

    // Metadata
    readCount: number;
    fileSize: number;
    fileSizeFormatted: string;

    // Feature flags
    hasAlignments: boolean;
    hasReference: boolean;
    hasMods: boolean;      // Base modifications in BAM
    hasEvents: boolean;    // Event alignment in BAM

    // Sample-specific
    sampleName?: string;   // Only when type === 'sample'
}
```

### ID Scheme

IDs follow a prefixed format for unified tracking:
- **POD5 files**: `pod5:/absolute/path/to/file.pod5`
- **Samples**: `sample:sampleName`
- **FASTA files**: `fasta:/absolute/path/to/file.fasta`

This allows mixing types in a single Map while maintaining uniqueness.

## Event-Driven Communication

### Event Emitters

ExtensionState exposes three event emitters for reactive updates:

#### 1. onLoadedItemsChanged
Fired when items are added, removed, or updated:

```typescript
state.onLoadedItemsChanged((items: LoadedItem[]) => {
    // items = current array of all loaded items
    // Use this to update any UI showing loaded files/samples
});
```

**When fired**:
- `addLoadedItem(item)` - Item added or updated
- `removeLoadedItem(id)` - Item removed
- `clearLoadedItems()` - All items cleared
- File panel and samples panel both subscribe to this

#### 2. onSelectionChanged
Fired when UI selections change (less commonly used):

```typescript
state.onSelectionChanged((ids: string[]) => {
    // ids = current set of selected item IDs
});
```

#### 3. onComparisonChanged
Fired when comparison mode selections change:

```typescript
state.onComparisonChanged((ids: string[]) => {
    // ids = current set of "sample:" prefixed IDs in comparison mode
});
```

**When fired**:
- `setComparisonItems(ids)` - Comparison items replaced
- `addToComparison(id)` - Item added to comparison
- `removeFromComparison(id)` - Item removed from comparison
- `clearComparison()` - Comparison cleared
- Samples panel uses this to highlight selected samples

## Panel Integration Pattern

### File Panel Implementation

File Panel subscribes to unified state changes:

```typescript
export class FilePanelProvider extends BaseWebviewProvider {
    constructor(extensionUri: vscode.Uri, private state?: ExtensionState) {
        super(extensionUri);

        if (this.state) {
            // Subscribe to state changes
            const disposable = this.state.onLoadedItemsChanged((items) => {
                this._handleLoadedItemsChanged(items);
            });
            this._disposables.push(disposable);
        }
    }

    private _handleLoadedItemsChanged(items: LoadedItem[]): void {
        // Convert LoadedItem[] to FileItem[] for display
        this._files = items.map((item) => ({
            path: item.pod5Path,
            filename: path.basename(item.pod5Path),
            type: item.type === 'pod5' ? 'POD5' : 'SAMPLE',
            size: item.fileSize,
            numReads: item.readCount,
            // ... other conversions
        }));

        this.updateView();  // Update webview with new data
    }
}
```

### Samples Panel Implementation

Samples Panel filters for sample-type items:

```typescript
private _handleLoadedItemsChanged(items: LoadedItem[]): void {
    // Filter for samples only (as Samples Panel does)
    this._samples = items
        .filter((item) => item.type === 'sample')
        .map((item) => ({
            name: item.sampleName!,
            pod5Path: item.pod5Path,
            bamPath: item.bamPath,
            readCount: item.readCount,
            // ... other fields
        }));

    this.updateView();
}

private _handleComparisonChanged(ids: string[]): void {
    // Extract sample names from "sample:" prefixed IDs
    this._selectedSamples = new Set(
        ids
            .filter((id) => id.startsWith('sample:'))
            .map((id) => id.substring(7))  // Remove "sample:" prefix
    );

    this.updateView();  // Highlight selected samples
}
```

## Using the Unified State

### Adding Items

```typescript
// Create a LoadedItem
const item: LoadedItem = {
    id: `pod5:${filePath}`,
    type: 'pod5',
    pod5Path: filePath,
    readCount: reads,
    fileSize: size,
    fileSizeFormatted: formatFileSize(size),
    hasAlignments: false,
    hasReference: false,
    hasMods: false,
    hasEvents: false,
};

// Add to state
state.addLoadedItem(item);

// Both File Panel and Samples Panel are notified via onLoadedItemsChanged
// They each update their displays appropriately
```

### Querying Items

```typescript
// Get all loaded items
const allItems = state.getLoadedItems();

// Get a specific item by ID
const item = state.getLoadedItem(`pod5:${filePath}`);

// Get comparison selections
const comparisonIds = state.getComparisonItems();
```

### Updating Items

Items are immutable; to update, create a new LoadedItem and re-add:

```typescript
// Update existing item with new BAM association
const updated = {
    ...oldItem,
    bamPath: '/path/to/alignment.bam',
    hasAlignments: true,
};

state.addLoadedItem(updated);  // Replaces old item with same ID
```

### Removing Items

```typescript
state.removeLoadedItem(`pod5:${filePath}`);

// File Panel and Samples Panel both receive onLoadedItemsChanged event
// with updated items list (no longer contains removed item)
```

## Session State Integration

### Serialization (toSessionState)

Unified state is serialized to session format:

```typescript
const sessionState = state.toSessionState();

// sessionState.samples contains:
// {
//   "sampleName": {
//     pod5Paths: [...],
//     bamPath?: "...",
//     fastaPath?: "..."
//   },
//   ...
// }
```

The toSessionState() method:
1. **Prefers** unified state if `_loadedItems` has items
2. **Falls back** to legacy `_loadedSamples` if available
3. **Falls back** to legacy single-file mode (`_currentPod5File`, etc.)

### Restoration (fromSessionState)

When restoring a session:

```typescript
await state.fromSessionState(sessionState, context);

// This:
// 1. Clears all current state
// 2. Loads each sample from session data
// 3. Creates LoadedItem for each and adds to _loadedItems
// 4. Fires onLoadedItemsChanged for each item
// 5. Restores comparison selections via setComparisonItems()
// 6. Restores plot options and filters
```

## Common Patterns

### "When does my event fire?"

**onLoadedItemsChanged fires on**:
- Any `addLoadedItem()` call (new or update)
- Any `removeLoadedItem()` call
- Any `clearLoadedItems()` call

Frequency: Could be called multiple times during session restore. Listeners should be efficient.

**onComparisonChanged fires on**:
- `setComparisonItems()` call
- Any `addToComparison()` call
- Any `removeFromComparison()` call
- `clearComparison()` call
- `removeLoadedItem()` that's in comparison (removes from comparison too)

### "How do I know when a file is a POD5 vs a sample?"

Check the `type` field:
```typescript
if (item.type === 'pod5') {
    // Standalone POD5 file
} else if (item.type === 'sample') {
    // Sample bundle (might have BAM, FASTA, etc.)
}
```

### "How do I filter items like the panels do?"

```typescript
// File Panel: Show all items
const allItems = items;

// Samples Panel: Show only samples
const samples = items.filter((item) => item.type === 'sample');

// Single file view: Get first POD5
const pod5 = items.find((item) => item.id.startsWith('pod5:'));
```

### "How do I handle item updates with metadata?"

When loading a file completes, update the LoadedItem with real metadata:

```typescript
// Initial add (metadata empty)
state.addLoadedItem({
    id: `pod5:${filePath}`,
    type: 'pod5',
    pod5Path: filePath,
    readCount: 0,  // Will update
    fileSize: 0,   // Will update
    fileSizeFormatted: 'Unknown',
    // ... feature flags
});

// Later, when file is loaded
state.addLoadedItem({
    id: `pod5:${filePath}`,  // Same ID!
    type: 'pod5',
    pod5Path: filePath,
    readCount: actualReads,      // Updated
    fileSize: actualSize,        // Updated
    fileSizeFormatted: '123 MB', // Updated
    hasAlignments: !!bamPath,
    // ... rest of item
});

// Listeners get notified with updated item
```

## Error Handling

Unified state is generally error-tolerant:

```typescript
// Safe operations (don't throw)
state.removeLoadedItem('nonexistent-id');  // No-op, no error
state.clearLoadedItems();                   // Safe if empty
state.addLoadedItem(item);                  // Replaces if ID exists

// Always have items in event handlers
state.onLoadedItemsChanged((items) => {
    // items is always an array (possibly empty)
    if (items.length === 0) {
        // No items loaded
    }
});
```

## Performance Considerations

1. **Event frequency**: onLoadedItemsChanged can fire many times during session restore. Handlers should be O(n) or better.

2. **Subscription cleanup**: Always clean up subscriptions when panels are disposed:
   ```typescript
   public dispose(): void {
       for (const disposable of this._disposables) {
           disposable.dispose();  // Unsubscribe from events
       }
   }
   ```

3. **Large file lists**: With thousands of items, consider optimizing filters:
   ```typescript
   // Good: O(n) filter once
   const samples = items.filter(i => i.type === 'sample');

   // Bad: O(n²) in loop
   for (const id of ids) {
       if (items.find(i => i.id === id)) { }  // Don't do this
   }
   ```

## Debugging

To inspect unified state at runtime:

```typescript
// In any context with access to state:
console.log('Loaded items:', state.getLoadedItems());
console.log('Selected IDs:', state.getSelectedItems());
console.log('Comparison IDs:', state.getComparisonItems());
console.log('Full session state:', state.toSessionState());
```

## Future Enhancements

Possible improvements to the unified state pattern:

1. **Derived items**: Computed properties based on other items
2. **Batch updates**: Multiple items in one event
3. **Item history**: Track item changes over time
4. **Filtering API**: Built-in query builders
5. **Persistence**: Direct localStorage/file save
6. **Undo/Redo**: State history management

## Related Files

- `src/state/extension-state.ts` - Main implementation
- `src/types/loaded-item.ts` - LoadedItem interface
- `src/views/squiggy-file-panel.ts` - File Panel example
- `src/views/squiggy-samples-panel.ts` - Samples Panel example
- `src/__tests__/cross-panel-sync.test.ts` - Test examples
- `src/__tests__/session-state.test.ts` - Session integration tests
