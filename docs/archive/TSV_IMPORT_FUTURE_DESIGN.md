# TSV Import Design - Future Enhancement (Issue #79)

**Phase:** Phase 3+ (post-Phase 3 implementation)
**Related Issues:** #79 (Sample-Based File Explorer Redesign)
**Status:** Design document - no implementation yet

---

## Overview

This document describes the TSV import pattern for Phase 3-prepared data structures. The groundwork has been laid in Phase 3 (Task 3.1) with the refactored `SampleInfo` interface; this document outlines how to implement TSV loading when needed.

## TSV Format Specification

Expected format from external sources (e.g., Google Sheets export):

```tsv
sample_name	fasta	bam	pod5
sample_A	reference.fa	sample_A.bam	sample_A.pod5
sample_B	reference.fa	sample_B.bam	sample_B_run1.pod5,sample_B_run2.pod5
sample_C	-	-	sample_C.pod5
```

**Columns:**
- `sample_name` (required): User-facing sample identifier
- `fasta` (optional): Path to FASTA reference file
- `bam` (optional): Path to BAM alignment file
- `pod5` (required): Path to POD5 file(s) - comma-separated for technical replicates

**Special values:**
- `-` or empty cell = no file for that type (BAM/FASTA optional)

## Phase 3 Foundation (Already Done)

The `SampleInfo` interface is already designed to support TSV:

```typescript
interface SampleInfo {
    // Phase 3: Manual sample creation
    sampleId: string;
    displayName: string;      // TSV: sample_name
    pod5Path: string;         // TSV: first pod5 from comma-separated list
    bamPath?: string;         // TSV: bam column
    fastaPath?: string;       // TSV: fasta column
    isLoaded: boolean;        // TSV: will be false initially (lazy loading)

    // Phase 3: Future-ready metadata
    metadata?: {
        autoDetected?: boolean;
        displayColor?: string;
        sourceType?: 'manual' | 'tsv';  // TSV: will be 'tsv'
        tsvGroup?: string;              // TSV: batch ID (all from same TSV)
        tags?: string[];
    };
}
```

**Key differences for TSV:**
- `sampleId`: UUID or batch ID + row number
- `displayName`: Directly from `sample_name` column (not auto-extracted from filename)
- `isLoaded`: Set to `false` initially (lazy loading - files registered but not in kernel)
- `metadata.sourceType`: Set to `'tsv'`
- `metadata.tsvGroup`: All samples from same TSV share a batch ID (for grouping)

## Implementation Plan (When Ready for #79)

### Step 1: Add TSV Parsing Module

**File:** `src/services/tsv-parser.ts`

```typescript
interface TSVSampleSpec {
    sampleName: string;
    fastaPath?: string;
    bamPath?: string;
    pod5Paths: string[];  // For future multi-POD5 support
}

export function parseSamplesTSV(tsvContent: string): TSVSampleSpec[] {
    // Parse tab-separated values
    // Validate columns (sample_name + pod5 required)
    // Handle comma-separated pod5 lists
    // Return validated specs or throw with helpful error
}

export function validateTSVSample(spec: TSVSampleSpec): string[] {
    // Check file existence
    // Return array of validation errors (empty = valid)
}
```

### Step 2: Add TSV Import UI

**Options:**
1. **Copy/paste text area** (MVP)
   - User copies TSV from Google Sheets
   - Pastes into text input dialog
   - Parse and preview

2. **File picker** (enhanced)
   - Select `.tsv` file from disk
   - Auto-parse and preview

3. **Hybrid** (best UX)
   - Text input with paste + file picker
   - Show preview of parsed samples
   - Allow path editing before loading

### Step 3: Add Import Command

**File:** `src/commands/file-commands.ts`

```typescript
// New command
vscode.commands.registerCommand('squiggy.importSamplesFromTSV', async () => {
    // 1. Show text input or file picker
    // 2. Parse TSV content
    // 3. Validate all file paths
    // 4. Show preview dialog with:
    //    - Sample names
    //    - File associations
    //    - Validation warnings (missing files, etc.)
    // 5. Create SampleInfo objects with:
    //    - sourceType: 'tsv'
    //    - tsvGroup: batchId
    //    - isLoaded: false (don't load to kernel yet)
    // 6. Add to unified state
    // 7. Files loaded on-demand when user plots (lazy loading)
});
```

### Step 4: Lazy Loading Integration

When user initiates a comparison or plot, check which samples are not loaded:

```typescript
// Before plotting
for (const sampleId of selectedSampleIds) {
    const sample = state.getSample(sampleId);
    if (!sample.isLoaded) {
        // Load to kernel now
        await loadSampleToKernel(sample);
        sample.isLoaded = true;
    }
}
```

### Step 5: Session Persistence

Extend `SessionState` to track TSV metadata:

```typescript
interface SessionState {
    // ... existing fields ...
    tsvMetadata?: {
        batchId: string;
        originalTSVPath?: string;
        importedAt: string;
        samplesCreated: number;
    };
}
```

## Batch Grouping (`tsvGroup`)

All samples loaded from a single TSV share a `metadata.tsvGroup` ID. Benefits:

1. **Visual grouping** - Display samples from same TSV batch together
2. **Batch operations** - Unload all samples from a TSV at once
3. **Reload/refresh** - Reapply TSV after changes
4. **Metadata tracking** - Know origin of samples

Example UI:

```
Samples (6)
├─ Batch: tsv_20250104_195600 (4 samples)
│  ├─ ☑ sample_A
│  ├─ ☐ sample_B
│  ├─ ☑ sample_C
│  └─ ☐ sample_D (missing files ⚠️)
└─ Manual (2 samples)
   ├─ ☑ my_sample
   └─ ☐ test_sample
```

## Error Handling & Validation

**File validation failures:**
- Missing POD5: Block loading (required)
- Missing BAM: Warn, allow loading without alignment
- Missing FASTA: Warn, allow loading without reference

**Path resolution:**
- Relative paths: Resolve relative to TSV location (or configurable base)
- Absolute paths: Use as-is
- Environment variables: Support `${HOME}`, `${WORKSPACEDIR}`, etc.

**Duplicate handling:**
- Same sample name across multiple TSVs: Allow (different sources)
- Same sample name within single TSV: Error (must be unique)

## Future Enhancements

1. **Multi-POD5 per sample** - Support comma-separated lists as in TSV format
2. **FASTA association** - Per-sample FASTA instead of session-level
3. **Validation reports** - Export CSV of missing files, validation issues
4. **TSV editing UI** - Edit sample associations in-place before loading
5. **Remote TSV** - Load TSV from HTTP URL (Google Sheets export link)

## Testing Strategy

### Unit Tests

```typescript
// src/services/__tests__/tsv-parser.test.ts
describe('TSV Parser', () => {
    test('parses valid TSV with all columns');
    test('parses TSV with optional columns missing');
    test('handles comma-separated pod5 files');
    test('validates required sample_name and pod5 columns');
    test('rejects invalid TSV format with helpful error');
});

describe('TSV Validation', () => {
    test('detects missing POD5 files');
    test('detects missing BAM files (warning only)');
    test('detects duplicate sample names');
    test('resolves relative paths correctly');
});
```

### Integration Tests

```typescript
// Load TSV → validate → create samples → add to state → load to kernel
// Verify samples appear in all panels
// Verify lazy loading defers kernel load until plot requested
```

### Manual Testing

1. Export sample data from Google Sheets as TSV
2. Import via Sample Manager
3. Verify preview dialog shows correct samples
4. Verify samples load lazily (not immediately in kernel)
5. Verify comparison works when samples loaded on-demand
6. Verify session persistence includes TSV metadata

## Implementation Notes

- **Lazy loading** is critical for TSV - don't load all files to kernel immediately
- **Path resolution** needs flexibility for different environments (local, remote, mounted)
- **Batch grouping** helps with organization when mixing manual and TSV samples
- **Validation early** - check all files exist before creating samples
- **Reuse existing logic** - file matching algorithm can run on TSV-parsed paths

## References

- Issue #79: Sample-Based File Explorer Redesign
- Phase 3 Plan: UI Consolidation & Workflow Refinement
- `src/state/extension-state.ts` - SampleInfo interface
- `src/commands/file-commands.ts` - File loading patterns
