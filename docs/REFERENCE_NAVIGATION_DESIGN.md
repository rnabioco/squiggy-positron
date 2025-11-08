# Reference Navigation Design

**Status**: Design Recommendation
**Date**: 2025-11-08
**Context**: Improving user experience for navigating references with large assemblies (many contigs/chromosomes)

---

## Overview

This document provides design recommendations for improving reference navigation in Squiggy, particularly for assemblies with many contigs or long chromosomes. The recommendations are based on analysis of industry-standard genome browsers (IGV and UCSC Genome Browser) and the current Squiggy implementation.

## Problem Statement

### Current State

**Reads Explorer Panel** (`squiggy-reads-core.tsx`):
- References displayed as collapsible groups
- Text search box with two modes:
  - **Reference mode**: Search by reference name (string matching)
  - **Read mode**: Search by read ID
- Sorting by name or read count
- Lazy loading when expanding references (500 reads at a time)

**Plot Options Panel** (`squiggy-plot-options-core.tsx`):
- Simple `<select>` dropdown for reference selection
- All references listed alphabetically
- Used for Aggregate and Comparison plot types

### Current Limitations

1. **No coordinate-based navigation**: Users can't jump to `chr1:1000-5000` or similar genomic coordinates
2. **Dropdown becomes unwieldy**: With hundreds of contigs (common in draft genomes), scrolling through a `<select>` is tedious
3. **No fuzzy search**: Reference names must match exactly (e.g., can't type "chrX" to find "NC_000023.11")
4. **No context about reference sizes**: Users don't see chromosome/contig lengths
5. **No visual hierarchy**: All references treated equally (no indication which are primary chromosomes vs. scaffolds)

---

## Industry Best Practices

### IGV (Integrative Genomics Viewer)

**Multi-Modal Navigation**:

1. **Dropdown menu** for reference/chromosome selection (top-left corner)
2. **Search bar** accepting:
   - Chromosome names: `chr1`, `X`, `chrM`
   - Coordinate ranges: `chr1:1,000-5,000`
   - Gene symbols: `BRCA1`
3. **Visual ideogram**: Shows current chromosome with red box indicating viewport
4. **Coordinate ruler**: Displays current region with base positions

### UCSC Genome Browser

**Unified Search Box**:

- Accepts gene symbols, mRNA accessions, SNP IDs, author names, keywords
- Coordinate formats:
  - **Position format** (1-start, fully-closed): `chr1:1000-5000` (with colon and dash)
  - **BED format** (0-start, half-open): `chr1 999 5000` (with spaces)
- Smart parsing: Detects format automatically
- Returns suggestions/matches when ambiguous

### Common Patterns

1. **Hybrid approach**: Dropdown + search box (not either/or)
2. **Autocomplete/typeahead**: Filter as you type
3. **Recent/favorites**: Quick access to commonly used references
4. **Coordinate validation**: Immediate feedback if range is invalid
5. **Context display**: Show reference length, current position

---

## Recommendations

### Priority 1: Enhanced Reference Selector (Quick Win)

Replace the simple `<select>` dropdown with a **searchable combobox** in the Plot Options panel.

#### Features

```typescript
// New component: <ReferenceSelector>
interface ReferenceSelectorProps {
    references: ReferenceInfo[];  // name, length, readCount
    selectedReference: string;
    onSelect: (ref: string) => void;
}

interface ReferenceInfo {
    name: string;
    length: number;       // From BAM header or FASTA index
    readCount: number;    // From BAM alignment counts
}
```

#### UI Layout

```
┌─────────────────────────────────────────────────┐
│ Reference                                       │
│ ┌─────────────────────────────────────────────┐ │
│ │ 🔍 Search or select...          ▼          │ │  ← Combobox (click opens dropdown)
│ └─────────────────────────────────────────────┘ │
│                                                 │
│ Dropdown (when focused):                        │
│ ┌─────────────────────────────────────────────┐ │
│ │ chr1                      248,956,422 bp    │ │  ← Length shown for context
│ │ chr2                      242,193,529 bp    │ │
│ │ chrX                      156,040,895 bp    │ │
│ │ ─────────────────────────────────────────── │ │
│ │ scaffold_42                   12,453 bp     │ │  ← Smaller contigs grouped
│ │ scaffold_107                   8,921 bp     │ │
│ └─────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

#### Implementation

- Use **fuzzy search** (e.g., `chr1` matches `NC_000001.11`, `chromosome_1`)
- Show **top 10 matches** while typing
- Display **reference length** from BAM/FASTA metadata (already available in `bam_info.references`)
- Consider **VSCode's QuickPick API** for native Positron integration

#### Benefits

- ✅ Handles hundreds of references gracefully
- ✅ Fast keyboard navigation
- ✅ Zero new backend code (uses existing metadata)
- ✅ Minimal UI change (replaces existing dropdown)

#### Effort

**Estimated time**: 1-2 days
**Backend changes**: None

---

### Priority 2: Coordinate-Based Navigation (Medium Effort)

Add a **coordinate input field** to the Plot Options panel for Aggregate plots.

#### UI Addition

```
┌─────────────────────────────────────────────────┐
│ Reference                                       │
│ ┌─────────────────────────────────────────────┐ │
│ │ chr1                                        │ │  ← Reference selector
│ └─────────────────────────────────────────────┘ │
│                                                 │
│ Region (optional)                               │
│ ┌─────────────────────────────────────────────┐ │
│ │ 1000-5000                                   │ │  ← Coordinate input
│ └─────────────────────────────────────────────┘ │
│ Format: start-end (e.g., 1000-5000)             │
│                                                 │
│ ☑ Clip x-axis to consensus region               │
│ ☑ Transform to relative coordinates             │
└─────────────────────────────────────────────────┘
```

#### Backend Changes

```python
# New function in squiggy/io.py
def get_reads_for_reference_region(
    reference_name: str,
    start: int | None = None,
    end: int | None = None,
    max_reads: int = 100
) -> list[str]:
    """
    Get reads overlapping a genomic region

    Args:
        reference_name: Reference sequence name
        start: Start coordinate (1-based, inclusive)
        end: End coordinate (1-based, inclusive)
        max_reads: Maximum reads to return

    Returns:
        List of read IDs overlapping the region
    """
    if _squiggy_session.bam_path is None:
        raise RuntimeError("No BAM file loaded")

    with pysam.AlignmentFile(_squiggy_session.bam_path, "rb") as bam:
        read_ids = []
        for read in bam.fetch(reference_name, start - 1, end):  # Convert to 0-based
            if not read.is_unmapped:
                read_ids.append(read.query_name)
                if len(read_ids) >= max_reads:
                    break
        return read_ids
```

#### Coordinate Formats to Support

1. **Relative**: `1000-5000` (positions along reference)
2. **UCSC-style**: `chr1:1,000-5,000` (with commas, chromosome prefix)
3. **Single position**: `1000` (expand to ±500bp window)

#### Frontend Implementation

```typescript
// Parse coordinate input
function parseCoordinates(input: string): { start: number; end: number } | null {
    // Support formats:
    // - "1000-5000"
    // - "1000"  → 1000±500
    // - "chr1:1000-5000"  → ignore chr prefix
    const match = input.match(/(\d+)(?:-(\d+))?/);
    if (!match) return null;

    const start = parseInt(match[1]);
    const end = match[2] ? parseInt(match[2]) : start + 1000;
    return { start, end };
}

// Add to PlotOptionsState
interface PlotOptionsState {
    // ... existing fields
    referenceRegion: { start: number | null; end: number | null };
}
```

#### Benefits

- ✅ Users can navigate directly to regions of interest
- ✅ Essential for large references (chromosomes)
- ✅ Enables reproducible analysis (share exact coordinates)
- ✅ Leverages existing BAM indexing (fast lookups via `pysam.fetch()`)

#### Effort

**Estimated time**: 2-3 days
**Backend changes**: New Python function, modify aggregate plot generation

---

### Priority 3: Reference Metadata Panel (Low Priority)

Add a collapsible **Reference Info** section to provide context about the selected reference.

#### UI Layout

```
┌─────────────────────────────────────────────────┐
│ Reference                                       │
│ ┌─────────────────────────────────────────────┐ │
│ │ chr1                                  ▼     │ │
│ └─────────────────────────────────────────────┘ │
│                                                 │
│ ▶ Reference Details                             │  ← Collapsible section
│                                                 │
│ (When expanded:)                                │
│ ┌─────────────────────────────────────────────┐ │
│ │ Length: 248,956,422 bp                      │ │
│ │ Mapped reads: 1,234 reads                   │ │
│ │ Coverage: 45.2× (if calculable)             │ │
│ └─────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

#### Data Sources

- Length: `bam_info['references'][i]['length']` or `fasta_info['reference_lengths'][ref]`
- Read count: `bam_info['references'][i]['read_count']`
- Coverage: Calculated from read lengths and reference length

#### Benefits

- ✅ Helps users choose appropriate references
- ✅ Distinguishes primary chromosomes from small scaffolds
- ✅ Educational for users unfamiliar with the assembly

#### Effort

**Estimated time**: 0.5 days
**Backend changes**: None

---

### Priority 4: Smart Reference Grouping (Advanced)

For assemblies with many contigs, **group references by type** in the dropdown.

#### Categorization Logic

```typescript
interface ReferenceCategory {
    name: string;              // "Primary Chromosomes", "Scaffolds", "Unplaced"
    references: ReferenceInfo[];
    expanded: boolean;
}

// Auto-categorize based on patterns
function categorizeReferences(refs: ReferenceInfo[]): ReferenceCategory[] {
    return [
        {
            name: "Primary Chromosomes",
            references: refs.filter(r =>
                r.name.match(/^(chr)?[0-9XYM]+$/) && r.length > 1_000_000
            ),
            expanded: true
        },
        {
            name: "Scaffolds",
            references: refs.filter(r =>
                r.name.includes("scaffold") || r.name.includes("contig")
            ),
            expanded: false  // Collapsed by default
        },
        {
            name: "Unplaced",
            references: refs.filter(r => r.name.includes("_random")),
            expanded: false
        }
    ];
}
```

#### UI

```
┌─────────────────────────────────────────────────┐
│ ▼ Primary Chromosomes (23)                      │  ← Expanded group
│   chr1                      248,956,422 bp      │
│   chr2                      242,193,529 bp      │
│   ...                                           │
│                                                 │
│ ▶ Scaffolds (1,247)                             │  ← Collapsed group
│ ▶ Unplaced (82)                                 │
└─────────────────────────────────────────────────┘
```

#### Benefits

- ✅ Reduces visual clutter for draft genomes
- ✅ Surfaces important references first
- ✅ Matches user mental model ("I want chr1, not scaffold_42")

#### Effort

**Estimated time**: 1-2 days
**Backend changes**: None

---

## Implementation Phases

### Phase 1: Searchable Combobox (1-2 days)

#### Frontend Changes

1. Create `src/views/components/reference-selector.tsx`:

```typescript
export const ReferenceSelector: React.FC<Props> = ({ references, selected, onSelect }) => {
    const [searchText, setSearchText] = useState('');
    const [isOpen, setIsOpen] = useState(false);

    const filtered = useMemo(() => {
        return references.filter(ref =>
            ref.name.toLowerCase().includes(searchText.toLowerCase())
        ).slice(0, 10);  // Top 10 matches
    }, [searchText, references]);

    return (
        <div className="reference-selector">
            <input
                value={searchText || selected}
                onChange={e => setSearchText(e.target.value)}
                onFocus={() => setIsOpen(true)}
                placeholder="Search references..."
            />
            {isOpen && (
                <div className="reference-dropdown">
                    {filtered.map(ref => (
                        <div key={ref.name} onClick={() => handleSelect(ref.name)}>
                            <span>{ref.name}</span>
                            <span className="ref-length">{formatLength(ref.length)}</span>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};
```

2. Replace dropdown in `squiggy-plot-options-core.tsx`:

```typescript
// Old:
<select value={options.aggregateReference} onChange={...}>
    {options.availableReferences.map(...)}
</select>

// New:
<ReferenceSelector
    references={options.availableReferences}
    selected={options.aggregateReference}
    onSelect={ref => setOptions(prev => ({ ...prev, aggregateReference: ref }))}
/>
```

#### Backend Changes

None required! Reference metadata already available in `bam_info['references']`

---

### Phase 2: Coordinate Input (2-3 days)

#### Frontend Changes

1. Add coordinate input to Plot Options:

```typescript
interface PlotOptionsState {
    // ... existing fields
    referenceRegion: { start: number | null; end: number | null };
}

// Parse coordinate input
function parseCoordinates(input: string): { start: number; end: number } | null {
    // Support formats:
    // - "1000-5000"
    // - "1000"  → 1000±500
    // - "chr1:1000-5000"  → ignore chr prefix
    const match = input.match(/(\d+)(?:-(\d+))?/);
    if (!match) return null;

    const start = parseInt(match[1]);
    const end = match[2] ? parseInt(match[2]) : start + 1000;
    return { start, end };
}
```

#### Backend Changes

1. Add `get_reads_for_reference_region()` to `squiggy/io.py` (see earlier example)
2. Modify aggregate plot generation to accept optional `region` parameter:

```python
def plot_aggregate(
    reference_name: str,
    region: tuple[int, int] | None = None,  # (start, end) in 1-based coords
    max_reads: int = 100,
    ...
):
    if region:
        read_ids = get_reads_for_reference_region(
            reference_name, region[0], region[1], max_reads
        )
    else:
        read_ids = get_reads_for_reference_paginated(
            reference_name, offset=0, limit=max_reads
        )
    # ... rest of plotting logic
```

---

### Phase 3: Reference Grouping (Optional, 1-2 days)

#### Frontend Changes

1. Categorize references when BAM loads:

```typescript
useEffect(() => {
    if (message.type === 'updateReferences') {
        const categorized = categorizeReferences(message.references);
        setOptions(prev => ({
            ...prev,
            referenceCategories: categorized
        }));
    }
}, []);
```

2. Render grouped dropdown:

```typescript
{referenceCategories.map(category => (
    <optgroup key={category.name} label={category.name}>
        {category.references.map(ref => (
            <option key={ref.name} value={ref.name}>
                {ref.name} ({formatLength(ref.length)})
            </option>
        ))}
    </optgroup>
))}
```

---

## Additional Considerations

### FASTA Integration (Future Enhancement)

If/when FASTA loading is more tightly integrated:

- Enable **gene name search** (requires gene annotation file like GFF/GTF)
- Add **sequence context** to plots (show bases at top of aggregate plot)
- Support **motif-based navigation** (already have `squiggy.motif` module!)

### Performance

- **Coordinate queries**: `pysam.fetch()` uses BAM index → O(log n) lookup, very fast
- **Reference metadata**: Already loaded during `load_bam()` → no additional cost
- **Dropdown rendering**: Virtualize if >1000 references (use `react-window`)

### User Experience

- **Validation feedback**: Show error if coordinates exceed reference length
- **Autocompletion**: Suggest common formats as user types
- **Keyboard shortcuts**: Arrow keys to navigate dropdown, Enter to select
- **Persistence**: Remember last-used reference across sessions (store in extension state)

---

## Summary Table

| Feature | Priority | Effort | User Impact | Backend Changes |
|---------|----------|--------|-------------|-----------------|
| Searchable combobox | **High** | 1-2 days | High (handles 100s of refs) | None |
| Coordinate input | **Medium** | 2-3 days | High (enables precise navigation) | New Python function |
| Reference metadata panel | Low | 0.5 days | Medium (contextual info) | None |
| Smart grouping | Low | 1-2 days | Medium (reduces clutter) | None |

---

## Recommended Approach

**Start with Phase 1** (searchable combobox) as a quick win. This addresses the immediate pain point of navigating large reference lists with minimal code changes.

**Follow with Phase 2** (coordinate input) once Phase 1 is validated. This unlocks the full power of genomic navigation and aligns Squiggy with professional genome browsers.

**Defer Phase 3** (grouping) until user feedback indicates it's needed—it's mostly useful for highly fragmented assemblies (e.g., draft genomes with 10,000+ scaffolds).

---

## References

- [IGV Genome Browser Documentation](https://igv.org/)
- [UCSC Genome Browser User Guide](https://genome.ucsc.edu/goldenPath/help/hgTracksHelp.html)
- [UCSC Coordinate Counting Systems](https://genome-blog.gi.ucsc.edu/blog/2016/12/12/the-ucsc-genome-browser-coordinate-counting-systems/)
