/**
 * Type definitions for Squiggy Reads Panel
 *
 * Defines data structures for the React-based reads panel with
 * virtualized rendering and multi-column table layout.
 */

/**
 * Item type in the flat list (for virtualization)
 */
export type ReadListItemType = 'reference' | 'read';

/**
 * Reference group header in the reads list
 */
export interface ReferenceGroupItem {
    type: 'reference';
    referenceName: string;
    readCount: number;
    isExpanded: boolean;
    indentLevel: 0;
}

/**
 * Individual read item in the reads list
 */
export interface ReadItem {
    type: 'read';
    readId: string;
    referenceName?: string; // Present when grouped by reference (BAM loaded)
    genomicPosition?: string; // Format: "chr:start-end" (when BAM loaded)
    quality?: number; // Mapping quality (when BAM loaded)
    indentLevel: 0 | 1; // 0 for flat list, 1 for nested under reference
}

/**
 * Union type for items in the virtualized list
 */
export type ReadListItem = ReferenceGroupItem | ReadItem;

/**
 * State for the reads panel
 */
export interface ReadsViewState {
    // Data
    items: ReadListItem[]; // Flattened list for virtualization
    hasReferences: boolean; // True when BAM loaded (grouped mode)
    totalReadCount: number;

    // Search/Filter
    searchText: string;
    searchMode: 'reference' | 'read'; // Search in reference names or read IDs
    filteredItems: ReadListItem[]; // Filtered subset of items

    // Selection
    selectedReadIds: Set<string>;
    focusedIndex: number | null;

    // Expansion state (for reference groups)
    expandedReferences: Set<string>;

    // Sorting (for reference groups)
    sortBy: 'name' | 'reads';
    sortOrder: 'asc' | 'desc';

    // Column widths
    nameColumnWidth: number;
    detailsColumnWidth: number;
}

/**
 * Message types for communication between extension and webview
 */
export type ReadsViewMessage =
    | { type: 'setReads'; reads: ReadItem[] }
    | { type: 'setReadsGrouped'; references: Map<string, ReadItem[]> }
    | { type: 'setReferencesOnly'; references: { referenceName: string; readCount: number }[] }
    | { type: 'appendReads'; reads: ReadItem[] }
    | {
          type: 'setReadsForReference';
          referenceName: string;
          reads: ReadItem[];
          offset: number;
          totalCount: number;
      }
    | { type: 'updateSearch'; searchText: string }
    | { type: 'selectRead'; readId: string; multiSelect: boolean }
    | { type: 'plotRead'; readId: string }
    | { type: 'toggleReference'; referenceName: string }
    | { type: 'expandReference'; referenceName: string; offset: number; limit: number }
    | { type: 'updateColumnWidths'; nameWidth: number; detailsWidth: number }
    | { type: 'loadMore' }
    | { type: 'setLoading'; isLoading: boolean; message?: string };

/**
 * Props for React components
 */
export interface ReadsViewProps {
    onPlotRead: (readId: string) => void;
    onSelectRead: (readId: string, multiSelect: boolean) => void;
    onToggleReference: (referenceName: string) => void;
    onSearch: (searchText: string) => void;
    onLoadMore: () => void;
}

export interface ReadsInstanceProps extends ReadsViewProps {
    items: ReadListItem[];
    selectedReadIds: Set<string>;
    focusedIndex: number | null;
    nameColumnWidth: number;
    detailsColumnWidth: number;
    hasReferences: boolean;
    sortBy: 'name' | 'reads';
    sortOrder: 'asc' | 'desc';
    onUpdateColumnWidths: (nameWidth: number, detailsWidth: number) => void;
    onSort: (column: 'name' | 'reads') => void;
}

export interface ReadItemProps {
    item: ReadItem;
    isSelected: boolean;
    isFocused: boolean;
    isEvenRow: boolean;
    nameColumnWidth: number;
    detailsColumnWidth: number;
    onPlotRead: (readId: string) => void;
    onClick: (readId: string, multiSelect: boolean) => void;
}

export interface ReferenceGroupProps {
    item: ReferenceGroupItem;
    isEvenRow: boolean;
    nameColumnWidth: number;
    detailsColumnWidth: number;
    onToggle: (referenceName: string) => void;
}

/**
 * Constants
 */
export const CONSTANTS = {
    // Virtualization
    ROW_HEIGHT: 26, // px
    OVERSCAN_COUNT: 5, // Number of items to render outside visible area

    // Default column widths
    DEFAULT_NAME_WIDTH: 300, // px
    DEFAULT_DETAILS_WIDTH: 200, // px
    MIN_COLUMN_WIDTH: 100, // px
    ACTIONS_COLUMN_WIDTH: 80, // px (fixed)

    // Lazy loading
    INITIAL_LOAD_COUNT: 1000,
    LOAD_MORE_COUNT: 500,

    // Indent
    INDENT_SIZE: 20, // px per level
};
