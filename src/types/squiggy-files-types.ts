/**
 * Type definitions for Squiggy Files Panel
 *
 * Defines data structures for the React-based files panel with
 * sortable table layout showing file metadata.
 */

/**
 * Individual file item in the files table
 */
export interface FileItem {
    path: string;
    filename: string;
    type: 'POD5' | 'BAM' | 'FASTA';
    size: number; // bytes
    sizeFormatted: string; // e.g., "2.5 MB"
    numReads?: number;
    numRefs?: number; // BAM only
    hasMods?: boolean; // BAM only - has MM/ML tags
    hasEvents?: boolean; // BAM only - has mv tag
}

/**
 * Sortable column identifiers
 */
export type SortColumn = 'filename' | 'type' | 'size' | 'reads' | 'refs';

/**
 * Sort direction
 */
export type SortDirection = 'asc' | 'desc';

/**
 * State for the files panel
 */
export interface FilesViewState {
    // Data
    files: FileItem[];

    // Sorting
    sortColumn: SortColumn;
    sortDirection: SortDirection;
}

/**
 * Message types for communication between extension and webview
 */
export type FilesViewMessage =
    | { type: 'updateFiles'; files: FileItem[] }
    | { type: 'closeFile'; fileType: 'POD5' | 'BAM' | 'FASTA' }
    | { type: 'openFile'; fileType: 'POD5' | 'BAM' | 'FASTA' };

/**
 * Props for React components
 */
export interface FilesViewProps {
    onCloseFile: (fileType: 'POD5' | 'BAM' | 'FASTA') => void;
    onOpenFile: (fileType: 'POD5' | 'BAM' | 'FASTA') => void;
}

export interface FilesTableProps extends FilesViewProps {
    files: FileItem[];
    sortColumn: SortColumn;
    sortDirection: SortDirection;
    onSort: (column: SortColumn) => void;
}

export interface FileRowProps extends FilesViewProps {
    file: FileItem;
    isEvenRow: boolean;
}

export interface FilesToolbarProps {
    onOpenPOD5: () => void;
    onOpenBAM: () => void;
    onOpenFASTA: () => void;
}

/**
 * Constants
 */
export const CONSTANTS = {
    // Default sort
    DEFAULT_SORT_COLUMN: 'filename' as SortColumn,
    DEFAULT_SORT_DIRECTION: 'asc' as SortDirection,
};
