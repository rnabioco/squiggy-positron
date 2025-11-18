/**
 * FilesTable Component
 *
 * Renders the files table with sortable column headers and file rows.
 * Uses flexbox-based layout (not HTML <table>) for easier styling.
 */

import * as React from 'react';
import { FilesTableProps, SortColumn } from '../../types/squiggy-files-types';
import { FileRow } from './squiggy-file-row';
import './squiggy-files-table.css';

export const FilesTable: React.FC<FilesTableProps> = ({
    files,
    sortColumn,
    sortDirection,
    onSort,
    onCloseFile,
}) => {
    const renderSortIndicator = (column: SortColumn) => {
        if (sortColumn !== column) {
            return null;
        }
        return <span className="sort-indicator">{sortDirection === 'asc' ? ' ▲' : ' ▼'}</span>;
    };

    /**
     * Handle keyboard activation of sortable headers
     * Supports Enter and Space keys for accessibility
     */
    const handleKeyDown = (event: React.KeyboardEvent, column: SortColumn) => {
        if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            onSort(column);
        }
    };

    /**
     * Get ARIA sort value for column header
     */
    const getAriaSort = (column: SortColumn): 'ascending' | 'descending' | 'none' => {
        if (sortColumn !== column) {
            return 'none';
        }
        return sortDirection === 'asc' ? 'ascending' : 'descending';
    };

    return (
        <div className="files-table">
            {/* Column Headers */}
            <div className="files-table-header">
                <div
                    className="files-header-cell files-col-filename"
                    onClick={() => onSort('filename')}
                    onKeyDown={(e) => handleKeyDown(e, 'filename')}
                    role="button"
                    tabIndex={0}
                    aria-label={`Sort by filename ${getAriaSort('filename')}`}
                    aria-sort={getAriaSort('filename')}
                    title="Sort by filename (press Enter or Space)"
                >
                    Filename{renderSortIndicator('filename')}
                </div>
                <div
                    className="files-header-cell files-col-type"
                    onClick={() => onSort('type')}
                    onKeyDown={(e) => handleKeyDown(e, 'type')}
                    role="button"
                    tabIndex={0}
                    aria-label={`Sort by type ${getAriaSort('type')}`}
                    aria-sort={getAriaSort('type')}
                    title="Sort by type (press Enter or Space)"
                >
                    Type{renderSortIndicator('type')}
                </div>
                <div
                    className="files-header-cell files-col-size"
                    onClick={() => onSort('size')}
                    onKeyDown={(e) => handleKeyDown(e, 'size')}
                    role="button"
                    tabIndex={0}
                    aria-label={`Sort by size ${getAriaSort('size')}`}
                    aria-sort={getAriaSort('size')}
                    title="Sort by size (press Enter or Space)"
                >
                    Size{renderSortIndicator('size')}
                </div>
                <div
                    className="files-header-cell files-col-reads"
                    onClick={() => onSort('reads')}
                    onKeyDown={(e) => handleKeyDown(e, 'reads')}
                    role="button"
                    tabIndex={0}
                    aria-label={`Sort by read count ${getAriaSort('reads')}`}
                    aria-sort={getAriaSort('reads')}
                    title="Sort by read count (press Enter or Space)"
                >
                    Reads{renderSortIndicator('reads')}
                </div>
                <div
                    className="files-header-cell files-col-refs"
                    onClick={() => onSort('refs')}
                    onKeyDown={(e) => handleKeyDown(e, 'refs')}
                    role="button"
                    tabIndex={0}
                    aria-label={`Sort by reference count ${getAriaSort('refs')}`}
                    aria-sort={getAriaSort('refs')}
                    title="Sort by reference count (press Enter or Space)"
                >
                    Refs{renderSortIndicator('refs')}
                </div>
                <div
                    className="files-header-cell files-col-mods"
                    aria-label="Modifications (not sortable)"
                    title="Modifications"
                >
                    Mods
                </div>
                <div
                    className="files-header-cell files-col-events"
                    aria-label="Event alignment move table (not sortable)"
                    title="Event alignment (move table)"
                >
                    Moves
                </div>
                <div className="files-header-cell files-col-actions" aria-label="Actions">
                    Close
                </div>
            </div>

            {/* File Rows */}
            <div className="files-table-body">
                {files.map((file, index) => (
                    <FileRow
                        key={file.path}
                        file={file}
                        isEvenRow={index % 2 === 0}
                        onCloseFile={onCloseFile}
                        onOpenFile={() => {}}
                    />
                ))}
            </div>
        </div>
    );
};
