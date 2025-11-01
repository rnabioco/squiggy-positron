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
        return (
            <span className="sort-indicator">{sortDirection === 'asc' ? ' ▲' : ' ▼'}</span>
        );
    };

    return (
        <div className="files-table">
            {/* Column Headers */}
            <div className="files-table-header">
                <div
                    className="files-header-cell files-col-filename"
                    onClick={() => onSort('filename')}
                    title="Sort by filename"
                >
                    Filename{renderSortIndicator('filename')}
                </div>
                <div
                    className="files-header-cell files-col-type"
                    onClick={() => onSort('type')}
                    title="Sort by type"
                >
                    Type{renderSortIndicator('type')}
                </div>
                <div
                    className="files-header-cell files-col-size"
                    onClick={() => onSort('size')}
                    title="Sort by size"
                >
                    Size{renderSortIndicator('size')}
                </div>
                <div
                    className="files-header-cell files-col-reads"
                    onClick={() => onSort('reads')}
                    title="Sort by read count"
                >
                    Reads{renderSortIndicator('reads')}
                </div>
                <div
                    className="files-header-cell files-col-refs"
                    onClick={() => onSort('refs')}
                    title="Sort by reference count"
                >
                    Refs{renderSortIndicator('refs')}
                </div>
                <div className="files-header-cell files-col-mods" title="Modifications">
                    Mods
                </div>
                <div className="files-header-cell files-col-events" title="Event alignment (move table)">
                    Moves
                </div>
                <div className="files-header-cell files-col-actions">Close</div>
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
