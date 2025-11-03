/**
 * FileRow Component
 *
 * Renders a single file row with all columns and close button
 */

import * as React from 'react';
import { FileRowProps } from '../../types/squiggy-files-types';
import './squiggy-file-row.css';

export const FileRow: React.FC<FileRowProps> = ({ file, isEvenRow, onCloseFile }) => {
    const handleClose = (e: React.MouseEvent) => {
        e.stopPropagation();
        onCloseFile(file.type);
    };

    return (
        <div className={`file-row ${isEvenRow ? 'even' : 'odd'}`}>
            {/* Filename */}
            <div className="file-cell file-col-filename" title={file.path}>
                {file.filename}
            </div>

            {/* Type */}
            <div className="file-cell file-col-type">
                <span className={`file-type-badge ${file.type.toLowerCase()}`}>{file.type}</span>
            </div>

            {/* Size */}
            <div className="file-cell file-col-size file-cell-numeric">{file.sizeFormatted}</div>

            {/* Reads */}
            <div className="file-cell file-col-reads file-cell-numeric">
                {file.numReads !== undefined ? file.numReads.toLocaleString() : '-'}
            </div>

            {/* Refs */}
            <div className="file-cell file-col-refs file-cell-numeric">
                {file.numRefs !== undefined ? file.numRefs.toLocaleString() : '-'}
            </div>

            {/* Modifications */}
            <div className="file-cell file-col-mods file-cell-centered">
                {file.hasMods ? (
                    <span style={{ color: '#22c55e', fontWeight: 'bold', fontSize: '1.2em' }}>
                        ✓
                    </span>
                ) : (
                    '-'
                )}
            </div>

            {/* Event Alignment */}
            <div className="file-cell file-col-events file-cell-centered">
                {file.hasEvents ? (
                    <span style={{ color: '#22c55e', fontWeight: 'bold', fontSize: '1.2em' }}>
                        ✓
                    </span>
                ) : (
                    '-'
                )}
            </div>

            {/* Actions */}
            <div className="file-cell file-col-actions file-cell-centered">
                <button
                    className="file-close-button"
                    onClick={handleClose}
                    title={`Close ${file.type} file`}
                    aria-label={`Close ${file.type} file`}
                >
                    ×
                </button>
            </div>
        </div>
    );
};
