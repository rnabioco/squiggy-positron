/**
 * FilesToolbar Component
 *
 * Icon-only toolbar in top-right with Open POD5/BAM buttons
 */

import * as React from 'react';
import { FilesToolbarProps } from '../../types/squiggy-files-types';
import './squiggy-files-toolbar.css';

export const FilesToolbar: React.FC<FilesToolbarProps> = ({ onOpenPOD5, onOpenBAM }) => {
    console.log('FilesToolbar rendering');
    return (
        <div
            className="files-toolbar"
            style={{
                display: 'flex',
                gap: '8px',
                padding: '8px',
                justifyContent: 'flex-end',
                borderBottom: '1px solid var(--vscode-panel-border)',
                backgroundColor: 'var(--vscode-sideBar-background)',
            }}
        >
            <button
                className="files-toolbar-button"
                onClick={onOpenPOD5}
                title="Open POD5 file"
                aria-label="Open POD5 file"
                style={{
                    padding: '6px 12px',
                    background: 'var(--vscode-button-secondaryBackground)',
                    color: 'var(--vscode-button-secondaryForeground)',
                    border: 'none',
                    borderRadius: '2px',
                    cursor: 'pointer',
                    fontSize: '0.9em',
                }}
            >
                📂 POD5
            </button>
            <button
                className="files-toolbar-button"
                onClick={onOpenBAM}
                title="Open BAM file"
                aria-label="Open BAM file"
                style={{
                    padding: '6px 12px',
                    background: 'var(--vscode-button-secondaryBackground)',
                    color: 'var(--vscode-button-secondaryForeground)',
                    border: 'none',
                    borderRadius: '2px',
                    cursor: 'pointer',
                    fontSize: '0.9em',
                }}
            >
                📂 BAM
            </button>
        </div>
    );
};
