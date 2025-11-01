/**
 * FilesCore Component
 *
 * Top-level React component for the files panel.
 * Manages state and handles communication with the extension host.
 */

import * as React from 'react';
import {
    FileItem,
    FilesViewState,
    SortColumn,
    SortDirection,
    CONSTANTS,
} from '../../types/squiggy-files-types';
import { FilesTable } from './squiggy-files-table';
import { FilesToolbar } from './squiggy-files-toolbar';
import { vscode } from './vscode-api';
import './squiggy-files-core.css';

export const FilesCore: React.FC = () => {
    console.log('FilesCore component mounted');

    // State
    const [state, setState] = React.useState<FilesViewState>({
        files: [],
        sortColumn: CONSTANTS.DEFAULT_SORT_COLUMN,
        sortDirection: CONSTANTS.DEFAULT_SORT_DIRECTION,
    });

    // Handle messages from extension
    React.useEffect(() => {
        console.log('FilesCore: Setting up message listener');

        const messageHandler = (event: MessageEvent) => {
            console.log('FilesCore: Received message:', event.data);
            const message = event.data;

            switch (message.type) {
                case 'updateFiles':
                    handleUpdateFiles(message.files);
                    break;
            }
        };

        window.addEventListener('message', messageHandler);

        // Notify extension that webview is ready
        console.log('FilesCore: Sending ready message');
        vscode.postMessage({ type: 'ready' });

        return () => window.removeEventListener('message', messageHandler);
    }, []);

    const handleUpdateFiles = (files: FileItem[]) => {
        console.log('FilesCore: Received updateFiles with', files.length, 'files:', files);
        setState((prev) => ({
            ...prev,
            files: sortFiles(files, prev.sortColumn, prev.sortDirection),
        }));
    };

    const handleSort = (column: SortColumn) => {
        setState((prev) => {
            // Toggle direction if clicking same column, otherwise default to asc
            const newDirection: SortDirection =
                prev.sortColumn === column && prev.sortDirection === 'asc' ? 'desc' : 'asc';

            return {
                ...prev,
                sortColumn: column,
                sortDirection: newDirection,
                files: sortFiles(prev.files, column, newDirection),
            };
        });
    };

    const handleCloseFile = (fileType: 'POD5' | 'BAM') => {
        vscode.postMessage({ type: 'closeFile', fileType });
    };

    const handleOpenPOD5 = () => {
        vscode.postMessage({ type: 'openFile', fileType: 'POD5' });
    };

    const handleOpenBAM = () => {
        vscode.postMessage({ type: 'openFile', fileType: 'BAM' });
    };

    return (
        <div className="files-core-container">
            {/* Toolbar with Open POD5/BAM buttons - Always visible */}
            <FilesToolbar onOpenPOD5={handleOpenPOD5} onOpenBAM={handleOpenBAM} />

            {/* Files table or empty state */}
            {state.files.length > 0 ? (
                <FilesTable
                    files={state.files}
                    sortColumn={state.sortColumn}
                    sortDirection={state.sortDirection}
                    onSort={handleSort}
                    onCloseFile={handleCloseFile}
                    onOpenFile={() => {}}
                />
            ) : (
                <div className="files-empty-state">
                    <p>No files loaded</p>
                    <p className="files-empty-hint">
                        Use the buttons above to open POD5 or BAM files, or open the Command Palette
                        (Cmd+Shift+P) and search for "Squiggy: Load Test Data"
                    </p>
                </div>
            )}
        </div>
    );
};

/**
 * Sort files by specified column and direction
 */
function sortFiles(files: FileItem[], column: SortColumn, direction: SortDirection): FileItem[] {
    const sorted = [...files];

    sorted.sort((a, b) => {
        let comparison = 0;

        switch (column) {
            case 'filename':
                comparison = a.filename.localeCompare(b.filename);
                break;
            case 'type':
                comparison = a.type.localeCompare(b.type);
                break;
            case 'size':
                comparison = a.size - b.size;
                break;
            case 'reads':
                comparison = a.numReads - b.numReads;
                break;
            case 'refs': {
                // Handle undefined for POD5 files (treat as -1 for sorting)
                const aRefs = a.numRefs ?? -1;
                const bRefs = b.numRefs ?? -1;
                comparison = aRefs - bRefs;
                break;
            }
        }

        return direction === 'asc' ? comparison : -comparison;
    });

    return sorted;
}
