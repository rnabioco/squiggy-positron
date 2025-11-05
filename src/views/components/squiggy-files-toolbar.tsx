/**
 * FilesToolbar Component
 *
 * Toolbar with "Add Files" (POD5/BAM) and "Add Reference" (FASTA) buttons
 * Files are loaded and appear in File Explorer, auto-matched in Sample Manager
 */

import * as React from 'react';
import { FilesToolbarProps } from '../../types/squiggy-files-types';
import './squiggy-files-toolbar.css';

export const FilesToolbar: React.FC<FilesToolbarProps> = ({
    onOpenPOD5,
    onOpenBAM,
    onOpenFASTA,
}) => {
    // onOpenPOD5 and onOpenBAM now trigger the combined file picker workflow
    // onOpenFASTA triggers the FASTA-specific workflow
    const handleAddFiles = onOpenPOD5; // Use the first handler for add files
    const handleAddReference = onOpenFASTA; // Use FASTA handler for reference

    return (
        <div className="files-toolbar">
            <button
                className="files-toolbar-button"
                onClick={handleAddFiles}
                title="Add POD5 and/or BAM files. Files will be loaded to File Explorer and auto-matched as samples in Sample Manager."
            >
                + Add Files
            </button>
            <button
                className="files-toolbar-button"
                onClick={handleAddReference}
                title="Add FASTA reference file"
            >
                + Add Reference
            </button>
        </div>
    );
};
