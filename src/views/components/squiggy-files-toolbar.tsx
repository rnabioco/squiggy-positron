/**
 * FilesToolbar Component
 *
 * Toolbar with POD5 and BAM file open buttons
 */

import * as React from 'react';
import { FilesToolbarProps } from '../../types/squiggy-files-types';
import './squiggy-files-toolbar.css';

export const FilesToolbar: React.FC<FilesToolbarProps> = ({ onOpenPOD5, onOpenBAM }) => {
    return (
        <div className="files-toolbar">
            <button className="files-toolbar-button" onClick={onOpenPOD5} title="Open POD5 file">
                POD5 📁
            </button>
            <button className="files-toolbar-button" onClick={onOpenBAM} title="Open BAM file">
                BAM 📁
            </button>
        </div>
    );
};
