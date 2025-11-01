/**
 * Webview Entry Point
 *
 * Initializes the React app for webview panels.
 * Detects which panel to render based on document title.
 * This file is bundled separately by webpack for the browser environment.
 */

import * as React from 'react';
import * as ReactDOM from 'react-dom';
import { ReadsCore } from './squiggy-reads-core';
import { FilesCore } from './squiggy-files-core';

// Log to help with debugging
console.log('Webview entry point loaded, document title:', document.title);

// Initialize React app when DOM is ready
const root = document.getElementById('root');
if (root) {
    // Detect which panel to render based on document title
    const title = document.title;

    if (title.includes('File Explorer') || title.includes('Files')) {
        console.log('Rendering FilesCore component');
        ReactDOM.render(<FilesCore />, root);
    } else if (title.includes('Reads')) {
        console.log('Rendering ReadsCore component');
        ReactDOM.render(<ReadsCore />, root);
    } else {
        // Default to reads panel for backward compatibility
        console.log('Rendering ReadsCore component (default)');
        ReactDOM.render(<ReadsCore />, root);
    }
} else {
    console.error('Root element not found!');
}
