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
import { PlotOptionsCore } from './squiggy-plot-options-core';
import { ModificationsCore } from './squiggy-modifications-core';

// Log to help with debugging
console.log('Webview entry point loaded');
console.log('Document title:', document.title);
console.log('Root element:', document.getElementById('root'));

// Initialize React app when DOM is ready
const root = document.getElementById('root');
if (root) {
    // Detect which panel to render based on document title
    const title = document.title;
    console.log('Checking title:', title);

    if (title.includes('File Explorer') || title.includes('Files')) {
        console.log('✓ Rendering FilesCore component');
        ReactDOM.render(<FilesCore />, root);
    } else if (title.includes('Reads')) {
        console.log('✓ Rendering ReadsCore component');
        ReactDOM.render(<ReadsCore />, root);
    } else if (title.includes('Plot Options')) {
        console.log('✓ Rendering PlotOptionsCore component');
        ReactDOM.render(<PlotOptionsCore />, root);
    } else if (title.includes('Modifications')) {
        console.log('✓ Rendering ModificationsCore component');
        try {
            ReactDOM.render(<ModificationsCore />, root);
            console.log('✓ ModificationsCore rendered successfully');
        } catch (error) {
            console.error('✗ Error rendering ModificationsCore:', error);
        }
    } else {
        // Default to reads panel for backward compatibility
        console.log('⚠ No match found, rendering ReadsCore component (default)');
        ReactDOM.render(<ReadsCore />, root);
    }
} else {
    console.error('✗ Root element not found!');
}
