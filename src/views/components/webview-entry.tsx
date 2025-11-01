/**
 * Webview Entry Point
 *
 * Initializes the React app for the reads panel webview.
 * This file is bundled separately by webpack for the browser environment.
 */

import * as React from 'react';
import * as ReactDOM from 'react-dom';
import { ReadsCore } from './squiggy-reads-core';

// Initialize React app when DOM is ready
const root = document.getElementById('root');
if (root) {
    ReactDOM.render(<ReadsCore />, root);
}
