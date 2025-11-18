/*---------------------------------------------------------------------------------------------
 *  Copyright (C) 2024 Posit Software, PBC. All rights reserved.
 *  Licensed under the Elastic License 2.0. See LICENSE.txt for license information.
 *--------------------------------------------------------------------------------------------*/

/**
 * Webview Entry Point
 *
 * Initializes the React app for webview panels using React 18's createRoot() API.
 * Detects which panel to render based on document title.
 * This file is bundled separately by webpack for the browser environment.
 *
 * Pattern follows Positron's positronReactRenderer.tsx for proper React 18 usage.
 */

import * as React from 'react';
import { createRoot } from 'react-dom/client';
import { ErrorBoundary } from './error-boundary';
import { ReadsCore } from './squiggy-reads-core';
import { FilesCore } from './squiggy-files-core';
import { PlotOptionsCore } from './squiggy-plot-options-core';
import { ModificationsCore } from './squiggy-modifications-core';
import { SamplesCore } from './squiggy-samples-core';
import { SessionCore } from './squiggy-session-core';

// Initialize React app when DOM is ready
const rootElement = document.getElementById('root');
if (rootElement) {
    // Create React 18 root (following Positron's pattern)
    const reactRoot = createRoot(rootElement);

    // Detect which panel to render based on document title
    const title = document.title;

    // Render the appropriate component wrapped in ErrorBoundary
    if (title.includes('Session Manager') || title.includes('Session')) {
        reactRoot.render(
            <ErrorBoundary>
                <SessionCore />
            </ErrorBoundary>
        );
    } else if (title.includes('File Explorer') || title.includes('Files')) {
        reactRoot.render(
            <ErrorBoundary>
                <FilesCore />
            </ErrorBoundary>
        );
    } else if (title.includes('Reads')) {
        reactRoot.render(
            <ErrorBoundary>
                <ReadsCore />
            </ErrorBoundary>
        );
    } else if (title.includes('Plotting')) {
        reactRoot.render(
            <ErrorBoundary>
                <PlotOptionsCore />
            </ErrorBoundary>
        );
    } else if (title.includes('Modifications')) {
        reactRoot.render(
            <ErrorBoundary>
                <ModificationsCore />
            </ErrorBoundary>
        );
    } else if (title.includes('Sample')) {
        reactRoot.render(
            <ErrorBoundary>
                <SamplesCore />
            </ErrorBoundary>
        );
    } else {
        // Default to reads panel for backward compatibility
        reactRoot.render(
            <ErrorBoundary>
                <ReadsCore />
            </ErrorBoundary>
        );
    }
}
