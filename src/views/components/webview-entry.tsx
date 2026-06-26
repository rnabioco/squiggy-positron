/*---------------------------------------------------------------------------------------------
 *  Copyright (C) 2024 Posit Software, PBC. All rights reserved.
 *  Licensed under the Elastic License 2.0. See LICENSE.txt for license information.
 *--------------------------------------------------------------------------------------------*/

/**
 * Webview Entry Point
 *
 * Initializes the React app for webview panels using React 19's createRoot() API.
 * Detects which panel to render based on document title.
 * This file is bundled separately by webpack for the browser environment.
 *
 * Pattern follows Positron's positronReactRenderer.tsx for proper React 19 usage.
 */

import * as React from 'react';
import { createRoot } from 'react-dom/client';
import { ErrorBoundary } from './error-boundary';
import { ExtensionErrorBanner } from './extension-error-banner';
import { ReadsCore } from './squiggy-reads-core';
import { PlotOptionsCore } from './squiggy-plot-options-core';
import { ModificationsCore } from './squiggy-modifications-core';
import { SamplesCore } from './squiggy-samples-core';
import { SessionCore } from './squiggy-session-core';

/**
 * Pick the panel component to render based on the document title.
 * Order matters: more specific titles must be checked first.
 */
function selectPanel(title: string): React.ReactElement {
    if (title.includes('Session Manager') || title.includes('Session')) {
        return <SessionCore />;
    } else if (title.includes('Reads')) {
        return <ReadsCore />;
    } else if (title.includes('Plotting')) {
        return <PlotOptionsCore />;
    } else if (title.includes('Modifications')) {
        return <ModificationsCore />;
    } else if (title.includes('Sample')) {
        return <SamplesCore />;
    }
    // Default to reads panel for backward compatibility
    return <ReadsCore />;
}

// Initialize React app when DOM is ready
const rootElement = document.getElementById('root');
if (rootElement) {
    // Create React 19 root (following Positron's pattern)
    const reactRoot = createRoot(rootElement);

    // Render the selected panel wrapped in an ErrorBoundary (catches render
    // crashes) and an ExtensionErrorBanner (surfaces errors posted by the
    // extension host instead of dropping them silently).
    reactRoot.render(
        <ErrorBoundary>
            <ExtensionErrorBanner>{selectPanel(document.title)}</ExtensionErrorBanner>
        </ErrorBoundary>
    );
}
