/**
 * Squiggle Plot Webview Panel
 *
 * Displays interactive Bokeh plots in a webview
 */

import * as vscode from 'vscode';
import { promises as fs } from 'fs';

export class SquigglePlotPanel {
    public static currentPanel: SquigglePlotPanel | undefined;
    private readonly _panel: vscode.WebviewPanel;
    private _disposables: vscode.Disposable[] = [];
    private _currentHtml: string = '';
    private _currentReadIds: string[] = [];

    // State for single read toggle
    private _currentCoordinateSpace: 'signal' | 'sequence' = 'signal';
    private _hasBam: boolean = false;
    private _onToggleCoordinateSpace?: (
        readId: string,
        coordinateSpace: 'signal' | 'sequence'
    ) => Promise<void>;

    private constructor(panel: vscode.WebviewPanel, _extensionUri: vscode.Uri) {
        this._panel = panel;

        // Set up event listeners
        this._panel.onDidDispose(() => this.dispose(), null, this._disposables);

        // Handle messages from the webview
        this._panel.webview.onDidReceiveMessage(
            async (message) => {
                switch (message.command) {
                    case 'alert':
                        vscode.window.showInformationMessage(message.text);
                        return;
                    case 'toggleCoordinateSpace': {
                        // Toggle between 'signal' and 'sequence' for single read plots
                        const newSpace = message.coordinateSpace as 'signal' | 'sequence';
                        this._currentCoordinateSpace = newSpace;

                        // Call the callback to regenerate the plot
                        if (this._onToggleCoordinateSpace && this._currentReadIds.length === 1) {
                            await this._onToggleCoordinateSpace(this._currentReadIds[0], newSpace);
                        }
                        return;
                    }
                }
            },
            null,
            this._disposables
        );
    }

    /**
     * Create or show the plot panel
     */
    public static createOrShow(extensionUri: vscode.Uri): SquigglePlotPanel {
        const column = vscode.window.activeTextEditor
            ? vscode.window.activeTextEditor.viewColumn
            : undefined;

        // If we already have a panel, show it
        if (SquigglePlotPanel.currentPanel) {
            SquigglePlotPanel.currentPanel._panel.reveal(column);
            return SquigglePlotPanel.currentPanel;
        }

        // Otherwise, create a new panel
        const panel = vscode.window.createWebviewPanel(
            'squigglePlot',
            'Squiggle Plot',
            column || vscode.ViewColumn.One,
            {
                enableScripts: true,
                retainContextWhenHidden: true,
                localResourceRoots: [extensionUri],
            }
        );

        SquigglePlotPanel.currentPanel = new SquigglePlotPanel(panel, extensionUri);
        return SquigglePlotPanel.currentPanel;
    }

    /**
     * Set the toggle callback for single read plots
     */
    public setToggleCallback(
        callback: (readId: string, coordinateSpace: 'signal' | 'sequence') => Promise<void>,
        hasBam: boolean
    ): void {
        this._onToggleCoordinateSpace = callback;
        this._hasBam = hasBam;
    }

    /**
     * Set the plot HTML content
     */
    public setPlot(
        bokehHtml: string,
        readIds: string[],
        coordinateSpace: 'signal' | 'sequence' = 'signal'
    ): void {
        this._currentHtml = bokehHtml;
        this._currentReadIds = readIds;
        this._currentCoordinateSpace = coordinateSpace;

        // Update panel title
        const title =
            readIds.length === 1
                ? `Squiggle Plot: ${readIds[0]}`
                : `Squiggle Plot: ${readIds.length} reads`;
        this._panel.title = title;

        // Show loading state first to make the update visible
        this._panel.webview.html = this._getLoadingContent();

        // Then set the actual content after a brief delay
        // This ensures the webview visibly refreshes
        setTimeout(() => {
            this._panel.webview.html = this._getWebviewContent(bokehHtml);
        }, 100);
    }

    /**
     * Export the current plot
     */
    public async exportPlot(outputPath: string): Promise<void> {
        // For HTML export, just write the current HTML
        if (outputPath.endsWith('.html')) {
            // Using imported fs.promises
            await fs.writeFile(outputPath, this._currentHtml, 'utf-8');
            vscode.window.showInformationMessage(`Plot exported to ${outputPath}`);
        } else {
            // PNG/SVG export would require calling Python backend
            vscode.window.showErrorMessage('PNG/SVG export not yet implemented');
        }
    }

    /**
     * Dispose of the panel
     */
    public dispose(): void {
        SquigglePlotPanel.currentPanel = undefined;

        this._panel.dispose();

        while (this._disposables.length) {
            const disposable = this._disposables.pop();
            if (disposable) {
                disposable.dispose();
            }
        }
    }

    /**
     * Get loading indicator HTML
     */
    private _getLoadingContent(): string {
        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Loading...</title>
    <style>
        body {
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            font-family: var(--vscode-font-family);
            background-color: var(--vscode-editor-background);
            color: var(--vscode-editor-foreground);
        }
        .spinner {
            border: 4px solid var(--vscode-progressBar-background);
            border-top: 4px solid var(--vscode-progressBar-foreground);
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .loading-text {
            margin-left: 20px;
        }
    </style>
</head>
<body>
    <div class="spinner"></div>
    <div class="loading-text">Loading plot...</div>
</body>
</html>`;
    }

    /**
     * Get the webview HTML content
     */
    private _getWebviewContent(bokehHtml: string): string {
        // Extract the Bokeh plot HTML and wrap it with proper CSP
        // The Bokeh HTML from Python includes CDN resources which we need to allow

        // Show toggle for single read plots with BAM
        const showToggle = this._currentReadIds.length === 1 && this._hasBam;
        const isSequenceMode = this._currentCoordinateSpace === 'sequence';

        const toggleHtml = showToggle
            ? `
            <div id="coordinate-toggle">
                <button
                    id="signal-btn"
                    class="toggle-btn ${!isSequenceMode ? 'active' : ''}"
                    onclick="setCoordinateSpace('signal')"
                >
                    Signal
                </button>
                <button
                    id="sequence-btn"
                    class="toggle-btn ${isSequenceMode ? 'active' : ''}"
                    onclick="setCoordinateSpace('sequence')"
                >
                    Sequence
                </button>
            </div>
        `
            : '';

        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="Content-Security-Policy" content="
        default-src 'none';
        script-src 'unsafe-inline' 'unsafe-eval' https://cdn.bokeh.org https://cdn.pydata.org;
        style-src 'unsafe-inline' https://cdn.bokeh.org https://cdn.pydata.org;
        img-src data: https:;
        font-src https://cdn.bokeh.org;
    ">
    <title>Squiggle Plot</title>
    <style>
        body {
            margin: 0;
            padding: 0;
            overflow: hidden;
        }
        #plot-container {
            width: 100vw;
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            position: relative;
        }
        #coordinate-toggle {
            position: absolute;
            top: 10px;
            right: 10px;
            display: flex;
            gap: 0;
            background: var(--vscode-editor-background);
            border: 1px solid var(--vscode-panel-border);
            border-radius: 4px;
            padding: 2px;
            z-index: 1000;
        }
        .toggle-btn {
            background: transparent;
            border: none;
            color: var(--vscode-foreground);
            padding: 6px 12px;
            cursor: pointer;
            font-size: 13px;
            font-family: var(--vscode-font-family);
            border-radius: 2px;
            transition: background-color 0.2s;
        }
        .toggle-btn:hover {
            background: var(--vscode-list-hoverBackground);
        }
        .toggle-btn.active {
            background: var(--vscode-button-background);
            color: var(--vscode-button-foreground);
        }
    </style>
</head>
<body>
    <div id="plot-container">
        ${toggleHtml}
        ${bokehHtml}
    </div>
    <script>
        const vscode = acquireVsCodeApi();

        function setCoordinateSpace(space) {
            vscode.postMessage({
                command: 'toggleCoordinateSpace',
                coordinateSpace: space
            });
        }
    </script>
</body>
</html>`;
    }
}
