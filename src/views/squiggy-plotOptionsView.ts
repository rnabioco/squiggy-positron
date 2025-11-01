/**
 * Plot Options Webview View
 *
 * Provides controls for plot configuration in the sidebar
 */

import * as vscode from 'vscode';

export class PlotOptionsViewProvider implements vscode.WebviewViewProvider {
    public static readonly viewType = 'squiggyPlotOptions';

    private _view?: vscode.WebviewView;
    private _plotMode: string = 'SINGLE';
    private _normalization: string = 'ZNORM';
    private _showDwellTime: boolean = false;
    private _showBaseAnnotations: boolean = true;
    private _scaleDwellTime: boolean = false;

    // Event emitter for when options change that should trigger refresh
    private _onDidChangeOptions = new vscode.EventEmitter<void>();
    public readonly onDidChangeOptions = this._onDidChangeOptions.event;

    constructor(private readonly _extensionUri: vscode.Uri) {}

    public resolveWebviewView(
        webviewView: vscode.WebviewView,
        context: vscode.WebviewViewResolveContext,
        _token: vscode.CancellationToken
    ) {
        this._view = webviewView;

        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [this._extensionUri],
        };

        webviewView.webview.html = this._getHtmlForWebview(webviewView.webview);

        // Handle messages from the webview
        webviewView.webview.onDidReceiveMessage((data) => {
            switch (data.type) {
                case 'plotModeChanged':
                    this._plotMode = data.value;
                    this._updateConfig('defaultPlotMode', data.value);
                    this._onDidChangeOptions.fire();
                    break;
                case 'normalizationChanged':
                    this._normalization = data.value;
                    this._updateConfig('defaultNormalization', data.value);
                    this._onDidChangeOptions.fire();
                    break;
                case 'dwellTimeChanged':
                    console.log('dwellTimeChanged:', data.value);
                    this._showDwellTime = data.value;
                    // Mutually exclusive with scaleDwellTime
                    if (data.value) {
                        console.log('Unchecking scaleDwellTime');
                        this._scaleDwellTime = false;
                        if (this._view) {
                            console.log('Sending updateScaleDwellTime message');
                            this._view.webview.postMessage({
                                type: 'updateScaleDwellTime',
                                value: false,
                            });
                        } else {
                            console.log('Warning: _view is undefined');
                        }
                    }
                    this._onDidChangeOptions.fire();
                    break;
                case 'baseAnnotationsChanged':
                    this._showBaseAnnotations = data.value;
                    this._onDidChangeOptions.fire();
                    break;
                case 'scaleDwellTimeChanged':
                    console.log('scaleDwellTimeChanged:', data.value);
                    this._scaleDwellTime = data.value;
                    // Mutually exclusive with showDwellTime
                    if (data.value) {
                        console.log('Unchecking showDwellTime');
                        this._showDwellTime = false;
                        if (this._view) {
                            console.log('Sending updateShowDwellTime message');
                            this._view.webview.postMessage({
                                type: 'updateShowDwellTime',
                                value: false,
                            });
                        } else {
                            console.log('Warning: _view is undefined');
                        }
                    }
                    this._onDidChangeOptions.fire();
                    break;
            }
        });
    }

    /**
     * Get current plot options
     */
    public getOptions() {
        return {
            mode: this._plotMode,
            normalization: this._normalization,
            showDwellTime: this._showDwellTime,
            showBaseAnnotations: this._showBaseAnnotations,
            scaleDwellTime: this._scaleDwellTime,
        };
    }

    private _updateConfig(key: string, value: any) {
        const config = vscode.workspace.getConfiguration('squiggy');
        config.update(key, value, vscode.ConfigurationTarget.Global);
    }

    private _getHtmlForWebview(webview: vscode.Webview) {
        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Plot Options</title>
    <style>
        body {
            padding: 10px;
            font-family: var(--vscode-font-family);
            font-size: var(--vscode-font-size);
            color: var(--vscode-foreground);
        }
        .section {
            margin-bottom: 20px;
        }
        .section-title {
            font-weight: bold;
            margin-bottom: 8px;
            color: var(--vscode-foreground);
        }
        .file-info {
            font-size: 0.9em;
            color: var(--vscode-descriptionForeground);
            margin-bottom: 4px;
            word-break: break-all;
        }
        .file-label {
            font-weight: bold;
        }
        label {
            display: block;
            margin-bottom: 4px;
            font-size: 0.9em;
        }
        select, input[type="checkbox"] {
            width: 100%;
            padding: 4px;
            margin-bottom: 10px;
            background: var(--vscode-input-background);
            color: var(--vscode-input-foreground);
            border: 1px solid var(--vscode-input-border);
        }
        input[type="checkbox"] {
            width: auto;
            margin-right: 6px;
        }
        .checkbox-label {
            display: flex;
            align-items: center;
            margin-bottom: 8px;
        }
        .info-text {
            font-size: 0.85em;
            color: var(--vscode-descriptionForeground);
            font-style: italic;
            margin-top: -6px;
            margin-bottom: 10px;
        }
    </style>
</head>
<body>
    <!-- Plot Mode Section -->
    <div class="section">
        <label for="plotMode">Plot mode:</label>
        <select id="plotMode">
            <option value="SINGLE">Single Read</option>
            <option value="EVENTALIGN">Event-Aligned (requires BAM)</option>
        </select>
    </div>

    <!-- Normalization Section -->
    <div class="section">
        <label for="normalization">Normalization method:</label>
        <select id="normalization">
            <option value="NONE">None (raw signal)</option>
            <option value="ZNORM" selected>Z-score</option>
            <option value="MEDIAN">Median-centered</option>
            <option value="MAD">Median Absolute Deviation</option>
        </select>
    </div>

    <!-- Display Options Section -->
    <div class="section">
        <div class="section-title">Display Options</div>
        <div class="checkbox-label">
            <input type="checkbox" id="showBaseAnnotations" checked>
            <label for="showBaseAnnotations">Show base labels</label>
        </div>
        <div class="info-text">Display base letters on signal (event-aligned mode)</div>

        <div class="checkbox-label">
            <input type="checkbox" id="showDwellTime">
            <label for="showDwellTime">Color by dwell time</label>
        </div>
        <div class="info-text">Color bases by dwell time instead of base type</div>

        <div class="checkbox-label">
            <input type="checkbox" id="scaleDwellTime">
            <label for="scaleDwellTime">Scale x-axis by dwell time</label>
        </div>
        <div class="info-text">X-axis shows cumulative dwell time instead of base positions</div>
    </div>

    <script>
        const vscode = acquireVsCodeApi();

        // Get elements
        const plotModeEl = document.getElementById('plotMode');
        const normalizationEl = document.getElementById('normalization');
        const showDwellTimeEl = document.getElementById('showDwellTime');
        const showBaseAnnotationsEl = document.getElementById('showBaseAnnotations');
        const scaleDwellTimeEl = document.getElementById('scaleDwellTime');

        // Listen for messages from extension (for mutual exclusion)
        window.addEventListener('message', event => {
            const message = event.data;
            console.log('Received message:', message);
            switch (message.type) {
                case 'updateShowDwellTime':
                    console.log('Updating showDwellTime to:', message.value);
                    showDwellTimeEl.checked = message.value;
                    break;
                case 'updateScaleDwellTime':
                    console.log('Updating scaleDwellTime to:', message.value);
                    scaleDwellTimeEl.checked = message.value;
                    break;
            }
        });

        // Send updates to extension
        plotModeEl.addEventListener('change', (e) => {
            vscode.postMessage({
                type: 'plotModeChanged',
                value: e.target.value
            });
        });

        normalizationEl.addEventListener('change', (e) => {
            vscode.postMessage({
                type: 'normalizationChanged',
                value: e.target.value
            });
        });

        showDwellTimeEl.addEventListener('change', (e) => {
            vscode.postMessage({
                type: 'dwellTimeChanged',
                value: e.target.checked
            });
        });

        showBaseAnnotationsEl.addEventListener('change', (e) => {
            vscode.postMessage({
                type: 'baseAnnotationsChanged',
                value: e.target.checked
            });
        });

        scaleDwellTimeEl.addEventListener('change', (e) => {
            vscode.postMessage({
                type: 'scaleDwellTimeChanged',
                value: e.target.checked
            });
        });
    </script>
</body>
</html>`;
    }
}
