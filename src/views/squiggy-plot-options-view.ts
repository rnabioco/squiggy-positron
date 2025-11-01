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
    private _downsample: number = 1;
    private _showSignalPoints: boolean = false;
    private _hasBamFile: boolean = false;

    // Event emitter for when options change that should trigger refresh
    private _onDidChangeOptions = new vscode.EventEmitter<void>();
    public readonly onDidChangeOptions = this._onDidChangeOptions.event;

    constructor(private readonly _extensionUri: vscode.Uri) {}

    public resolveWebviewView(
        webviewView: vscode.WebviewView,
        _context: vscode.WebviewViewResolveContext,
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
                    this._showDwellTime = data.value;
                    // Mutually exclusive with scaleDwellTime
                    if (data.value) {
                        this._scaleDwellTime = false;
                        if (this._view) {
                            this._view.webview.postMessage({
                                type: 'updateScaleDwellTime',
                                value: false,
                            });
                        }
                    }
                    this._onDidChangeOptions.fire();
                    break;
                case 'baseAnnotationsChanged':
                    this._showBaseAnnotations = data.value;
                    this._onDidChangeOptions.fire();
                    break;
                case 'scaleDwellTimeChanged':
                    this._scaleDwellTime = data.value;
                    // Mutually exclusive with showDwellTime
                    if (data.value) {
                        this._showDwellTime = false;
                        if (this._view) {
                            this._view.webview.postMessage({
                                type: 'updateShowDwellTime',
                                value: false,
                            });
                        }
                    }
                    this._onDidChangeOptions.fire();
                    break;
                case 'downsampleChanged':
                    this._downsample = data.value;
                    this._onDidChangeOptions.fire();
                    break;
                case 'showSignalPointsChanged':
                    this._showSignalPoints = data.value;
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
            downsample: this._downsample,
            showSignalPoints: this._showSignalPoints,
        };
    }

    /**
     * Update BAM file status and available plot modes
     */
    public updateBamStatus(hasBam: boolean) {
        this._hasBamFile = hasBam;

        // If BAM loaded, default to EVENTALIGN
        if (hasBam && this._plotMode === 'SINGLE') {
            this._plotMode = 'EVENTALIGN';
            this._updateConfig('defaultPlotMode', 'EVENTALIGN');
        }
        // If BAM unloaded and currently in EVENTALIGN mode, switch to SINGLE
        else if (!hasBam && this._plotMode === 'EVENTALIGN') {
            this._plotMode = 'SINGLE';
            this._updateConfig('defaultPlotMode', 'SINGLE');
        }

        // Update webview if available
        if (this._view) {
            this._view.webview.postMessage({
                type: 'updateBamStatus',
                hasBam: hasBam,
                plotMode: this._plotMode,
            });
        }
    }

    private _updateConfig(key: string, value: unknown) {
        const config = vscode.workspace.getConfiguration('squiggy');
        config.update(key, value, vscode.ConfigurationTarget.Global);
    }

    private _getHtmlForWebview(_webview: vscode.Webview) {
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
        .slider-container {
            margin-bottom: 8px;
        }
        .slider-label {
            display: flex;
            justify-content: space-between;
            margin-bottom: 4px;
            font-size: 0.9em;
        }
        .slider-value {
            font-weight: bold;
            color: var(--vscode-input-foreground);
        }
        input[type="range"] {
            width: 100%;
            margin-bottom: 4px;
        }
    </style>
</head>
<body>
    <!-- Plot Mode Section -->
    <div class="section">
        <div class="section-title">Plot Mode</div>
        <select id="plotMode">
            <option value="SINGLE">Single Read</option>
            <option value="EVENTALIGN" id="eventalignOption">Event-Aligned</option>
        </select>
    </div>

    <!-- Normalization Section -->
    <div class="section">
        <div class="section-title">Normalization</div>
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

        <div class="slider-container">
            <div class="slider-label">
                <span>Downsample signal:</span>
                <span class="slider-value" id="downsampleValue">1x (no downsampling)</span>
            </div>
            <input type="range" id="downsample" min="1" max="40" value="1" step="1">
            <div class="info-text">Reduce signal points for faster rendering (1 = all points)</div>
        </div>

        <div class="checkbox-label">
            <input type="checkbox" id="showSignalPoints">
            <label for="showSignalPoints">Show individual signal points</label>
        </div>
        <div class="info-text">Display circles at each signal sample point</div>
    </div>

    <script>
        const vscode = acquireVsCodeApi();

        // Get elements
        const plotModeEl = document.getElementById('plotMode');
        const eventalignOptionEl = document.getElementById('eventalignOption');
        const normalizationEl = document.getElementById('normalization');
        const showDwellTimeEl = document.getElementById('showDwellTime');
        const showBaseAnnotationsEl = document.getElementById('showBaseAnnotations');
        const scaleDwellTimeEl = document.getElementById('scaleDwellTime');
        const downsampleEl = document.getElementById('downsample');
        const downsampleValueEl = document.getElementById('downsampleValue');
        const showSignalPointsEl = document.getElementById('showSignalPoints');

        // Initialize: hide EVENTALIGN option by default
        eventalignOptionEl.style.display = 'none';

        // Listen for messages from extension
        window.addEventListener('message', event => {
            const message = event.data;
            switch (message.type) {
                case 'updateBamStatus':
                    // Show/hide EVENTALIGN option based on BAM status
                    if (message.hasBam) {
                        eventalignOptionEl.style.display = '';
                        // Set plot mode to the value sent by extension
                        plotModeEl.value = message.plotMode;
                    } else {
                        eventalignOptionEl.style.display = 'none';
                        // Force to SINGLE if currently EVENTALIGN
                        if (plotModeEl.value === 'EVENTALIGN') {
                            plotModeEl.value = 'SINGLE';
                        }
                    }
                    break;
                case 'updateShowDwellTime':
                    showDwellTimeEl.checked = message.value;
                    break;
                case 'updateScaleDwellTime':
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

        // Update downsample value display and send message
        function updateDownsampleDisplay(value) {
            if (value == 1) {
                downsampleValueEl.textContent = '1x (no downsampling)';
            } else {
                downsampleValueEl.textContent = '1/' + value + 'x';
            }
        }

        downsampleEl.addEventListener('input', (e) => {
            const value = parseInt(e.target.value);
            updateDownsampleDisplay(value);
            vscode.postMessage({
                type: 'downsampleChanged',
                value: value
            });
        });

        showSignalPointsEl.addEventListener('change', (e) => {
            vscode.postMessage({
                type: 'showSignalPointsChanged',
                value: e.target.checked
            });
        });
    </script>
</body>
</html>`;
    }
}
