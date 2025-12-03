/**
 * Motif Search Panel Webview View
 *
 * Provides UI for searching motifs in FASTA files and generating aggregate plots
 */

import * as vscode from 'vscode';
import { MotifMatch } from '../types/motif-types';
import { ExtensionState } from '../state/extension-state';

export class MotifSearchPanelProvider implements vscode.WebviewViewProvider {
    public static readonly viewType = 'squiggyMotifSearch';

    private _view?: vscode.WebviewView;
    private _matches: MotifMatch[] = [];
    private _currentMotif: string = 'DRACH';
    private _searching: boolean = false;

    constructor(
        private readonly extensionUri: vscode.Uri,
        private readonly state: ExtensionState
    ) {}

    public resolveWebviewView(
        webviewView: vscode.WebviewView,
        _context: vscode.WebviewViewResolveContext,
        _token: vscode.CancellationToken
    ): void {
        this._view = webviewView;

        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [this.extensionUri],
        };

        webviewView.webview.html = this.getHtmlContent();

        // Handle messages from the webview
        webviewView.webview.onDidReceiveMessage(async (message) => {
            switch (message.type) {
                case 'searchMotif':
                    await this.searchMotif(message.motif, message.strand || 'both');
                    break;
                case 'plotAllMotifs':
                    await this.plotAllMotifs(message.motif, message.upstream, message.downstream);
                    break;
            }
        });
    }

    private getHtmlContent(): string {
        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Motif Search</title>
    <style>
        body {
            padding: 10px;
            color: var(--vscode-foreground);
            font-family: var(--vscode-font-family);
            font-size: var(--vscode-font-size);
        }
        .search-box {
            margin-bottom: 15px;
        }
        input[type="text"] {
            width: 60%;
            padding: 5px;
            background: var(--vscode-input-background);
            color: var(--vscode-input-foreground);
            border: 1px solid var(--vscode-input-border);
            font-family: monospace;
        }
        button {
            padding: 5px 15px;
            background: var(--vscode-button-background);
            color: var(--vscode-button-foreground);
            border: none;
            cursor: pointer;
            margin-left: 5px;
        }
        button:hover {
            background: var(--vscode-button-hoverBackground);
        }
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .matches-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
            font-size: 12px;
        }
        .matches-table th {
            background: var(--vscode-editor-background);
            padding: 5px;
            text-align: left;
            border-bottom: 1px solid var(--vscode-panel-border);
        }
        .matches-table td {
            padding: 5px;
            border-bottom: 1px solid var(--vscode-panel-border);
        }
        .matches-table tr:hover {
            background: var(--vscode-list-hoverBackground);
        }
        .sequence {
            font-family: monospace;
            font-weight: bold;
        }
        .status {
            margin: 10px 0;
            font-style: italic;
            color: var(--vscode-descriptionForeground);
        }
        .window-control {
            margin: 15px 0;
            padding: 10px;
            background: var(--vscode-editor-background);
            border: 1px solid var(--vscode-panel-border);
            border-radius: 3px;
        }
        .window-control label {
            display: block;
            margin-bottom: 10px;
            font-weight: bold;
        }
        .slider-container {
            display: flex;
            flex-direction: column;
            gap: 8px;
            margin-bottom: 10px;
        }
        .slider-row {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .slider-group {
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 3px;
        }
        .slider-label {
            font-size: 11px;
            font-weight: bold;
        }
        input[type="range"] {
            width: 100%;
        }
        .slider-center {
            font-size: 11px;
            text-align: center;
            color: var(--vscode-descriptionForeground);
            padding: 0 10px;
            white-space: nowrap;
        }
        .plot-all-btn {
            width: 100%;
            margin-top: 10px;
            padding: 8px;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="search-box">
        <label>Motif Pattern (IUPAC):</label><br/>
        <div style="display: flex; gap: 8px; align-items: center;">
            <input type="text" id="motifInput" value="" placeholder="e.g., DRACH or CCA" style="width: 50%;"/>
            <button onclick="searchMotif()">Search</button>
            <label style="display: flex; align-items: center; gap: 5px; margin: 0;">
                <input type="checkbox" id="plusStrandOnly" checked style="width: auto; margin: 0;"/>
                + strand only
            </label>
        </div>
    </div>

    <div id="status" class="status"></div>

    <div id="windowControlContainer"></div>

    <div id="resultsContainer"></div>

    <script>
        const vscode = acquireVsCodeApi();
        let currentMotif = '';

        function searchMotif() {
            const motif = document.getElementById('motifInput').value.trim();
            if (!motif) {
                document.getElementById('status').textContent = 'Please enter a motif pattern';
                return;
            }
            currentMotif = motif;
            const plusStrandOnly = document.getElementById('plusStrandOnly').checked;
            const strand = plusStrandOnly ? '+' : 'both';
            document.getElementById('status').textContent = 'Searching...';
            vscode.postMessage({ type: 'searchMotif', motif: motif, strand: strand });
        }

        function plotAllMotifs() {
            const upstream = parseInt(document.getElementById('upstreamSlider').value) || 10;
            const downstream = parseInt(document.getElementById('downstreamSlider').value) || 10;

            vscode.postMessage({
                type: 'plotAllMotifs',
                motif: currentMotif,
                upstream: upstream,
                downstream: downstream
            });
        }

        function updateUpstreamValue() {
            const value = document.getElementById('upstreamSlider').value;
            document.getElementById('upstreamValue').textContent = value;
            document.getElementById('upstreamInput').value = value;
        }

        function updateDownstreamValue() {
            const value = document.getElementById('downstreamSlider').value;
            document.getElementById('downstreamValue').textContent = value;
            document.getElementById('downstreamInput').value = value;
        }

        function updateUpstreamFromInput() {
            const value = document.getElementById('upstreamInput').value;
            document.getElementById('upstreamSlider').value = value;
            document.getElementById('upstreamValue').textContent = value;
        }

        function updateDownstreamFromInput() {
            const value = document.getElementById('downstreamInput').value;
            document.getElementById('downstreamSlider').value = value;
            document.getElementById('downstreamValue').textContent = value;
        }

        window.addEventListener('message', event => {
            const message = event.data;
            if (message.type === 'updateMatches') {
                updateMatchesTable(message.matches, message.searching, message.motif);
            }
        });

        function updateMatchesTable(matches, searching, motif) {
            const statusEl = document.getElementById('status');
            const windowControlContainer = document.getElementById('windowControlContainer');
            const containerEl = document.getElementById('resultsContainer');

            if (searching) {
                statusEl.textContent = 'Searching...';
                windowControlContainer.innerHTML = '';
                return;
            }

            if (!matches || matches.length === 0) {
                statusEl.textContent = 'No matches found';
                containerEl.innerHTML = '';
                windowControlContainer.innerHTML = '';
                return;
            }

            currentMotif = motif;
            statusEl.textContent = \`Found \${matches.length} matches for \${motif}\`;

            // Add window control sliders
            let windowHtml = \`
                <div class="window-control">
                    <label>Window Size (bp):</label>
                    <div class="slider-container">
                        <div class="slider-row">
                            <div class="slider-group">
                                <span class="slider-label">Upstream: <span id="upstreamValue">10</span>bp</span>
                                <div style="display: flex; gap: 5px; align-items: center;">
                                    <input type="number" id="upstreamInput" min="0" max="50" value="10"
                                           style="width: 50px; padding: 2px;" oninput="updateUpstreamFromInput()" />
                                    <input type="range" id="upstreamSlider" min="0" max="50" value="10"
                                           style="flex: 1;" oninput="updateUpstreamValue()" />
                                </div>
                            </div>
                            <span class="slider-center">Motif<br/>Center</span>
                            <div class="slider-group">
                                <span class="slider-label">Downstream: <span id="downstreamValue">10</span>bp</span>
                                <div style="display: flex; gap: 5px; align-items: center;">
                                    <input type="range" id="downstreamSlider" min="0" max="50" value="10"
                                           style="flex: 1;" oninput="updateDownstreamValue()" />
                                    <input type="number" id="downstreamInput" min="0" max="50" value="10"
                                           style="width: 50px; padding: 2px;" oninput="updateDownstreamFromInput()" />
                                </div>
                            </div>
                        </div>
                    </div>
                    <button class="plot-all-btn" onclick="plotAllMotifs()">Plot All Matches</button>
                </div>
            \`;
            windowControlContainer.innerHTML = windowHtml;

            // Build table without window/action columns
            let html = '<table class="matches-table">';
            html += '<tr><th>#</th><th>Chrom</th><th>Position</th><th>Sequence</th><th>Strand</th></tr>';

            matches.forEach((match, index) => {
                html += \`<tr>
                    <td>\${index + 1}</td>
                    <td>\${match.chrom}</td>
                    <td>\${match.position.toLocaleString()}</td>
                    <td class="sequence">\${match.sequence}</td>
                    <td>\${match.strand}</td>
                </tr>\`;
            });

            html += '</table>';
            containerEl.innerHTML = html;
        }

        // Allow Enter key to search
        document.getElementById('motifInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                searchMotif();
            }
        });
    </script>
</body>
</html>`;
    }

    /**
     * Search for motif matches
     */
    private async searchMotif(motif: string, strand: string = 'both'): Promise<void> {
        if (!this.state.currentFastaFile) {
            vscode.window.showErrorMessage('No FASTA file loaded. Use "Open FASTA File" first.');
            return;
        }

        this._currentMotif = motif;
        this._searching = true;
        this.updateView();

        try {
            const api = await this.state.ensureBackgroundKernel();
            const matches = await api.searchMotif(
                this.state.currentFastaFile,
                motif,
                undefined, // region
                strand
            );
            this._matches = matches;
        } catch (error) {
            vscode.window.showErrorMessage(
                `Failed to search motif: ${error instanceof Error ? error.message : String(error)}`
            );
            this._matches = [];
        } finally {
            this._searching = false;
            this.updateView();
        }
    }

    /**
     * Generate aggregate plot for all motif matches
     */
    private async plotAllMotifs(
        motif: string,
        upstream: number,
        downstream: number
    ): Promise<void> {
        if (!this.state.currentFastaFile) {
            vscode.window.showErrorMessage('No FASTA file loaded.');
            return;
        }

        if (!this._matches || this._matches.length === 0) {
            vscode.window.showErrorMessage('No motif matches to plot. Search for a motif first.');
            return;
        }

        try {
            await vscode.commands.executeCommand('squiggy.plotMotifAggregateAll', {
                fastaFile: this.state.currentFastaFile,
                motif: motif,
                upstream: upstream,
                downstream: downstream,
            });
        } catch (error) {
            vscode.window.showErrorMessage(
                `Failed to plot motif aggregate: ${error instanceof Error ? error.message : String(error)}`
            );
        }
    }

    /**
     * Update the webview with current matches
     */
    private updateView(): void {
        if (!this._view) {
            return;
        }

        this._view.webview.postMessage({
            type: 'updateMatches',
            matches: this._matches,
            searching: this._searching,
            motif: this._currentMotif,
        });
    }

    /**
     * Public API to set matches (if needed from external commands)
     */
    public setMatches(matches: MotifMatch[]): void {
        this._matches = matches;
        this.updateView();
    }

    /**
     * Clear matches
     */
    public clear(): void {
        this._matches = [];
        this._currentMotif = 'DRACH';
        this._searching = false;
        this.updateView();
    }
}
