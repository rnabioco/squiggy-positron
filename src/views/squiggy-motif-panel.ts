/**
 * Motif Search Panel Webview View
 *
 * Provides UI for searching motifs in FASTA files and generating aggregate plots
 */

import * as vscode from 'vscode';
import { MotifMatch } from '../types/motif-types';

export class MotifSearchPanelProvider implements vscode.WebviewViewProvider {
    public static readonly viewType = 'squiggyMotifSearch';

    private _view?: vscode.WebviewView;
    private _matches: MotifMatch[] = [];
    private _currentMotif: string = 'DRACH';
    private _searching: boolean = false;

    constructor(
        private readonly extensionUri: vscode.Uri,
        private readonly state: any // ExtensionState
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
            display: grid;
            grid-template-columns: auto 1fr auto 1fr auto;
            gap: 10px;
            align-items: center;
            margin-bottom: 10px;
        }
        .slider-label {
            font-size: 11px;
            white-space: nowrap;
        }
        input[type="range"] {
            width: 100%;
        }
        .slider-center {
            font-size: 11px;
            text-align: center;
            color: var(--vscode-descriptionForeground);
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
        <input type="text" id="motifInput" value="DRACH" placeholder="e.g., DRACH, YGCY"/>
        <button onclick="searchMotif()">Search</button>
        <br/>
        <label style="margin-top: 8px; display: inline-block;">
            <input type="checkbox" id="plusStrandOnly" checked style="width: auto; margin-right: 5px;"/>
            + strand only
        </label>
    </div>

    <div id="status" class="status"></div>

    <div id="windowControlContainer"></div>

    <div id="resultsContainer"></div>

    <script>
        const vscode = acquireVsCodeApi();
        let currentMotif = 'DRACH';

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
        }

        function updateDownstreamValue() {
            const value = document.getElementById('downstreamSlider').value;
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
                        <span class="slider-label">Upstream: <span id="upstreamValue">10</span>bp</span>
                        <input type="range" id="upstreamSlider" min="0" max="100" value="10" oninput="updateUpstreamValue()" />
                        <span class="slider-center">← Motif Center →</span>
                        <input type="range" id="downstreamSlider" min="0" max="100" value="10" oninput="updateDownstreamValue()" />
                        <span class="slider-label">Downstream: <span id="downstreamValue">10</span>bp</span>
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
            if (this.state.usePositron && this.state.squiggyAPI) {
                const matches = await this.state.squiggyAPI.searchMotif(
                    this.state.currentFastaFile,
                    motif,
                    undefined, // region
                    strand
                );
                this._matches = matches;
            } else {
                vscode.window.showErrorMessage(
                    'Motif search requires Positron runtime. Please use Positron IDE.'
                );
                this._matches = [];
            }
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
