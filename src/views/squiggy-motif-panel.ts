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
                case 'plotMotif':
                    await this.plotMotif(message.motif, message.matchIndex, message.window);
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
        input {
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
        .plot-btn {
            padding: 2px 8px;
            font-size: 11px;
        }
        .status {
            margin: 10px 0;
            font-style: italic;
            color: var(--vscode-descriptionForeground);
        }
        .window-input {
            width: 50px;
            padding: 2px;
            margin-right: 5px;
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

    <div id="resultsContainer"></div>

    <script>
        const vscode = acquireVsCodeApi();

        function searchMotif() {
            const motif = document.getElementById('motifInput').value.trim();
            if (!motif) {
                document.getElementById('status').textContent = 'Please enter a motif pattern';
                return;
            }
            const plusStrandOnly = document.getElementById('plusStrandOnly').checked;
            const strand = plusStrandOnly ? '+' : 'both';
            document.getElementById('status').textContent = 'Searching...';
            vscode.postMessage({ type: 'searchMotif', motif: motif, strand: strand });
        }

        function plotMotif(motif, matchIndex) {
            const window = parseInt(document.getElementById('window-' + matchIndex).value) || 50;
            vscode.postMessage({
                type: 'plotMotif',
                motif: motif,
                matchIndex: matchIndex,
                window: window
            });
        }

        window.addEventListener('message', event => {
            const message = event.data;
            if (message.type === 'updateMatches') {
                updateMatchesTable(message.matches, message.searching, message.motif);
            }
        });

        function updateMatchesTable(matches, searching, motif) {
            const statusEl = document.getElementById('status');
            const containerEl = document.getElementById('resultsContainer');

            if (searching) {
                statusEl.textContent = 'Searching...';
                return;
            }

            if (!matches || matches.length === 0) {
                statusEl.textContent = 'No matches found';
                containerEl.innerHTML = '';
                return;
            }

            statusEl.textContent = \`Found \${matches.length} matches for \${motif}\`;

            let html = '<table class="matches-table">';
            html += '<tr><th>#</th><th>Chrom</th><th>Position</th><th>Sequence</th><th>Strand</th><th>Window</th><th>Action</th></tr>';

            matches.forEach((match, index) => {
                html += \`<tr>
                    <td>\${index + 1}</td>
                    <td>\${match.chrom}</td>
                    <td>\${match.position.toLocaleString()}</td>
                    <td class="sequence">\${match.sequence}</td>
                    <td>\${match.strand}</td>
                    <td><input type="number" id="window-\${index}" class="window-input" value="50" min="10" max="500"/> bp</td>
                    <td><button class="plot-btn" onclick="plotMotif('\${motif}', \${index})">Plot</button></td>
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
     * Generate motif aggregate plot
     */
    private async plotMotif(motif: string, matchIndex: number, window: number): Promise<void> {
        if (!this.state.currentFastaFile) {
            vscode.window.showErrorMessage('No FASTA file loaded.');
            return;
        }

        try {
            await vscode.commands.executeCommand('squiggy.plotMotifAggregate', {
                fastaFile: this.state.currentFastaFile,
                motif: motif,
                matchIndex: matchIndex,
                window: window,
            });
        } catch (error) {
            vscode.window.showErrorMessage(
                `Failed to plot motif: ${error instanceof Error ? error.message : String(error)}`
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
