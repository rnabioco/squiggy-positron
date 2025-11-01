/**
 * Reads View Pane - WebviewViewProvider for React-based reads panel
 *
 * Provides a multi-column table view of reads from POD5 files,
 * optionally grouped by reference (when BAM loaded).
 *
 * Follows Positron Variables panel pattern with ViewPane container.
 */

import * as vscode from 'vscode';
import { ReadItem, ReadListItem, ReferenceGroupItem } from '../types/squiggy-reads-types';

export class ReadsViewPane implements vscode.WebviewViewProvider {
    public static readonly viewType = 'squiggyReadList';

    private _view?: vscode.WebviewView;
    private _hasReferences: boolean = false;
    private _readItems: ReadListItem[] = [];
    private _referenceToReads?: Map<string, ReadItem[]>;

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

        // Handle visibility changes - restore state when view becomes visible
        webviewView.onDidChangeVisibility(() => {
            if (webviewView.visible) {
                this._restoreState();
            }
        });

        // Handle messages from the webview
        webviewView.webview.onDidReceiveMessage((data) => {
            switch (data.type) {
                case 'plotRead':
                    vscode.commands.executeCommand('squiggy.plotRead', data.readId);
                    break;
                case 'plotAggregate':
                    vscode.commands.executeCommand('squiggy.plotAggregate', data.referenceName);
                    break;
                case 'selectRead':
                    // Handle selection (could emit event for other panels to react)
                    break;
                case 'toggleReference':
                    // Expansion state handled in webview
                    break;
                case 'updateColumnWidths':
                    // Persist column widths to workspace state
                    this._saveColumnWidths(data.nameWidth, data.detailsWidth);
                    break;
                case 'loadMore':
                    // Request more reads from Python backend
                    this._loadMoreReads();
                    break;
            }
        });

        // Send initial state if we have data
        this._restoreState();
    }

    /**
     * Set reads without reference grouping (flat list)
     */
    public setReads(readIds: string[]): void {
        this._hasReferences = false;
        this._readItems = readIds.map((readId) => ({
            type: 'read' as const,
            readId,
            indentLevel: 0,
        }));

        if (this._view) {
            this._view.webview.postMessage({
                type: 'setReads',
                items: this._readItems,
                hasReferences: false,
            });
        }
    }

    /**
     * Set reads grouped by reference (BAM loaded)
     */
    public setReadsGrouped(referenceToReads: Map<string, ReadItem[]>): void {
        this._hasReferences = true;
        this._referenceToReads = referenceToReads;

        // Flatten into list with reference headers (initially all collapsed)
        const items: ReadListItem[] = [];
        for (const [referenceName, reads] of referenceToReads.entries()) {
            // Add reference group header
            items.push({
                type: 'reference',
                referenceName,
                readCount: reads.length,
                isExpanded: false,
                indentLevel: 0,
            } as ReferenceGroupItem);

            // Reads will be shown when expanded (handled in webview)
        }

        this._readItems = items;

        if (this._view) {
            this._view.webview.postMessage({
                type: 'setReadsGrouped',
                items: items,
                referenceToReads: Array.from(referenceToReads.entries()),
                hasReferences: true,
            });
        }
    }

    /**
     * Refresh the view
     */
    public refresh(): void {
        if (this._view) {
            this._view.webview.postMessage({
                type: 'refresh',
            });
        }
    }

    /**
     * Filter reads by search text
     */
    public filterReads(searchText: string): void {
        if (this._view) {
            this._view.webview.postMessage({
                type: 'updateSearch',
                searchText,
            });
        }
    }

    /**
     * Restore state to webview (called when view becomes visible)
     */
    private _restoreState(): void {
        if (!this._view) {
            return;
        }

        // Re-send data based on current mode
        if (this._hasReferences && this._referenceToReads) {
            this._view.webview.postMessage({
                type: 'setReadsGrouped',
                items: this._readItems,
                referenceToReads: Array.from(this._referenceToReads.entries()),
                hasReferences: true,
            });
        } else if (this._readItems.length > 0) {
            this._view.webview.postMessage({
                type: 'setReads',
                items: this._readItems,
                hasReferences: false,
            });
        }
    }

    private _saveColumnWidths(nameWidth: number, detailsWidth: number): void {
        // TODO: Persist to workspace state
        console.log(`Column widths updated: name=${nameWidth}, details=${detailsWidth}`);
    }

    private async _loadMoreReads(): Promise<void> {
        // TODO: Request more reads from Python backend via PositronRuntime
        console.log('Loading more reads...');
    }

    private _getHtmlForWebview(webview: vscode.Webview): string {
        // Get URIs for script and style
        const scriptUri = webview.asWebviewUri(
            vscode.Uri.joinPath(this._extensionUri, 'build', 'webview.js')
        );

        // Use a nonce for CSP
        const nonce = getNonce();

        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="Content-Security-Policy"
          content="default-src 'none';
                   style-src ${webview.cspSource} 'unsafe-inline';
                   script-src 'nonce-${nonce}';">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Squiggy Reads</title>
    <style>
        body, html {
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
            overflow: hidden;
            font-family: var(--vscode-font-family);
            font-size: var(--vscode-font-size);
            color: var(--vscode-foreground);
            background-color: var(--vscode-editor-background);
        }
        #root {
            width: 100%;
            height: 100%;
        }
    </style>
</head>
<body>
    <div id="root"></div>
    <script nonce="${nonce}" src="${scriptUri}"></script>
</body>
</html>`;
    }
}

function getNonce(): string {
    let text = '';
    const possible = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    for (let i = 0; i < 32; i++) {
        text += possible.charAt(Math.floor(Math.random() * possible.length));
    }
    return text;
}
