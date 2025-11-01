/**
 * File Panel Webview View - React-based
 *
 * Provides file management with sortable table layout
 */

import * as vscode from 'vscode';
import * as path from 'path';
import { FileItem } from '../types/squiggy-files-types';

export class FilePanelProvider implements vscode.WebviewViewProvider {
    public static readonly viewType = 'squiggyFilePanel';

    private _view?: vscode.WebviewView;
    private _files: FileItem[] = [];

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
                this._updateView();
            }
        });

        // Handle messages from the webview
        webviewView.webview.onDidReceiveMessage(async (data) => {
            switch (data.type) {
                case 'openFile':
                    if (data.fileType === 'POD5') {
                        vscode.commands.executeCommand('squiggy.openPOD5');
                    } else if (data.fileType === 'BAM') {
                        vscode.commands.executeCommand('squiggy.openBAM');
                    }
                    break;
                case 'closeFile':
                    if (data.fileType === 'POD5') {
                        vscode.commands.executeCommand('squiggy.closePOD5');
                    } else if (data.fileType === 'BAM') {
                        vscode.commands.executeCommand('squiggy.closeBAM');
                    }
                    break;
                case 'ready':
                    // Webview is ready, send initial state
                    this._updateView();
                    break;
            }
        });
    }

    /**
     * Set POD5 file info
     */
    public setPOD5(fileInfo: { path: string; numReads: number; size: number }) {
        console.log('FilePanelProvider.setPOD5 called with:', fileInfo);
        // Remove existing POD5 file
        this._files = this._files.filter((f) => f.type !== 'POD5');

        // Add new POD5 file
        this._files.push({
            path: fileInfo.path,
            filename: path.basename(fileInfo.path),
            type: 'POD5',
            size: fileInfo.size,
            sizeFormatted: formatFileSize(fileInfo.size),
            numReads: fileInfo.numReads,
        });

        console.log('FilePanelProvider._files after setPOD5:', this._files);
        this._updateView();
    }

    /**
     * Set BAM file info
     */
    public setBAM(fileInfo: {
        path: string;
        numReads: number;
        numRefs: number;
        size: number;
        hasMods: boolean;
        hasEvents: boolean;
    }) {
        // Remove existing BAM file
        this._files = this._files.filter((f) => f.type !== 'BAM');

        // Add new BAM file
        this._files.push({
            path: fileInfo.path,
            filename: path.basename(fileInfo.path),
            type: 'BAM',
            size: fileInfo.size,
            sizeFormatted: formatFileSize(fileInfo.size),
            numReads: fileInfo.numReads,
            numRefs: fileInfo.numRefs,
            hasMods: fileInfo.hasMods,
            hasEvents: fileInfo.hasEvents,
        });

        this._updateView();
    }

    /**
     * Clear POD5 file
     */
    public clearPOD5() {
        this._files = this._files.filter((f) => f.type !== 'POD5');
        this._updateView();
    }

    /**
     * Clear BAM file
     */
    public clearBAM() {
        this._files = this._files.filter((f) => f.type !== 'BAM');
        this._updateView();
    }

    private _updateView() {
        if (this._view) {
            console.log('FilePanelProvider: Sending updateFiles with', this._files.length, 'files');
            this._view.webview.postMessage({
                type: 'updateFiles',
                files: this._files,
            });
        } else {
            console.log('FilePanelProvider: No view to update');
        }
    }

    private _getHtmlForWebview(webview: vscode.Webview): string {
        // Get URIs for script
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
                   font-src ${webview.cspSource};
                   script-src 'nonce-${nonce}';">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Squiggy File Explorer</title>
    <style>
        body, html {
            margin: 0;
            padding: 0;
            width: 100%;
            font-family: var(--vscode-font-family);
            font-size: var(--vscode-font-size);
            color: var(--vscode-foreground);
            background-color: var(--vscode-editor-background);
        }
        #root {
            width: 100%;
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

function formatFileSize(bytes: number): string {
    if (bytes < 1024) {
        return `${bytes} B`;
    } else if (bytes < 1024 * 1024) {
        return `${(bytes / 1024).toFixed(1)} KB`;
    } else if (bytes < 1024 * 1024 * 1024) {
        return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    } else {
        return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
    }
}
