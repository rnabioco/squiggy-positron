/**
 * File Panel Webview View
 *
 * Provides file management controls in the sidebar
 */

import * as vscode from 'vscode';
import * as path from 'path';

export class FilePanelProvider implements vscode.WebviewViewProvider {
    public static readonly viewType = 'squiggyFilePanel';

    private _view?: vscode.WebviewView;
    private _pod5File?: string;
    private _bamFile?: string;
    private _pod5Info?: { numReads: number; fileSize: string };
    private _bamInfo?: { numReads: number; fileSize: string };

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
                case 'openPOD5':
                    vscode.commands.executeCommand('squiggy.openPOD5');
                    break;
                case 'openBAM':
                    vscode.commands.executeCommand('squiggy.openBAM');
                    break;
            }
        });
    }

    /**
     * Update POD5 file info display
     */
    public setPOD5Info(filePath: string, numReads: number, fileSize: string) {
        this._pod5File = filePath;
        this._pod5Info = { numReads, fileSize };
        this._updateView();
    }

    /**
     * Update BAM file info display
     */
    public setBAMInfo(filePath: string, numReads: number, fileSize: string) {
        this._bamFile = filePath;
        this._bamInfo = { numReads, fileSize };
        this._updateView();
    }

    private _updateView() {
        if (this._view) {
            this._view.webview.postMessage({
                type: 'updateFileInfo',
                pod5File: this._pod5File,
                pod5Info: this._pod5Info,
                bamFile: this._bamFile,
                bamInfo: this._bamInfo,
            });
        }
    }

    private _getHtmlForWebview(webview: vscode.Webview) {
        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>File Management</title>
    <style>
        body {
            padding: 10px;
            font-family: var(--vscode-font-family);
            font-size: var(--vscode-font-size);
            color: var(--vscode-foreground);
        }
        .button-container {
            display: flex;
            gap: 8px;
            margin-bottom: 16px;
        }
        button {
            flex: 1;
            padding: 6px 12px;
            background: var(--vscode-button-background);
            color: var(--vscode-button-foreground);
            border: none;
            cursor: pointer;
            font-size: 0.9em;
            border-radius: 2px;
        }
        button:hover {
            background: var(--vscode-button-hoverBackground);
        }
        .file-section {
            margin-bottom: 16px;
            padding: 10px;
            background: var(--vscode-editor-background);
            border: 1px solid var(--vscode-panel-border);
            border-radius: 4px;
        }
        .file-section.empty {
            opacity: 0.6;
        }
        .file-label {
            font-weight: bold;
            font-size: 0.85em;
            color: var(--vscode-descriptionForeground);
            margin-bottom: 4px;
        }
        .file-path {
            font-size: 0.85em;
            word-break: break-all;
            margin-bottom: 6px;
            font-family: var(--vscode-editor-font-family);
        }
        .file-meta {
            font-size: 0.8em;
            color: var(--vscode-descriptionForeground);
            margin-top: 4px;
        }
        .file-meta-item {
            margin-right: 12px;
        }
    </style>
</head>
<body>
    <!-- File Open Buttons -->
    <div class="button-container">
        <button id="openPOD5">Open POD5</button>
        <button id="openBAM">Open BAM</button>
    </div>

    <!-- POD5 File Section -->
    <div class="file-section" id="pod5Section">
        <div class="file-label">POD5 File</div>
        <div class="file-path" id="pod5Path">No file loaded</div>
        <div class="file-meta" id="pod5Meta"></div>
    </div>

    <!-- BAM File Section -->
    <div class="file-section" id="bamSection">
        <div class="file-label">BAM File</div>
        <div class="file-path" id="bamPath">No file loaded</div>
        <div class="file-meta" id="bamMeta"></div>
    </div>

    <script>
        const vscode = acquireVsCodeApi();

        // Get elements
        const openPOD5Btn = document.getElementById('openPOD5');
        const openBAMBtn = document.getElementById('openBAM');
        const pod5Section = document.getElementById('pod5Section');
        const pod5Path = document.getElementById('pod5Path');
        const pod5Meta = document.getElementById('pod5Meta');
        const bamSection = document.getElementById('bamSection');
        const bamPath = document.getElementById('bamPath');
        const bamMeta = document.getElementById('bamMeta');

        // Button click handlers
        openPOD5Btn.addEventListener('click', () => {
            vscode.postMessage({ type: 'openPOD5' });
        });

        openBAMBtn.addEventListener('click', () => {
            vscode.postMessage({ type: 'openBAM' });
        });

        // Receive updates from extension
        window.addEventListener('message', event => {
            const message = event.data;
            switch (message.type) {
                case 'updateFileInfo':
                    // Update POD5 file info
                    if (message.pod5File) {
                        const fileName = message.pod5File.split('/').pop();
                        pod5Path.textContent = fileName;
                        pod5Section.classList.remove('empty');

                        if (message.pod5Info) {
                            pod5Meta.innerHTML =
                                '<span class="file-meta-item">Reads: ' + message.pod5Info.numReads + '</span>' +
                                '<span class="file-meta-item">Size: ' + message.pod5Info.fileSize + '</span>';
                        }
                    } else {
                        pod5Path.textContent = 'No file loaded';
                        pod5Meta.textContent = '';
                        pod5Section.classList.add('empty');
                    }

                    // Update BAM file info
                    if (message.bamFile) {
                        const fileName = message.bamFile.split('/').pop();
                        bamPath.textContent = fileName;
                        bamSection.classList.remove('empty');

                        if (message.bamInfo) {
                            bamMeta.innerHTML =
                                '<span class="file-meta-item">Reads: ' + message.bamInfo.numReads + '</span>' +
                                '<span class="file-meta-item">Size: ' + message.bamInfo.fileSize + '</span>';
                        }
                    } else {
                        bamPath.textContent = 'No file loaded';
                        bamMeta.textContent = '';
                        bamSection.classList.add('empty');
                    }
                    break;
            }
        });
    </script>
</body>
</html>`;
    }
}
