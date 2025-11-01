/**
 * Read Search Webview View
 *
 * Provides a search box for filtering reads in the tree view
 */

import * as vscode from 'vscode';

export class ReadSearchViewProvider implements vscode.WebviewViewProvider {
    public static readonly viewType = 'squiggyReadSearch';

    private _view?: vscode.WebviewView;

    // Event emitter for when search text changes
    private _onDidChangeSearchText = new vscode.EventEmitter<string>();
    public readonly onDidChangeSearchText = this._onDidChangeSearchText.event;

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
                case 'searchChanged':
                    this._onDidChangeSearchText.fire(data.value);
                    break;
            }
        });
    }

    private _getHtmlForWebview(_webview: vscode.Webview) {
        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Read Search</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            padding: 4px 8px;
            font-family: var(--vscode-font-family);
            font-size: var(--vscode-font-size);
            overflow: hidden;
        }
        .search-container {
            display: flex;
            align-items: center;
            gap: 4px;
        }
        .search-icon {
            color: var(--vscode-foreground);
            opacity: 0.7;
            flex-shrink: 0;
            font-size: 12px;
            line-height: 1;
        }
        input[type="text"] {
            flex: 1;
            padding: 2px 6px;
            background: var(--vscode-input-background);
            color: var(--vscode-input-foreground);
            border: 1px solid var(--vscode-input-border);
            outline: none;
            font-family: var(--vscode-font-family);
            font-size: 12px;
            line-height: 1.4;
            min-height: 20px;
        }
        input[type="text"]:focus {
            border-color: var(--vscode-focusBorder);
        }
        input[type="text"]::placeholder {
            color: var(--vscode-input-placeholderForeground);
        }
        .clear-button {
            background: transparent;
            border: none;
            color: var(--vscode-foreground);
            cursor: pointer;
            padding: 0 4px;
            opacity: 0.7;
            flex-shrink: 0;
            display: none;
            font-size: 12px;
            line-height: 1;
        }
        .clear-button:hover {
            opacity: 1;
        }
        .clear-button.visible {
            display: block;
        }
    </style>
</head>
<body>
    <div class="search-container">
        <span class="search-icon">🔍</span>
        <input
            type="text"
            id="searchInput"
            placeholder="Filter by reference or read ID..."
            autocomplete="off"
        />
        <button id="clearButton" class="clear-button" title="Clear filter">✕</button>
    </div>

    <script>
        const vscode = acquireVsCodeApi();
        const searchInput = document.getElementById('searchInput');
        const clearButton = document.getElementById('clearButton');

        // Send search updates to extension
        searchInput.addEventListener('input', (e) => {
            const value = e.target.value;
            vscode.postMessage({
                type: 'searchChanged',
                value: value
            });

            // Show/hide clear button
            if (value) {
                clearButton.classList.add('visible');
            } else {
                clearButton.classList.remove('visible');
            }
        });

        // Clear button
        clearButton.addEventListener('click', () => {
            searchInput.value = '';
            clearButton.classList.remove('visible');
            vscode.postMessage({
                type: 'searchChanged',
                value: ''
            });
            searchInput.focus();
        });

        // Focus input on load
        searchInput.focus();
    </script>
</body>
</html>`;
    }
}
