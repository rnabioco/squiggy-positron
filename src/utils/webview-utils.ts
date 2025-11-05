/**
 * Shared utilities for webview creation and management
 *
 * Consolidates duplicate HTML generation and CSP nonce logic
 * from individual webview providers.
 */

import * as vscode from 'vscode';

/**
 * Generate a cryptographically random nonce for Content Security Policy
 * @returns 32-character random string
 */
export function getNonce(): string {
    let text = '';
    const possible = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    for (let i = 0; i < 32; i++) {
        text += possible.charAt(Math.floor(Math.random() * possible.length));
    }
    return text;
}

/**
 * Generate HTML for a React-based webview
 *
 * All webviews use the same bundled React app (build/webview.js) and
 * mount to #root with common styles.
 *
 * @param webview - VSCode webview instance
 * @param extensionUri - Extension root URI
 * @param title - HTML page title
 * @returns Complete HTML string
 */
export function getReactWebviewHtml(
    webview: vscode.Webview,
    extensionUri: vscode.Uri,
    title: string
): string {
    const scriptUri = webview.asWebviewUri(
        vscode.Uri.joinPath(extensionUri, 'build', 'webview.js')
    );

    const nonce = getNonce();

    return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="Content-Security-Policy"
          content="default-src 'none';
                   style-src ${webview.cspSource} 'unsafe-inline';
                   font-src ${webview.cspSource};
                   script-src 'nonce-${nonce}';
                   connect-src 'self';">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${title}</title>
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

/**
 * Configure common webview options
 *
 * @param extensionUri - Extension root URI
 * @returns WebviewOptions with enableScripts and localResourceRoots
 */
export function getWebviewOptions(extensionUri: vscode.Uri): vscode.WebviewOptions {
    return {
        enableScripts: true,
        localResourceRoots: [extensionUri],
    };
}
