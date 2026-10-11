/**
 * Shared utilities for webview creation and management
 *
 * Consolidates duplicate HTML generation and CSP nonce logic
 * from individual webview providers.
 */

import { randomBytes } from 'crypto';
import * as vscode from 'vscode';

/**
 * Generate a cryptographically random nonce for Content Security Policy
 *
 * Uses Node's CSPRNG (the extension host runs in Node) rather than
 * Math.random(), which is not suitable for security tokens.
 *
 * @returns 32-character hex string (128 bits of entropy)
 */
export function getNonce(): string {
    return randomBytes(16).toString('hex');
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
                   script-src 'nonce-${nonce}';">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${title}</title>
    <style>
        *, *::before, *::after {
            box-sizing: border-box;
        }
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
            display: flex;
            flex-direction: column;
            overflow-y: auto;
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
        // Only the bundled webview script is loaded from disk; restrict access
        // to the build output rather than the whole extension directory.
        localResourceRoots: [vscode.Uri.joinPath(extensionUri, 'build')],
    };
}
