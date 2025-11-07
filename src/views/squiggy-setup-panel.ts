/**
 * Squiggy Setup Panel
 *
 * Displays installation instructions when squiggy Python package is not installed.
 * Provides options to install automatically, view manual instructions, or retry detection.
 */

import * as vscode from 'vscode';
import { BaseWebviewProvider } from './base-webview-provider';
import { ExtensionState } from '../state/extension-state';
import { SetupPanelIncomingMessage } from '../types/messages';

export class SquiggySetupPanelProvider extends BaseWebviewProvider {
    public static readonly viewType = 'squiggySetupPanel';

    constructor(
        extensionUri: vscode.Uri,
        private readonly state: ExtensionState,
        private readonly context: vscode.ExtensionContext
    ) {
        super(extensionUri);
    }

    protected getTitle(): string {
        return 'Squiggy Setup';
    }

    protected async handleMessage(message: SetupPanelIncomingMessage): Promise<void> {
        switch (message.type) {
            case 'ready':
                this.updateView();
                break;

            case 'install':
                await this.handleInstall();
                break;

            case 'manual':
                await this.showManualInstructions();
                break;

            case 'retry':
                await this.handleRetry();
                break;
        }
    }

    protected updateView(): void {
        if (!this._view) {
            return;
        }

        // Send installation status to webview
        this.postMessage({
            type: 'updateStatus',
            installed: false,
            message:
                'The squiggy Python package is not installed in your active Python environment.',
        });
    }

    /**
     * Handle automatic installation request
     */
    private async handleInstall(): Promise<void> {
        if (!this.state.packageManager) {
            vscode.window.showErrorMessage('Package manager not available');
            return;
        }

        const extensionPath = this.context.extensionPath;
        const success = await this.state.packageManager.installSquiggyWithProgress(extensionPath);

        if (success) {
            // Trigger extension reload
            await vscode.commands.executeCommand('squiggy.checkInstallation');
        }
    }

    /**
     * Show manual installation instructions
     */
    private async showManualInstructions(): Promise<void> {
        if (!this.state.packageManager) {
            vscode.window.showErrorMessage('Package manager not available');
            return;
        }

        await this.state.packageManager.showManualInstallationGuide(this.context.extensionPath);
    }

    /**
     * Retry installation detection
     */
    private async handleRetry(): Promise<void> {
        await vscode.commands.executeCommand('squiggy.checkInstallation');
    }

    /**
     * Override resolveWebviewView to provide custom HTML for setup panel
     */
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

        // Set custom HTML for setup panel (not React-based)
        webviewView.webview.html = this.getSetupHtml();

        // Handle messages from webview
        webviewView.webview.onDidReceiveMessage(async (message: SetupPanelIncomingMessage) => {
            try {
                await this.handleMessage(message);
            } catch (error) {
                const err = error instanceof Error ? error : new Error(String(error));
                this.handleMessageError(err, message as any);
            }
        });
    }

    /**
     * Generate HTML for setup panel
     */
    private getSetupHtml(): string {
        return `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Squiggy Setup</title>
    <style>
        body {
            padding: 16px;
            font-family: var(--vscode-font-family);
            color: var(--vscode-foreground);
            line-height: 1.6;
        }

        .warning-icon {
            color: var(--vscode-editorWarning-foreground);
            font-size: 48px;
            text-align: center;
            margin-bottom: 16px;
        }

        h2 {
            color: var(--vscode-foreground);
            margin-top: 0;
            text-align: center;
        }

        .message {
            background-color: var(--vscode-editorWidget-background);
            border: 1px solid var(--vscode-editorWidget-border);
            border-radius: 4px;
            padding: 12px;
            margin-bottom: 16px;
        }

        .info-box {
            background-color: var(--vscode-textBlockQuote-background);
            border-left: 3px solid var(--vscode-textLink-foreground);
            padding: 12px 16px;
            margin-top: 16px;
            margin-bottom: 16px;
        }

        .info-box p {
            margin: 8px 0;
        }

        .info-box p:first-child {
            margin-top: 0;
        }

        .info-box p:last-child {
            margin-bottom: 0;
        }

        .button-group {
            display: flex;
            flex-direction: column;
            gap: 8px;
            margin-top: 16px;
        }

        button {
            background-color: var(--vscode-button-background);
            color: var(--vscode-button-foreground);
            border: none;
            padding: 8px 16px;
            cursor: pointer;
            border-radius: 2px;
            font-size: 13px;
            text-align: center;
        }

        button:hover {
            background-color: var(--vscode-button-hoverBackground);
        }

        button.primary-large {
            background-color: var(--vscode-button-background);
            color: var(--vscode-button-foreground);
            padding: 12px 20px;
            font-size: 15px;
            font-weight: 600;
            border: 2px solid var(--vscode-focusBorder);
        }

        button.primary-large:hover {
            background-color: var(--vscode-button-hoverBackground);
            border-color: var(--vscode-button-hoverBackground);
        }

        button.secondary {
            background-color: var(--vscode-button-secondaryBackground);
            color: var(--vscode-button-secondaryForeground);
        }

        button.secondary:hover {
            background-color: var(--vscode-button-secondaryHoverBackground);
        }

        .note {
            color: var(--vscode-descriptionForeground);
            font-size: 12px;
            margin-top: 16px;
            font-style: italic;
        }
    </style>
</head>
<body>
    <div class="warning-icon">⚠️</div>
    <h2>Squiggy Python Package Required</h2>

    <div class="message">
        The Squiggy extension requires the <code>squiggy</code> Python package to be installed
        in your active Python environment.
    </div>

    <div class="button-group">
        <button class="primary-large" onclick="manual()">📖 View Detailed Instructions</button>
        <button class="secondary" onclick="retry()">Check Again</button>
    </div>

    <div class="info-box">
        <p>Click "📖 View Detailed Instructions" above to see step-by-step commands for installing with <strong>uv</strong> (modern, fast package manager).</p>
        <p>Each command can be clicked to copy to your clipboard.</p>
    </div>

    <div class="note">
        After installing, click "Check Again" to activate the extension.
    </div>

    <script>
        const vscode = acquireVsCodeApi();

        function manual() {
            vscode.postMessage({ type: 'manual' });
        }

        function retry() {
            vscode.postMessage({ type: 'retry' });
        }

        // Signal ready
        vscode.postMessage({ type: 'ready' });
    </script>
</body>
</html>
        `;
    }
}
