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

            case 'selectInterpreter':
                await this.selectPythonInterpreter();
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
     * Handle installation instructions request
     */
    private async handleInstall(): Promise<void> {
        // Deprecated - kept for backward compatibility but does nothing
    }

    /**
     * Open Python interpreter selection dialog
     */
    private async selectPythonInterpreter(): Promise<void> {
        await vscode.commands.executeCommand('python.setInterpreter');
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

        .warning-box {
            background-color: var(--vscode-inputValidation-warningBackground);
            border: 1px solid var(--vscode-inputValidation-warningBorder);
            border-radius: 4px;
            padding: 16px;
            margin-top: 16px;
            margin-bottom: 24px;
        }

        .warning-box h3 {
            margin-top: 0;
            margin-bottom: 12px;
            color: var(--vscode-foreground);
            font-size: 14px;
        }

        .warning-box ol {
            margin: 0;
            padding-left: 20px;
        }

        .warning-box li {
            margin: 10px 0;
            line-height: 1.5;
        }

        .warning-box code {
            background-color: var(--vscode-textCodeBlock-background);
            padding: 6px 10px;
            border-radius: 3px;
            font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace;
            font-size: 13px;
            display: block;
            margin-top: 6px;
            overflow-x: auto;
        }

        .command-wrapper {
            display: flex;
            align-items: stretch;
            gap: 6px;
            margin-top: 6px;
        }

        .command-wrapper code {
            flex: 1;
            margin-top: 0;
        }

        .copy-button {
            background: var(--vscode-button-secondaryBackground);
            border: none;
            color: var(--vscode-button-secondaryForeground);
            cursor: pointer;
            padding: 6px 10px;
            font-size: 14px;
            border-radius: 3px;
            white-space: nowrap;
            flex-shrink: 0;
            align-self: stretch;
            width: auto;
            margin-top: 0;
        }

        .copy-button:hover {
            background-color: var(--vscode-button-secondaryHoverBackground);
        }

        .copy-button.copied {
            background-color: var(--vscode-testing-iconPassed);
            color: #ffffff;
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
            font-weight: 500;
            display: block;
            margin-top: 8px;
            width: 100%;
        }

        button:hover {
            background-color: var(--vscode-button-hoverBackground);
        }

        button.select-interpreter {
            background-color: var(--vscode-button-background);
            color: var(--vscode-button-foreground);
        }

        button.select-interpreter:hover {
            background-color: var(--vscode-button-hoverBackground);
        }

        button.check-again {
            background-color: #d4a500;
            color: #000000;
            font-weight: 600;
        }

        button.check-again:hover {
            background-color: #e6b800;
        }
    </style>
</head>
<body>
    <div class="warning-icon">⚠️</div>
    <h2>Squiggy Python Package Required</h2>

    <div class="message">
        The Squiggy extension requires the <code>squiggy-positron</code> Python package to be installed
        in your active Python environment.
    </div>

    <div class="warning-box">
        <h3>📋 Setup Instructions</h3>
        <ol>
            <li>
                Navigate to a project folder for virtual environment installation
            </li>
            <li>
                (Optional) Install uv if not already installed
                <div class="command-wrapper">
                    <code>curl -LsSf https://astral.sh/uv/install.sh | sh</code>
                    <button class="copy-button" onclick="copyCommand('curl -LsSf https://astral.sh/uv/install.sh | sh', this)" title="Copy to clipboard">📋</button>
                </div>
            </li>
            <li>
                Create virtual environment
                <div class="command-wrapper">
                    <code>uv venv</code>
                    <button class="copy-button" onclick="copyCommand('uv venv', this)" title="Copy to clipboard">📋</button>
                </div>
            </li>
            <li>
                Install Squiggy
                <div class="command-wrapper">
                    <code>uv pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ squiggy-positron</code>
                    <button class="copy-button" onclick="copyCommand('uv pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ squiggy-positron', this)" title="Copy to clipboard">📋</button>
                </div>
            </li>
            <li>
                Select Python interpreter from new environment
                <div style="margin-top: 6px; font-size: 12px; color: var(--vscode-descriptionForeground);">
                    (Optional) Activate in terminal first: <code style="font-size: 11px; padding: 2px 4px;">source .venv/bin/activate</code>
                </div>
                <div style="margin-top: 6px; font-size: 12px;">
                    Click button below and select the interpreter from your workspace (look for <strong>.venv</strong> path or <strong>"uv:"</strong> label):
                </div>
                <button class="select-interpreter" onclick="selectInterpreter()">🐍 Select Python Interpreter</button>
            </li>
            <li>
                If setup panel doesn't disappear automatically, click:
                <button class="check-again" onclick="retry()">✓ Check Again</button>
            </li>
        </ol>
    </div>

    <script>
        const vscode = acquireVsCodeApi();

        function selectInterpreter() {
            vscode.postMessage({ type: 'selectInterpreter' });
        }

        function retry() {
            vscode.postMessage({ type: 'retry' });
        }

        function copyCommand(command, button) {
            // Use the Clipboard API to copy the command
            navigator.clipboard.writeText(command).then(() => {
                // Visual feedback: change emoji temporarily
                button.textContent = '✓';
                button.classList.add('copied');

                setTimeout(() => {
                    button.textContent = '📋';
                    button.classList.remove('copied');
                }, 1500);
            }).catch(err => {
                console.error('Failed to copy command:', err);
                button.textContent = '✗';
                setTimeout(() => {
                    button.textContent = '📋';
                }, 1500);
            });
        }

        // Signal ready
        vscode.postMessage({ type: 'ready' });
    </script>
</body>
</html>
        `;
    }
}
