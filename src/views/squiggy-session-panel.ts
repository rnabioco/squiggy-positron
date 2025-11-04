/**
 * Session Manager Panel
 *
 * Displays current session state and provides session management actions via React UI
 */

import * as vscode from 'vscode';
import { BaseWebviewProvider } from './base-webview-provider';
import { SessionStateManager } from '../state/session-state-manager';
import { ExtensionState } from '../state/extension-state';
import { SessionPanelIncomingMessage, UpdateSessionMessage } from '../types/messages';

export class SessionPanelProvider extends BaseWebviewProvider {
    public static readonly viewType = 'squiggySessionPanel';

    constructor(
        extensionUri: vscode.Uri,
        private context: vscode.ExtensionContext,
        private state: ExtensionState
    ) {
        super(extensionUri);
    }

    protected getTitle(): string {
        return 'Squiggy Session Manager';
    }

    protected async handleMessage(message: SessionPanelIncomingMessage): Promise<void> {
        switch (message.type) {
            case 'ready':
                this.updateView();
                break;
            case 'loadDemo':
                await vscode.commands.executeCommand('squiggy.loadDemoSession');
                this.updateView();
                break;
            case 'save':
                await vscode.commands.executeCommand('squiggy.saveSession');
                this.updateView();
                break;
            case 'restore':
                await vscode.commands.executeCommand('squiggy.restoreSession');
                this.updateView();
                break;
            case 'export':
                await vscode.commands.executeCommand('squiggy.exportSession');
                break;
            case 'import':
                await vscode.commands.executeCommand('squiggy.importSession');
                this.updateView();
                break;
            case 'clear':
                await vscode.commands.executeCommand('squiggy.clearSession');
                this.updateView();
                break;
        }
    }

    protected updateView(): void {
        if (!this._view) {
            return;
        }

        // Get current session state
        const currentState = this.state.toSessionState();
        const hasSamples = Object.keys(currentState.samples).length > 0;

        // Check if there's a saved session
        SessionStateManager.loadSession(this.context).then((savedSession) => {
            const hasSavedSession = savedSession !== null;

            const message: UpdateSessionMessage = {
                type: 'updateSession',
                hasSamples,
                hasSavedSession,
                sampleCount: Object.keys(currentState.samples).length,
                sampleNames: Object.keys(currentState.samples),
            };

            this.postMessage(message);
        });
    }
}
