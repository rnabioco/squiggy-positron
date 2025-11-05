/**
 * Session Management Commands
 *
 * Commands for saving, restoring, exporting, importing, and loading demo sessions
 */

import * as vscode from 'vscode';
import { ExtensionState } from '../state/extension-state';
import { SessionStateManager } from '../state/session-state-manager';

/**
 * Register all session management commands
 */
export function registerSessionCommands(
    context: vscode.ExtensionContext,
    state: ExtensionState
): void {
    context.subscriptions.push(
        vscode.commands.registerCommand('squiggy.saveSession', () =>
            saveSessionCommand(state, context)
        ),
        vscode.commands.registerCommand('squiggy.restoreSession', () =>
            restoreSessionCommand(state, context)
        ),
        vscode.commands.registerCommand('squiggy.exportSession', () =>
            exportSessionCommand(state, context)
        ),
        vscode.commands.registerCommand('squiggy.importSession', () =>
            importSessionCommand(state, context)
        ),
        vscode.commands.registerCommand('squiggy.clearSession', () => clearSessionCommand(context)),
        vscode.commands.registerCommand('squiggy.loadDemoSession', () =>
            loadDemoSessionCommand(state, context)
        )
    );
}

/**
 * Save current session to workspace state
 */
export async function saveSessionCommand(
    extensionState: ExtensionState,
    context: vscode.ExtensionContext
): Promise<void> {
    try {
        // Check if there's anything to save
        const sessionState = extensionState.toSessionState();

        if (Object.keys(sessionState.samples).length === 0) {
            vscode.window.showWarningMessage('No data loaded to save. Load a POD5 file first.');
            return;
        }

        // Prompt for optional session name
        const sessionName = await vscode.window.showInputBox({
            prompt: 'Enter a name for this session (optional)',
            placeHolder: 'My Analysis Session',
        });

        if (sessionName !== undefined) {
            // User didn't cancel, add name to session
            if (sessionName.trim()) {
                sessionState.sessionName = sessionName.trim();
            }

            // Save to workspace state
            await SessionStateManager.saveSession(sessionState, context);

            // Get workspace name for better feedback
            const workspaceName = vscode.workspace.name || 'this workspace';

            vscode.window.showInformationMessage(
                `Session ${sessionName ? `"${sessionName}" ` : ''}saved to ${workspaceName} workspace state`
            );
        }
    } catch (error) {
        vscode.window.showErrorMessage(`Failed to save session: ${error}`);
    }
}

/**
 * Restore session from workspace state
 */
export async function restoreSessionCommand(
    extensionState: ExtensionState,
    context: vscode.ExtensionContext
): Promise<void> {
    try {
        // Load session from workspace state
        const savedSession = await SessionStateManager.loadSession(context);

        if (!savedSession) {
            vscode.window.showWarningMessage('No saved session found for this workspace.');
            return;
        }

        // Check for unsaved changes
        const currentState = extensionState.toSessionState();
        const hasUnsavedChanges = await SessionStateManager.hasUnsavedChanges(
            currentState,
            context
        );

        if (hasUnsavedChanges && Object.keys(currentState.samples).length > 0) {
            const response = await vscode.window.showWarningMessage(
                'Current session has unsaved changes. Continue restoring?',
                { modal: true },
                'Restore',
                'Cancel'
            );

            if (response !== 'Restore') {
                return;
            }
        }

        // Show progress while restoring
        await vscode.window.withProgress(
            {
                location: vscode.ProgressLocation.Notification,
                title: 'Restoring session...',
                cancellable: false,
            },
            async (progress) => {
                progress.report({ message: 'Loading files...' });
                await extensionState.fromSessionState(savedSession, context);
            }
        );
    } catch (error) {
        vscode.window.showErrorMessage(`Failed to restore session: ${error}`);
    }
}

/**
 * Export session to JSON file
 */
export async function exportSessionCommand(
    extensionState: ExtensionState,
    context: vscode.ExtensionContext
): Promise<void> {
    try {
        // Check if there's anything to export
        const sessionState = extensionState.toSessionState();

        if (Object.keys(sessionState.samples).length === 0) {
            vscode.window.showWarningMessage('No data loaded to export. Load a POD5 file first.');
            return;
        }

        // Prompt for file location
        const uri = await vscode.window.showSaveDialog({
            defaultUri: vscode.Uri.file('squiggy-session.json'),
            filters: {
                'Squiggy Session': ['json'],
                'All Files': ['*'],
            },
            title: 'Export Session',
        });

        if (!uri) {
            return; // User cancelled
        }

        // Export session with full metadata
        await SessionStateManager.exportSession(sessionState, uri.fsPath, context);

        vscode.window.showInformationMessage(`Session exported to ${uri.fsPath}`);
    } catch (error) {
        vscode.window.showErrorMessage(`Failed to export session: ${error}`);
    }
}

/**
 * Import session from JSON file
 */
export async function importSessionCommand(
    extensionState: ExtensionState,
    context: vscode.ExtensionContext
): Promise<void> {
    try {
        // Prompt for file location
        const uris = await vscode.window.showOpenDialog({
            canSelectMany: false,
            filters: {
                'Squiggy Session': ['json'],
                'All Files': ['*'],
            },
            title: 'Import Session',
        });

        if (!uris || uris.length === 0) {
            return; // User cancelled
        }

        // Check for unsaved changes
        const currentState = extensionState.toSessionState();
        const hasData = Object.keys(currentState.samples).length > 0;

        if (hasData) {
            const response = await vscode.window.showWarningMessage(
                'Current session will be replaced. Continue?',
                { modal: true },
                'Import',
                'Cancel'
            );

            if (response !== 'Import') {
                return;
            }
        }

        // Import session
        const importedSession = await SessionStateManager.importSession(uris[0].fsPath);

        // Show progress while restoring
        await vscode.window.withProgress(
            {
                location: vscode.ProgressLocation.Notification,
                title: 'Importing session...',
                cancellable: false,
            },
            async (progress) => {
                progress.report({ message: 'Loading files...' });
                await extensionState.fromSessionState(importedSession, context);
            }
        );
    } catch (error) {
        vscode.window.showErrorMessage(`Failed to import session: ${error}`);
    }
}

/**
 * Clear saved session from workspace state
 */
export async function clearSessionCommand(context: vscode.ExtensionContext): Promise<void> {
    try {
        const response = await vscode.window.showWarningMessage(
            'Clear saved session? This will not affect currently loaded files.',
            { modal: true },
            'Clear',
            'Cancel'
        );

        if (response !== 'Clear') {
            return;
        }

        await SessionStateManager.clearSession(context);
        vscode.window.showInformationMessage('Saved session cleared');
    } catch (error) {
        vscode.window.showErrorMessage(`Failed to clear session: ${error}`);
    }
}

/**
 * Load demo session with packaged test data
 */
export async function loadDemoSessionCommand(
    extensionState: ExtensionState,
    context: vscode.ExtensionContext
): Promise<void> {
    try {
        // Check for unsaved changes
        const currentState = extensionState.toSessionState();
        const hasData = Object.keys(currentState.samples).length > 0;

        if (hasData) {
            const response = await vscode.window.showWarningMessage(
                'Loading demo session will replace current data. Continue?',
                { modal: true },
                'Load Demo',
                'Cancel'
            );

            if (response !== 'Load Demo') {
                return;
            }
        }

        // Show progress while loading
        await vscode.window.withProgress(
            {
                location: vscode.ProgressLocation.Notification,
                title: 'Loading demo session...',
                cancellable: false,
            },
            async (progress) => {
                progress.report({ message: 'Loading yeast tRNA reads...' });
                await extensionState.loadDemoSession(context);
            }
        );
    } catch (error) {
        vscode.window.showErrorMessage(`Failed to load demo session: ${error}`);
    }
}
