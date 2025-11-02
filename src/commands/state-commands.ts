/**
 * State Commands
 *
 * Handles refresh and clear state commands.
 * Extracted from extension.ts to improve modularity.
 */

import * as vscode from 'vscode';
import { ExtensionState } from '../state/extension-state';

/**
 * Register state management commands
 */
export function registerStateCommands(
    context: vscode.ExtensionContext,
    state: ExtensionState
): void {
    // Refresh reads
    context.subscriptions.push(
        vscode.commands.registerCommand('squiggy.refreshReads', () => {
            state.readsViewPane?.refresh();
        })
    );

    // Clear state (useful after kernel restart)
    context.subscriptions.push(
        vscode.commands.registerCommand('squiggy.clearState', async () => {
            await state.clearAll();

            vscode.window.showInformationMessage(
                'Squiggy state cleared. Load new files to continue.'
            );
        })
    );
}
