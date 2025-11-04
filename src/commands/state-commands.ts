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
    // Refresh reads - query Python backend for fresh data
    context.subscriptions.push(
        vscode.commands.registerCommand('squiggy.refreshReads', async () => {
            await refreshReadsFromBackend(state);
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

/**
 * Refresh reads list by querying Python backend
 * Re-fetches data from _squiggy_session instead of using cached data
 */
async function refreshReadsFromBackend(state: ExtensionState): Promise<void> {
    if (!state.usePositron || !state.squiggyAPI) {
        vscode.window.showWarningMessage(
            'Refresh requires Positron runtime with active Python kernel'
        );
        return;
    }

    try {
        // Show loading state
        state.readsViewPane?.setLoading(true, 'Refreshing read list...');

        // Check if POD5 is loaded in Python session
        const hasPod5 = await state.squiggyAPI.client.getVariable(
            '_squiggy_session.reader is not None'
        );

        if (!hasPod5) {
            vscode.window.showInformationMessage('No POD5 file loaded in Python session');
            state.readsViewPane?.setLoading(false);
            return;
        }

        // Check if BAM is loaded
        const hasBAM = await state.squiggyAPI.client.getVariable(
            '_squiggy_session.bam_path is not None'
        );

        if (hasBAM) {
            // BAM mode: grouped by reference with lazy loading
            await refreshWithBAM(state);
        } else {
            // POD5-only mode: flat list with pagination
            await refreshPOD5Only(state);
        }

        vscode.window.showInformationMessage('Read list refreshed successfully');
    } catch (error) {
        vscode.window.showErrorMessage(`Failed to refresh reads: ${error}`);
    } finally {
        state.readsViewPane?.setLoading(false);
    }
}

/**
 * Refresh POD5-only mode (flat list)
 */
async function refreshPOD5Only(state: ExtensionState): Promise<void> {
    if (!state.squiggyAPI) {
        return;
    }

    // Get total read count
    const totalReads = await state.squiggyAPI.client.getVariable('len(_squiggy_session.read_ids)');

    // Reset pagination context
    state.pod5LoadContext = {
        currentOffset: 1000, // First batch loaded
        pageSize: 500,
        totalReads: totalReads as number,
    };

    // Fetch first 1000 read IDs (same as initial load)
    const readIds = await state.squiggyAPI.getReadIds(0, 1000);

    // Display in reads view
    state.readsViewPane?.setReads(readIds);
}

/**
 * Refresh BAM mode (grouped by reference, lazy loading)
 */
async function refreshWithBAM(state: ExtensionState): Promise<void> {
    if (!state.squiggyAPI) {
        return;
    }

    // Get references from Python session
    const references = await state.squiggyAPI.getReferences();

    // Build reference list with read counts
    const referenceList: { referenceName: string; readCount: number }[] = [];

    for (const ref of references) {
        const readCount = await state.squiggyAPI.client.getVariable(
            `len(_squiggy_session.ref_mapping.get('${ref.replace(/'/g, "\\'")}', []))`
        );
        referenceList.push({
            referenceName: ref,
            readCount: readCount as number,
        });
    }

    // Display in reads view (lazy loading mode)
    state.readsViewPane?.setReferencesOnly(referenceList);
}
