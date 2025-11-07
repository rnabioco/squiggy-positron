/**
 * Kernel Event Listeners
 *
 * Handles Positron runtime session changes (kernel restarts, session switches)
 * and clears extension state appropriately.
 */

import * as vscode from 'vscode';
import { ExtensionState } from '../state/extension-state';
import { logger } from '../utils/logger';

/**
 * Register listeners for Positron kernel/session events
 *
 * Monitors session changes and kernel restarts, clearing extension state
 * when the Python environment changes.
 *
 * @param context VSCode extension context
 * @param state Extension state manager
 */
export function registerKernelListeners(
    context: vscode.ExtensionContext,
    state: ExtensionState
): void {
    if (!state.usePositron) {
        // Only register listeners when using Positron runtime
        return;
    }

    try {
        // eslint-disable-next-line @typescript-eslint/no-var-requires, @typescript-eslint/no-require-imports
        const positron = require('positron');

        // Helper function to clear extension state
        const clearExtensionState = async (reason: string) => {
            await state.clearAll();
            logger.debug(`Squiggy: ${reason}, state cleared`);
        };

        // Listen for session changes (kernel switches)
        context.subscriptions.push(
            positron.runtime.onDidChangeForegroundSession((_sessionId: string | undefined) => {
                clearExtensionState('Python session changed');
            })
        );

        // Also listen to runtime state changes on the current session
        // This catches kernel restarts within the same session
        const setupSessionListeners = async () => {
            try {
                const session = await positron.runtime.getForegroundSession();
                logger.debug(
                    'Squiggy: Setting up session listeners, session:',
                    session?.metadata.sessionId
                );

                if (session && session.onDidChangeRuntimeState) {
                    context.subscriptions.push(
                        session.onDidChangeRuntimeState((runtimeState: string) => {
                            // Only log important state changes (not idle/busy cycles)
                            if (runtimeState === 'restarting' || runtimeState === 'exited') {
                                logger.debug('Squiggy: Kernel state changed to:', runtimeState);
                                clearExtensionState(`Kernel ${runtimeState}`);
                            }
                        })
                    );
                    logger.debug('Squiggy: Successfully attached runtime state listener');
                } else {
                    logger.debug(
                        'Squiggy: No session or no onDidChangeRuntimeState event available'
                    );
                }
            } catch (error) {
                logger.error('Squiggy: Error setting up session listeners:', error);
            }
        };
        setupSessionListeners();

        // Re-setup listeners when session changes
        context.subscriptions.push(
            positron.runtime.onDidChangeForegroundSession(async () => {
                await setupSessionListeners();
            })
        );
    } catch (_error) {
        // Positron API not available - not running in Positron
    }
}
