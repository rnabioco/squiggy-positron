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
 * @param onSessionChange Optional callback to run when session changes (e.g., re-check installation)
 */
export function registerKernelListeners(
    context: vscode.ExtensionContext,
    state: ExtensionState,
    onSessionChange?: () => Promise<void>
): void {
    try {
        // eslint-disable-next-line @typescript-eslint/no-var-requires, @typescript-eslint/no-require-imports
        const positron = require('positron');

        // Helper function to clear extension state and re-check installation
        const clearExtensionState = async (reason: string) => {
            await state.clearAll();
            logger.debug(`[KernelListeners] ${reason}, state cleared`);

            // Re-check installation after session change if callback provided
            if (onSessionChange) {
                try {
                    await onSessionChange();
                } catch (error) {
                    logger.error('Error in session change callback', error);
                }
            }
        };

        // Listen for session changes (kernel switches)
        context.subscriptions.push(
            positron.runtime.onDidChangeForegroundSession(async (sessionId: string | undefined) => {
                // With dedicated kernel architecture, we should NOT clear state when
                // the user switches between their interactive console and our kernel.
                // The dedicated kernel maintains state independently.
                if (state.kernelManager) {
                    const dedicatedSessionId = state.kernelManager.getSessionId();
                    if (dedicatedSessionId) {
                        // Dedicated kernel exists - don't clear state for foreground changes
                        // State is only cleared when the dedicated kernel itself restarts/exits
                        logger.debug(
                            `[KernelListeners] Foreground session changed to ${sessionId}, dedicated kernel (${dedicatedSessionId}) still active - NOT clearing state`
                        );
                        return;
                    }
                }

                // Only clear state if there's no dedicated kernel (legacy behavior)
                await clearExtensionState('Python session changed (no dedicated kernel)');
            })
        );

        // Also listen to runtime state changes on the current session
        // This catches kernel restarts within the same session
        const setupSessionListeners = async () => {
            try {
                const session = await positron.runtime.getForegroundSession();
                logger.debug(
                    '[KernelListeners] Setting up session listeners, session:',
                    session?.metadata.sessionId
                );

                if (session && session.onDidChangeRuntimeState) {
                    context.subscriptions.push(
                        session.onDidChangeRuntimeState((runtimeState: string) => {
                            // Only log important state changes (not idle/busy cycles)
                            if (runtimeState === 'restarting' || runtimeState === 'exited') {
                                logger.debug(
                                    '[KernelListeners] Kernel state changed to:',
                                    runtimeState
                                );
                                clearExtensionState(`Kernel ${runtimeState}`);
                            } else if (runtimeState === 'ready' && onSessionChange) {
                                // Kernel restarted and is now ready - re-check installation
                                logger.info('Kernel ready after restart - rechecking installation');
                                onSessionChange().catch((error) => {
                                    logger.error(
                                        'Error rechecking installation after restart',
                                        error
                                    );
                                });
                            }
                        })
                    );
                    logger.debug('[KernelListeners] Successfully attached runtime state listener');
                } else {
                    logger.debug(
                        '[KernelListeners] No session or no onDidChangeRuntimeState event available'
                    );
                }
            } catch (error) {
                logger.error('[KernelListeners] Error setting up session listeners:', error);
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
