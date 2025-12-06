/**
 * Squiggy Kernel Manager
 *
 * Manages a dedicated console Python kernel for Squiggy extension operations.
 * This creates a separate kernel session to isolate extension state from the
 * user's foreground workspace.
 *
 * Architecture:
 * - Extension UI → Dedicated console kernel (separate session)
 * - Notebook API → User's foreground kernel (full programmatic access)
 *
 * Note: Uses Console session mode because Positron's public API doesn't expose
 * Background session mode through positron.runtime.startLanguageRuntime().
 */

import * as vscode from 'vscode';
import * as positron from 'positron';
import { logger } from '../utils/logger';
import { RuntimeClient } from './runtime-client-interface';

export enum SquiggyKernelState {
    Uninitialized = 'uninitialized',
    Starting = 'starting',
    Ready = 'ready',
    Error = 'error',
    Restarting = 'restarting',
}

// Key for storing session ID in workspace state
const SESSION_ID_KEY = 'squiggy.dedicatedKernelSessionId';

export class SquiggyKernelManager implements RuntimeClient {
    private session: positron.LanguageRuntimeSession | undefined;
    // For reconnected sessions (from window reload), we only have the session ID
    private reconnectedSessionId: string | undefined;
    private state: SquiggyKernelState = SquiggyKernelState.Uninitialized;
    private stateChangeEmitter = new vscode.EventEmitter<SquiggyKernelState>();
    private extensionPath: string;
    private isDisposed = false;
    private context: vscode.ExtensionContext | undefined;

    readonly onDidChangeState = this.stateChangeEmitter.event;

    // Session name used to identify our dedicated kernel
    private static readonly SESSION_NAME = 'Squiggy Dedicated Kernel';

    constructor(extensionPath: string, context?: vscode.ExtensionContext) {
        this.extensionPath = extensionPath;
        this.context = context;
    }

    /**
     * Save session ID to workspace state for reconnection after reload
     */
    private saveSessionId(sessionId: string): void {
        if (this.context) {
            this.context.workspaceState.update(SESSION_ID_KEY, sessionId);
            logger.debug(`Saved session ID to workspace state: ${sessionId}`);
        }
    }

    /**
     * Get saved session ID from workspace state
     */
    private getSavedSessionId(): string | undefined {
        return this.context?.workspaceState.get<string>(SESSION_ID_KEY);
    }

    /**
     * Clear saved session ID from workspace state
     */
    private clearSavedSessionId(): void {
        if (this.context) {
            this.context.workspaceState.update(SESSION_ID_KEY, undefined);
            logger.debug('Cleared saved session ID from workspace state');
        }
    }

    /**
     * Get current kernel state
     */
    getState(): SquiggyKernelState {
        return this.state;
    }

    /**
     * Get the dedicated kernel session ID (if running)
     */
    getSessionId(): string | undefined {
        return this.session?.metadata.sessionId ?? this.reconnectedSessionId;
    }

    /**
     * Start the background Python kernel
     * First tries to reconnect to an existing session, then creates a new one if needed
     */
    async start(): Promise<void> {
        if (this.isDisposed) {
            throw new Error('Kernel manager has been disposed');
        }

        if (this.session) {
            logger.info('Squiggy kernel already running');
            return;
        }

        this.setState(SquiggyKernelState.Starting);

        // Try to reconnect to an existing session first
        if (await this.tryReconnectToExistingSession()) {
            return;
        }

        logger.info('Starting new Squiggy dedicated kernel...');

        try {
            // Poll for squiggy venv to appear in registered runtimes
            // Positron may not have discovered it yet after creation
            let runtime: positron.LanguageRuntimeMetadata | undefined;
            const maxAttempts = 10;
            const pollIntervalMs = 500;

            for (let attempt = 1; attempt <= maxAttempts; attempt++) {
                const allRuntimes = await positron.runtime.getRegisteredRuntimes();
                const pythonRuntimes = allRuntimes.filter(
                    (r: positron.LanguageRuntimeMetadata) => r.languageId === 'python'
                );

                if (attempt === 1) {
                    logger.info(
                        `Available Python runtimes: ${pythonRuntimes.map((r: positron.LanguageRuntimeMetadata) => `${r.runtimeName} @ ${r.runtimePath}`).join(', ')}`
                    );
                }

                // Look for squiggy venv
                runtime = pythonRuntimes.find((r: positron.LanguageRuntimeMetadata) =>
                    r.runtimePath.includes('.venvs/squiggy')
                );

                if (runtime) {
                    logger.info(`Found squiggy venv on attempt ${attempt}`);
                    break;
                }

                if (attempt < maxAttempts) {
                    logger.debug(
                        `Squiggy venv not found, polling... (attempt ${attempt}/${maxAttempts})`
                    );
                    await new Promise((resolve) => setTimeout(resolve, pollIntervalMs));
                }
            }

            // Fall back to preferred runtime if squiggy venv not found after polling
            if (!runtime) {
                logger.warning('Squiggy venv not found after polling, using preferred runtime');
                runtime = await positron.runtime.getPreferredRuntime('python');
            }

            if (!runtime) {
                throw new Error('No Python runtime available');
            }

            const runtimeId = runtime.runtimeId;
            const runtimeName = `${runtime.runtimeName} (${runtime.runtimeVersion})`;

            logger.info(`Using Python runtime: ${runtimeName} @ ${runtime.runtimePath}`);

            // Start a dedicated console session for Squiggy
            // NOTE: positron.runtime.startLanguageRuntime() only supports Console and Notebook modes
            // Background mode is not exposed through the public API
            // This creates a Console session which is separate from the foreground session
            this.session = await positron.runtime.startLanguageRuntime(
                runtimeId,
                SquiggyKernelManager.SESSION_NAME,
                undefined // notebookUri - undefined means Console mode
            );

            if (!this.session) {
                throw new Error('Failed to create dedicated session');
            }

            logger.info(`Dedicated session created: ${this.session.metadata.sessionId}`);

            // Listen for session state changes
            this.session.onDidChangeRuntimeState(this.handleRuntimeStateChange.bind(this));
            this.session.onDidEndSession(this.handleSessionEnd.bind(this));

            // Wait for kernel to be ready before executing code
            logger.info('Waiting for dedicated kernel to be ready...');
            await this.waitForReady();
            logger.info('Dedicated kernel is ready, setting up environment...');

            // Set up PYTHONPATH to include squiggy package
            await this.setupPythonPath();

            // Verify squiggy is importable
            await this.verifySquiggyAvailable();

            // State should already be Ready from handleRuntimeStateChange
            // Don't set it manually to avoid interfering with event flow
            logger.info('Squiggy dedicated kernel ready');

            // Save session ID for reconnection after window reload
            this.saveSessionId(this.session.metadata.sessionId);
        } catch (error) {
            this.setState(SquiggyKernelState.Error);
            logger.error(`Failed to start dedicated kernel: ${error}`);
            throw error;
        }
    }

    /**
     * Try to reconnect to an existing Squiggy session (e.g., after window reload)
     * @returns true if successfully reconnected, false otherwise
     */
    private async tryReconnectToExistingSession(): Promise<boolean> {
        try {
            // Get saved session ID from workspace state
            const savedSessionId = this.getSavedSessionId();
            if (!savedSessionId) {
                logger.debug('No saved session ID found in workspace state');
                return false;
            }

            logger.debug(`Looking for saved session ID: ${savedSessionId}`);

            // First, check if sessions are already available
            let sessions = await positron.runtime.getActiveSessions();
            logger.debug(
                `Active sessions (${sessions.length}): ${sessions.map((s) => `${s.metadata.sessionName ?? 'unnamed'} [${s.metadata.sessionId}] (${s.metadata.sessionMode})`).join(', ')}`
            );

            // If no sessions yet, wait for Positron to restore them (event-driven)
            if (sessions.length === 0) {
                logger.debug('No sessions yet, waiting for session restoration...');
                sessions = await this.waitForSessionRestoration(savedSessionId);
            }

            // Find our saved session
            const existingSession = sessions.find((s) => s.metadata.sessionId === savedSessionId);

            if (!existingSession) {
                logger.debug(`Saved session ${savedSessionId} no longer exists, clearing state`);
                this.clearSavedSessionId();
                return false;
            }

            logger.info(`Found existing Squiggy session: ${existingSession.metadata.sessionId}`);

            // Store the session ID for reconnected session
            this.reconnectedSessionId = existingSession.metadata.sessionId;

            // Set up PYTHONPATH and verify squiggy is available
            await this.setupPythonPathForReconnected();
            await this.verifySquiggyAvailableForReconnected();

            this.setState(SquiggyKernelState.Ready);
            logger.info('Successfully reconnected to existing Squiggy session');
            return true;
        } catch (error) {
            logger.warning(`Failed to reconnect to existing session: ${error}`);
            this.reconnectedSessionId = undefined;
            return false;
        }
    }

    /**
     * Wait for Positron to restore sessions after window reload.
     * Uses event subscription rather than polling.
     */
    private async waitForSessionRestoration(
        savedSessionId: string
    ): Promise<positron.BaseLanguageRuntimeSession[]> {
        const timeoutMs = 5000;

        return new Promise((resolve) => {
            let resolved = false;

            // Set up timeout - if no sessions appear, return empty array
            const timeout = setTimeout(async () => {
                if (!resolved) {
                    resolved = true;
                    disposable.dispose();
                    logger.debug('Timeout waiting for session restoration');
                    // Final check before giving up
                    const sessions = await positron.runtime.getActiveSessions();
                    resolve(sessions);
                }
            }, timeoutMs);

            // Listen for foreground session changes - this fires when Positron restores sessions
            const disposable = positron.runtime.onDidChangeForegroundSession(async (sessionId) => {
                if (resolved) return;

                logger.debug(`Foreground session changed: ${sessionId ?? 'none'}`);

                // Check if our saved session is now available
                const sessions = await positron.runtime.getActiveSessions();
                const found = sessions.find((s) => s.metadata.sessionId === savedSessionId);

                if (found || sessions.length > 0) {
                    // Either found our session, or sessions exist (ours might not have survived)
                    resolved = true;
                    clearTimeout(timeout);
                    disposable.dispose();
                    logger.debug(
                        `Sessions restored (${sessions.length}): ${sessions.map((s) => s.metadata.sessionId).join(', ')}`
                    );
                    resolve(sessions);
                }
            });
        });
    }

    /**
     * Set up PYTHONPATH for a reconnected session (uses sessionId instead of session object)
     */
    private async setupPythonPathForReconnected(): Promise<void> {
        const sessionId = this.reconnectedSessionId;
        if (!sessionId) {
            return;
        }

        const squiggyPath = this.extensionPath;

        try {
            await positron.runtime.executeCode(
                'python',
                `
import sys
import os

# Add extension path to sys.path if not already present (temporary variable)
_squiggy_temp_path = ${JSON.stringify(squiggyPath)}
if _squiggy_temp_path not in sys.path:
    sys.path.insert(0, _squiggy_temp_path)
del _squiggy_temp_path
`,
                false,
                true,
                positron.RuntimeCodeExecutionMode.Silent,
                positron.RuntimeErrorBehavior.Continue,
                undefined,
                sessionId
            );
            logger.info(`Added ${squiggyPath} to PYTHONPATH in reconnected session`);
        } catch (error) {
            logger.warning(`Failed to set up PYTHONPATH for reconnected session: ${error}`);
        }
    }

    /**
     * Verify squiggy package is available in reconnected session
     */
    private async verifySquiggyAvailableForReconnected(): Promise<void> {
        const sessionId = this.reconnectedSessionId;
        if (!sessionId) {
            throw new Error('No reconnected session ID');
        }

        try {
            await positron.runtime.executeCode(
                'python',
                `
import sys

# Force fresh import by clearing cached modules (ensures code changes are picked up)
_to_remove = [key for key in sys.modules if key.startswith('squiggy')]
for _key in _to_remove:
    del sys.modules[_key]

# Now import fresh from disk
import squiggy
assert hasattr(squiggy, 'load_pod5'), "squiggy.load_pod5 not found"
assert hasattr(squiggy, 'load_bam'), "squiggy.load_bam not found"
assert hasattr(squiggy, 'plot_read'), "squiggy.plot_read not found"
`,
                false,
                true,
                positron.RuntimeCodeExecutionMode.Silent,
                positron.RuntimeErrorBehavior.Continue,
                undefined,
                sessionId
            );
            logger.info('Squiggy package verified in reconnected session');
        } catch (error) {
            throw new Error(`Squiggy package not available in reconnected session: ${error}`);
        }
    }

    /**
     * Restart the dedicated kernel
     */
    async restart(): Promise<void> {
        if (this.isDisposed) {
            throw new Error('Kernel manager has been disposed');
        }

        logger.info('Restarting Squiggy dedicated kernel...');
        this.setState(SquiggyKernelState.Restarting);

        try {
            const sessionId = this.getSessionId();
            if (sessionId) {
                // Use Positron API to restart the session
                await positron.runtime.restartSession(sessionId);

                // For sessions with full object (has event listeners), wait for ready
                if (this.session) {
                    logger.info('Waiting for restarted kernel to be ready...');
                    await this.waitForReady();
                    logger.info('Restarted kernel is ready, setting up environment...');
                    await this.setupPythonPath();
                    await this.verifySquiggyAvailable();
                } else {
                    // For reconnected sessions (only have sessionId), wait a bit then verify
                    logger.info('Waiting for reconnected session to restart...');
                    await new Promise((resolve) => setTimeout(resolve, 2000));
                    await this.setupPythonPathForReconnected();
                    await this.verifySquiggyAvailableForReconnected();
                    this.setState(SquiggyKernelState.Ready);
                }

                logger.info('Squiggy dedicated kernel restarted');
            } else {
                // No session - start fresh
                await this.start();
            }
        } catch (error) {
            logger.error(`Failed to restart dedicated kernel: ${error}`);
            this.setState(SquiggyKernelState.Error);

            // Try to recover by starting fresh
            this.session = undefined;
            this.reconnectedSessionId = undefined;
            await this.start();
        }
    }

    /**
     * Execute code in the dedicated kernel
     *
     * @param code Python code to execute
     * @param mode Execution mode (default: Silent)
     * @returns Promise that resolves with execution result
     */
    async execute(
        code: string,
        mode: positron.RuntimeCodeExecutionMode = positron.RuntimeCodeExecutionMode.Silent
    ): Promise<Record<string, unknown>> {
        const sessionId = this.getSessionId();
        if (!sessionId) {
            throw new Error('Dedicated kernel not started');
        }

        if (this.state !== SquiggyKernelState.Ready) {
            throw new Error(`Dedicated kernel not ready (state: ${this.state})`);
        }

        try {
            // Execute code using Positron runtime API with explicit sessionId
            return await positron.runtime.executeCode(
                'python',
                code,
                false, // focus
                true, // allowIncomplete
                mode,
                positron.RuntimeErrorBehavior.Continue,
                undefined, // observer
                sessionId
            );
        } catch (error) {
            logger.error(`Dedicated kernel execution failed: ${error}`);
            throw error;
        }
    }

    /**
     * Execute code silently (no console output)
     * @param code Python code to execute
     * @param _enableRetry Ignored (for RuntimeClient interface compatibility)
     */
    async executeSilent(code: string, _enableRetry?: boolean): Promise<void> {
        await this.execute(code, positron.RuntimeCodeExecutionMode.Silent);
    }

    /**
     * Get a variable value from the dedicated kernel
     * @param varName Python variable name or expression
     * @param _enableRetry Ignored (for RuntimeClient interface compatibility)
     */
    async getVariable(varName: string, _enableRetry?: boolean): Promise<unknown> {
        const sessionId = this.getSessionId();
        if (!sessionId) {
            throw new Error('Dedicated kernel not started');
        }

        const tempVar = '_squiggy_temp_' + Math.random().toString(36).substr(2, 9);

        try {
            await this.executeSilent(`
import json
${tempVar} = json.dumps(${varName})
`);

            const [[variable]] = await positron.runtime.getSessionVariables(sessionId, [[tempVar]]);

            await this.executeSilent(`
if '${tempVar}' in globals():
    del ${tempVar}
`);

            if (!variable) {
                throw new Error(`Variable ${varName} not found`);
            }

            const jsonString = variable.display_value;
            const cleaned = jsonString.replace(/^['"]|['"]$/g, '');
            return JSON.parse(cleaned);
        } catch (error) {
            await this.executeSilent(
                `
if '${tempVar}' in globals():
    del ${tempVar}
`
            ).catch(() => {});
            throw new Error(`Failed to get variable ${varName}: ${error}`);
        }
    }

    /**
     * Shutdown the dedicated kernel
     */
    async shutdown(): Promise<void> {
        const sessionId = this.getSessionId();
        if (sessionId) {
            logger.info('Shutting down Squiggy dedicated kernel...');
            try {
                await positron.runtime.deleteSession(sessionId);
            } catch (error) {
                logger.error(`Error shutting down kernel: ${error}`);
            }
            this.session = undefined;
            this.reconnectedSessionId = undefined;
        }
    }

    /**
     * Dispose of resources
     */
    dispose(): void {
        this.isDisposed = true;
        this.shutdown();
        this.stateChangeEmitter.dispose();
    }

    /**
     * Wait for the kernel to be ready before executing code
     * Uses the onDidChangeRuntimeState event to wait for idle/ready state
     */
    private async waitForReady(timeoutMs: number = 10000): Promise<void> {
        if (!this.session) {
            throw new Error('No session to wait for');
        }

        return new Promise((resolve, reject) => {
            const startTime = Date.now();

            // Set up timeout
            const timeout = setTimeout(() => {
                disposable.dispose();
                reject(new Error(`Timeout waiting for kernel to be ready (${timeoutMs}ms)`));
            }, timeoutMs);

            // Listen for state changes
            const disposable = this.session!.onDidChangeRuntimeState((state) => {
                logger.debug(`Waiting for ready: current state = ${state}`);

                if (state === positron.RuntimeState.Ready || state === positron.RuntimeState.Idle) {
                    clearTimeout(timeout);
                    disposable.dispose();
                    logger.debug(`Kernel ready after ${Date.now() - startTime}ms`);
                    resolve();
                } else if (
                    state === positron.RuntimeState.Exited ||
                    state === positron.RuntimeState.Offline
                ) {
                    clearTimeout(timeout);
                    disposable.dispose();
                    reject(new Error(`Kernel failed to start (state: ${state})`));
                }
            });
        });
    }

    /**
     * Set up PYTHONPATH to include squiggy package
     */
    private async setupPythonPath(): Promise<void> {
        if (!this.session) {
            return;
        }

        // In development, add extension's squiggy source to PYTHONPATH
        // In production, squiggy should be installed via pip
        const squiggyPath = this.extensionPath; // Contains squiggy/ subdirectory

        try {
            await this.executeSilent(`
import sys
import os

# Add extension path to sys.path if not already present (temporary variable)
_squiggy_temp_path = ${JSON.stringify(squiggyPath)}
if _squiggy_temp_path not in sys.path:
    sys.path.insert(0, _squiggy_temp_path)
del _squiggy_temp_path
`);
            logger.info(`Added ${squiggyPath} to PYTHONPATH in dedicated kernel`);
        } catch (error) {
            logger.warning(`Failed to set up PYTHONPATH: ${error}`);
        }
    }

    /**
     * Verify squiggy package is available
     */
    private async verifySquiggyAvailable(): Promise<void> {
        try {
            await this.executeSilent(`
import sys

# Force fresh import by clearing cached modules (ensures code changes are picked up)
_to_remove = [key for key in sys.modules if key.startswith('squiggy')]
for _key in _to_remove:
    del sys.modules[_key]

# Now import fresh from disk
import squiggy
assert hasattr(squiggy, 'load_pod5'), "squiggy.load_pod5 not found"
assert hasattr(squiggy, 'load_bam'), "squiggy.load_bam not found"
assert hasattr(squiggy, 'plot_read'), "squiggy.plot_read not found"
`);
            logger.info('Squiggy package verified in dedicated kernel');
        } catch (error) {
            throw new Error(`Squiggy package not available in dedicated kernel: ${error}`);
        }
    }

    /**
     * Handle runtime state changes
     */
    private handleRuntimeStateChange(state: positron.RuntimeState): void {
        logger.debug(`Dedicated kernel state changed: ${state}`);

        switch (state) {
            case positron.RuntimeState.Ready:
            case positron.RuntimeState.Idle:
                if (this.state !== SquiggyKernelState.Ready) {
                    this.setState(SquiggyKernelState.Ready);
                }
                break;
            case positron.RuntimeState.Busy:
                // Stay in current state (Ready or Starting)
                break;
            case positron.RuntimeState.Starting:
            case positron.RuntimeState.Initializing:
                if (this.state !== SquiggyKernelState.Starting) {
                    this.setState(SquiggyKernelState.Starting);
                }
                break;
            case positron.RuntimeState.Restarting:
                this.setState(SquiggyKernelState.Restarting);
                break;
            case positron.RuntimeState.Exited:
            case positron.RuntimeState.Offline:
                this.setState(SquiggyKernelState.Error);
                logger.error(`Dedicated kernel exited unexpectedly (state: ${state})`);
                // TODO: Implement auto-restart logic
                break;
        }
    }

    /**
     * Handle session end
     */
    private handleSessionEnd(exit: any): void {
        logger.warning(`Dedicated kernel session ended: ${JSON.stringify(exit)}`);
        this.session = undefined;
        this.setState(SquiggyKernelState.Error);
        // TODO: Implement auto-restart logic
    }

    /**
     * Set state and emit event
     */
    private setState(newState: SquiggyKernelState): void {
        if (this.state !== newState) {
            this.state = newState;
            this.stateChangeEmitter.fire(newState);
        }
    }
}
