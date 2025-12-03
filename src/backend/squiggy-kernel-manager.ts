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

export class SquiggyKernelManager implements RuntimeClient {
    private session: positron.LanguageRuntimeSession | undefined;
    private state: SquiggyKernelState = SquiggyKernelState.Uninitialized;
    private stateChangeEmitter = new vscode.EventEmitter<SquiggyKernelState>();
    private extensionPath: string;
    private isDisposed = false;

    readonly onDidChangeState = this.stateChangeEmitter.event;

    constructor(extensionPath: string) {
        this.extensionPath = extensionPath;
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
        return this.session?.metadata.sessionId;
    }

    /**
     * Start the background Python kernel
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
        logger.info('Starting Squiggy dedicated kernel...');

        try {
            // Get the preferred Python runtime
            // Positron auto-discovers venvs in ~/.venvs/ and sets the squiggy venv as preferred
            const runtime = await positron.runtime.getPreferredRuntime('python');

            if (!runtime) {
                throw new Error('No Python runtime available');
            }

            const runtimeId = runtime.runtimeId;
            const runtimeName = `${runtime.runtimeName} (${runtime.runtimeVersion})`;

            logger.info(`Using Python runtime: ${runtimeName}`);

            // Start a dedicated console session for Squiggy
            // NOTE: positron.runtime.startLanguageRuntime() only supports Console and Notebook modes
            // Background mode is not exposed through the public API
            // This creates a Console session which is separate from the foreground session
            const sessionName = 'Squiggy Dedicated Kernel';
            this.session = await positron.runtime.startLanguageRuntime(
                runtimeId,
                sessionName,
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
        } catch (error) {
            this.setState(SquiggyKernelState.Error);
            logger.error(`Failed to start dedicated kernel: ${error}`);
            throw error;
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
            if (this.session) {
                // Use Positron API to restart the session
                await positron.runtime.restartSession(this.session.metadata.sessionId);

                // Wait for kernel to be ready after restart
                logger.info('Waiting for restarted kernel to be ready...');
                await this.waitForReady();
                logger.info('Restarted kernel is ready, setting up environment...');

                await this.setupPythonPath();
                await this.verifySquiggyAvailable();

                // State should already be Ready from handleRuntimeStateChange
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
        if (!this.session) {
            throw new Error('Dedicated kernel not started');
        }

        if (this.state !== SquiggyKernelState.Ready) {
            throw new Error(`Dedicated kernel not ready (state: ${this.state})`);
        }

        try {
            // Execute code using Positron runtime API
            return await positron.runtime.executeCode(
                'python',
                code,
                false, // focus
                true, // allowIncomplete
                mode,
                positron.RuntimeErrorBehavior.Continue
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
        if (!this.session) {
            throw new Error('Dedicated kernel not started');
        }

        const tempVar = '_squiggy_temp_' + Math.random().toString(36).substr(2, 9);

        try {
            await this.executeSilent(`
import json
${tempVar} = json.dumps(${varName})
`);

            const [[variable]] = await positron.runtime.getSessionVariables(
                this.session.metadata.sessionId,
                [[tempVar]]
            );

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
        if (this.session) {
            logger.info('Shutting down Squiggy dedicated kernel...');
            try {
                await positron.runtime.deleteSession(this.session.metadata.sessionId);
            } catch (error) {
                logger.error(`Error shutting down kernel: ${error}`);
            }
            this.session = undefined;
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
