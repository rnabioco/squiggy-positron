/**
 * Squiggy Kernel Manager
 *
 * Manages a dedicated background Python kernel for Squiggy extension operations.
 * This isolates Squiggy state from the user's workspace while maintaining
 * notebook API compatibility.
 *
 * Architecture:
 * - Extension UI → Background kernel (isolated state, no Variables pane clutter)
 * - Notebook API → User's kernel (full programmatic access, as before)
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
        logger.info('Starting Squiggy background kernel...');

        try {
            // Get the Python runtime manager from Positron
            const pythonExt = vscode.extensions.getExtension('positron.positron-python');
            if (!pythonExt) {
                throw new Error('Positron Python extension not found');
            }

            await pythonExt.activate();
            const runtimeManager = pythonExt.exports as any; // PositronPythonRuntimeManager

            if (!runtimeManager || !runtimeManager.getRegisteredRuntimes) {
                throw new Error('Python runtime manager not available');
            }

            // Get available Python runtimes
            const runtimes: positron.LanguageRuntimeMetadata[] =
                await runtimeManager.getRegisteredRuntimes();

            if (runtimes.length === 0) {
                throw new Error('No Python runtimes available');
            }

            // Use the first available Python runtime
            // TODO: In future, could match foreground session's Python version
            const runtime = runtimes[0];
            logger.info(
                `Using Python runtime: ${runtime.runtimeName} (${runtime.runtimeVersion})`
            );

            // Create session metadata for background mode
            const sessionMetadata = {
                sessionId: `squiggy-background-${Date.now()}`,
                sessionName: 'Squiggy Background',
                sessionMode: 'background', // RuntimeSessionMode.Background
                createdTimestamp: Date.now(),
                notebookUri: undefined,
                startReason: 'Extension Activation',
            };

            // Create the background session
            this.session = await runtimeManager.createSession(runtime, sessionMetadata);

            if (!this.session) {
                throw new Error('Failed to create background session');
            }

            logger.info(`Background session created: ${sessionMetadata.sessionId}`);

            // Listen for session state changes
            this.session.onDidChangeRuntimeState(this.handleRuntimeStateChange.bind(this));
            this.session.onDidEndSession(this.handleSessionEnd.bind(this));

            // Start the session
            await (this.session as any).start();

            // Set up PYTHONPATH to include squiggy package
            await this.setupPythonPath();

            // Verify squiggy is importable
            await this.verifySquiggyAvailable();

            this.setState(SquiggyKernelState.Ready);
            logger.info('Squiggy background kernel ready');
        } catch (error) {
            this.setState(SquiggyKernelState.Error);
            logger.error(`Failed to start background kernel: ${error}`);
            throw error;
        }
    }

    /**
     * Restart the background kernel
     */
    async restart(): Promise<void> {
        if (this.isDisposed) {
            throw new Error('Kernel manager has been disposed');
        }

        logger.info('Restarting Squiggy background kernel...');
        this.setState(SquiggyKernelState.Restarting);

        try {
            if (this.session) {
                // Try to restart existing session
                await (this.session as any).restart();
                await this.setupPythonPath();
                await this.verifySquiggyAvailable();
                this.setState(SquiggyKernelState.Ready);
                logger.info('Squiggy background kernel restarted');
            } else {
                // No session - start fresh
                await this.start();
            }
        } catch (error) {
            logger.error(`Failed to restart background kernel: ${error}`);
            this.setState(SquiggyKernelState.Error);

            // Try to recover by starting fresh
            this.session = undefined;
            await this.start();
        }
    }

    /**
     * Execute code in the background kernel
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
            throw new Error('Background kernel not started');
        }

        if (this.state !== SquiggyKernelState.Ready) {
            throw new Error(`Background kernel not ready (state: ${this.state})`);
        }

        try {
            // Execute in the background session
            return await (this.session as any).execute(
                code,
                'auto', // executionId
                mode,
                positron.RuntimeErrorBehavior.Continue
            );
        } catch (error) {
            logger.error(`Background kernel execution failed: ${error}`);
            throw error;
        }
    }

    /**
     * Execute code silently (no console output)
     * @param code Python code to execute
     * @param enableRetry Ignored (for RuntimeClient interface compatibility)
     */
    async executeSilent(code: string, enableRetry?: boolean): Promise<void> {
        await this.execute(code, positron.RuntimeCodeExecutionMode.Silent);
    }

    /**
     * Get a variable value from the background kernel
     * @param varName Python variable name or expression
     * @param enableRetry Ignored (for RuntimeClient interface compatibility)
     */
    async getVariable(varName: string, enableRetry?: boolean): Promise<unknown> {
        if (!this.session) {
            throw new Error('Background kernel not started');
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
            await this.executeSilent(`
if '${tempVar}' in globals():
    del ${tempVar}
`).catch(() => {});
            throw new Error(`Failed to get variable ${varName}: ${error}`);
        }
    }

    /**
     * Shutdown the background kernel
     */
    async shutdown(): Promise<void> {
        if (this.session) {
            logger.info('Shutting down Squiggy background kernel...');
            try {
                await (this.session as any).shutdown();
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

# Add extension path to sys.path if not already present
squiggy_path = ${JSON.stringify(squiggyPath)}
if squiggy_path not in sys.path:
    sys.path.insert(0, squiggy_path)
    print(f"Added {squiggy_path} to sys.path")
`);
            logger.info(`Added ${squiggyPath} to PYTHONPATH in background kernel`);
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
            logger.info('Squiggy package verified in background kernel');
        } catch (error) {
            throw new Error(`Squiggy package not available in background kernel: ${error}`);
        }
    }

    /**
     * Handle runtime state changes
     */
    private handleRuntimeStateChange(state: positron.RuntimeState): void {
        logger.debug(`Background kernel state changed: ${state}`);

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
                logger.error(`Background kernel exited unexpectedly (state: ${state})`);
                // TODO: Implement auto-restart logic
                break;
        }
    }

    /**
     * Handle session end
     */
    private handleSessionEnd(exit: any): void {
        logger.warning(`Background kernel session ended: ${JSON.stringify(exit)}`);
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
