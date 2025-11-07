/**
 * Low-level Positron Runtime Client
 *
 * Provides thin wrapper around Positron's runtime API for executing Python code
 * and reading kernel variables. Handles kernel readiness checking and basic
 * code execution patterns.
 *
 * This module should NOT contain any squiggy-specific logic - just generic
 * Positron kernel communication primitives.
 */

import * as positron from 'positron';
import { KernelNotAvailableError, retryOperation, isTransientError } from '../utils/error-handler';
import { logger } from '../utils/logger';

/**
 * Low-level client for Positron runtime API
 *
 * Manages kernel communication, code execution, and variable access.
 */
export class PositronRuntimeClient {
    private kernelReadyCache: { ready: boolean; timestamp: number } | null = null;
    private readonly KERNEL_CACHE_TTL = 1000; // 1 second - kernel state is fairly stable

    /**
     * Check if Positron runtime is available
     */
    isAvailable(): boolean {
        try {
            return typeof positron !== 'undefined' && typeof positron.runtime !== 'undefined';
        } catch {
            return false;
        }
    }

    /**
     * Ensure the Python kernel is ready to execute code
     *
     * Waits up to 10 seconds for kernel to be ready after restart.
     * Uses event-based approach if available, falls back to polling.
     *
     * Uses a 1-second cache to avoid repeated readiness checks within the same
     * logical operation (e.g., getVariable() makes 3 executeSilent calls sequentially).
     */
    private async ensureKernelReady(): Promise<void> {
        // Check cache first - if kernel was ready in the last 1 second, assume still ready
        const now = Date.now();
        if (
            this.kernelReadyCache &&
            now - this.kernelReadyCache.timestamp < this.KERNEL_CACHE_TTL
        ) {
            if (this.kernelReadyCache.ready) {
                return; // Kernel is cached as ready, skip check
            }
        }
        const session = await positron.runtime.getForegroundSession();

        if (!session) {
            throw new KernelNotAvailableError();
        }

        // Check if session has onDidChangeRuntimeState event (may not exist in all Positron versions)
        const hasStateEvent = typeof (session as any).onDidChangeRuntimeState === 'function';

        if (hasStateEvent) {
            return this.ensureKernelReadyViaEvents(session);
        } else {
            return this.ensureKernelReadyViaPolling();
        }
    }

    /**
     * Ensure kernel ready using event-based approach (Positron API with onDidChangeRuntimeState)
     */
    private async ensureKernelReadyViaEvents(session: any): Promise<void> {
        return new Promise<void>((resolve, reject) => {
            const timeout = setTimeout(() => {
                this.kernelReadyCache = { ready: false, timestamp: Date.now() };
                reject(new Error('Timeout waiting for Python kernel to be ready'));
            }, 10000); // 10 second timeout

            // Function to check if current state is ready
            const checkState = (state: string) => {
                logger.debug(`Squiggy: Kernel state is ${state}`);

                // Ready states - can execute code
                if (state === 'ready' || state === 'idle' || state === 'busy') {
                    clearTimeout(timeout);
                    this.kernelReadyCache = { ready: true, timestamp: Date.now() };
                    resolve();
                    return true;
                }

                // Failed states - cannot execute code
                if (state === 'offline' || state === 'exited') {
                    clearTimeout(timeout);
                    this.kernelReadyCache = { ready: false, timestamp: Date.now() };
                    reject(new Error(`Python kernel is ${state}. Please start a Python console.`));
                    return true;
                }

                // Transitioning states - wait
                // uninitialized, initializing, starting, restarting
                return false;
            };

            // Listen for state changes
            const disposable = session.onDidChangeRuntimeState((state: string) => {
                if (checkState(state)) {
                    disposable.dispose();
                }
            });

            // Also check current state immediately (in case already ready)
            Promise.resolve(
                positron.runtime.executeCode(
                    'python',
                    '1+1',
                    false,
                    true,
                    positron.RuntimeCodeExecutionMode.Silent
                )
            )
                .then(() => {
                    // Kernel responded - it's ready
                    clearTimeout(timeout);
                    disposable.dispose();
                    this.kernelReadyCache = { ready: true, timestamp: Date.now() };
                    resolve();
                })
                .catch(() => {
                    // Kernel didn't respond - wait for state change
                    // The onDidChangeRuntimeState listener will handle it
                });
        });
    }

    /**
     * Ensure kernel ready using polling approach (fallback for older Positron versions)
     */
    private async ensureKernelReadyViaPolling(): Promise<void> {
        const startTime = Date.now();
        const maxWaitMs = 10000; // 10 seconds
        const retryDelayMs = 500; // 500ms between retries

        while (Date.now() - startTime < maxWaitMs) {
            try {
                // Try to execute a simple test command
                await positron.runtime.executeCode(
                    'python',
                    '1+1',
                    false,
                    true,
                    positron.RuntimeCodeExecutionMode.Silent
                );
                // Success - kernel is ready
                logger.debug('Squiggy: Kernel is ready (polling check)');
                this.kernelReadyCache = { ready: true, timestamp: Date.now() };
                return;
            } catch (_error) {
                // Kernel not ready yet, wait and retry
                await new Promise((resolve) => setTimeout(resolve, retryDelayMs));
            }
        }

        // Timeout reached
        this.kernelReadyCache = { ready: false, timestamp: Date.now() };
        throw new Error('Timeout waiting for Python kernel to be ready');
    }

    /**
     * Execute Python code in the active kernel
     *
     * @param code Python code to execute
     * @param focus Whether to focus the console
     * @param allowIncomplete Whether to allow incomplete statements
     * @param mode Execution mode (silent by default to hide imports)
     * @param observer Optional observer for capturing output
     * @param enableRetry Whether to retry on transient failures (default: false)
     * @returns Promise that resolves with the result object containing MIME type mappings
     */
    async executeCode(
        code: string,
        focus: boolean = false,
        allowIncomplete: boolean = true,
        mode: positron.RuntimeCodeExecutionMode = positron.RuntimeCodeExecutionMode.Silent,
        observer?: positron.RuntimeCodeExecutionObserver,
        enableRetry: boolean = false
    ): Promise<Record<string, unknown>> {
        if (!this.isAvailable()) {
            throw new Error('Positron runtime not available');
        }

        // Ensure kernel is ready before executing code
        await this.ensureKernelReady();

        const executeOperation = async () => {
            try {
                return await positron.runtime.executeCode(
                    'python',
                    code,
                    focus,
                    allowIncomplete,
                    mode,
                    undefined, // errorBehavior
                    observer
                );
            } catch (error) {
                // Extract error message properly - error might be an object with message property
                const errorMessage = error instanceof Error ? error.message : String(error);
                throw new Error(`Failed to execute Python code: ${errorMessage}`);
            }
        };

        // If retry is enabled, wrap the operation with retry logic
        if (enableRetry) {
            return await retryOperation(executeOperation, {
                maxAttempts: 3,
                baseDelayMs: 500,
                shouldRetry: (error, attempt) => {
                    // Only retry transient errors
                    return isTransientError(error) && attempt < 3;
                },
            });
        }

        return await executeOperation();
    }

    /**
     * Execute code silently without console output
     *
     * @param code Python code to execute
     * @param enableRetry Whether to retry on transient failures (default: false)
     * @returns Promise that resolves when execution completes
     */
    async executeSilent(code: string, enableRetry: boolean = false): Promise<void> {
        await this.executeCode(
            code,
            false, // focus=false
            true,
            positron.RuntimeCodeExecutionMode.Silent,
            undefined, // observer
            enableRetry
        );
    }

    /**
     * Execute code and capture printed output via observer
     *
     * NOTE: This will show output in console. Use only when absolutely necessary.
     * Prefer getVariable() for reading data from Python memory.
     *
     * @param code Python code to execute
     * @returns Promise that resolves with captured output
     */
    async executeWithOutput(code: string): Promise<string> {
        return new Promise((resolve, reject) => {
            let output = '';

            this.executeCode(code, false, true, positron.RuntimeCodeExecutionMode.Silent, {
                onOutput: (message: string) => {
                    output += message;
                },
                onFinished: () => {
                    resolve(output);
                },
            }).catch(reject);
        });
    }

    /**
     * Get a Python variable value directly from the kernel
     *
     * Uses Positron's getSessionVariables API to read kernel memory directly
     * without polluting the console with print() statements.
     *
     * @param varName Python variable name (can include indexing like 'var[0:10]')
     * @param enableRetry Whether to retry on transient failures (default: false)
     * @returns Promise that resolves with the variable value
     */
    async getVariable(varName: string, enableRetry: boolean = false): Promise<unknown> {
        const session = await positron.runtime.getForegroundSession();
        if (!session || session.runtimeMetadata.languageId !== 'python') {
            throw new Error('No active Python session');
        }

        // Convert the Python value to JSON in Python, then read that
        const tempVar = '_squiggy_temp_' + Math.random().toString(36).substr(2, 9);

        try {
            await this.executeSilent(
                `
import json
${tempVar} = json.dumps(${varName})
`,
                enableRetry
            );

            const [[variable]] = await positron.runtime.getSessionVariables(
                session.metadata.sessionId,
                [[tempVar]]
            );

            // Clean up temp variable
            await this.executeSilent(
                `
if '${tempVar}' in globals():
    del ${tempVar}
`,
                false // Don't retry cleanup
            );

            if (!variable) {
                throw new Error(`Variable ${varName} not found`);
            }

            // display_value contains the JSON string (as a Python string repr)
            // We need to parse it: Python repr -> actual string -> JSON parse
            // e.g., "'[1,2,3]'" -> "[1,2,3]" -> [1,2,3]
            const jsonString = variable.display_value;

            // Remove outer quotes if present (Python string repr)
            const cleaned = jsonString.replace(/^['"]|['"]$/g, '');

            return JSON.parse(cleaned);
        } catch (error) {
            // Clean up temp variable on error
            await this.executeSilent(
                `
if '${tempVar}' in globals():
    del ${tempVar}
`
            ).catch(() => {}); // Ignore cleanup errors
            throw new Error(`Failed to get variable ${varName}: ${error}`);
        }
    }
}
