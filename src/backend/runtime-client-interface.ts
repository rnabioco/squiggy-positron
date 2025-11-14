/**
 * Common interface for Python runtime clients
 *
 * Defines the minimal interface needed for executing Python code and
 * reading variables. Implemented by both:
 * - PositronRuntimeClient (foreground session - for notebook API)
 * - SquiggyKernelManager (background session - for extension UI)
 */

export interface RuntimeClient {
    /**
     * Execute Python code silently (no console output)
     * @param code Python code to execute
     * @param enableRetry Whether to retry on transient failures (optional)
     */
    executeSilent(code: string, enableRetry?: boolean): Promise<void>;

    /**
     * Get a variable value from the Python kernel
     * @param varName Python variable name or expression
     * @param enableRetry Whether to retry on transient failures (optional)
     * @returns The variable value (parsed from JSON)
     */
    getVariable(varName: string, enableRetry?: boolean): Promise<unknown>;
}
