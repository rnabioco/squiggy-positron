/**
 * Centralized error handling with consistent user messaging
 *
 * Provides type-safe error handling and user-friendly error messages
 * for common failure scenarios.
 */

import * as vscode from 'vscode';

/**
 * Error context for better error messages
 */
export enum ErrorContext {
    POD5_LOAD = 'loading POD5 file',
    BAM_LOAD = 'loading BAM file',
    FASTA_LOAD = 'loading FASTA file',
    POD5_CLOSE = 'closing POD5 file',
    BAM_CLOSE = 'closing BAM file',
    FASTA_CLOSE = 'closing FASTA file',
    PLOT_GENERATE = 'generating plot',
    KERNEL_COMMUNICATION = 'communicating with Python kernel',
    PACKAGE_INSTALL = 'installing squiggy package',
    STATE_CLEAR = 'clearing extension state',
    MOTIF_SEARCH = 'searching for motif matches',
    MOTIF_PLOT = 'generating motif aggregate plot',
}

/**
 * Custom error types for specific failure modes
 */
export class SquiggyError extends Error {
    constructor(
        message: string,
        public readonly context: ErrorContext,
        public readonly cause?: Error
    ) {
        super(message);
        this.name = 'SquiggyError';
    }
}

export class KernelNotAvailableError extends SquiggyError {
    constructor() {
        super(
            'No Python kernel is running. Please start a Python console first.',
            ErrorContext.KERNEL_COMMUNICATION
        );
        this.name = 'KernelNotAvailableError';
    }
}

export class PackageNotInstalledError extends SquiggyError {
    constructor() {
        super(
            'squiggy Python package is not installed in the active environment.',
            ErrorContext.PACKAGE_INSTALL
        );
        this.name = 'PackageNotInstalledError';
    }
}

export class ExternallyManagedEnvironmentError extends SquiggyError {
    constructor(pythonPath: string) {
        super(
            `Cannot install squiggy: Python environment is externally managed by your system package manager.\n\n` +
                `Your Python installation (${pythonPath}) cannot be modified directly.\n\n` +
                `Please create a virtual environment first:\n` +
                `1. Run: python3 -m venv .venv\n` +
                `2. Select the new environment in Positron (Interpreter selector)\n` +
                `3. Restart the Python console\n` +
                `4. Try opening the file again`,
            ErrorContext.PACKAGE_INSTALL
        );
        this.name = 'ExternallyManagedEnvironmentError';
    }
}

/**
 * Handle errors with context-aware user messaging
 *
 * @param error - Error instance
 * @param context - What operation was being performed
 */
export function handleError(error: unknown, context: ErrorContext): void {
    const errorMessage = formatErrorMessage(error, context);

    // Show error message to user
    vscode.window.showErrorMessage(errorMessage);

    // Log to extension output channel for debugging
    console.error(`[Squiggy] Error while ${context}:`, error);
}

/**
 * Handle errors with progress notification
 *
 * Useful for long-running operations that show progress dialogs.
 *
 * @param error - Error instance
 * @param context - What operation was being performed
 */
export function handleErrorWithProgress(error: unknown, context: ErrorContext): void {
    const errorMessage = formatErrorMessage(error, context);
    vscode.window.showErrorMessage(errorMessage);
    console.error(`[Squiggy] Error while ${context}:`, error);
}

/**
 * Format error message for user display
 *
 * @param error - Error instance
 * @param context - Operation context
 * @returns User-friendly error message
 */
function formatErrorMessage(error: unknown, context: ErrorContext): string {
    // Handle custom Squiggy errors
    if (error instanceof SquiggyError) {
        return `Failed ${error.context}: ${error.message}`;
    }

    // Handle standard errors
    if (error instanceof Error) {
        // Check for specific error patterns
        if (error.message.includes('kernel') || error.message.includes('session')) {
            return `Failed ${context}: No Python kernel is running. Please start a Python console first.`;
        }

        if (
            error.message.includes('EXTERNALLY_MANAGED') ||
            error.message.includes('externally-managed')
        ) {
            return `Failed ${context}: ${error.message}`;
        }

        if (error.message.includes('squiggy') && error.message.includes('not installed')) {
            return `Failed ${context}: squiggy Python package is not installed.`;
        }

        // Generic error with message
        return `Failed ${context}: ${error.message}`;
    }

    // Unknown error type
    return `Failed ${context}: ${String(error)}`;
}

/**
 * Safely execute an async operation with error handling
 *
 * @param operation - Async function to execute
 * @param context - Operation context for error messages
 * @param onSuccess - Optional callback on success
 * @returns Promise that resolves to the operation result or undefined on error
 */
export async function safeExecute<T>(
    operation: () => Promise<T>,
    context: ErrorContext,
    onSuccess?: (result: T) => void
): Promise<T | undefined> {
    try {
        const result = await operation();
        if (onSuccess) {
            onSuccess(result);
        }
        return result;
    } catch (error) {
        handleError(error, context);
        return undefined;
    }
}

/**
 * Safely execute an async operation with progress notification
 *
 * @param operation - Async function to execute
 * @param context - Operation context for error messages
 * @param progressMessage - Message to show during execution
 * @returns Promise that resolves to the operation result or undefined on error
 */
export async function safeExecuteWithProgress<T>(
    operation: () => Promise<T>,
    context: ErrorContext,
    progressMessage: string
): Promise<T | undefined> {
    try {
        return await vscode.window.withProgress(
            {
                location: vscode.ProgressLocation.Notification,
                title: progressMessage,
                cancellable: false,
            },
            async () => {
                return await operation();
            }
        );
    } catch (error) {
        handleErrorWithProgress(error, context);
        return undefined;
    }
}
