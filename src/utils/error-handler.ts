/**
 * Centralized error handling with consistent user messaging
 *
 * Provides type-safe error handling and user-friendly error messages
 * for common failure scenarios.
 */

import * as vscode from 'vscode';
import { logger } from './logger';

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

export class POD5Error extends SquiggyError {
    constructor(message: string, cause?: Error) {
        super(message, ErrorContext.POD5_LOAD, cause);
        this.name = 'POD5Error';
    }
}

export class BAMError extends SquiggyError {
    constructor(message: string, cause?: Error) {
        super(message, ErrorContext.BAM_LOAD, cause);
        this.name = 'BAMError';
    }
}

export class FASTAError extends SquiggyError {
    constructor(message: string, cause?: Error) {
        super(message, ErrorContext.FASTA_LOAD, cause);
        this.name = 'FASTAError';
    }
}

export class PlottingError extends SquiggyError {
    constructor(message: string, cause?: Error) {
        super(message, ErrorContext.PLOT_GENERATE, cause);
        this.name = 'PlottingError';
    }
}

export class ValidationError extends SquiggyError {
    constructor(message: string, parameterName?: string) {
        const fullMessage = parameterName
            ? `Invalid parameter '${parameterName}': ${message}`
            : message;
        super(fullMessage, ErrorContext.KERNEL_COMMUNICATION);
        this.name = 'ValidationError';
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

    // Show error message to user with option to show logs
    vscode.window.showErrorMessage(errorMessage, 'Show Logs').then((selection) => {
        if (selection === 'Show Logs') {
            logger.show();
        }
    });

    // Log to Output Channel (Output panel → Squiggy)
    logger.error(`Error while ${context}`, error);
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
    vscode.window.showErrorMessage(errorMessage, 'Show Logs').then((selection) => {
        if (selection === 'Show Logs') {
            logger.show();
        }
    });
    logger.error(`Error while ${context}`, error);
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

/**
 * Options for retry behavior
 */
export interface RetryOptions {
    /** Maximum number of retry attempts (default: 3) */
    maxAttempts?: number;
    /** Base delay in milliseconds between retries (default: 1000) */
    baseDelayMs?: number;
    /** Whether to use exponential backoff (default: true) */
    exponentialBackoff?: boolean;
    /** Predicate to determine if an error should trigger a retry (default: all errors) */
    shouldRetry?: (error: Error, attempt: number) => boolean;
}

/**
 * Retry an async operation with exponential backoff
 *
 * Useful for handling transient failures like network issues or temporary kernel unavailability.
 *
 * @param operation - Async function to execute
 * @param options - Retry configuration options
 * @returns Promise that resolves with the operation result
 * @throws Error if all retry attempts fail
 */
export async function retryOperation<T>(
    operation: () => Promise<T>,
    options: RetryOptions = {}
): Promise<T> {
    const {
        maxAttempts = 3,
        baseDelayMs = 1000,
        exponentialBackoff = true,
        shouldRetry = () => true,
    } = options;

    let lastError: Error | undefined;

    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        try {
            return await operation();
        } catch (error) {
            lastError = error instanceof Error ? error : new Error(String(error));

            // Check if we should retry this error
            if (!shouldRetry(lastError, attempt)) {
                throw lastError;
            }

            // If this was the last attempt, throw the error
            if (attempt === maxAttempts) {
                throw lastError;
            }

            // Calculate delay with optional exponential backoff
            const delay = exponentialBackoff ? baseDelayMs * Math.pow(2, attempt - 1) : baseDelayMs;

            logger.warning(
                `Retry attempt ${attempt}/${maxAttempts} failed: ${lastError.message}. Retrying in ${delay}ms...`
            );

            // Wait before retrying
            await new Promise((resolve) => setTimeout(resolve, delay));
        }
    }

    // This should never be reached due to the throw in the loop, but TypeScript doesn't know that
    throw lastError || new Error('Retry operation failed with no error');
}

/**
 * Check if an error is a transient failure that should be retried
 *
 * Transient errors include:
 * - Network timeouts
 * - Kernel busy/not ready
 * - Temporary file locks
 *
 * @param error - Error to check
 * @returns True if the error indicates a transient failure
 */
export function isTransientError(error: Error): boolean {
    const message = error.message.toLowerCase();

    // Kernel-related transient errors
    if (message.includes('timeout') || message.includes('kernel') || message.includes('busy')) {
        return true;
    }

    // File lock errors (can be transient)
    if (message.includes('lock') || message.includes('in use')) {
        return true;
    }

    // Network errors (if using subprocess backend)
    if (message.includes('econnrefused') || message.includes('econnreset')) {
        return true;
    }

    return false;
}
