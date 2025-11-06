/**
 * Tests for Error Handler
 *
 * Tests centralized error handling with custom error types,
 * formatting, safe execution wrappers, and retry logic.
 * Target: >80% coverage of error-handler.ts
 */

import { describe, it, expect, jest, beforeEach, afterEach } from '@jest/globals';
import * as vscode from 'vscode';
import {
    ErrorContext,
    SquiggyError,
    KernelNotAvailableError,
    PackageNotInstalledError,
    ExternallyManagedEnvironmentError,
    POD5Error,
    BAMError,
    FASTAError,
    PlottingError,
    ValidationError,
    handleError,
    handleErrorWithProgress,
    safeExecute,
    safeExecuteWithProgress,
    retryOperation,
    isTransientError,
} from '../error-handler';

describe('Error Handler', () => {
    beforeEach(() => {
        jest.clearAllMocks();
    });

    describe('Custom Error Classes', () => {
        describe('SquiggyError', () => {
            it('should create error with message and context', () => {
                const error = new SquiggyError('Test error', ErrorContext.POD5_LOAD);

                expect(error.message).toBe('Test error');
                expect(error.context).toBe(ErrorContext.POD5_LOAD);
                expect(error.name).toBe('SquiggyError');
                expect(error.cause).toBeUndefined();
            });

            it('should create error with cause', () => {
                const cause = new Error('Root cause');
                const error = new SquiggyError('Test error', ErrorContext.BAM_LOAD, cause);

                expect(error.cause).toBe(cause);
            });
        });

        describe('KernelNotAvailableError', () => {
            it('should create error with appropriate message', () => {
                const error = new KernelNotAvailableError();

                expect(error.name).toBe('KernelNotAvailableError');
                expect(error.message).toContain('No Python kernel is running');
                expect(error.context).toBe(ErrorContext.KERNEL_COMMUNICATION);
            });
        });

        describe('PackageNotInstalledError', () => {
            it('should create error with package message', () => {
                const error = new PackageNotInstalledError();

                expect(error.name).toBe('PackageNotInstalledError');
                expect(error.message).toContain('squiggy Python package is not installed');
                expect(error.context).toBe(ErrorContext.PACKAGE_INSTALL);
            });
        });

        describe('ExternallyManagedEnvironmentError', () => {
            it('should create error with virtual environment instructions', () => {
                const error = new ExternallyManagedEnvironmentError('/usr/bin/python3');

                expect(error.name).toBe('ExternallyManagedEnvironmentError');
                expect(error.message).toContain('externally managed');
                expect(error.message).toContain('/usr/bin/python3');
                expect(error.message).toContain('python3 -m venv .venv');
                expect(error.context).toBe(ErrorContext.PACKAGE_INSTALL);
            });
        });

        describe('POD5Error', () => {
            it('should create POD5 error with message', () => {
                const error = new POD5Error('Invalid POD5 file');

                expect(error.name).toBe('POD5Error');
                expect(error.message).toBe('Invalid POD5 file');
                expect(error.context).toBe(ErrorContext.POD5_LOAD);
            });

            it('should include cause if provided', () => {
                const cause = new Error('File not found');
                const error = new POD5Error('Invalid POD5 file', cause);

                expect(error.cause).toBe(cause);
            });
        });

        describe('BAMError', () => {
            it('should create BAM error with message', () => {
                const error = new BAMError('Invalid BAM index');

                expect(error.name).toBe('BAMError');
                expect(error.message).toBe('Invalid BAM index');
                expect(error.context).toBe(ErrorContext.BAM_LOAD);
            });
        });

        describe('FASTAError', () => {
            it('should create FASTA error with message', () => {
                const error = new FASTAError('Invalid FASTA format');

                expect(error.name).toBe('FASTAError');
                expect(error.message).toBe('Invalid FASTA format');
                expect(error.context).toBe(ErrorContext.FASTA_LOAD);
            });
        });

        describe('PlottingError', () => {
            it('should create plotting error with message', () => {
                const error = new PlottingError('Bokeh generation failed');

                expect(error.name).toBe('PlottingError');
                expect(error.message).toBe('Bokeh generation failed');
                expect(error.context).toBe(ErrorContext.PLOT_GENERATE);
            });
        });

        describe('ValidationError', () => {
            it('should create validation error with message only', () => {
                const error = new ValidationError('Value must be positive');

                expect(error.name).toBe('ValidationError');
                expect(error.message).toBe('Value must be positive');
                expect(error.context).toBe(ErrorContext.KERNEL_COMMUNICATION);
            });

            it('should include parameter name in message', () => {
                const error = new ValidationError('Value must be positive', 'readCount');

                expect(error.message).toContain("Invalid parameter 'readCount'");
                expect(error.message).toContain('Value must be positive');
            });
        });
    });

    describe('handleError', () => {
        it('should show error message for SquiggyError', () => {
            const error = new SquiggyError('Test error', ErrorContext.POD5_LOAD);

            handleError(error, ErrorContext.POD5_LOAD);

            expect(vscode.window.showErrorMessage).toHaveBeenCalledWith(
                'Failed loading POD5 file: Test error',
                'Show Logs'
            );
        });

        it('should show error message for standard Error', () => {
            const error = new Error('Standard error');

            handleError(error, ErrorContext.BAM_LOAD);

            expect(vscode.window.showErrorMessage).toHaveBeenCalledWith(
                'Failed loading BAM file: Standard error',
                'Show Logs'
            );
        });

        it('should detect kernel-related errors', () => {
            const error = new Error('kernel is not available');

            handleError(error, ErrorContext.PLOT_GENERATE);

            expect(vscode.window.showErrorMessage).toHaveBeenCalledWith(
                expect.stringContaining('No Python kernel is running'),
                'Show Logs'
            );
        });

        it('should detect session-related errors', () => {
            const error = new Error('No active session found');

            handleError(error, ErrorContext.PLOT_GENERATE);

            expect(vscode.window.showErrorMessage).toHaveBeenCalledWith(
                expect.stringContaining('No Python kernel is running'),
                'Show Logs'
            );
        });

        it('should detect externally managed environment errors', () => {
            const error = new Error('error: EXTERNALLY_MANAGED environment');

            handleError(error, ErrorContext.PACKAGE_INSTALL);

            expect(vscode.window.showErrorMessage).toHaveBeenCalledWith(
                expect.stringContaining('EXTERNALLY_MANAGED'),
                'Show Logs'
            );
        });

        it('should detect package not installed errors', () => {
            const error = new Error('squiggy package is not installed');

            handleError(error, ErrorContext.KERNEL_COMMUNICATION);

            expect(vscode.window.showErrorMessage).toHaveBeenCalledWith(
                expect.stringContaining('squiggy Python package is not installed'),
                'Show Logs'
            );
        });

        it('should handle non-Error objects', () => {
            handleError('string error', ErrorContext.STATE_CLEAR);

            expect(vscode.window.showErrorMessage).toHaveBeenCalledWith(
                'Failed clearing extension state: string error',
                'Show Logs'
            );
        });

        it('should handle null errors', () => {
            handleError(null, ErrorContext.MOTIF_SEARCH);

            expect(vscode.window.showErrorMessage).toHaveBeenCalledWith(
                'Failed searching for motif matches: null',
                'Show Logs'
            );
        });
    });

    describe('handleErrorWithProgress', () => {
        it('should show error message with progress context', () => {
            const error = new Error('Progress error');

            handleErrorWithProgress(error, ErrorContext.PLOT_GENERATE);

            expect(vscode.window.showErrorMessage).toHaveBeenCalledWith(
                'Failed generating plot: Progress error',
                'Show Logs'
            );
        });
    });

    describe('safeExecute', () => {
        it('should execute operation and return result on success', async () => {
            const operation = jest.fn(async () => 'success');
            const onSuccess = jest.fn();

            const result = await safeExecute(operation, ErrorContext.POD5_LOAD, onSuccess);

            expect(result).toBe('success');
            expect(operation).toHaveBeenCalled();
            expect(onSuccess).toHaveBeenCalledWith('success');
        });

        it('should return undefined on error', async () => {
            const operation = jest.fn(async () => {
                throw new Error('Operation failed');
            });

            const result = await safeExecute(operation, ErrorContext.BAM_LOAD);

            expect(result).toBeUndefined();
            expect(vscode.window.showErrorMessage).toHaveBeenCalled();
        });

        it('should work without onSuccess callback', async () => {
            const operation = jest.fn(async () => 'success');

            const result = await safeExecute(operation, ErrorContext.POD5_LOAD);

            expect(result).toBe('success');
        });

        it('should handle errors from operation', async () => {
            const operation = jest.fn(async () => {
                throw new PlottingError('Bokeh failed');
            });

            const result = await safeExecute(operation, ErrorContext.PLOT_GENERATE);

            expect(result).toBeUndefined();
            expect(vscode.window.showErrorMessage).toHaveBeenCalledWith(
                expect.stringContaining('Bokeh failed'),
                'Show Logs'
            );
        });
    });

    describe('safeExecuteWithProgress', () => {
        it('should execute operation with progress notification', async () => {
            const operation = jest.fn(async () => 'success');

            const result = await safeExecuteWithProgress(
                operation,
                ErrorContext.POD5_LOAD,
                'Loading POD5...'
            );

            expect(result).toBe('success');
            expect(vscode.window.withProgress).toHaveBeenCalledWith(
                expect.objectContaining({
                    location: vscode.ProgressLocation.Notification,
                    title: 'Loading POD5...',
                    cancellable: false,
                }),
                expect.any(Function)
            );
        });

        it('should return undefined on error', async () => {
            const operation = jest.fn(async () => {
                throw new Error('Operation failed');
            });

            const result = await safeExecuteWithProgress(
                operation,
                ErrorContext.BAM_LOAD,
                'Loading BAM...'
            );

            expect(result).toBeUndefined();
            expect(vscode.window.showErrorMessage).toHaveBeenCalled();
        });
    });

    describe('retryOperation', () => {
        it('should succeed on first attempt', async () => {
            const operation = jest.fn(async () => 'success');

            const result = await retryOperation(operation, { maxAttempts: 3 });

            expect(result).toBe('success');
            expect(operation).toHaveBeenCalledTimes(1);
        });

        it('should retry on failure and eventually succeed', async () => {
            let attemptCount = 0;
            const operation = jest.fn(async () => {
                attemptCount++;
                if (attemptCount < 3) {
                    throw new Error('Temporary failure');
                }
                return 'success';
            });

            const result = await retryOperation(operation, {
                maxAttempts: 3,
                baseDelayMs: 10, // Short delay for testing
            });

            expect(result).toBe('success');
            expect(operation).toHaveBeenCalledTimes(3);
        });

        it('should throw error after max attempts', async () => {
            const operation = jest.fn(async () => {
                throw new Error('Persistent failure');
            });

            await expect(
                retryOperation(operation, { maxAttempts: 3, baseDelayMs: 10 })
            ).rejects.toThrow('Persistent failure');
            expect(operation).toHaveBeenCalledTimes(3);
        });

        it('should use exponential backoff by default', async () => {
            let attemptCount = 0;
            const operation = jest.fn(async () => {
                attemptCount++;
                if (attemptCount < 3) {
                    throw new Error('Failure');
                }
                return 'success';
            });

            const result = await retryOperation(operation, {
                maxAttempts: 3,
                baseDelayMs: 10,
                exponentialBackoff: true,
            });

            expect(result).toBe('success');
            expect(operation).toHaveBeenCalledTimes(3);
        });

        it('should use constant delay when exponential backoff is disabled', async () => {
            let attemptCount = 0;
            const operation = jest.fn(async () => {
                attemptCount++;
                if (attemptCount < 3) {
                    throw new Error('Failure');
                }
                return 'success';
            });

            const result = await retryOperation(operation, {
                maxAttempts: 3,
                baseDelayMs: 10,
                exponentialBackoff: false,
            });

            expect(result).toBe('success');
            expect(operation).toHaveBeenCalledTimes(3);
        });

        it('should respect shouldRetry predicate', async () => {
            const operation = jest.fn(async () => {
                throw new Error('Non-retryable error');
            });

            const shouldRetry = jest.fn(() => false);

            await expect(
                retryOperation(operation, { maxAttempts: 3, shouldRetry })
            ).rejects.toThrow('Non-retryable error');
            expect(operation).toHaveBeenCalledTimes(1);
            expect(shouldRetry).toHaveBeenCalled();
            // Verify it was called with an Error instance and attempt number
            const call = shouldRetry.mock.calls[0] as any[];
            expect(call[0]).toBeInstanceOf(Error);
            expect(call[1]).toBe(1);
        });

        it('should convert non-Error to Error', async () => {
            const operation = jest.fn(async () => {
                throw 'string error';
            });

            await expect(
                retryOperation(operation, { maxAttempts: 2, baseDelayMs: 10 })
            ).rejects.toThrow('string error');
        });
    });

    describe('isTransientError', () => {
        it('should identify timeout errors as transient', () => {
            const error = new Error('Request timeout after 5000ms');

            expect(isTransientError(error)).toBe(true);
        });

        it('should identify kernel errors as transient', () => {
            const error = new Error('Kernel is busy');

            expect(isTransientError(error)).toBe(true);
        });

        it('should identify busy errors as transient', () => {
            const error = new Error('Resource is busy');

            expect(isTransientError(error)).toBe(true);
        });

        it('should identify file lock errors as transient', () => {
            const error = new Error('File is locked by another process');

            expect(isTransientError(error)).toBe(true);
        });

        it('should identify "in use" errors as transient', () => {
            const error = new Error('Port is already in use');

            expect(isTransientError(error)).toBe(true);
        });

        it('should identify connection refused errors as transient', () => {
            const error = new Error('ECONNREFUSED: Connection refused');

            expect(isTransientError(error)).toBe(true);
        });

        it('should identify connection reset errors as transient', () => {
            const error = new Error('ECONNRESET: Connection reset by peer');

            expect(isTransientError(error)).toBe(true);
        });

        it('should not identify permanent errors as transient', () => {
            const error = new Error('File not found');

            expect(isTransientError(error)).toBe(false);
        });

        it('should not identify validation errors as transient', () => {
            const error = new Error('Invalid parameter value');

            expect(isTransientError(error)).toBe(false);
        });
    });
});
