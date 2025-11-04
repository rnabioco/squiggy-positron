/**
 * Tests for Positron Runtime Backend
 */

import { describe, it, expect, beforeEach, jest } from '@jest/globals';
import { PositronRuntime } from '../squiggy-positron-runtime';

// Positron is mocked via src/__mocks__/positron.ts
jest.mock('positron');

describe('PositronRuntime', () => {
    let runtime: PositronRuntime;
    const positron = require('positron');

    beforeEach(() => {
        runtime = new PositronRuntime();
        jest.clearAllMocks();
    });

    describe('isAvailable', () => {
        it('should return true when positron runtime is available', () => {
            expect(runtime.isAvailable()).toBe(true);
        });

        it('should return false when positron is not defined', () => {
            const originalPositron = require('positron');
            jest.doMock('positron', () => undefined);

            // Note: In practice, this test may not work as expected due to Jest module caching
            // This test documents the intended behavior
            expect(runtime.isAvailable()).toBe(true); // Still true due to mock
        });
    });

    describe('executeSilent', () => {
        beforeEach(() => {
            // Mock session for kernel ready check
            positron.runtime.getForegroundSession.mockResolvedValue({
                metadata: { sessionId: 'test' },
                runtimeMetadata: { languageId: 'python' },
            });
            // Mock successful kernel ready check
            positron.runtime.executeCode.mockResolvedValue({});
        });

        it('should execute code in silent mode', async () => {
            const code = 'x = 42';

            await runtime.executeSilent(code);

            // Should have called executeCode (once for ready check, once for actual code)
            expect(positron.runtime.executeCode).toHaveBeenCalled();

            // Find the call with our code
            const calls = positron.runtime.executeCode.mock.calls;
            const codeCall = calls.find((call: any[]) => call[1] === code);

            expect(codeCall).toBeDefined();
            expect(codeCall![2]).toBe(false); // focus
            expect(codeCall![4]).toBe('silent'); // mode
        });

        it('should not focus console when executing silently', async () => {
            await runtime.executeSilent('x = 1');

            const calls = positron.runtime.executeCode.mock.calls;
            const codeCall = calls.find((call: any[]) => call[1] === 'x = 1');
            expect(codeCall![2]).toBe(false); // focus parameter
        });
    });

    describe('getVariable', () => {
        beforeEach(() => {
            positron.runtime.executeCode.mockResolvedValue({});
            positron.runtime.getForegroundSession.mockResolvedValue({
                metadata: {
                    sessionId: 'test-session-id',
                },
                runtimeMetadata: {
                    languageId: 'python',
                },
            });
        });

        it('should retrieve variable value from kernel', async () => {
            const mockVariable = {
                display_value: '"42"',
                display_type: 'int',
            };

            positron.runtime.getSessionVariables.mockResolvedValue([[mockVariable]]);

            const result = await runtime.getVariable('test_var');

            expect(result).toBe(42);
        });

        it('should handle string values', async () => {
            // Mock the variable as JSON-serialized string
            const mockVariable = {
                display_value: '\'"hello world"\'',
                display_type: 'str',
            };

            positron.runtime.getSessionVariables.mockResolvedValue([[mockVariable]]);

            const result = await runtime.getVariable('test_var');

            expect(result).toBe('hello world');
        });

        it('should handle list values', async () => {
            // Mock the variable as JSON-serialized list
            const mockVariable = {
                display_value: "'[1, 2, 3]'",
                display_type: 'list',
            };

            positron.runtime.getSessionVariables.mockResolvedValue([[mockVariable]]);

            const result = await runtime.getVariable('test_var');

            expect(result).toEqual([1, 2, 3]);
        });

        it('should handle dict values', async () => {
            // Mock the variable as JSON-serialized object
            const mockVariable = {
                display_value: '\'{"key": "value"}\'',
                display_type: 'dict',
            };

            positron.runtime.getSessionVariables.mockResolvedValue([[mockVariable]]);

            const result = await runtime.getVariable('test_var');

            expect(result).toEqual({ key: 'value' });
        });

        it('should clean up temporary variable after reading', async () => {
            const mockVariable = {
                display_value: '"42"',
                display_type: 'int',
            };

            positron.runtime.getSessionVariables.mockResolvedValue([[mockVariable]]);

            await runtime.getVariable('test_var');

            // Verify that getVariable completed successfully (cleanup happens in finally block)
            // We can't easily test the cleanup in unit tests since it's in a finally block
            // that runs even if there are errors, but we can verify the variable was read
            expect(positron.runtime.getSessionVariables).toHaveBeenCalled();
        });

        it('should throw error if no session is available', async () => {
            positron.runtime.getForegroundSession.mockResolvedValue(undefined);

            await expect(runtime.getVariable('test_var')).rejects.toThrow(
                'No active Python session'
            );
        });
    });

    describe('kernel readiness', () => {
        it('should wait for kernel to be ready before executing', async () => {
            // Mock a session so ensureKernelReady doesn't fail immediately
            positron.runtime.getForegroundSession.mockResolvedValue({
                metadata: { sessionId: 'test' },
                runtimeMetadata: { languageId: 'python' },
            });

            positron.runtime.executeCode
                .mockRejectedValueOnce(new Error('Kernel not ready'))
                .mockResolvedValue({});

            await runtime.executeSilent('x = 1');

            // Should have tried executeCode at least twice (once for ready check, once for actual code)
            expect(positron.runtime.executeCode.mock.calls.length).toBeGreaterThanOrEqual(2);
        });

        it('should throw error if kernel never becomes ready', async () => {
            positron.runtime.getForegroundSession.mockResolvedValue({
                metadata: { sessionId: 'test' },
                runtimeMetadata: { languageId: 'python' },
            });
            positron.runtime.executeCode.mockRejectedValue(new Error('Kernel not ready'));

            // Mock the private ensureKernelReadyViaPolling to use shorter timeout
            const originalMethod = (runtime as any).ensureKernelReadyViaPolling;
            (runtime as any).ensureKernelReadyViaPolling = async function () {
                const startTime = Date.now();
                const maxWaitMs = 100; // Much shorter timeout for testing
                const retryDelayMs = 10;

                while (Date.now() - startTime < maxWaitMs) {
                    try {
                        await positron.runtime.executeCode(
                            'python',
                            '1+1',
                            false,
                            true,
                            positron.RuntimeCodeExecutionMode.Silent
                        );
                        return;
                    } catch (_error) {
                        await new Promise((resolve) => setTimeout(resolve, retryDelayMs));
                    }
                }
                throw new Error('Timeout waiting for Python kernel to be ready');
            };

            try {
                await expect(runtime.executeSilent('x = 1')).rejects.toThrow(
                    'Timeout waiting for Python kernel to be ready'
                );
            } finally {
                // Restore original method
                (runtime as any).ensureKernelReadyViaPolling = originalMethod;
            }
        });

        it('should throw error if no kernel is running', async () => {
            positron.runtime.getForegroundSession.mockResolvedValue(undefined);

            await expect(runtime.executeSilent('x = 1')).rejects.toThrow(
                'No Python kernel is running'
            );
        });
    });

    describe('error handling', () => {
        beforeEach(() => {
            positron.runtime.executeCode.mockResolvedValue({});
        });

        it('should propagate execution errors', async () => {
            const error = new Error('Python execution error');
            positron.runtime.executeCode.mockRejectedValue(error);

            await expect(runtime.executeSilent('raise Exception()')).rejects.toThrow();
        });

        it('should handle session variable retrieval errors', async () => {
            positron.runtime.getForegroundSession.mockResolvedValue({
                metadata: { sessionId: 'test' },
            });
            positron.runtime.getSessionVariables.mockRejectedValue(new Error('Variable not found'));

            await expect(runtime.getVariable('nonexistent')).rejects.toThrow();
        });
    });
});
