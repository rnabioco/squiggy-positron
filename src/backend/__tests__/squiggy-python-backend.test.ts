/**
 * Tests for Python Backend Communication
 */

import { describe, it, expect, beforeEach, jest, afterEach } from '@jest/globals';
import { PythonBackend } from '../squiggy-python-backend';
import { EventEmitter } from 'events';

// Mock child_process
jest.mock('child_process', () => {
    const EventEmitter = require('events').EventEmitter;

    class MockChildProcess extends EventEmitter {
        stdin: any = {
            write: jest.fn(),
        };
        stdout = new EventEmitter();
        stderr = new EventEmitter();
        kill = jest.fn();
    }

    return {
        spawn: jest.fn(() => new MockChildProcess()),
    };
});

// Mock readline
jest.mock('readline', () => ({
    createInterface: jest.fn((config: any) => {
        const EventEmitter = require('events').EventEmitter;
        const emitter = new EventEmitter();
        (emitter as any).close = jest.fn();

        // Simulate line events when data comes in on stdout
        config.input.on('data', (data: Buffer) => {
            const lines = data.toString().split('\n');
            lines.forEach((line: string) => {
                if (line.trim()) {
                    emitter.emit('line', line);
                }
            });
        });

        return emitter;
    }),
}));

describe('PythonBackend', () => {
    let backend: PythonBackend;
    let mockChildProcess: any;
    const mockPythonPath = '/usr/bin/python3';
    const mockServerScript = '/path/to/server.py';

    beforeEach(() => {
        // Use legacy fake timers to prevent leaks and speed up tests
        jest.useFakeTimers({ legacyFakeTimers: true });

        // Silence console output during tests
        jest.spyOn(console, 'log').mockImplementation(() => {});
        jest.spyOn(console, 'warn').mockImplementation(() => {});
        jest.spyOn(console, 'error').mockImplementation(() => {});

        const { spawn } = require('child_process');
        backend = new PythonBackend(mockPythonPath, mockServerScript);

        // Get reference to the mock process that will be created
        mockChildProcess = null;
        (spawn as jest.Mock).mockImplementation(() => {
            const EventEmitter = require('events').EventEmitter;

            class MockChildProcess extends EventEmitter {
                stdin: any = {
                    write: jest.fn(),
                };
                stdout = new EventEmitter();
                stderr = new EventEmitter();
                kill = jest.fn();
            }

            mockChildProcess = new MockChildProcess();
            return mockChildProcess;
        });
    });

    afterEach(() => {
        // Clean up backend to clear any pending timers
        backend.stop();

        // Run only pending timers to avoid hanging
        jest.runOnlyPendingTimers();

        // Restore real timers
        jest.useRealTimers();

        // Clear all mocks
        jest.clearAllMocks();
    });

    describe('start', () => {
        it('should start the Python backend process', async () => {
            const { spawn } = require('child_process');
            const startPromise = backend.start();

            // Process should spawn
            expect(spawn).toHaveBeenCalledWith(mockPythonPath, [mockServerScript], {
                stdio: ['pipe', 'pipe', 'pipe'],
            });

            // Fast-forward the 500ms startup timer
            jest.advanceTimersByTime(500);

            await startPromise;
        });

        it('should reject if process pipes are not created', async () => {
            const { spawn } = require('child_process');
            (spawn as jest.Mock).mockImplementation(() => ({
                stdin: null,
                stdout: null,
                stderr: null,
            }));

            await expect(backend.start()).rejects.toThrow('Failed to create Python process pipes');
        });
    });

    describe('stop', () => {
        it('should kill the process and close readline', async () => {
            const startPromise = backend.start();
            jest.advanceTimersByTime(500);
            await startPromise;

            backend.stop();

            expect(mockChildProcess.kill).toHaveBeenCalled();
        });

        it('should handle stop when process is not running', () => {
            expect(() => backend.stop()).not.toThrow();
        });
    });

    describe('call', () => {
        beforeEach(async () => {
            const startPromise = backend.start();
            jest.advanceTimersByTime(500);
            await startPromise;
        });

        it('should send JSON-RPC request and return result', async () => {
            const method = 'test_method';
            const params = { arg1: 'value1' };

            const callPromise = backend.call(method, params);

            // Immediately emit the response (no need to wait with fake timers)
            const response = {
                jsonrpc: '2.0',
                result: { success: true },
                id: 1,
            };
            mockChildProcess.stdout.emit('data', Buffer.from(JSON.stringify(response) + '\n'));

            const result = await callPromise;

            expect(result).toEqual({ success: true });
            expect(mockChildProcess.stdin.write).toHaveBeenCalledWith(
                expect.stringContaining('"method":"test_method"')
            );
            expect(mockChildProcess.stdin.write).toHaveBeenCalledWith(
                expect.stringContaining('"params":{"arg1":"value1"}')
            );
        });

        it('should reject on JSON-RPC error response', async () => {
            const method = 'failing_method';

            const callPromise = backend.call(method);

            // Immediately emit the error response
            const response = {
                jsonrpc: '2.0',
                error: {
                    code: -32600,
                    message: 'Invalid Request',
                },
                id: 1,
            };
            mockChildProcess.stdout.emit('data', Buffer.from(JSON.stringify(response) + '\n'));

            await expect(callPromise).rejects.toThrow('Invalid Request');
        });

        it('should reject on timeout', async () => {
            const method = 'slow_method';

            // Don't send any response - let it timeout
            const promise = backend.call(method);

            expect(mockChildProcess.stdin.write).toHaveBeenCalledWith(
                expect.stringContaining('"method":"slow_method"')
            );

            // Fast-forward past the 30 second timeout
            jest.advanceTimersByTime(30000);

            await expect(promise).rejects.toThrow('Request timeout for method: slow_method');
        });

        it('should throw error when backend is not running', async () => {
            backend.stop();

            await expect(backend.call('test')).rejects.toThrow('Python backend is not running');
        });

        it('should handle multiple concurrent requests', async () => {
            // Send multiple requests
            const promise1 = backend.call('method1');
            const promise2 = backend.call('method2');
            const promise3 = backend.call('method3');

            // Send responses in different order (immediately with fake timers)
            mockChildProcess.stdout.emit(
                'data',
                Buffer.from(
                    JSON.stringify({
                        jsonrpc: '2.0',
                        result: { value: 2 },
                        id: 2,
                    }) + '\n'
                )
            );

            mockChildProcess.stdout.emit(
                'data',
                Buffer.from(
                    JSON.stringify({
                        jsonrpc: '2.0',
                        result: { value: 1 },
                        id: 1,
                    }) + '\n'
                )
            );

            mockChildProcess.stdout.emit(
                'data',
                Buffer.from(
                    JSON.stringify({
                        jsonrpc: '2.0',
                        result: { value: 3 },
                        id: 3,
                    }) + '\n'
                )
            );

            const [result1, result2, result3] = await Promise.all([promise1, promise2, promise3]);

            expect(result1).toEqual({ value: 1 });
            expect(result2).toEqual({ value: 2 });
            expect(result3).toEqual({ value: 3 });
        });
    });

    describe('process exit handling', () => {
        it('should reject pending requests when process exits', async () => {
            const startPromise = backend.start();
            jest.advanceTimersByTime(500);
            await startPromise;

            const promise = backend.call('test_method');

            // Simulate process exit
            mockChildProcess.emit('exit', 1);

            await expect(promise).rejects.toThrow('Python backend process exited');
        });
    });
});
