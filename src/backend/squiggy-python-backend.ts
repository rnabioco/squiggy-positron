/**
 * Python Backend Communication
 *
 * Manages subprocess communication with the Python JSON-RPC server
 */

import { spawn, ChildProcess } from 'child_process';
import * as readline from 'readline';

interface JSONRPCRequest {
    jsonrpc: '2.0';
    method: string;
    params?: unknown;
    id: number;
}

interface JSONRPCResponse {
    jsonrpc: '2.0';
    result?: unknown;
    error?: {
        code: number;
        message: string;
        data?: unknown;
    };
    id: number;
}

export class PythonBackend {
    private process: ChildProcess | null = null;
    private requestId = 0;
    private pendingRequests = new Map<
        number,
        {
            resolve: (result: unknown) => void;
            reject: (error: Error) => void;
        }
    >();
    private lineReader: readline.Interface | null = null;
    private startupTimer: NodeJS.Timeout | null = null;
    private requestTimers = new Map<number, NodeJS.Timeout>();

    constructor(
        private pythonPath: string,
        private serverScriptPath: string
    ) {}

    /**
     * Start the Python backend process
     */
    async start(): Promise<void> {
        return new Promise((resolve, reject) => {
            console.log(`Starting Python backend: ${this.pythonPath} ${this.serverScriptPath}`);

            this.process = spawn(this.pythonPath, [this.serverScriptPath], {
                stdio: ['pipe', 'pipe', 'pipe'],
            });

            if (!this.process.stdin || !this.process.stdout || !this.process.stderr) {
                reject(new Error('Failed to create Python process pipes'));
                return;
            }

            // Setup line reader for stdout
            this.lineReader = readline.createInterface({
                input: this.process.stdout,
                crlfDelay: Infinity,
            });

            this.lineReader.on('line', (line) => {
                this.handleResponse(line);
            });

            // Handle stderr (logging)
            this.process.stderr.on('data', (data) => {
                console.log(`[Python Backend] ${data.toString()}`);
            });

            // Handle process exit
            this.process.on('exit', (code) => {
                console.log(`Python backend exited with code ${code}`);
                this.process = null;

                // Clear all request timers
                for (const timer of this.requestTimers.values()) {
                    clearTimeout(timer);
                }
                this.requestTimers.clear();

                // Reject all pending requests
                for (const [_id, callbacks] of this.pendingRequests) {
                    callbacks.reject(new Error('Python backend process exited'));
                }
                this.pendingRequests.clear();
            });

            // Handle errors
            this.process.on('error', (error) => {
                console.error('Python backend error:', error);
                reject(error);
            });

            // Give it a moment to start, then resolve
            this.startupTimer = setTimeout(() => resolve(), 500);
        });
    }

    /**
     * Stop the Python backend process
     */
    stop(): void {
        // Clear startup timer if it exists
        if (this.startupTimer) {
            clearTimeout(this.startupTimer);
            this.startupTimer = null;
        }

        // Clear all request timers
        for (const timer of this.requestTimers.values()) {
            clearTimeout(timer);
        }
        this.requestTimers.clear();

        if (this.process) {
            if (typeof this.process.kill === 'function') {
                this.process.kill();
            }
            this.process = null;
        }
        if (this.lineReader) {
            this.lineReader.close();
            this.lineReader = null;
        }
    }

    /**
     * Call a method on the Python backend
     */
    async call(method: string, params?: unknown): Promise<unknown> {
        if (!this.process || !this.process.stdin) {
            throw new Error('Python backend is not running');
        }

        const id = ++this.requestId;
        const request: JSONRPCRequest = {
            jsonrpc: '2.0',
            method,
            params,
            id,
        };

        return new Promise((resolve, reject) => {
            // Store callbacks
            this.pendingRequests.set(id, { resolve, reject });

            // Send request
            const requestJson = JSON.stringify(request) + '\n';
            this.process!.stdin!.write(requestJson);

            // Set timeout and track it
            const timer = setTimeout(() => {
                if (this.pendingRequests.has(id)) {
                    this.pendingRequests.delete(id);
                    this.requestTimers.delete(id);
                    reject(new Error(`Request timeout for method: ${method}`));
                }
            }, 30000); // 30 second timeout

            this.requestTimers.set(id, timer);
        });
    }

    /**
     * Handle a response from the Python backend
     */
    private handleResponse(line: string): void {
        try {
            const response: JSONRPCResponse = JSON.parse(line);

            const callbacks = this.pendingRequests.get(response.id);
            if (!callbacks) {
                console.warn(`Received response for unknown request ID: ${response.id}`);
                return;
            }

            this.pendingRequests.delete(response.id);

            // Clear the timeout timer for this request
            const timer = this.requestTimers.get(response.id);
            if (timer) {
                clearTimeout(timer);
                this.requestTimers.delete(response.id);
            }

            if (response.error) {
                callbacks.reject(new Error(response.error.message));
            } else {
                callbacks.resolve(response.result);
            }
        } catch (error) {
            console.error('Failed to parse JSON-RPC response:', error);
            console.error('Line was:', line);
        }
    }
}
