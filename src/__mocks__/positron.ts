/**
 * Mock for Positron runtime API
 *
 * Used in tests to mock Positron-specific functionality
 */

export const runtime = {
    getForegroundSession: jest.fn(),
    executeCode: jest.fn(),
    getSessionVariables: jest.fn(),
};

export enum RuntimeCodeExecutionMode {
    Silent = 'silent',
    Interactive = 'interactive',
}

export enum RuntimeErrorBehavior {
    Stop = 'stop',
    Continue = 'continue',
}

export interface RuntimeCodeExecutionObserver {
    onResult?: (result: any) => void;
    onError?: (error: any) => void;
    onOutput?: (output: string) => void;
}
