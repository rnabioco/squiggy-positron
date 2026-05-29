/**
 * Mock for Positron runtime API
 *
 * Used in tests to mock Positron-specific functionality
 */

export const runtime = {
    getForegroundSession: jest.fn(),
    executeCode: jest.fn(),
    evaluateCode: jest.fn(),
    getSessionVariables: jest.fn(),
};

export enum RuntimeCodeExecutionMode {
    Interactive = 'interactive',
    NonInteractive = 'non-interactive',
    Transient = 'transient',
    Silent = 'silent',
}

export enum RuntimeErrorBehavior {
    Stop = 'stop',
    Continue = 'continue',
}

export interface ExecutionObserver {
    token?: any;
    onStarted?: () => void;
    onOutput?: (message: string) => void;
    onError?: (message: string) => void;
    onPlot?: (plotData: string) => void;
    onData?: (data: any) => void;
    onCompleted?: (result: Record<string, any>) => void;
    onFailed?: (error: Error) => void;
    onFinished?: () => void;
}
