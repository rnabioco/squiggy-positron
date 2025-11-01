/**
 * Type declarations for Positron API
 *
 * These are minimal types to allow compilation. The actual 'positron' module
 * is only available when running in Positron's Extension Development Host.
 */

declare module 'positron' {
    export interface RuntimeCodeExecutionObserver {
        onOutput?: (message: string) => void;
        onError?: (message: string) => void;
        onFinished?: () => void;
        token?: any; // CancellationToken
    }

    export enum RuntimeCodeExecutionMode {
        Interactive = 'interactive',
        Silent = 'silent',
    }

    export enum RuntimeErrorBehavior {
        Stop = 'stop',
        Continue = 'continue',
    }

    export namespace runtime {
        export function executeCode(
            languageId: string,
            code: string,
            focus: boolean,
            allowIncomplete?: boolean,
            mode?: RuntimeCodeExecutionMode,
            errorBehavior?: RuntimeErrorBehavior,
            observer?: RuntimeCodeExecutionObserver
        ): Thenable<Record<string, any>>;
    }

    export interface LanguageRuntimeMetadata {
        runtimeId: string;
        runtimeName: string;
        runtimePath: string;
        runtimeVersion: string;
        languageId: string;
        languageName: string;
        languageVersion: string;
    }
}
