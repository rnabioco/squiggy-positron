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

    export interface BaseLanguageRuntimeSession {
        metadata: {
            sessionId: string;
            sessionName: string;
            sessionMode: string;
        };
        runtimeMetadata: {
            languageId: string;
            languageName: string;
            runtimeId: string;
            runtimeName: string;
            runtimeVersion: string;
        };
    }

    export interface RuntimeVariable {
        access_key: string[];
        display_name: string;
        display_value: string;
        display_type: string;
        type_info: string;
        size: number;
        kind: number;
        length: number;
        has_children: boolean;
        has_viewer: boolean;
        is_truncated: boolean;
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

        export function getForegroundSession(): Thenable<BaseLanguageRuntimeSession | undefined>;

        export function getSessionVariables(
            sessionId: string,
            accessKeys?: Array<Array<string>>
        ): Thenable<Array<Array<RuntimeVariable>>>;
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
