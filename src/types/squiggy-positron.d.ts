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

    export enum RuntimeState {
        Uninitialized = 'uninitialized',
        Initializing = 'initializing',
        Starting = 'starting',
        Ready = 'ready',
        Idle = 'idle',
        Busy = 'busy',
        Restarting = 'restarting',
        Offline = 'offline',
        Exited = 'exited',
    }

    export interface LanguageRuntimeSession {
        metadata: { sessionId: string; sessionName: string; sessionMode: string };
        runtimeMetadata: { languageId: string };
        onDidChangeRuntimeState: import('vscode').Event<RuntimeState>;
        onDidEndSession: import('vscode').Event<any>;
    }

    export namespace runtime {
        export function executeCode(
            languageId: string,
            code: string,
            focus: boolean,
            allowIncomplete?: boolean,
            mode?: RuntimeCodeExecutionMode,
            errorBehavior?: RuntimeErrorBehavior,
            observer?: RuntimeCodeExecutionObserver,
            sessionId?: string
        ): Thenable<Record<string, any>>;

        export function getForegroundSession(): Thenable<LanguageRuntimeSession | undefined>;

        export function getSessionVariables(
            sessionId: string,
            accessKeys?: Array<Array<string>>
        ): Thenable<Array<Array<RuntimeVariable>>>;

        /**
         * Get the preferred runtime for a given language
         */
        export function getPreferredRuntime(
            languageId: string
        ): Thenable<LanguageRuntimeMetadata | undefined>;

        /**
         * Get all registered runtimes
         */
        export function getRegisteredRuntimes(): Thenable<LanguageRuntimeMetadata[]>;

        /**
         * Select a language runtime by its ID
         * This will start a new session with the specified runtime
         */
        export function selectLanguageRuntime(runtimeId: string): Thenable<void>;

        /**
         * Start a new language runtime session
         */
        export function startLanguageRuntime(
            runtimeId: string,
            sessionName: string,
            notebookUri?: import('vscode').Uri
        ): Thenable<LanguageRuntimeSession>;

        /**
         * Restart an existing session
         */
        export function restartSession(sessionId: string): Thenable<void>;

        /**
         * Delete/shutdown a session
         */
        export function deleteSession(sessionId: string): Thenable<void>;

        /**
         * List all active sessions
         */
        export function getActiveSessions(): Thenable<BaseLanguageRuntimeSession[]>;

        /**
         * Get a specific session by its ID
         */
        export function getSession(
            sessionId: string
        ): Thenable<BaseLanguageRuntimeSession | undefined>;

        /**
         * Event that fires when the foreground session changes (including kernel restarts)
         */
        export const onDidChangeForegroundSession: import('vscode').Event<string | undefined>;
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

    export namespace window {
        /**
         * Show a simple modal dialog with a custom prompt and action button
         * @param title Dialog title
         * @param message Dialog message (supports HTML)
         * @param okButtonTitle Text for the OK/action button
         * @returns Promise<boolean> true if user clicked OK, false if cancelled
         */
        export function showSimpleModalDialogPrompt(
            title: string,
            message: string,
            okButtonTitle: string
        ): Thenable<boolean>;
    }
}
