/**
 * Type declarations for Positron API
 *
 * These are minimal types to allow compilation. The actual 'positron' module
 * is only available when running in Positron's Extension Development Host.
 *
 * Synced with: posit-dev/positron src/positron-dts/positron.d.ts
 * (main as of 2026-10-11; Positron 2026.10.0 release)
 *
 * This is intentionally a minimal subset of the upstream API — only the
 * declarations the extension actually consumes. Add to it when adopting a
 * new Positron API rather than vendoring the full ~5,500-line upstream file.
 *
 * Members added in recent Positron releases are marked with the release that
 * introduced them. Callers must feature-detect (or rely on the extra argument
 * being ignored) when engines.positron allows older versions.
 */

declare module 'positron' {
    import * as vscode from 'vscode';

    /** The current Positron calendar version, e.g. "2026.10.0". */
    export const version: string;

    /** Monotonically increasing build number within a release. */
    export const buildNumber: number;

    // --- Execution Observer ---

    /**
     * An object that observes an ongoing code execution invoked from the
     * `executeCode` API.
     */
    export interface ExecutionObserver {
        /** Optional cancellation token to cancel the execution. */
        token?: vscode.CancellationToken;

        /** Called when execution has actually started (may differ from when executeCode was called). */
        onStarted?: () => void;

        /** Called when execution emits text output (zero or more times). */
        onOutput?: (message: string) => void;

        /** Called when execution emits stderr output (zero or more times). */
        onError?: (message: string) => void;

        /** Called when execution emits a static plot. */
        onPlot?: (plotData: string) => void;

        /** Called when execution emits rectangular data. NOTE: Not currently fired. */
        onData?: (data: any) => void;

        /** Called when execution completed successfully. */
        onCompleted?: (result: Record<string, any>) => void;

        /** Called when execution failed. */
        onFailed?: (error: Error) => void;

        /** Called when execution finished, regardless of success or failure. */
        onFinished?: () => void;
    }

    // --- Enums ---

    export enum RuntimeCodeExecutionMode {
        /** Displayed, combined with pending code, stored in history. */
        Interactive = 'interactive',
        /** Displayed, not combined, stored in history. */
        NonInteractive = 'non-interactive',
        /** Displayed, not combined, NOT stored in history. */
        Transient = 'transient',
        /** NOT displayed, not combined, NOT stored in history. */
        Silent = 'silent',
        /** Like Interactive, but completeness is checked by the session. (2026.09+) */
        Unprocessed = 'unprocessed',
    }

    /** Error disposition. Upstream notes this is currently ignored by kernels. */
    export enum RuntimeErrorBehavior {
        Stop = 'stop',
        Continue = 'continue',
    }

    /** What evaluateCode() does when the target runtime is busy. (2026.09+) */
    export enum RuntimeBusyBehavior {
        Queue = 'queue',
        Reject = 'reject',
    }

    export enum LanguageRuntimeSessionMode {
        Console = 'console',
        Notebook = 'notebook',
        /** Not creatable through the public API (startLanguageRuntime). */
        Background = 'background',
    }

    export enum RuntimeExitReason {
        StartupFailed = 'startupFailed',
        Shutdown = 'shutdown',
        ForcedQuit = 'forcedQuit',
        Restart = 'restart',
        SwitchRuntime = 'switchRuntime',
        Error = 'error',
        Transferred = 'transferred',
        ExtensionHost = 'extensionHost',
        Unknown = 'unknown',
    }

    export enum RuntimeState {
        Uninitialized = 'uninitialized',
        Initializing = 'initializing',
        Starting = 'starting',
        Ready = 'ready',
        Idle = 'idle',
        Busy = 'busy',
        Restarting = 'restarting',
        Exiting = 'exiting',
        Exited = 'exited',
        Offline = 'offline',
        Interrupting = 'interrupting',
    }

    export enum RuntimeClientType {
        Variables = 'positron.variables',
        Lsp = 'positron.lsp',
        Plot = 'positron.plot',
        DataExplorer = 'positron.dataExplorer',
        Ui = 'positron.ui',
        Help = 'positron.help',
        IPyWidget = 'positron.ipyWidget',
        Connection = 'positron.connection',
    }

    export enum RuntimeClientState {
        Uninitialized = 'uninitialized',
        Opening = 'opening',
        Connected = 'connected',
        Closing = 'closing',
        Closed = 'closed',
    }

    // --- Interfaces ---

    /** Who a session belongs to. (2026.10+) */
    export type RuntimeSessionOwner = 'user' | 'agent';

    /** Options for {@link runtime.startLanguageRuntime}. (2026.10+) */
    export interface RuntimeSessionStartOptions {
        /** Who the session belongs to. Defaults to `user`. */
        readonly owner?: RuntimeSessionOwner;
        /**
         * Whether the session becomes the foreground session once started.
         * Defaults to `true`. Pass `false` to leave the user's session in place.
         */
        readonly activate?: boolean;
    }

    export interface LanguageRuntimeExit {
        runtime_name: string;
        session_name: string;
        exit_code: number;
        reason: RuntimeExitReason;
        message: string;
    }

    export interface BaseLanguageRuntimeSession {
        // Upstream: `readonly metadata: RuntimeSessionMetadata`. `sessionName` is
        // not part of upstream RuntimeSessionMetadata (it lives in getDynState()),
        // so it is optional here.
        readonly metadata: {
            readonly sessionId: string;
            readonly sessionMode: LanguageRuntimeSessionMode;
            readonly sessionName?: string;
            readonly notebookUri?: vscode.Uri;
            readonly workingDirectory?: string;
        };
        readonly runtimeMetadata: LanguageRuntimeMetadata;

        /** Last known runtime state; synchronous. (2026.09+) */
        getRuntimeState?(): RuntimeState;

        /** Fires on runtime state transitions. (2026.09+ on the base interface) */
        onDidChangeRuntimeState?: vscode.Event<RuntimeState>;

        /** Fires when the connection to the runtime is lost. (2026.09+) */
        onDidDisconnect?: vscode.Event<void>;

        /** Fires when the connection is re-established. (2026.09+) */
        onDidReconnect?: vscode.Event<void>;
    }

    export interface LanguageRuntimeSession extends BaseLanguageRuntimeSession {
        onDidChangeRuntimeState: vscode.Event<RuntimeState>;
        onDidEndSession: vscode.Event<LanguageRuntimeExit>;
    }

    export interface RuntimeVariable {
        access_key: string;
        display_name: string;
        display_value: string;
        display_type: string;
        type_info?: string;
        size: number;
        length: number;
        has_children: boolean;
    }

    export interface EvalResult {
        /** The value resulting from the code evaluation. */
        result: any;
        /** The output emitted during code evaluation, if any. */
        output: string;
    }

    export interface QueryTableSummaryResult {
        num_rows: number;
        num_columns: number;
    }

    export interface RuntimeClientOutput<T> {
        data: T;
        buffers?: Array<Uint8Array>;
    }

    export interface RuntimeClientInstance extends vscode.Disposable {
        onDidChangeClientState: vscode.Event<RuntimeClientState>;
        onDidSendEvent: vscode.Event<RuntimeClientOutput<object>>;
        performRpcWithBuffers<T>(data: object): Thenable<RuntimeClientOutput<T>>;
        performRpc<T>(data: object): Thenable<T>;
        getClientState(): RuntimeClientState;
        getClientId(): string;
        getClientType(): RuntimeClientType;
    }

    export type RuntimeClientHandlerCallback = (
        client: RuntimeClientInstance,
        params: Object
    ) => boolean;

    export interface RuntimeClientHandler {
        clientType: string;
        callback: RuntimeClientHandlerCallback;
    }

    export interface LanguageRuntimeMetadata {
        /** Absolute, fully expanded path to the runtime executable (never contains ~). */
        runtimePath: string;
        runtimeId: string;
        runtimeName: string;
        runtimeShortName: string;
        runtimeVersion: string;
        /** e.g. "Venv", "Conda", "uv", "System". */
        runtimeSource: string;
        languageId: string;
        languageName: string;
        languageVersion: string;
    }

    // --- Runtime namespace ---

    export namespace runtime {
        export function executeCode(
            languageId: string,
            code: string,
            focus: boolean,
            allowIncomplete?: boolean,
            mode?: RuntimeCodeExecutionMode,
            errorBehavior?: RuntimeErrorBehavior,
            observer?: ExecutionObserver,
            sessionId?: string,
            documentUri?: vscode.Uri,
            executionMetadata?: Record<string, any>,
            /** (2026.09+) */
            attributionMetadata?: Record<string, any>
        ): Thenable<Record<string, any>>;

        /** (2026.04+) */
        export function evaluateCode(
            languageId: string,
            code: string,
            cancellationToken?: vscode.CancellationToken,
            sessionId?: string,
            /** (2026.09+) */
            whenBusy?: RuntimeBusyBehavior
        ): Thenable<EvalResult>;

        export function getForegroundSession(): Thenable<BaseLanguageRuntimeSession | undefined>;

        export function getNotebookSession(
            notebookUri: vscode.Uri
        ): Thenable<BaseLanguageRuntimeSession | undefined>;

        export function getSessionVariables(
            sessionId: string,
            accessKeys?: Array<Array<string>>
        ): Thenable<Array<Array<RuntimeVariable>>>;

        export function querySessionTables(
            sessionId: string,
            accessKeys: Array<Array<string>>,
            queryTypes: Array<string>
        ): Thenable<Array<QueryTableSummaryResult>>;

        export function getPreferredRuntime(
            languageId: string
        ): Thenable<LanguageRuntimeMetadata | undefined>;

        export function getRegisteredRuntimes(): Thenable<LanguageRuntimeMetadata[]>;

        export function selectLanguageRuntime(runtimeId: string): Thenable<void>;

        export function startLanguageRuntime(
            runtimeId: string,
            sessionName: string,
            notebookUri?: vscode.Uri,
            /** (2026.10+; ignored by older versions) */
            options?: RuntimeSessionStartOptions
        ): Thenable<LanguageRuntimeSession>;

        export function interruptSession(sessionId: string): Thenable<void>;

        /**
         * Resolves `true` if restarted, `false` if the user declined to
         * interrupt a busy session. (Returned `void` before 2026.04.)
         */
        export function restartSession(sessionId: string): Thenable<boolean>;

        export function focusSession(sessionId: string): void;

        /** (2025.10+) Resolves `false` if the user declines to interrupt a busy session. */
        export function deleteSession(sessionId: string): Thenable<boolean>;

        export function getActiveSessions(): Thenable<BaseLanguageRuntimeSession[]>;

        export function getSession(
            sessionId: string
        ): Thenable<BaseLanguageRuntimeSession | undefined>;

        export function registerClientHandler(handler: RuntimeClientHandler): vscode.Disposable;

        export function registerClientInstance(clientInstanceId: string): vscode.Disposable;

        export function emitPerfMark(name: string): void;

        export const onDidRegisterRuntime: vscode.Event<LanguageRuntimeMetadata>;

        export const onDidChangeForegroundSession: vscode.Event<string | undefined>;

        export const onDidExecuteCode: vscode.Event<any>;
    }

    // --- Window namespace ---

    export namespace window {
        export function showSimpleModalDialogPrompt(
            title: string,
            message: string,
            okButtonTitle: string
        ): Thenable<boolean>;
    }
}
