/**
 * Type declarations for Positron API
 *
 * These are minimal types to allow compilation. The actual 'positron' module
 * is only available when running in Positron's Extension Development Host.
 *
 * Synced with: posit-dev/positron src/positron-dts/positron.d.ts
 * (upstream commit 1212541, 2026-06-26)
 *
 * This is intentionally a minimal subset of the upstream API — only the
 * declarations the extension actually consumes. Add to it when adopting a
 * new Positron API rather than vendoring the full 3,600-line upstream file.
 */

declare module 'positron' {
    import * as vscode from 'vscode';

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
    }

    export enum RuntimeErrorBehavior {
        Stop = 'stop',
        Continue = 'continue',
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

    // Upstream models this as `LanguageRuntimeSession extends BaseLanguageRuntimeSession`.
    export interface LanguageRuntimeSession extends BaseLanguageRuntimeSession {
        onDidChangeRuntimeState: vscode.Event<RuntimeState>;
        onDidEndSession: vscode.Event<any>;
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
        runtimeId: string;
        runtimeName: string;
        runtimePath: string;
        runtimeVersion: string;
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
            executionMetadata?: Record<string, any>
        ): Thenable<Record<string, any>>;

        export function evaluateCode(
            languageId: string,
            code: string,
            cancellationToken?: vscode.CancellationToken,
            sessionId?: string
        ): Thenable<EvalResult>;

        // Upstream returns BaseLanguageRuntimeSession; deliberately widened to
        // LanguageRuntimeSession because callers defensively access the optional
        // onDidChangeRuntimeState event (guarded by typeof checks at runtime).
        export function getForegroundSession(): Thenable<LanguageRuntimeSession | undefined>;

        export function getNotebookSession(
            notebookUri: vscode.Uri
        ): Thenable<LanguageRuntimeSession | undefined>;

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
            notebookUri?: vscode.Uri
        ): Thenable<LanguageRuntimeSession>;

        export function interruptSession(sessionId: string): Thenable<void>;

        export function restartSession(sessionId: string): Thenable<boolean>;

        export function focusSession(sessionId: string): void;

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
