/**
 * Mock VSCode API for testing
 */

export class TreeItem {
    label: string;
    collapsibleState: TreeItemCollapsibleState | number;
    tooltip?: string;
    contextValue?: string;
    iconPath?: ThemeIcon | string | { light: string; dark: string };
    command?: Command;

    constructor(label: string, collapsibleState: TreeItemCollapsibleState | number) {
        this.label = label;
        this.collapsibleState = collapsibleState;
    }
}

export enum TreeItemCollapsibleState {
    None = 0,
    Collapsed = 1,
    Expanded = 2,
}

export class EventEmitter<T> {
    private listeners: ((e: T) => void)[] = [];

    get event() {
        return (listener: (e: T) => void) => {
            this.listeners.push(listener);
            return { dispose: () => {} };
        };
    }

    fire(data: T) {
        this.listeners.forEach((l) => l(data));
    }
}

export class ThemeIcon {
    constructor(public id: string) {}
}

export interface Command {
    command: string;
    title: string;
    arguments?: unknown[];
}

export class Uri {
    scheme: string;
    authority: string;
    path: string;
    query: string;
    fragment: string;
    fsPath: string;

    private constructor(
        scheme: string,
        authority: string,
        path: string,
        query: string,
        fragment: string
    ) {
        this.scheme = scheme;
        this.authority = authority;
        this.path = path;
        this.query = query;
        this.fragment = fragment;
        this.fsPath = path;
    }

    static file(path: string): Uri {
        return new Uri('file', '', path, '', '');
    }

    static parse(value: string): Uri {
        const match = value.match(/^(\w+):\/\/([^/]*)(.*)$/);
        if (match) {
            return new Uri(match[1], match[2], match[3], '', '');
        }
        return new Uri('file', '', value, '', '');
    }

    static joinPath(uri: Uri, ...pathSegments: string[]): Uri {
        const newPath = [uri.path, ...pathSegments].join('/').replace(/\/+/g, '/');
        return new Uri(uri.scheme, uri.authority, newPath, uri.query, uri.fragment);
    }

    with(change: {
        scheme?: string;
        authority?: string;
        path?: string;
        query?: string;
        fragment?: string;
    }): Uri {
        return new Uri(
            change.scheme ?? this.scheme,
            change.authority ?? this.authority,
            change.path ?? this.path,
            change.query ?? this.query,
            change.fragment ?? this.fragment
        );
    }

    toString(): string {
        return `${this.scheme}://${this.authority}${this.path}`;
    }
}

export interface WebviewOptions {
    enableScripts?: boolean;
    localResourceRoots?: Uri[];
}

export interface Webview {
    options: WebviewOptions;
    html: string;
    asWebviewUri(uri: Uri): Uri;
    postMessage(message: any): Thenable<boolean>;
    onDidReceiveMessage: any;
    cspSource: string;
}

export interface WebviewView {
    webview: Webview;
    visible: boolean;
    onDidChangeVisibility: any;
}

export interface WebviewViewProvider {
    resolveWebviewView(webviewView: WebviewView, context: any, token: any): void | Thenable<void>;
}

// Export ProgressLocation enum for error handler
export enum ProgressLocation {
    SourceControl = 1,
    Window = 10,
    Notification = 15,
}

// Mock Color Theme Kind
export enum ColorThemeKind {
    Light = 1,
    Dark = 2,
    HighContrast = 3,
}

// Mock window API
export const window = {
    showErrorMessage: jest.fn(),
    showInformationMessage: jest.fn(),
    showWarningMessage: jest.fn(),
    showOpenDialog: jest.fn(),
    withProgress: jest.fn((options, task) => task()),
    activeColorTheme: {
        kind: ColorThemeKind.Light,
    },
    onDidChangeActiveColorTheme: jest.fn(() => ({ dispose: jest.fn() })),
    registerWebviewViewProvider: jest.fn(() => ({ dispose: jest.fn() })),
};

// Command registry for testing
const commandRegistry = new Map<string, (...args: any[]) => any>();

// Mock commands API
export const commands = {
    registerCommand: jest.fn((command: string, callback: (...args: any[]) => any) => {
        commandRegistry.set(command, callback);
        return { dispose: jest.fn() };
    }),
    executeCommand: jest.fn(async (command: string, ...args: any[]) => {
        const handler = commandRegistry.get(command);
        if (handler) {
            return await handler(...args);
        }
        throw new Error(`Command '${command}' not found`);
    }),
    getCommands: jest.fn(async (_filterInternal?: boolean) => {
        return Array.from(commandRegistry.keys());
    }),
};
