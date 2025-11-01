/**
 * Shared VSCode API instance
 *
 * acquireVsCodeApi() can only be called once per webview,
 * so we acquire it here and export it for all components to use.
 */

// VSCode API (injected by webview)
declare const acquireVsCodeApi: () => {
    postMessage: (message: any) => void;
    getState: () => any;
    setState: (state: any) => void;
};

// Acquire the API once and export it
export const vscode = acquireVsCodeApi();
