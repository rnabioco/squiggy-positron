/**
 * Status Bar Messenger
 *
 * Shows transient informational messages in the status bar instead of pop-ups.
 * Reduces notification noise by displaying routine confirmations in a less
 * intrusive location, while errors/warnings remain as pop-ups.
 */

import * as vscode from 'vscode';

/**
 * Singleton class that manages status bar messages
 */
class StatusBarMessenger {
    private static _instance: StatusBarMessenger | undefined;
    private _statusItem: vscode.StatusBarItem;
    private _messageTimeout?: NodeJS.Timeout;
    private _defaultText: string = '$(squirrel) Squiggy';
    private _defaultTooltip: string = 'Squiggy extension ready';
    private _isDisposed: boolean = false;

    private constructor() {
        // Create status bar item on the right side, lower priority than kernel status
        this._statusItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 50);
        this._statusItem.text = this._defaultText;
        this._statusItem.tooltip = this._defaultTooltip;
        this._statusItem.show();
    }

    /**
     * Get the singleton instance
     */
    static getInstance(): StatusBarMessenger {
        if (!StatusBarMessenger._instance) {
            StatusBarMessenger._instance = new StatusBarMessenger();
        }
        return StatusBarMessenger._instance;
    }

    /**
     * Initialize the messenger (should be called during extension activation)
     * Returns the disposable for cleanup
     */
    static initialize(): vscode.Disposable {
        const instance = StatusBarMessenger.getInstance();
        return {
            dispose: () => instance.dispose(),
        };
    }

    /**
     * Show a message with auto-hide after ~8 seconds
     * @param text - Message text (e.g., "Exported")
     * @param icon - Optional codicon (e.g., "check", "export"). If not provided, defaults to no icon.
     */
    showMessage(text: string, icon?: string): void {
        if (this._isDisposed) {
            return;
        }

        // Clear any existing timeout
        if (this._messageTimeout) {
            clearTimeout(this._messageTimeout);
        }

        // Set the message with optional icon
        const displayText = icon ? `$(${icon}) ${text}` : text;
        this._statusItem.text = displayText;
        this._statusItem.tooltip = text;

        // Auto-hide after 8 seconds
        this._messageTimeout = setTimeout(() => {
            this.clear();
        }, 8000);
    }

    /**
     * Show a message that persists until explicitly cleared
     * @param text - Message text
     * @param icon - Optional codicon
     */
    showPersistent(text: string, icon?: string): void {
        if (this._isDisposed) {
            return;
        }

        // Clear any existing timeout
        if (this._messageTimeout) {
            clearTimeout(this._messageTimeout);
            this._messageTimeout = undefined;
        }

        // Clear error state
        this._statusItem.backgroundColor = undefined;

        // Set the message with optional icon
        const displayText = icon ? `$(${icon}) ${text}` : text;
        this._statusItem.text = displayText;
        this._statusItem.tooltip = text;
    }

    /**
     * Show an error message with red background that persists until cleared
     * @param text - Error message text
     */
    showError(text: string): void {
        if (this._isDisposed) {
            return;
        }

        // Clear any existing timeout
        if (this._messageTimeout) {
            clearTimeout(this._messageTimeout);
            this._messageTimeout = undefined;
        }

        // Set error styling with red background
        this._statusItem.backgroundColor = new vscode.ThemeColor('statusBarItem.errorBackground');
        this._statusItem.text = `$(error) ${text}`;
        this._statusItem.tooltip = `Error: ${text}\n\nClick to dismiss`;
        this._statusItem.command = 'squiggy.clearStatusBarError';
    }

    /**
     * Clear message and show default
     */
    clear(): void {
        if (this._isDisposed) {
            return;
        }

        if (this._messageTimeout) {
            clearTimeout(this._messageTimeout);
            this._messageTimeout = undefined;
        }

        this._statusItem.backgroundColor = undefined;
        this._statusItem.command = undefined;
        this._statusItem.text = this._defaultText;
        this._statusItem.tooltip = this._defaultTooltip;
    }

    /**
     * Dispose of the status bar item
     */
    dispose(): void {
        if (this._isDisposed) {
            return;
        }

        this._isDisposed = true;

        if (this._messageTimeout) {
            clearTimeout(this._messageTimeout);
        }

        this._statusItem.dispose();
        StatusBarMessenger._instance = undefined;
    }
}

// Export the singleton instance getter for convenience
export const statusBarMessenger = {
    /**
     * Initialize the status bar messenger (call during extension activation)
     */
    initialize: (): vscode.Disposable => StatusBarMessenger.initialize(),

    /**
     * Show a message with auto-hide after ~8 seconds
     */
    show: (text: string, icon?: string): void =>
        StatusBarMessenger.getInstance().showMessage(text, icon),

    /**
     * Show a message that persists until explicitly cleared
     */
    showPersistent: (text: string, icon?: string): void =>
        StatusBarMessenger.getInstance().showPersistent(text, icon),

    /**
     * Show an error with red background (persists until clicked)
     */
    showError: (text: string): void => StatusBarMessenger.getInstance().showError(text),

    /**
     * Clear message and show default
     */
    clear: (): void => StatusBarMessenger.getInstance().clear(),
};
