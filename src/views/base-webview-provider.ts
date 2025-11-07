/**
 * Base Webview Provider
 *
 * Abstract base class consolidating duplicate webview setup code.
 * All squiggy webview providers should extend this class.
 */

import * as vscode from 'vscode';
import { getReactWebviewHtml, getWebviewOptions } from '../utils/webview-utils';
import { IncomingWebviewMessage, OutgoingWebviewMessage } from '../types/messages';
import { SquiggyError, handleError } from '../utils/error-handler';
import { logger } from '../utils/logger';

/**
 * Abstract base class for webview providers
 *
 * Handles common setup: HTML generation, options, visibility, and message routing.
 * Subclasses implement abstract methods for message handling and view updates.
 */
export abstract class BaseWebviewProvider implements vscode.WebviewViewProvider {
    protected _view?: vscode.WebviewView;

    constructor(protected readonly extensionUri: vscode.Uri) {}

    /**
     * Abstract: Get the title for the webview HTML page
     */
    protected abstract getTitle(): string;

    /**
     * Abstract: Handle incoming messages from the webview
     * @param message Typed message from webview
     */
    protected abstract handleMessage(message: IncomingWebviewMessage): Promise<void>;

    /**
     * Abstract: Update the webview content
     * Called when view becomes visible or state changes
     */
    protected abstract updateView(): void;

    /**
     * Resolve the webview view (called by VSCode when view is created)
     */
    public resolveWebviewView(
        webviewView: vscode.WebviewView,
        _context: vscode.WebviewViewResolveContext,
        _token: vscode.CancellationToken
    ): void {
        logger.debug(`[${this.getTitle()}] resolveWebviewView called`);
        this._view = webviewView;

        // Set webview options
        webviewView.webview.options = getWebviewOptions(this.extensionUri);

        // Set HTML content
        const title = this.getTitle();
        logger.debug(`[${title}] Setting HTML with title: "${title}"`);
        webviewView.webview.html = getReactWebviewHtml(
            webviewView.webview,
            this.extensionUri,
            title
        );

        // Handle visibility changes - restore state when view becomes visible
        webviewView.onDidChangeVisibility(() => {
            if (webviewView.visible) {
                try {
                    this.updateView();
                } catch (error) {
                    const err = error instanceof Error ? error : new Error(String(error));
                    this.handleUpdateError(err);
                }
            }
        });

        // Handle messages from the webview
        webviewView.webview.onDidReceiveMessage(async (message: IncomingWebviewMessage) => {
            try {
                await this.handleMessage(message);
            } catch (error) {
                const err = error instanceof Error ? error : new Error(String(error));
                this.handleMessageError(err, message);
            }
        });
    }

    /**
     * Send a typed message to the webview
     * @param message Outgoing message
     */
    protected postMessage(message: OutgoingWebviewMessage): void {
        if (this._view) {
            this._view.webview.postMessage(message);
        }
    }

    /**
     * Check if the webview is currently visible
     */
    protected get isVisible(): boolean {
        return this._view?.visible ?? false;
    }

    /**
     * Send an error message to the webview
     * Subclasses can override to customize error display
     * @param error Error to display
     * @param context Optional context about what operation failed
     */
    protected sendErrorToWebview(error: Error, context?: string): void {
        const errorMessage: OutgoingWebviewMessage = {
            command: 'error',
            error: {
                message: error.message,
                context: context || (error instanceof SquiggyError ? error.context : undefined),
                type: error.name,
            },
        };
        this.postMessage(errorMessage);
    }

    /**
     * Handle errors that occur during message processing
     * Provides consistent error handling and logging across all webview providers
     * @param error Error that occurred
     * @param message Original message that caused the error
     * @param showUserNotification Whether to show VSCode notification (default: true)
     */
    protected handleMessageError(
        error: Error,
        message: IncomingWebviewMessage,
        showUserNotification: boolean = true
    ): void {
        // Get command from message (if available)
        const command = (message as any).command || 'unknown';

        // Log the error with context
        logger.error(`[${this.getTitle()}] Error handling message '${command}':`, error);

        // Send error to webview for UI display
        this.sendErrorToWebview(error, `Failed to handle '${command}' command`);

        // Show user notification if requested
        if (showUserNotification) {
            if (error instanceof SquiggyError) {
                handleError(error, error.context);
            } else {
                vscode.window.showErrorMessage(`${this.getTitle()}: ${error.message}`);
            }
        }
    }

    /**
     * Handle errors that occur during view updates
     * @param error Error that occurred during update
     */
    protected handleUpdateError(error: Error): void {
        logger.error(`[${this.getTitle()}] Error updating view:`, error);

        // Send error to webview
        this.sendErrorToWebview(error, 'Failed to update view');

        // Don't show user notification for update errors unless they're critical
        // (to avoid spamming the user during rapid state changes)
        if (error instanceof SquiggyError) {
            // Only show notifications for specific error types
            handleError(error, error.context);
        }
    }

    /**
     * Safely execute an async operation with error handling
     * Wraps operations to provide consistent error handling
     * @param operation Operation to execute
     * @param errorContext Description of what operation is being performed
     * @returns Result of operation or undefined on error
     */
    protected async safeExecute<T>(
        operation: () => Promise<T>,
        errorContext: string
    ): Promise<T | undefined> {
        try {
            return await operation();
        } catch (error) {
            const err = error instanceof Error ? error : new Error(String(error));
            logger.error(`[${this.getTitle()}] ${errorContext}:`, err);
            this.sendErrorToWebview(err, errorContext);
            return undefined;
        }
    }
}
