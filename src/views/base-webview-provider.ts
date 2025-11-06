/**
 * Base Webview Provider
 *
 * Abstract base class consolidating duplicate webview setup code.
 * All squiggy webview providers should extend this class.
 */

import * as vscode from 'vscode';
import { getReactWebviewHtml, getWebviewOptions } from '../utils/webview-utils';
import { IncomingWebviewMessage, OutgoingWebviewMessage } from '../types/messages';

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
        console.log(`[${this.getTitle()}] resolveWebviewView called`);
        this._view = webviewView;

        // Set webview options
        webviewView.webview.options = getWebviewOptions(this.extensionUri);

        // Set HTML content
        const title = this.getTitle();
        console.log(`[${title}] Setting HTML with title: "${title}"`);
        webviewView.webview.html = getReactWebviewHtml(
            webviewView.webview,
            this.extensionUri,
            title
        );

        // Handle visibility changes - restore state when view becomes visible
        webviewView.onDidChangeVisibility(() => {
            if (webviewView.visible) {
                this.updateView();
            }
        });

        // Handle messages from the webview
        webviewView.webview.onDidReceiveMessage(async (message: IncomingWebviewMessage) => {
            try {
                await this.handleMessage(message);
            } catch (error) {
                console.error(`Error handling webview message:`, error);
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
}
