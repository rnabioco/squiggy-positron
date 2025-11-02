/**
 * Tests for BaseWebviewProvider
 */

import { describe, it, expect, beforeEach, jest } from '@jest/globals';
import * as vscode from 'vscode';
import { BaseWebviewProvider } from '../base-webview-provider';
import { IncomingWebviewMessage, OutgoingWebviewMessage } from '../../types/messages';

// Concrete test implementation of BaseWebviewProvider
class TestWebviewProvider extends BaseWebviewProvider {
    public messageHandler = jest.fn();
    public updateViewHandler = jest.fn();

    protected getTitle(): string {
        return 'Test Panel';
    }

    protected async handleMessage(message: IncomingWebviewMessage): Promise<void> {
        this.messageHandler(message);
    }

    protected updateView(): void {
        this.updateViewHandler();
    }

    // Expose protected methods for testing
    public testPostMessage(message: OutgoingWebviewMessage): void {
        this.postMessage(message);
    }

    public get testIsVisible(): boolean {
        return this.isVisible;
    }
}

describe('BaseWebviewProvider', () => {
    let provider: TestWebviewProvider;
    let mockWebviewView: vscode.WebviewView;
    let mockWebview: vscode.Webview;
    let mockExtensionUri: vscode.Uri;

    beforeEach(() => {
        // Mock extension URI
        mockExtensionUri = vscode.Uri.file('/mock/extension/path');

        // Mock webview
        mockWebview = {
            options: {},
            html: '',
            cspSource: 'mock-csp-source',
            asWebviewUri: jest.fn((uri: vscode.Uri) => {
                return vscode.Uri.parse(`webview://resource${uri.fsPath}`);
            }),
            postMessage: jest.fn(),
            onDidReceiveMessage: jest.fn((callback) => {
                // Store callback for later invocation
                (mockWebview as any)._messageCallback = callback;
                return { dispose: jest.fn() };
            }),
        } as any;

        // Mock webview view with mutable visible property
        const mockView: any = {
            webview: mockWebview,
            _visible: true,
            onDidChangeVisibility: jest.fn((callback) => {
                mockView._visibilityCallback = callback;
                return { dispose: jest.fn() };
            }),
        };

        // Make visible a getter/setter
        Object.defineProperty(mockView, 'visible', {
            get() {
                return this._visible;
            },
            set(value) {
                this._visible = value;
            },
        });

        mockWebviewView = mockView;

        provider = new TestWebviewProvider(mockExtensionUri);
    });

    describe('resolveWebviewView', () => {
        it('should set up webview when resolved', () => {
            provider.resolveWebviewView(mockWebviewView, {} as any, {} as any);

            // Should set webview options
            expect(mockWebview.options).toBeDefined();

            // Should set HTML content
            expect(mockWebview.html).toContain('Test Panel');

            // Should register message handler
            expect(mockWebview.onDidReceiveMessage).toHaveBeenCalled();

            // Should register visibility change handler
            expect(mockWebviewView.onDidChangeVisibility).toHaveBeenCalled();
        });

        it('should call updateView when visibility changes to visible', () => {
            provider.resolveWebviewView(mockWebviewView, {} as any, {} as any);

            // Simulate visibility change using the internal property
            (mockWebviewView as any)._visible = true;
            (mockWebviewView as any)._visibilityCallback();

            expect(provider.updateViewHandler).toHaveBeenCalled();
        });

        it('should not call updateView when visibility changes to hidden', () => {
            provider.resolveWebviewView(mockWebviewView, {} as any, {} as any);

            // Simulate visibility change to hidden using internal property
            (mockWebviewView as any)._visible = false;
            (mockWebviewView as any)._visibilityCallback();

            expect(provider.updateViewHandler).not.toHaveBeenCalled();
        });
    });

    describe('message handling', () => {
        beforeEach(() => {
            provider.resolveWebviewView(mockWebviewView, {} as any, {} as any);
        });

        it('should handle messages from webview', async () => {
            const testMessage: IncomingWebviewMessage = {
                type: 'ready',
            };

            // Simulate message from webview
            await (mockWebview as any)._messageCallback(testMessage);

            expect(provider.messageHandler).toHaveBeenCalledWith(testMessage);
        });

        it('should handle message errors gracefully', async () => {
            const consoleSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
            const testMessage: IncomingWebviewMessage = {
                type: 'ready',
            };

            // Make handler throw error
            provider.messageHandler.mockImplementation(() => {
                throw new Error('Test error');
            });

            // Should not throw
            await expect((mockWebview as any)._messageCallback(testMessage)).resolves.not.toThrow();

            expect(consoleSpy).toHaveBeenCalled();
            consoleSpy.mockRestore();
        });
    });

    describe('postMessage', () => {
        it('should send messages to webview when view is available', () => {
            provider.resolveWebviewView(mockWebviewView, {} as any, {} as any);

            const testMessage: OutgoingWebviewMessage = {
                type: 'updateFiles',
                files: [],
            };

            provider.testPostMessage(testMessage);

            expect(mockWebview.postMessage).toHaveBeenCalledWith(testMessage);
        });

        it('should not crash when view is not available', () => {
            const testMessage: OutgoingWebviewMessage = {
                type: 'updateReads',
                reads: [],
                groupedByReference: false,
            };

            // Should not throw even though view not resolved
            expect(() => provider.testPostMessage(testMessage)).not.toThrow();
        });
    });

    describe('isVisible', () => {
        it('should return true when view is visible', () => {
            (mockWebviewView as any)._visible = true;
            provider.resolveWebviewView(mockWebviewView, {} as any, {} as any);

            expect(provider.testIsVisible).toBe(true);
        });

        it('should return false when view is hidden', () => {
            (mockWebviewView as any)._visible = false;
            provider.resolveWebviewView(mockWebviewView, {} as any, {} as any);

            expect(provider.testIsVisible).toBe(false);
        });

        it('should return false when view is not resolved', () => {
            expect(provider.testIsVisible).toBe(false);
        });
    });

    describe('getTitle', () => {
        it('should return the correct title from subclass', () => {
            // Access protected method via class internals for testing
            const title = (provider as any).getTitle();
            expect(title).toBe('Test Panel');
        });
    });
});
