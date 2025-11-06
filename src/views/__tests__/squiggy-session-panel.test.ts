/**
 * Tests for Session Panel Provider
 *
 * Tests the SessionPanelProvider webview implementation.
 * Target: >80% coverage of squiggy-session-panel.ts
 */

import { describe, it, expect, beforeEach, jest, afterEach } from '@jest/globals';
import * as vscode from 'vscode';
import { SessionPanelProvider } from '../squiggy-session-panel';
import { SessionStateManager } from '../../state/session-state-manager';
import { ExtensionState } from '../../state/extension-state';

// Mock SessionStateManager
jest.mock('../../state/session-state-manager', () => ({
    SessionStateManager: {
        loadSession: (jest.fn() as any).mockResolvedValue(null),
    },
}));

describe('SessionPanelProvider', () => {
    let provider: SessionPanelProvider;
    let mockContext: vscode.ExtensionContext;
    let mockState: any;
    let mockWebviewView: any;

    beforeEach(() => {
        mockContext = {
            extensionUri: vscode.Uri.file('/mock/extension'),
            subscriptions: [],
        } as any;

        mockState = {
            toSessionState: jest.fn().mockReturnValue({
                samples: {
                    sample1: { pod5Path: '/path/to/sample1.pod5' },
                    sample2: { pod5Path: '/path/to/sample2.pod5' },
                },
            }),
        } as any;

        provider = new SessionPanelProvider(mockContext.extensionUri, mockContext, mockState);

        // Mock webview view
        mockWebviewView = {
            webview: {
                options: {},
                html: '',
                postMessage: jest.fn(),
                asWebviewUri: (uri: vscode.Uri) => uri,
                onDidReceiveMessage: jest.fn(),
            },
            visible: true,
            onDidChangeVisibility: jest.fn(),
            onDidDispose: jest.fn(),
        };

        jest.clearAllMocks();
    });

    afterEach(() => {
        jest.clearAllMocks();
    });

    describe('Provider Properties', () => {
        it('should have correct viewType', () => {
            expect(SessionPanelProvider.viewType).toBe('squiggySessionPanel');
        });

        it('should return correct title', () => {
            // Access protected method via any cast for testing
            const title = (provider as any).getTitle();
            expect(title).toBe('Squiggy Session Manager');
        });
    });

    describe('resolveWebviewView', () => {
        it('should set up webview when resolved', () => {
            provider.resolveWebviewView(mockWebviewView, {} as any, {} as any);

            expect(mockWebviewView.webview.options).toBeDefined();
            expect(mockWebviewView.webview.html).toBeTruthy();
        });

        it('should register message handler', () => {
            provider.resolveWebviewView(mockWebviewView, {} as any, {} as any);

            expect(mockWebviewView.webview.onDidReceiveMessage).toHaveBeenCalled();
        });
    });

    describe('Message Handling', () => {
        beforeEach(() => {
            provider.resolveWebviewView(mockWebviewView, {} as any, {} as any);
        });

        it('should handle ready message and update view', async () => {
            const messageHandler = mockWebviewView.webview.onDidReceiveMessage.mock.calls[0][0];

            await messageHandler({ type: 'ready' });

            // Should call postMessage to update the view
            await new Promise((resolve) => setTimeout(resolve, 10)); // Wait for async
            expect(mockWebviewView.webview.postMessage).toHaveBeenCalled();
        });

        it('should handle loadDemo message', async () => {
            const messageHandler = mockWebviewView.webview.onDidReceiveMessage.mock.calls[0][0];

            await messageHandler({ type: 'loadDemo' });

            expect(vscode.commands.executeCommand).toHaveBeenCalledWith(
                'squiggy.loadDemoSession'
            );
        });

        it('should handle save message', async () => {
            const messageHandler = mockWebviewView.webview.onDidReceiveMessage.mock.calls[0][0];

            await messageHandler({ type: 'save' });

            expect(vscode.commands.executeCommand).toHaveBeenCalledWith('squiggy.saveSession');
        });

        it('should handle restore message', async () => {
            const messageHandler = mockWebviewView.webview.onDidReceiveMessage.mock.calls[0][0];

            await messageHandler({ type: 'restore' });

            expect(vscode.commands.executeCommand).toHaveBeenCalledWith('squiggy.restoreSession');
        });

        it('should handle export message', async () => {
            const messageHandler = mockWebviewView.webview.onDidReceiveMessage.mock.calls[0][0];

            await messageHandler({ type: 'export' });

            expect(vscode.commands.executeCommand).toHaveBeenCalledWith('squiggy.exportSession');
        });

        it('should handle import message', async () => {
            const messageHandler = mockWebviewView.webview.onDidReceiveMessage.mock.calls[0][0];

            await messageHandler({ type: 'import' });

            expect(vscode.commands.executeCommand).toHaveBeenCalledWith('squiggy.importSession');
        });

        it('should handle clear message', async () => {
            const messageHandler = mockWebviewView.webview.onDidReceiveMessage.mock.calls[0][0];

            await messageHandler({ type: 'clear' });

            expect(vscode.commands.executeCommand).toHaveBeenCalledWith('squiggy.clearSession');
        });
    });

    describe('updateView', () => {
        beforeEach(() => {
            provider.resolveWebviewView(mockWebviewView, {} as any, {} as any);
        });

        it('should send session state to webview when samples exist', async () => {
            (provider as any).updateView();

            // Wait for async operations
            await new Promise((resolve) => setTimeout(resolve, 10));

            expect(mockWebviewView.webview.postMessage).toHaveBeenCalledWith(
                expect.objectContaining({
                    type: 'updateSession',
                    hasSamples: true,
                    hasSavedSession: false,
                    sampleCount: 2,
                    sampleNames: ['sample1', 'sample2'],
                })
            );
        });

        it('should indicate no samples when state is empty', async () => {
            mockState.toSessionState.mockReturnValue({ samples: {} });

            (provider as any).updateView();

            await new Promise((resolve) => setTimeout(resolve, 10));

            expect(mockWebviewView.webview.postMessage).toHaveBeenCalledWith(
                expect.objectContaining({
                    type: 'updateSession',
                    hasSamples: false,
                    sampleCount: 0,
                    sampleNames: [],
                })
            );
        });

        it('should indicate when saved session exists', async () => {
            (SessionStateManager.loadSession as any).mockResolvedValue({
                samples: { saved: {} },
            });

            (provider as any).updateView();

            await new Promise((resolve) => setTimeout(resolve, 10));

            expect(mockWebviewView.webview.postMessage).toHaveBeenCalledWith(
                expect.objectContaining({
                    hasSavedSession: true,
                })
            );
        });

        it('should not post message if view is not available', () => {
            // Create provider without resolving view
            const newProvider = new SessionPanelProvider(
                mockContext.extensionUri,
                mockContext,
                mockState
            );

            (newProvider as any).updateView();

            // Should not throw and should not post
            expect(mockWebviewView.webview.postMessage).not.toHaveBeenCalled();
        });
    });
});
