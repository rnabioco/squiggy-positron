/**
 * Tests for Modifications Panel Provider
 *
 * Tests the ModificationsPanelProvider webview implementation.
 * Target: >80% coverage of squiggy-modifications-panel.ts
 */

import { describe, it, expect, beforeEach, jest, afterEach } from '@jest/globals';
import * as vscode from 'vscode';
import { ModificationsPanelProvider } from '../squiggy-modifications-panel';

describe('ModificationsPanelProvider', () => {
    let provider: ModificationsPanelProvider;
    let mockWebviewView: any;
    let filterChangeListener: jest.Mock;

    beforeEach(() => {
        const extensionUri = vscode.Uri.file('/mock/extension');
        provider = new ModificationsPanelProvider(extensionUri);

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

        filterChangeListener = jest.fn();
        provider.onDidChangeFilters(filterChangeListener);

        jest.clearAllMocks();
    });

    afterEach(() => {
        jest.clearAllMocks();
    });

    describe('Provider Properties', () => {
        it('should have correct viewType', () => {
            expect(ModificationsPanelProvider.viewType).toBe('squiggyModificationsPanel');
        });

        it('should return correct title', () => {
            const title = (provider as any).getTitle();
            expect(title).toBe('Squiggy Modifications');
        });
    });

    describe('resolveWebviewView', () => {
        it('should set up webview when resolved', () => {
            provider.resolveWebviewView(mockWebviewView, {} as any, {} as any);

            expect(mockWebviewView.webview.options).toBeDefined();
            expect(mockWebviewView.webview.html).toBeTruthy();
        });
    });

    describe('Message Handling', () => {
        beforeEach(() => {
            provider.resolveWebviewView(mockWebviewView, {} as any, {} as any);
        });

        it('should handle ready message and update view', async () => {
            const messageHandler = mockWebviewView.webview.onDidReceiveMessage.mock.calls[0][0];

            await messageHandler({ type: 'ready' });

            // Should send clearMods message when no modifications
            expect(mockWebviewView.webview.postMessage).toHaveBeenCalledWith({
                type: 'clearMods',
            });
        });

        it('should handle filtersChanged message', async () => {
            const messageHandler = mockWebviewView.webview.onDidReceiveMessage.mock.calls[0][0];

            await messageHandler({
                type: 'filtersChanged',
                minProbability: 0.8,
                enabledModTypes: ['5mC', '6mA'],
            });

            // Should update filters
            const filters = provider.getFilters();
            expect(filters.minProbability).toBe(0.8);
            expect(filters.enabledModTypes).toEqual(['5mC', '6mA']);

            // Should fire event
            expect(filterChangeListener).toHaveBeenCalled();
        });
    });

    describe('setModificationInfo', () => {
        beforeEach(() => {
            provider.resolveWebviewView(mockWebviewView, {} as any, {} as any);
        });

        it('should update modification info and send to webview', () => {
            provider.setModificationInfo(true, ['5mC', '6mA'], true);

            expect(mockWebviewView.webview.postMessage).toHaveBeenCalledWith({
                type: 'updateModInfo',
                hasModifications: true,
                modificationTypes: ['5mC', '6mA'],
                hasProbabilities: true,
            });
        });

        it('should enable all modification types by default', () => {
            provider.setModificationInfo(true, ['5mC', '6mA', 'm5C'], true);

            const filters = provider.getFilters();
            expect(filters.enabledModTypes).toEqual(['5mC', '6mA', 'm5C']);
        });

        it('should not post message if view not available', () => {
            const newProvider = new ModificationsPanelProvider(vscode.Uri.file('/mock'));

            newProvider.setModificationInfo(true, ['5mC'], true);

            expect(mockWebviewView.webview.postMessage).not.toHaveBeenCalled();
        });
    });

    describe('clear', () => {
        beforeEach(() => {
            provider.resolveWebviewView(mockWebviewView, {} as any, {} as any);
        });

        it('should clear modification info', () => {
            // Set some modifications first
            provider.setModificationInfo(true, ['5mC', '6mA'], true);
            jest.clearAllMocks();

            // Clear
            provider.clear();

            expect(mockWebviewView.webview.postMessage).toHaveBeenCalledWith({
                type: 'clearMods',
            });
        });

        it('should preserve filter settings (user preferences)', () => {
            provider.setModificationInfo(true, ['5mC', '6mA'], true);
            provider.clear();

            // Filter settings should be preserved
            const filters = provider.getFilters();
            expect(filters.enabledModTypes).toEqual(['5mC', '6mA']);
            expect(filters.minProbability).toBe(0.5);
        });
    });

    describe('getFilters', () => {
        it('should return current filter settings', () => {
            const filters = provider.getFilters();

            expect(filters).toEqual({
                minProbability: 0.5, // default
                enabledModTypes: [],
                minFrequency: 0.2, // default
                minModifiedReads: 5, // default
            });
        });

        it('should return updated filters after changes', () => {
            provider.setModificationInfo(true, ['5mC', '6mA'], true);

            provider.resolveWebviewView(mockWebviewView, {} as any, {} as any);
            const messageHandler = mockWebviewView.webview.onDidReceiveMessage.mock.calls[0][0];

            messageHandler({
                type: 'filtersChanged',
                minProbability: 0.9,
                enabledModTypes: ['5mC'],
            });

            const filters = provider.getFilters();
            expect(filters.minProbability).toBe(0.9);
            expect(filters.enabledModTypes).toEqual(['5mC']);
        });
    });

    describe('onDidChangeFilters event', () => {
        it('should fire when filters change', async () => {
            provider.resolveWebviewView(mockWebviewView, {} as any, {} as any);
            const messageHandler = mockWebviewView.webview.onDidReceiveMessage.mock.calls[0][0];

            await messageHandler({
                type: 'filtersChanged',
                minProbability: 0.7,
                enabledModTypes: ['5mC'],
            });

            expect(filterChangeListener).toHaveBeenCalledTimes(1);
        });

        it('should allow multiple listeners', async () => {
            const listener2 = jest.fn();
            provider.onDidChangeFilters(listener2);

            provider.resolveWebviewView(mockWebviewView, {} as any, {} as any);
            const messageHandler = mockWebviewView.webview.onDidReceiveMessage.mock.calls[0][0];

            await messageHandler({
                type: 'filtersChanged',
                minProbability: 0.6,
                enabledModTypes: [],
            });

            expect(filterChangeListener).toHaveBeenCalled();
            expect(listener2).toHaveBeenCalled();
        });
    });

    describe('updateView', () => {
        beforeEach(() => {
            provider.resolveWebviewView(mockWebviewView, {} as any, {} as any);
        });

        it('should send clearMods when no modifications', () => {
            (provider as any).updateView();

            expect(mockWebviewView.webview.postMessage).toHaveBeenCalledWith({
                type: 'clearMods',
            });
        });

        it('should send updateModInfo when modifications exist', () => {
            provider.setModificationInfo(true, ['5mC'], false);

            // Clear previous calls
            jest.clearAllMocks();

            (provider as any).updateView();

            expect(mockWebviewView.webview.postMessage).toHaveBeenCalledWith({
                type: 'updateModInfo',
                hasModifications: true,
                modificationTypes: ['5mC'],
                hasProbabilities: false,
            });
        });
    });
});
