/**
 * Tests for Reads View Pane Provider
 *
 * Tests the ReadsViewPane webview implementation.
 * Target: >80% coverage of squiggy-reads-view-pane.ts
 */

import { describe, it, expect, beforeEach, jest, afterEach } from '@jest/globals';
import * as vscode from 'vscode';
import { ReadsViewPane } from '../squiggy-reads-view-pane';
import { ExtensionState } from '../../state/extension-state';

describe('ReadsViewPane', () => {
    let provider: ReadsViewPane;
    let mockWebviewView: any;
    let mockState: any;

    beforeEach(() => {
        const extensionUri = vscode.Uri.file('/mock/extension');

        mockState = {
            getAllSampleNames: jest.fn().mockReturnValue(['Sample_A', 'Sample_B']),
            selectedReadExplorerSample: 'Sample_A',
        } as any;

        provider = new ReadsViewPane(extensionUri, mockState);

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
            expect(ReadsViewPane.viewType).toBe('squiggyReadList');
        });

        it('should return correct title', () => {
            const title = (provider as any).getTitle();
            expect(title).toBe('Squiggy Reads Explorer');
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
            // Mock executeCommand to resolve successfully for any command
            (vscode.commands.executeCommand as any).mockResolvedValue(undefined);
        });

        it('should handle plotRead message', async () => {
            const messageHandler = mockWebviewView.webview.onDidReceiveMessage.mock.calls[0][0];

            await messageHandler({ type: 'plotRead', readId: 'read_001' });

            expect(vscode.commands.executeCommand).toHaveBeenCalledWith(
                'squiggy.plotRead',
                'read_001'
            );
        });

        it('should handle plotAggregate message', async () => {
            const messageHandler = mockWebviewView.webview.onDidReceiveMessage.mock.calls[0][0];

            await messageHandler({ type: 'plotAggregate', referenceName: 'chr1' });

            expect(vscode.commands.executeCommand).toHaveBeenCalledWith(
                'squiggy.plotAggregate',
                'chr1'
            );
        });

        it('should handle loadMore message', async () => {
            const messageHandler = mockWebviewView.webview.onDidReceiveMessage.mock.calls[0][0];

            await messageHandler({ type: 'loadMore' });

            expect(vscode.commands.executeCommand).toHaveBeenCalledWith(
                'squiggy.internal.loadMoreReads'
            );
        });

        it('should handle expandReference message', async () => {
            const messageHandler = mockWebviewView.webview.onDidReceiveMessage.mock.calls[0][0];

            await messageHandler({
                type: 'expandReference',
                referenceName: 'chr1',
                offset: 0,
                limit: 100,
            });

            expect(vscode.commands.executeCommand).toHaveBeenCalledWith(
                'squiggy.internal.expandReference',
                'chr1',
                0,
                100
            );
        });

        it('should handle selectSample message', async () => {
            const messageHandler = mockWebviewView.webview.onDidReceiveMessage.mock.calls[0][0];

            await messageHandler({ type: 'selectSample', sampleName: 'Sample_B' });

            expect(mockState.selectedReadExplorerSample).toBe('Sample_B');
            expect(vscode.commands.executeCommand).toHaveBeenCalledWith(
                'squiggy.internal.loadReadsForSample',
                'Sample_B'
            );
        });

        it('should handle ready message and update view', async () => {
            const messageHandler = mockWebviewView.webview.onDidReceiveMessage.mock.calls[0][0];

            await messageHandler({ type: 'ready' });

            // Should send setAvailableSamples message
            expect(mockWebviewView.webview.postMessage).toHaveBeenCalledWith(
                expect.objectContaining({
                    type: 'setAvailableSamples',
                    samples: ['Sample_A', 'Sample_B'],
                    selectedSample: 'Sample_A',
                })
            );

            // Should send updateReads message
            expect(mockWebviewView.webview.postMessage).toHaveBeenCalledWith(
                expect.objectContaining({
                    type: 'updateReads',
                    reads: [],
                    groupedByReference: false,
                })
            );
        });
    });

    describe('setReads', () => {
        beforeEach(() => {
            provider.resolveWebviewView(mockWebviewView, {} as any, {} as any);
        });

        it('should set flat list of reads', () => {
            provider.setReads(['read_001', 'read_002', 'read_003']);

            expect(mockWebviewView.webview.postMessage).toHaveBeenCalledWith(
                expect.objectContaining({
                    type: 'updateReads',
                    reads: [
                        { type: 'read', readId: 'read_001', indentLevel: 0 },
                        { type: 'read', readId: 'read_002', indentLevel: 0 },
                        { type: 'read', readId: 'read_003', indentLevel: 0 },
                    ],
                    groupedByReference: false,
                })
            );
        });
    });

    describe('setReadsGrouped', () => {
        beforeEach(() => {
            provider.resolveWebviewView(mockWebviewView, {} as any, {} as any);
        });

        it('should set reads grouped by reference', () => {
            const referenceToReads = new Map([
                [
                    'chr1',
                    [
                        {
                            type: 'read' as const,
                            readId: 'read_001',
                            referenceName: 'chr1',
                            indentLevel: 1 as 0 | 1,
                        },
                    ],
                ],
                [
                    'chr2',
                    [
                        {
                            type: 'read' as const,
                            readId: 'read_002',
                            referenceName: 'chr2',
                            indentLevel: 1 as 0 | 1,
                        },
                    ],
                ],
            ]);

            provider.setReadsGrouped(referenceToReads);

            expect(mockWebviewView.webview.postMessage).toHaveBeenCalledWith(
                expect.objectContaining({
                    type: 'updateReads',
                    groupedByReference: true,
                    referenceToReads: Array.from(referenceToReads.entries()),
                })
            );
        });
    });

    describe('setReferencesOnly', () => {
        beforeEach(() => {
            provider.resolveWebviewView(mockWebviewView, {} as any, {} as any);
        });

        it('should set reference headers for lazy loading', () => {
            const references = [
                { referenceName: 'chr1', readCount: 100 },
                { referenceName: 'chr2', readCount: 50 },
            ];

            provider.setReferencesOnly(references);

            expect(mockWebviewView.webview.postMessage).toHaveBeenCalledWith({
                type: 'setReferencesOnly',
                references,
            });
        });
    });

    describe('appendReads', () => {
        beforeEach(() => {
            provider.resolveWebviewView(mockWebviewView, {} as any, {} as any);
            provider.setReads(['read_001', 'read_002']);
            jest.clearAllMocks();
        });

        it('should append new reads to existing list', () => {
            provider.appendReads(['read_003', 'read_004']);

            expect(mockWebviewView.webview.postMessage).toHaveBeenCalledWith({
                type: 'appendReads',
                reads: [
                    { type: 'read', readId: 'read_003', indentLevel: 0 },
                    { type: 'read', readId: 'read_004', indentLevel: 0 },
                ],
            });
        });
    });

    describe('setReadsForReference', () => {
        beforeEach(() => {
            provider.resolveWebviewView(mockWebviewView, {} as any, {} as any);
        });

        it('should set reads for specific reference with pagination info', () => {
            provider.setReadsForReference('chr1', ['read_001', 'read_002'], 0, 150);

            expect(mockWebviewView.webview.postMessage).toHaveBeenCalledWith({
                type: 'setReadsForReference',
                referenceName: 'chr1',
                reads: [
                    { type: 'read', readId: 'read_001', referenceName: 'chr1', indentLevel: 1 },
                    { type: 'read', readId: 'read_002', referenceName: 'chr1', indentLevel: 1 },
                ],
                offset: 0,
                totalCount: 150,
            });
        });
    });

    describe('setLoading', () => {
        beforeEach(() => {
            provider.resolveWebviewView(mockWebviewView, {} as any, {} as any);
        });

        it('should send loading state with message', () => {
            provider.setLoading(true, 'Loading reads...');

            expect(mockWebviewView.webview.postMessage).toHaveBeenCalledWith({
                type: 'setLoading',
                isLoading: true,
                message: 'Loading reads...',
            });
        });

        it('should send loading state without message', () => {
            provider.setLoading(false);

            expect(mockWebviewView.webview.postMessage).toHaveBeenCalledWith({
                type: 'setLoading',
                isLoading: false,
                message: undefined,
            });
        });
    });

    describe('getAvailableSamples', () => {
        it('should return samples from state', () => {
            const samples = provider.getAvailableSamples();

            expect(samples).toEqual(['Sample_A', 'Sample_B']);
            expect(mockState.getAllSampleNames).toHaveBeenCalled();
        });
    });

    describe('getSelectedSample', () => {
        it('should return selected sample from state', () => {
            const selected = provider.getSelectedSample();

            expect(selected).toBe('Sample_A');
        });

        it('should return null when no sample selected', () => {
            mockState.selectedReadExplorerSample = null;

            const selected = provider.getSelectedSample();

            expect(selected).toBeNull();
        });
    });

    describe('refresh', () => {
        beforeEach(() => {
            provider.resolveWebviewView(mockWebviewView, {} as any, {} as any);
            provider.setReads(['read_001']);
            jest.clearAllMocks();
        });

        it('should re-send current state to webview', () => {
            provider.refresh();

            expect(mockWebviewView.webview.postMessage).toHaveBeenCalledWith(
                expect.objectContaining({
                    type: 'setAvailableSamples',
                })
            );

            expect(mockWebviewView.webview.postMessage).toHaveBeenCalledWith(
                expect.objectContaining({
                    type: 'updateReads',
                })
            );
        });
    });

    describe('updateView without webview', () => {
        it('should not post message if view not available', () => {
            // Don't resolve webview view
            (provider as any).updateView();

            expect(mockWebviewView.webview.postMessage).not.toHaveBeenCalled();
        });
    });
});
