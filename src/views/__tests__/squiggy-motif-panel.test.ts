/**
 * Tests for Motif Search Panel Provider
 *
 * Tests the MotifSearchPanelProvider webview implementation.
 * Target: >80% coverage of squiggy-motif-panel.ts
 */

import { describe, it, expect, beforeEach, jest, afterEach } from '@jest/globals';
import * as vscode from 'vscode';
import { MotifSearchPanelProvider } from '../squiggy-motif-panel';
import { MotifMatch } from '../../types/motif-types';

describe('MotifSearchPanelProvider', () => {
    let provider: MotifSearchPanelProvider;
    let mockWebviewView: any;
    let mockState: any;

    beforeEach(() => {
        const extensionUri = vscode.Uri.file('/mock/extension');

        mockState = {
            currentFastaFile: null,
            usePositron: false,
            squiggyAPI: null,
            ensureBackgroundKernel: jest.fn(),
        } as any;

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

        provider = new MotifSearchPanelProvider(extensionUri, mockState);
    });

    afterEach(() => {
        jest.clearAllMocks();
    });

    describe('Provider Properties', () => {
        it('should have correct viewType', () => {
            expect(MotifSearchPanelProvider.viewType).toBe('squiggyMotifSearch');
        });
    });

    describe('resolveWebviewView', () => {
        it('should set up webview when resolved', () => {
            provider.resolveWebviewView(mockWebviewView, {} as any, {} as any);

            expect(mockWebviewView.webview.options).toBeDefined();
            expect(mockWebviewView.webview.options.enableScripts).toBe(true);
            expect(mockWebviewView.webview.html).toBeTruthy();
            expect(mockWebviewView.webview.html).toContain('Motif Search');
        });

        it('should register message handler', () => {
            provider.resolveWebviewView(mockWebviewView, {} as any, {} as any);

            expect(mockWebviewView.webview.onDidReceiveMessage).toHaveBeenCalled();
        });
    });

    describe('Message Handling - searchMotif', () => {
        beforeEach(() => {
            provider.resolveWebviewView(mockWebviewView, {} as any, {} as any);
            (vscode.commands.executeCommand as any).mockResolvedValue(undefined);
        });

        it('should show error when no FASTA file loaded', async () => {
            const messageHandler = mockWebviewView.webview.onDidReceiveMessage.mock.calls[0][0];

            await messageHandler({ type: 'searchMotif', motif: 'DRACH', strand: 'both' });

            expect(vscode.window.showErrorMessage).toHaveBeenCalledWith(
                'No FASTA file loaded. Use "Open FASTA File" first.'
            );
        });

        it('should show error when Positron runtime not available', async () => {
            mockState.currentFastaFile = '/path/to/reference.fasta';
            mockState.usePositron = false;

            const messageHandler = mockWebviewView.webview.onDidReceiveMessage.mock.calls[0][0];

            await messageHandler({ type: 'searchMotif', motif: 'DRACH', strand: 'both' });

            expect(vscode.window.showErrorMessage).toHaveBeenCalledWith(
                'Motif search requires Positron runtime. Please use Positron IDE.'
            );
        });

        it('should search motif successfully with Positron runtime', async () => {
            mockState.currentFastaFile = '/path/to/reference.fasta';
            mockState.usePositron = true;

            const mockSearchMotif = (jest.fn() as any).mockResolvedValue([
                { chrom: 'chr1', position: 100, sequence: 'GGACA', strand: '+' },
                { chrom: 'chr1', position: 500, sequence: 'AGACT', strand: '+' },
            ]);

            mockState.ensureBackgroundKernel.mockResolvedValue({
                searchMotif: mockSearchMotif,
            });

            const messageHandler = mockWebviewView.webview.onDidReceiveMessage.mock.calls[0][0];

            await messageHandler({ type: 'searchMotif', motif: 'DRACH', strand: 'both' });

            expect(mockSearchMotif).toHaveBeenCalledWith(
                '/path/to/reference.fasta',
                'DRACH',
                undefined,
                'both'
            );

            // Should update view with matches
            expect(mockWebviewView.webview.postMessage).toHaveBeenCalledWith(
                expect.objectContaining({
                    type: 'updateMatches',
                    matches: [
                        { chrom: 'chr1', position: 100, sequence: 'GGACA', strand: '+' },
                        { chrom: 'chr1', position: 500, sequence: 'AGACT', strand: '+' },
                    ],
                    searching: false,
                    motif: 'DRACH',
                })
            );
        });

        it('should handle default strand parameter', async () => {
            mockState.currentFastaFile = '/path/to/reference.fasta';
            mockState.usePositron = true;

            const mockSearchMotif = (jest.fn() as any).mockResolvedValue([]);

            mockState.ensureBackgroundKernel.mockResolvedValue({
                searchMotif: mockSearchMotif,
            });

            const messageHandler = mockWebviewView.webview.onDidReceiveMessage.mock.calls[0][0];

            // Call without strand parameter
            await messageHandler({ type: 'searchMotif', motif: 'CCA' });

            expect(mockSearchMotif).toHaveBeenCalledWith(
                '/path/to/reference.fasta',
                'CCA',
                undefined,
                'both'
            );
        });

        it('should handle errors during motif search', async () => {
            mockState.currentFastaFile = '/path/to/reference.fasta';
            mockState.usePositron = true;

            const mockSearchMotif = (jest.fn() as any).mockRejectedValue(new Error('Search failed'));

            mockState.ensureBackgroundKernel.mockResolvedValue({
                searchMotif: mockSearchMotif,
            });

            const messageHandler = mockWebviewView.webview.onDidReceiveMessage.mock.calls[0][0];

            await messageHandler({ type: 'searchMotif', motif: 'DRACH', strand: '+' });

            expect(vscode.window.showErrorMessage).toHaveBeenCalledWith(
                'Failed to search motif: Search failed'
            );

            // Should update view with empty matches
            expect(mockWebviewView.webview.postMessage).toHaveBeenCalledWith(
                expect.objectContaining({
                    type: 'updateMatches',
                    matches: [],
                    searching: false,
                })
            );
        });

        it('should handle non-Error exceptions during search', async () => {
            mockState.currentFastaFile = '/path/to/reference.fasta';
            mockState.usePositron = true;

            const mockSearchMotif = (jest.fn() as any).mockRejectedValue('String error');

            mockState.ensureBackgroundKernel.mockResolvedValue({
                searchMotif: mockSearchMotif,
            });

            const messageHandler = mockWebviewView.webview.onDidReceiveMessage.mock.calls[0][0];

            await messageHandler({ type: 'searchMotif', motif: 'DRACH', strand: '+' });

            expect(vscode.window.showErrorMessage).toHaveBeenCalledWith(
                'Failed to search motif: String error'
            );
        });

        it('should set searching state before and after search', async () => {
            mockState.currentFastaFile = '/path/to/reference.fasta';
            mockState.usePositron = true;

            const mockSearchMotif = (jest.fn() as any).mockImplementation(
                () => new Promise((resolve) => setTimeout(() => resolve([]), 10))
            );

            mockState.ensureBackgroundKernel.mockResolvedValue({
                searchMotif: mockSearchMotif,
            });

            const messageHandler = mockWebviewView.webview.onDidReceiveMessage.mock.calls[0][0];

            const searchPromise = messageHandler({
                type: 'searchMotif',
                motif: 'DRACH',
                strand: 'both',
            });

            // Should set searching=true before search
            await new Promise((resolve) => setTimeout(resolve, 0));
            expect(mockWebviewView.webview.postMessage).toHaveBeenCalledWith(
                expect.objectContaining({
                    type: 'updateMatches',
                    searching: true,
                })
            );

            // Wait for search to complete
            await searchPromise;

            // Should set searching=false after search
            expect(mockWebviewView.webview.postMessage).toHaveBeenCalledWith(
                expect.objectContaining({
                    type: 'updateMatches',
                    searching: false,
                })
            );
        });
    });

    describe('Message Handling - plotAllMotifs', () => {
        beforeEach(() => {
            provider.resolveWebviewView(mockWebviewView, {} as any, {} as any);
            (vscode.commands.executeCommand as any).mockResolvedValue(undefined);
        });

        it('should show error when no FASTA file loaded', async () => {
            const messageHandler = mockWebviewView.webview.onDidReceiveMessage.mock.calls[0][0];

            await messageHandler({
                type: 'plotAllMotifs',
                motif: 'DRACH',
                upstream: 10,
                downstream: 10,
            });

            expect(vscode.window.showErrorMessage).toHaveBeenCalledWith('No FASTA file loaded.');
        });

        it('should show error when no matches available', async () => {
            mockState.currentFastaFile = '/path/to/reference.fasta';

            const messageHandler = mockWebviewView.webview.onDidReceiveMessage.mock.calls[0][0];

            await messageHandler({
                type: 'plotAllMotifs',
                motif: 'DRACH',
                upstream: 10,
                downstream: 10,
            });

            expect(vscode.window.showErrorMessage).toHaveBeenCalledWith(
                'No motif matches to plot. Search for a motif first.'
            );
        });

        it('should execute plot command when matches available', async () => {
            mockState.currentFastaFile = '/path/to/reference.fasta';

            // Save message handler before clearing mocks
            const messageHandler = mockWebviewView.webview.onDidReceiveMessage.mock.calls[0][0];

            // Set some matches first
            const matches: MotifMatch[] = [
                { chrom: 'chr1', position: 100, sequence: 'GGACA', strand: '+' },
            ];
            provider.setMatches(matches);
            jest.clearAllMocks();

            await messageHandler({
                type: 'plotAllMotifs',
                motif: 'DRACH',
                upstream: 15,
                downstream: 20,
            });

            expect(vscode.commands.executeCommand).toHaveBeenCalledWith(
                'squiggy.plotMotifAggregateAll',
                {
                    fastaFile: '/path/to/reference.fasta',
                    motif: 'DRACH',
                    upstream: 15,
                    downstream: 20,
                }
            );
        });

        it('should handle errors during plot command', async () => {
            mockState.currentFastaFile = '/path/to/reference.fasta';

            // Save message handler before clearing mocks
            const messageHandler = mockWebviewView.webview.onDidReceiveMessage.mock.calls[0][0];

            const matches: MotifMatch[] = [
                { chrom: 'chr1', position: 100, sequence: 'GGACA', strand: '+' },
            ];
            provider.setMatches(matches);
            jest.clearAllMocks();

            (vscode.commands.executeCommand as any).mockRejectedValue(new Error('Plot failed'));

            await messageHandler({
                type: 'plotAllMotifs',
                motif: 'DRACH',
                upstream: 10,
                downstream: 10,
            });

            expect(vscode.window.showErrorMessage).toHaveBeenCalledWith(
                'Failed to plot motif aggregate: Plot failed'
            );
        });

        it('should handle non-Error exceptions during plot', async () => {
            mockState.currentFastaFile = '/path/to/reference.fasta';

            // Save message handler before clearing mocks
            const messageHandler = mockWebviewView.webview.onDidReceiveMessage.mock.calls[0][0];

            const matches: MotifMatch[] = [
                { chrom: 'chr1', position: 100, sequence: 'GGACA', strand: '+' },
            ];
            provider.setMatches(matches);
            jest.clearAllMocks();

            (vscode.commands.executeCommand as any).mockRejectedValue('String error');

            await messageHandler({
                type: 'plotAllMotifs',
                motif: 'DRACH',
                upstream: 10,
                downstream: 10,
            });

            expect(vscode.window.showErrorMessage).toHaveBeenCalledWith(
                'Failed to plot motif aggregate: String error'
            );
        });
    });

    describe('setMatches', () => {
        beforeEach(() => {
            provider.resolveWebviewView(mockWebviewView, {} as any, {} as any);
            jest.clearAllMocks();
        });

        it('should set matches and update view', () => {
            const matches: MotifMatch[] = [
                { chrom: 'chr1', position: 100, sequence: 'GGACA', strand: '+' },
                { chrom: 'chr2', position: 200, sequence: 'AGACT', strand: '-' },
            ];

            provider.setMatches(matches);

            expect(mockWebviewView.webview.postMessage).toHaveBeenCalledWith({
                type: 'updateMatches',
                matches: matches,
                searching: false,
                motif: 'DRACH',
            });
        });

        it('should set empty matches', () => {
            provider.setMatches([]);

            expect(mockWebviewView.webview.postMessage).toHaveBeenCalledWith({
                type: 'updateMatches',
                matches: [],
                searching: false,
                motif: 'DRACH',
            });
        });
    });

    describe('clear', () => {
        beforeEach(() => {
            provider.resolveWebviewView(mockWebviewView, {} as any, {} as any);
        });

        it('should clear matches and reset state', () => {
            // Set some matches first
            const matches: MotifMatch[] = [
                { chrom: 'chr1', position: 100, sequence: 'GGACA', strand: '+' },
            ];
            provider.setMatches(matches);
            jest.clearAllMocks();

            provider.clear();

            expect(mockWebviewView.webview.postMessage).toHaveBeenCalledWith({
                type: 'updateMatches',
                matches: [],
                searching: false,
                motif: 'DRACH',
            });
        });
    });

    describe('updateView', () => {
        it('should not post message if view not available', () => {
            // Don't resolve webview view
            (provider as any).updateView();

            expect(mockWebviewView.webview.postMessage).not.toHaveBeenCalled();
        });

        it('should post message when view is available', () => {
            provider.resolveWebviewView(mockWebviewView, {} as any, {} as any);
            jest.clearAllMocks();

            (provider as any).updateView();

            expect(mockWebviewView.webview.postMessage).toHaveBeenCalledWith({
                type: 'updateMatches',
                matches: [],
                searching: false,
                motif: 'DRACH',
            });
        });
    });

    describe('getHtmlContent', () => {
        it('should generate HTML with motif search UI', () => {
            provider.resolveWebviewView(mockWebviewView, {} as any, {} as any);

            const html = mockWebviewView.webview.html;

            expect(html).toContain('Motif Pattern (IUPAC)');
            expect(html).toContain('searchMotif');
            expect(html).toContain('plotAllMotifs');
            expect(html).toContain('updateMatches');
        });

        it('should include window control sliders', () => {
            provider.resolveWebviewView(mockWebviewView, {} as any, {} as any);

            const html = mockWebviewView.webview.html;

            expect(html).toContain('upstreamSlider');
            expect(html).toContain('downstreamSlider');
            expect(html).toContain('Window Size (bp)');
        });

        it('should include strand selection', () => {
            provider.resolveWebviewView(mockWebviewView, {} as any, {} as any);

            const html = mockWebviewView.webview.html;

            expect(html).toContain('plusStrandOnly');
            expect(html).toContain('+ strand only');
        });
    });
});
