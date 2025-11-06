/**
 * Tests for Plot Options View Provider
 *
 * Tests the PlotOptionsViewProvider webview implementation.
 * Target: >80% coverage of squiggy-plot-options-view.ts
 */

import { describe, it, expect, beforeEach, jest, afterEach } from '@jest/globals';
import * as vscode from 'vscode';
import { PlotOptionsViewProvider } from '../squiggy-plot-options-view';

describe('PlotOptionsViewProvider', () => {
    let provider: PlotOptionsViewProvider;
    let mockWebviewView: any;
    let optionsChangeListener: jest.Mock;
    let aggregatePlotListener: jest.Mock;
    let mockState: any;

    beforeEach(() => {
        const extensionUri = vscode.Uri.file('/mock/extension');

        // Create mock ExtensionState
        mockState = {
            onVisualizationSelectionChanged: jest.fn().mockReturnValue({
                dispose: jest.fn(),
            }),
        };

        provider = new PlotOptionsViewProvider(extensionUri, mockState);

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

        optionsChangeListener = jest.fn();
        aggregatePlotListener = jest.fn();
        provider.onDidChangeOptions(optionsChangeListener);
        provider.onDidRequestAggregatePlot(aggregatePlotListener);

        jest.clearAllMocks();
    });

    afterEach(() => {
        jest.clearAllMocks();
    });

    describe('Provider Properties', () => {
        it('should have correct viewType', () => {
            expect(PlotOptionsViewProvider.viewType).toBe('squiggyPlotOptions');
        });

        it('should return correct title', () => {
            const title = (provider as any).getTitle();
            expect(title).toBe('Plotting');
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

            // Should send updatePlotOptions message
            expect(mockWebviewView.webview.postMessage).toHaveBeenCalledWith(
                expect.objectContaining({
                    type: 'updatePlotOptions',
                    options: expect.objectContaining({
                        plotType: 'AGGREGATE',
                        mode: 'SINGLE',
                        normalization: 'ZNORM',
                    }),
                })
            );

            // Should send POD5 status
            expect(mockWebviewView.webview.postMessage).toHaveBeenCalledWith({
                type: 'updatePod5Status',
                hasPod5: false,
            });

            // Should send BAM status
            expect(mockWebviewView.webview.postMessage).toHaveBeenCalledWith({
                type: 'updateBamStatus',
                hasBam: false,
            });
        });

        it('should handle optionsChanged message and update state', async () => {
            const messageHandler = mockWebviewView.webview.onDidReceiveMessage.mock.calls[0][0];

            await messageHandler({
                type: 'optionsChanged',
                options: {
                    plotType: 'AGGREGATE',
                    mode: 'STACKED',
                    normalization: 'MEDIAN',
                    showDwellTime: true,
                    showBaseAnnotations: false,
                    scaleDwellTime: true,
                    downsample: 10,
                    showSignalPoints: true,
                    clipXAxisToAlignment: false,
                },
            });

            const options = provider.getOptions();
            expect(options.plotType).toBe('AGGREGATE');
            expect(options.mode).toBe('STACKED');
            expect(options.normalization).toBe('MEDIAN');
            expect(options.showDwellTime).toBe(true);
            expect(options.showBaseAnnotations).toBe(false);
            expect(options.scaleDwellTime).toBe(true);
            expect(options.downsample).toBe(10);
            expect(options.showSignalPoints).toBe(true);
            expect(options.clipXAxisToAlignment).toBe(false);

            // Should fire change event
            expect(optionsChangeListener).toHaveBeenCalled();
        });

        it('should handle optionsChanged with aggregate-specific options', async () => {
            const messageHandler = mockWebviewView.webview.onDidReceiveMessage.mock.calls[0][0];

            await messageHandler({
                type: 'optionsChanged',
                options: {
                    mode: 'AGGREGATE',
                    normalization: 'ZNORM',
                    showDwellTime: false,
                    showBaseAnnotations: true,
                    scaleDwellTime: false,
                    downsample: 5,
                    showSignalPoints: false,
                    aggregateReference: 'chr1',
                    aggregateMaxReads: 200,
                    showModifications: false,
                    showPileup: false,
                    showSignal: false,
                    showQuality: false,
                },
            });

            const options = provider.getOptions();
            expect(options.aggregateReference).toBe('chr1');
            expect(options.aggregateMaxReads).toBe(200);
            expect(options.showModifications).toBe(false);
            expect(options.showPileup).toBe(false);
            expect(options.showSignal).toBe(false);
            expect(options.showQuality).toBe(false);
        });

        it('should handle requestReferences message', async () => {
            const messageHandler = mockWebviewView.webview.onDidReceiveMessage.mock.calls[0][0];

            await messageHandler({ type: 'requestReferences' });

            // Should fire change event for extension.ts to handle
            expect(optionsChangeListener).toHaveBeenCalled();
        });

        it('should handle generateAggregatePlot message', async () => {
            const messageHandler = mockWebviewView.webview.onDidReceiveMessage.mock.calls[0][0];

            await messageHandler({
                type: 'generateAggregatePlot',
                reference: 'chr2',
                maxReads: 150,
                normalization: 'MAD',
                showModifications: true,
                showPileup: true,
                showDwellTime: false,
                showSignal: true,
                showQuality: true,
                clipXAxisToAlignment: true,
            });

            // Should fire event with plot parameters
            expect(aggregatePlotListener).toHaveBeenCalledWith({
                reference: 'chr2',
                maxReads: 150,
                normalization: 'MAD',
                showModifications: true,
                showPileup: true,
                showDwellTime: false,
                showSignal: true,
                showQuality: true,
                clipXAxisToAlignment: true,
            });
        });
    });

    describe('getOptions', () => {
        it('should return default options', () => {
            const options = provider.getOptions();

            // Use objectContaining to be resilient to new properties being added
            expect(options).toEqual(
                expect.objectContaining({
                    plotType: 'AGGREGATE',
                    mode: 'SINGLE',
                    normalization: 'ZNORM',
                    showDwellTime: false,
                    showBaseAnnotations: true,
                    scaleDwellTime: false,
                    downsample: 5,
                    showSignalPoints: false,
                    clipXAxisToAlignment: true,
                    aggregateReference: '',
                    aggregateMaxReads: 100,
                    showModifications: true,
                    showPileup: true,
                    showSignal: true,
                    showQuality: true,
                })
            );
        });

        it('should return updated options after changes', () => {
            provider.resolveWebviewView(mockWebviewView, {} as any, {} as any);
            const messageHandler = mockWebviewView.webview.onDidReceiveMessage.mock.calls[0][0];

            messageHandler({
                type: 'optionsChanged',
                options: {
                    mode: 'OVERLAY',
                    normalization: 'MEDIAN',
                    showDwellTime: true,
                    showBaseAnnotations: false,
                    scaleDwellTime: true,
                    downsample: 20,
                    showSignalPoints: true,
                    clipXAxisToAlignment: false,
                },
            });

            const options = provider.getOptions();
            expect(options.mode).toBe('OVERLAY');
            expect(options.normalization).toBe('MEDIAN');
            expect(options.showDwellTime).toBe(true);
        });
    });

    describe('updatePod5Status', () => {
        beforeEach(() => {
            provider.resolveWebviewView(mockWebviewView, {} as any, {} as any);
            jest.clearAllMocks();
        });

        it('should update POD5 status to true', () => {
            provider.updatePod5Status(true);

            expect(mockWebviewView.webview.postMessage).toHaveBeenCalledWith({
                type: 'updatePod5Status',
                hasPod5: true,
            });
        });

        it('should update POD5 status to false', () => {
            provider.updatePod5Status(false);

            expect(mockWebviewView.webview.postMessage).toHaveBeenCalledWith({
                type: 'updatePod5Status',
                hasPod5: false,
            });
        });

        it('should not post message if view not available', () => {
            const newProvider = new PlotOptionsViewProvider(vscode.Uri.file('/mock'), mockState);

            newProvider.updatePod5Status(true);

            expect(mockWebviewView.webview.postMessage).not.toHaveBeenCalled();
        });
    });

    describe('updateBamStatus', () => {
        beforeEach(() => {
            provider.resolveWebviewView(mockWebviewView, {} as any, {} as any);
            jest.clearAllMocks();
        });

        it('should update BAM status to true and switch to EVENTALIGN mode', () => {
            provider.updateBamStatus(true);

            expect(mockWebviewView.webview.postMessage).toHaveBeenCalledWith({
                type: 'updateBamStatus',
                hasBam: true,
            });

            const options = provider.getOptions();
            expect(options.plotType).toBe('AGGREGATE');
            expect(options.mode).toBe('EVENTALIGN');
        });

        it('should update BAM status to false and switch back to MULTI_READ_OVERLAY mode', () => {
            // First set to true
            provider.updateBamStatus(true);
            jest.clearAllMocks();

            // Then set to false
            provider.updateBamStatus(false);

            expect(mockWebviewView.webview.postMessage).toHaveBeenCalledWith({
                type: 'updateBamStatus',
                hasBam: false,
            });

            const options = provider.getOptions();
            expect(options.plotType).toBe('MULTI_READ_OVERLAY');
            expect(options.mode).toBe('SINGLE');
        });

        it('should update workspace configuration when BAM loaded', () => {
            const mockConfig = {
                update: jest.fn(),
            };
            (vscode.workspace.getConfiguration as jest.Mock).mockReturnValue(mockConfig);

            provider.updateBamStatus(true);

            expect(vscode.workspace.getConfiguration).toHaveBeenCalledWith('squiggy');
            expect(mockConfig.update).toHaveBeenCalledWith(
                'defaultPlotMode',
                'EVENTALIGN',
                vscode.ConfigurationTarget.Workspace
            );
        });

        it('should update workspace configuration when BAM unloaded', () => {
            const mockConfig = {
                update: jest.fn(),
            };
            (vscode.workspace.getConfiguration as jest.Mock).mockReturnValue(mockConfig);

            provider.updateBamStatus(false);

            expect(vscode.workspace.getConfiguration).toHaveBeenCalledWith('squiggy');
            expect(mockConfig.update).toHaveBeenCalledWith(
                'defaultPlotMode',
                'SINGLE',
                vscode.ConfigurationTarget.Workspace
            );
        });
    });

    describe('updateReferences', () => {
        beforeEach(() => {
            provider.resolveWebviewView(mockWebviewView, {} as any, {} as any);
            jest.clearAllMocks();
        });

        it('should update references and send to webview', () => {
            provider.updateReferences(['chr1', 'chr2', 'chr3']);

            expect(mockWebviewView.webview.postMessage).toHaveBeenCalledWith({
                type: 'updateReferences',
                references: ['chr1', 'chr2', 'chr3'],
            });
        });

        it('should set first reference as aggregate reference if none set', () => {
            provider.updateReferences(['chr1', 'chr2']);

            const options = provider.getOptions();
            expect(options.aggregateReference).toBe('chr1');
        });

        it('should not change aggregate reference if already set', () => {
            // Set reference via optionsChanged
            provider.resolveWebviewView(mockWebviewView, {} as any, {} as any);
            const messageHandler = mockWebviewView.webview.onDidReceiveMessage.mock.calls[0][0];
            messageHandler({
                type: 'optionsChanged',
                options: {
                    mode: 'AGGREGATE',
                    normalization: 'ZNORM',
                    showDwellTime: false,
                    showBaseAnnotations: true,
                    scaleDwellTime: false,
                    downsample: 5,
                    showSignalPoints: false,
                    aggregateReference: 'chr2',
                },
            });
            jest.clearAllMocks();

            // Update references
            provider.updateReferences(['chr1', 'chr2', 'chr3']);

            const options = provider.getOptions();
            expect(options.aggregateReference).toBe('chr2'); // Should remain chr2
        });

        it('should handle empty references array', () => {
            provider.updateReferences([]);

            expect(mockWebviewView.webview.postMessage).toHaveBeenCalledWith({
                type: 'updateReferences',
                references: [],
            });

            const options = provider.getOptions();
            expect(options.aggregateReference).toBe(''); // Should remain empty
        });
    });

    describe('updateView', () => {
        beforeEach(() => {
            provider.resolveWebviewView(mockWebviewView, {} as any, {} as any);
        });

        it('should send all current options to webview', () => {
            (provider as any).updateView();

            expect(mockWebviewView.webview.postMessage).toHaveBeenCalledWith(
                expect.objectContaining({
                    type: 'updatePlotOptions',
                    options: expect.objectContaining({
                        plotType: 'AGGREGATE',
                        mode: 'SINGLE',
                        normalization: 'ZNORM',
                    }),
                })
            );
        });

        it('should send references when BAM loaded and references available', () => {
            provider.updateBamStatus(true);
            provider.updateReferences(['chr1', 'chr2']);
            jest.clearAllMocks();

            (provider as any).updateView();

            expect(mockWebviewView.webview.postMessage).toHaveBeenCalledWith({
                type: 'updateReferences',
                references: ['chr1', 'chr2'],
            });
        });

        it('should not send references when no BAM loaded', () => {
            provider.updateReferences(['chr1', 'chr2']);
            jest.clearAllMocks();

            (provider as any).updateView();

            // Should not send updateReferences message
            expect(mockWebviewView.webview.postMessage).not.toHaveBeenCalledWith(
                expect.objectContaining({
                    type: 'updateReferences',
                })
            );
        });

        it('should not post message if view not available', () => {
            // Create provider without resolving view
            const newProvider = new PlotOptionsViewProvider(vscode.Uri.file('/mock'), mockState);

            (newProvider as any).updateView();

            // Should not throw and should not post
            expect(mockWebviewView.webview.postMessage).not.toHaveBeenCalled();
        });
    });

    describe('Event Emitters', () => {
        beforeEach(() => {
            provider.resolveWebviewView(mockWebviewView, {} as any, {} as any);
        });

        it('should fire onDidChangeOptions when options change', async () => {
            const messageHandler = mockWebviewView.webview.onDidReceiveMessage.mock.calls[0][0];

            await messageHandler({
                type: 'optionsChanged',
                options: {
                    mode: 'OVERLAY',
                    normalization: 'MAD',
                    showDwellTime: true,
                    showBaseAnnotations: true,
                    scaleDwellTime: false,
                    downsample: 5,
                    showSignalPoints: false,
                },
            });

            expect(optionsChangeListener).toHaveBeenCalledTimes(1);
        });

        it('should fire onDidRequestAggregatePlot when plot requested', async () => {
            const messageHandler = mockWebviewView.webview.onDidReceiveMessage.mock.calls[0][0];

            await messageHandler({
                type: 'generateAggregatePlot',
                reference: 'chr1',
                maxReads: 100,
                normalization: 'ZNORM',
                showModifications: true,
                showPileup: true,
                showDwellTime: false,
                showSignal: true,
                showQuality: true,
                clipXAxisToAlignment: false,
            });

            expect(aggregatePlotListener).toHaveBeenCalledTimes(1);
        });

        it('should allow multiple listeners for onDidChangeOptions', async () => {
            const listener2 = jest.fn();
            provider.onDidChangeOptions(listener2);

            const messageHandler = mockWebviewView.webview.onDidReceiveMessage.mock.calls[0][0];

            await messageHandler({
                type: 'optionsChanged',
                options: {
                    mode: 'SINGLE',
                    normalization: 'ZNORM',
                    showDwellTime: false,
                    showBaseAnnotations: true,
                    scaleDwellTime: false,
                    downsample: 5,
                    showSignalPoints: false,
                },
            });

            expect(optionsChangeListener).toHaveBeenCalled();
            expect(listener2).toHaveBeenCalled();
        });
    });
});
