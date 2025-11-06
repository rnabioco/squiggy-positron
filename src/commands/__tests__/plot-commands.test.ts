/**
 * Tests for Plot Commands
 *
 * Tests command handlers for plotting operations.
 * Target: >80% coverage of plot-commands.ts
 */

import { describe, it, expect, beforeEach, jest, afterEach } from '@jest/globals';
import * as vscode from 'vscode';
import { registerPlotCommands } from '../plot-commands';
import { ExtensionState } from '../../state/extension-state';

describe('Plot Commands', () => {
    let mockContext: vscode.ExtensionContext;
    let mockState: any;
    let commandHandlers: Map<string, Function>;

    // Helper to register commands
    const registerCommands = () => {
        registerPlotCommands(mockContext, mockState);
    };

    beforeEach(() => {
        // Clear command registry before each test
        commandHandlers = new Map();

        // Mock extension context
        mockContext = {
            subscriptions: [],
            extensionPath: '/mock/extension/path',
            extensionUri: vscode.Uri.file('/mock/extension/path'),
        } as any;

        // Mock state with minimal required properties
        mockState = {
            usePositron: true,
            squiggyAPI: {
                generatePlot: (jest.fn() as any).mockResolvedValue(undefined),
                generateAggregatePlot: (jest.fn() as any).mockResolvedValue(undefined),
            },
            currentPod5File: null,
            currentBamFile: null,
            currentFastaFile: null,
            getAllSampleNames: jest.fn().mockReturnValue([]),
            getSample: jest.fn().mockReturnValue(null),
            selectedReadExplorerSample: null,
            plotOptionsProvider: {
                getOptions: jest.fn().mockReturnValue({
                    mode: 'SINGLE',
                    normalization: 'ZNORM',
                    showDwellTime: true,
                    showBaseAnnotations: true,
                    scaleDwellTime: false,
                    downsample: false,
                    showSignalPoints: true,
                }),
            },
            modificationsProvider: {
                getFilters: jest.fn().mockReturnValue({
                    minProbability: 0.5,
                    enabledModTypes: [],
                }),
            },
            currentPlotReadIds: null,
        } as any;

        // Override registerCommand to capture handlers
        (vscode.commands.registerCommand as any).mockImplementation(
            (command: string, callback: Function) => {
                commandHandlers.set(command, callback);
                const disposable = { dispose: jest.fn() };
                return disposable;
            }
        );

        // Mock workspace configuration
        (vscode.workspace.getConfiguration as any).mockReturnValue({
            get: jest.fn((key: string, defaultValue: any) => {
                if (key === 'aggregateSampleSize') return 100;
                return defaultValue;
            }),
        });

        // Reset all vscode mocks
        jest.clearAllMocks();
    });

    afterEach(() => {
        commandHandlers.clear();
    });

    describe('Command Registration', () => {
        it('should register all plot commands', () => {
            registerCommands();

            const expectedCommands = [
                'squiggy.plotRead',
                'squiggy.plotAggregate',
                'squiggy.plotMotifAggregateAll',
                'squiggy.plotSignalOverlayComparison',
                'squiggy.plotDeltaComparison',
            ];

            expectedCommands.forEach((cmd) => {
                expect(vscode.commands.registerCommand).toHaveBeenCalledWith(
                    cmd,
                    expect.any(Function)
                );
            });
        });

        it('should add all command disposables to context', () => {
            registerCommands();

            // Verify registerCommand was called 5 times (once per command)
            expect(vscode.commands.registerCommand).toHaveBeenCalledTimes(5);
        });
    });

    describe('squiggy.plotRead', () => {
        it('should show error when no POD5 file is loaded', async () => {
            registerCommands();
            const handler = commandHandlers.get('squiggy.plotRead');

            await handler!();

            expect(vscode.window.showErrorMessage).toHaveBeenCalledWith(
                'No POD5 file loaded. Use "Load Sample(s)" in the Samples panel.'
            );
        });

        it('should show warning when no read is selected', async () => {
            mockState.currentPod5File = '/path/to/test.pod5';

            registerCommands();
            const handler = commandHandlers.get('squiggy.plotRead');

            await handler!();

            expect(vscode.window.showWarningMessage).toHaveBeenCalledWith(
                'Please click the Plot button on a read in the Reads panel'
            );
        });

        it('should plot when called with readId string', async () => {
            mockState.currentPod5File = '/path/to/test.pod5';

            registerCommands();
            const handler = commandHandlers.get('squiggy.plotRead');

            await handler!('read_001');

            expect(mockState.squiggyAPI.generatePlot).toHaveBeenCalled();
        });

        it('should plot when called with ReadItem object', async () => {
            mockState.currentPod5File = '/path/to/test.pod5';

            registerCommands();
            const handler = commandHandlers.get('squiggy.plotRead');

            await handler!({ readId: 'read_001' });

            expect(mockState.squiggyAPI.generatePlot).toHaveBeenCalled();
        });
    });

    describe('squiggy.plotAggregate', () => {
        it('should show error when no POD5 file is loaded', async () => {
            registerCommands();
            const handler = commandHandlers.get('squiggy.plotAggregate');

            await handler!('ref1');

            expect(vscode.window.showErrorMessage).toHaveBeenCalledWith(
                'No POD5 file loaded. Use "Load Sample(s)" in the Samples panel.'
            );
        });

        it('should show error when no BAM file is loaded (legacy mode)', async () => {
            mockState.currentPod5File = '/path/to/test.pod5';
            mockState.currentBamFile = null;

            registerCommands();
            const handler = commandHandlers.get('squiggy.plotAggregate');

            await handler!('ref1');

            expect(vscode.window.showErrorMessage).toHaveBeenCalledWith(
                'Aggregate plots require a BAM file. Use "Open BAM File" first.'
            );
        });

        it('should show error when sample has no BAM (multi-sample mode)', async () => {
            mockState.getAllSampleNames.mockReturnValue(['Sample_A']);
            mockState.selectedReadExplorerSample = 'Sample_A';
            mockState.getSample.mockReturnValue({
                sampleId: 'Sample_A',
                hasBam: false,
            });

            registerCommands();
            const handler = commandHandlers.get('squiggy.plotAggregate');

            await handler!('ref1');

            expect(vscode.window.showErrorMessage).toHaveBeenCalledWith(
                'Sample "Sample_A" does not have BAM alignment data. Aggregate plots require BAM files.'
            );
        });

        it('should show error when no reference name provided', async () => {
            mockState.currentPod5File = '/path/to/test.pod5';
            mockState.currentBamFile = '/path/to/test.bam';

            registerCommands();
            const handler = commandHandlers.get('squiggy.plotAggregate');

            await handler!();

            expect(vscode.window.showErrorMessage).toHaveBeenCalledWith(
                'Please click the Aggregate button on a reference in the Read Explorer panel'
            );
        });

        it('should generate aggregate plot when all conditions are met', async () => {
            mockState.currentPod5File = '/path/to/test.pod5';
            mockState.currentBamFile = '/path/to/test.bam';

            registerCommands();
            const handler = commandHandlers.get('squiggy.plotAggregate');

            await handler!('ref1');

            expect(mockState.squiggyAPI.generateAggregatePlot).toHaveBeenCalledWith(
                'ref1',
                100, // maxReads from config
                'ZNORM',
                expect.any(String), // theme
                true,
                0.5,
                [],
                true,
                true,
                true,
                true,
                true,
                undefined
            );
        });
    });

    describe('squiggy.plotMotifAggregateAll', () => {
        it('should show error when no POD5 file is loaded', async () => {
            registerCommands();
            const handler = commandHandlers.get('squiggy.plotMotifAggregateAll');

            await handler!({ fastaFile: 'test.fa', motif: 'ATCG', upstream: 10, downstream: 10 });

            expect(vscode.window.showErrorMessage).toHaveBeenCalledWith(
                'No POD5 file loaded. Use "Open POD5 File" first.'
            );
        });

        it('should show error when no BAM file is loaded', async () => {
            mockState.currentPod5File = '/path/to/test.pod5';

            registerCommands();
            const handler = commandHandlers.get('squiggy.plotMotifAggregateAll');

            await handler!({ fastaFile: 'test.fa', motif: 'ATCG', upstream: 10, downstream: 10 });

            expect(vscode.window.showErrorMessage).toHaveBeenCalledWith(
                'Motif aggregate plots require a BAM file. Use "Open BAM File" first.'
            );
        });

        it('should show error when no FASTA file is loaded', async () => {
            mockState.currentPod5File = '/path/to/test.pod5';
            mockState.currentBamFile = '/path/to/test.bam';

            registerCommands();
            const handler = commandHandlers.get('squiggy.plotMotifAggregateAll');

            await handler!({ fastaFile: 'test.fa', motif: 'ATCG', upstream: 10, downstream: 10 });

            expect(vscode.window.showErrorMessage).toHaveBeenCalledWith(
                'No FASTA file loaded. Use "Open FASTA File" first.'
            );
        });

        it('should show error when no params provided', async () => {
            mockState.currentPod5File = '/path/to/test.pod5';
            mockState.currentBamFile = '/path/to/test.bam';
            mockState.currentFastaFile = '/path/to/test.fa';

            registerCommands();
            const handler = commandHandlers.get('squiggy.plotMotifAggregateAll');

            await handler!();

            expect(vscode.window.showErrorMessage).toHaveBeenCalledWith(
                'Please select a motif and window from the Motif Explorer panel'
            );
        });
    });

    describe('squiggy.plotSignalOverlayComparison', () => {
        it('should show error when less than 2 samples are loaded', async () => {
            mockState.getAllSampleNames.mockReturnValue(['Sample_A']);

            registerCommands();
            const handler = commandHandlers.get('squiggy.plotSignalOverlayComparison');

            await handler!();

            expect(vscode.window.showErrorMessage).toHaveBeenCalledWith(
                expect.stringContaining('at least 2 loaded samples')
            );
        });
    });

    describe('squiggy.plotDeltaComparison', () => {
        it('should show error when less than 2 samples are loaded', async () => {
            mockState.getAllSampleNames.mockReturnValue(['Sample_A']);

            registerCommands();
            const handler = commandHandlers.get('squiggy.plotDeltaComparison');

            await handler!();

            expect(vscode.window.showErrorMessage).toHaveBeenCalledWith(
                expect.stringContaining('at least 2 loaded samples')
            );
        });
    });
});
