/**
 * Plot Commands
 *
 * Handles generating plots for individual reads and aggregates.
 * Extracted from extension.ts to improve modularity.
 */

import * as vscode from 'vscode';
import { ExtensionState } from '../state/extension-state';
import { ReadItem } from '../types/squiggy-reads-types';
import { ErrorContext, safeExecuteWithProgress } from '../utils/error-handler';

/**
 * Register plot-related commands
 */
export function registerPlotCommands(
    context: vscode.ExtensionContext,
    state: ExtensionState
): void {
    // Plot selected reads
    context.subscriptions.push(
        vscode.commands.registerCommand(
            'squiggy.plotRead',
            async (readIdOrItem?: string | ReadItem) => {
                let readIds: string[];

                // If called with a readId string (from React view)
                if (typeof readIdOrItem === 'string') {
                    readIds = [readIdOrItem];
                }
                // If called with a ReadItem (from command palette or context menu)
                else if (readIdOrItem && 'readId' in readIdOrItem) {
                    readIds = [readIdOrItem.readId];
                }
                // Otherwise, no selection available (React view handles selection internally)
                else {
                    vscode.window.showWarningMessage(
                        'Please click the Plot button on a read in the Reads panel'
                    );
                    return;
                }

                await plotReads(readIds, state);
            }
        )
    );

    // Plot aggregate for reference
    context.subscriptions.push(
        vscode.commands.registerCommand('squiggy.plotAggregate', async (referenceName?: string) => {
            // Validate that both POD5 and BAM are loaded
            if (!state.currentPod5File) {
                vscode.window.showErrorMessage('No POD5 file loaded. Use "Open POD5 File" first.');
                return;
            }
            if (!state.currentBamFile) {
                vscode.window.showErrorMessage(
                    'Aggregate plots require a BAM file. Use "Open BAM File" first.'
                );
                return;
            }

            // Validate reference name was provided
            if (!referenceName) {
                vscode.window.showErrorMessage(
                    'Please click the Aggregate button on a reference in the Read Explorer panel'
                );
                return;
            }

            await plotAggregate(referenceName, state);
        })
    );

    // Plot motif aggregate all
    context.subscriptions.push(
        vscode.commands.registerCommand(
            'squiggy.plotMotifAggregateAll',
            async (params?: {
                fastaFile: string;
                motif: string;
                upstream: number;
                downstream: number;
            }) => {
                // Validate that all required files are loaded
                if (!state.currentPod5File) {
                    vscode.window.showErrorMessage(
                        'No POD5 file loaded. Use "Open POD5 File" first.'
                    );
                    return;
                }
                if (!state.currentBamFile) {
                    vscode.window.showErrorMessage(
                        'Motif aggregate plots require a BAM file. Use "Open BAM File" first.'
                    );
                    return;
                }
                if (!state.currentFastaFile) {
                    vscode.window.showErrorMessage(
                        'No FASTA file loaded. Use "Open FASTA File" first.'
                    );
                    return;
                }

                // Validate params were provided
                if (!params) {
                    vscode.window.showErrorMessage(
                        'Please select a motif and window from the Motif Explorer panel'
                    );
                    return;
                }

                await plotMotifAggregateAll(params, state);
            }
        )
    );

    // Plot signal overlay comparison - Phase 1 (Default multi-sample comparison)
    context.subscriptions.push(
        vscode.commands.registerCommand(
            'squiggy.plotSignalOverlayComparison',
            async (sampleNames?: string[], maxReads?: number | null) => {
                // If no sample names provided, prompt user to select
                if (!sampleNames || sampleNames.length === 0) {
                    // Get list of loaded samples
                    const loadedSamples = state.getAllSampleNames();

                    if (loadedSamples.length < 2) {
                        vscode.window.showErrorMessage(
                            'Signal overlay comparison requires at least 2 loaded samples. ' +
                                'Use "Load Sample" to add samples for comparison.'
                        );
                        return;
                    }

                    sampleNames = loadedSamples;
                }

                if (sampleNames.length < 2) {
                    vscode.window.showErrorMessage(
                        'Signal overlay comparison requires at least 2 samples'
                    );
                    return;
                }

                await plotSignalOverlayComparison(sampleNames, state, maxReads);
            }
        )
    );

    // Plot delta comparison - Phase 4 (Optional, 2-sample only)
    context.subscriptions.push(
        vscode.commands.registerCommand(
            'squiggy.plotDeltaComparison',
            async (sampleNames?: string[], maxReads?: number | null) => {
                // If no sample names provided, prompt user to select
                if (!sampleNames || sampleNames.length === 0) {
                    // Get list of loaded samples
                    const loadedSamples = state.getAllSampleNames();

                    if (loadedSamples.length < 2) {
                        vscode.window.showErrorMessage(
                            'Delta comparison requires at least 2 loaded samples. ' +
                                'Use "Load Sample" to add samples for comparison.'
                        );
                        return;
                    }

                    // Let user select samples to compare
                    const selected = await vscode.window.showQuickPick(loadedSamples, {
                        canPickMany: true,
                        placeHolder: 'Select 2 or more samples to compare',
                        matchOnDetail: true,
                    });

                    if (!selected || selected.length < 2) {
                        vscode.window.showWarningMessage(
                            'Please select at least 2 samples for comparison'
                        );
                        return;
                    }

                    sampleNames = selected;
                }

                // Validate we have at least 2 samples
                if (sampleNames.length < 2) {
                    vscode.window.showErrorMessage('Delta comparison requires at least 2 samples');
                    return;
                }

                await plotDeltaComparison(sampleNames, state, maxReads);
            }
        )
    );
}

/**
 * Plot reads
 */
async function plotReads(readIds: string[], state: ExtensionState): Promise<void> {
    // Track current plot for refresh
    state.currentPlotReadIds = readIds;

    await safeExecuteWithProgress(
        async () => {
            // Get options from sidebar panel
            const options = state.plotOptionsProvider?.getOptions();
            if (!options) {
                throw new Error('Plot options not available');
            }

            const mode = options.mode;
            const normalization = options.normalization;

            // Get modification filters
            const modFilters = state.modificationsProvider?.getFilters();
            if (!modFilters) {
                throw new Error('Modification filters not available');
            }

            // Detect VS Code theme
            const colorThemeKind = vscode.window.activeColorTheme.kind;
            const theme = colorThemeKind === vscode.ColorThemeKind.Dark ? 'DARK' : 'LIGHT';

            if (state.usePositron && state.squiggyAPI) {
                // Use Positron kernel - plot appears in Plots pane automatically
                await state.squiggyAPI.generatePlot(
                    readIds,
                    mode,
                    normalization,
                    theme,
                    options.showDwellTime,
                    options.showBaseAnnotations,
                    options.scaleDwellTime,
                    modFilters.minProbability,
                    modFilters.enabledModTypes,
                    options.downsample,
                    options.showSignalPoints,
                    state.selectedReadExplorerSample || undefined // Pass current sample for multi-sample mode
                );
            } else if (state.pythonBackend) {
                // Use subprocess backend - still need webview fallback
                // TODO: subprocess backend doesn't have Plots pane integration
                vscode.window.showWarningMessage(
                    'Plot display in Plots pane requires Positron runtime. Subprocess backend not yet supported.'
                );
            } else {
                throw new Error('No backend available');
            }
        },
        ErrorContext.PLOT_GENERATE,
        `Generating plot for ${readIds.length} read(s)...`
    );
}

/**
 * Generate and display aggregate plot for a reference sequence
 */
async function plotAggregate(referenceName: string, state: ExtensionState): Promise<void> {
    await safeExecuteWithProgress(
        async () => {
            // Get normalization from sidebar panel
            const options = state.plotOptionsProvider?.getOptions();
            if (!options) {
                throw new Error('Plot options not available');
            }

            const normalization = options.normalization;

            // Detect VS Code theme
            const colorThemeKind = vscode.window.activeColorTheme.kind;
            const theme = colorThemeKind === vscode.ColorThemeKind.Dark ? 'DARK' : 'LIGHT';

            // Get max reads from config
            const config = vscode.workspace.getConfiguration('squiggy');
            const maxReads = config.get<number>('aggregateSampleSize', 100);

            if (state.usePositron && state.squiggyAPI) {
                // Use Positron kernel - plot appears in Plots pane automatically
                await state.squiggyAPI.generateAggregatePlot(
                    referenceName,
                    maxReads,
                    normalization,
                    theme,
                    true, // showModifications
                    0.5, // modificationThreshold
                    [], // enabledModTypes
                    true, // showPileup
                    true, // showDwellTime
                    true, // showSignal
                    true, // showQuality
                    true, // clipXAxisToAlignment
                    state.selectedReadExplorerSample || undefined // Pass current sample for multi-sample mode
                );
            } else if (state.pythonBackend) {
                // Subprocess backend not yet implemented for aggregate
                throw new Error(
                    'Aggregate plots are only available with Positron runtime. Please use Positron IDE.'
                );
            } else {
                throw new Error('No backend available');
            }
        },
        ErrorContext.PLOT_GENERATE,
        `Generating aggregate plot for ${referenceName}...`
    );
}

/**
 * Generate and display motif-centered aggregate plot
 */
async function plotMotifAggregateAll(
    params: {
        fastaFile: string;
        motif: string;
        upstream: number;
        downstream: number;
    },
    state: ExtensionState
): Promise<void> {
    await safeExecuteWithProgress(
        async () => {
            // Get normalization from sidebar panel
            const options = state.plotOptionsProvider?.getOptions();
            if (!options) {
                throw new Error('Plot options not available');
            }

            const normalization = options.normalization;

            // Detect VS Code theme
            const colorThemeKind = vscode.window.activeColorTheme.kind;
            const theme = colorThemeKind === vscode.ColorThemeKind.Dark ? 'DARK' : 'LIGHT';

            // Get max reads from config
            const config = vscode.workspace.getConfiguration('squiggy');
            const maxReadsPerMotif = config.get<number>('aggregateSampleSize', 100);

            if (state.usePositron && state.squiggyAPI) {
                // Use Positron kernel - plot appears in Plots pane automatically
                await state.squiggyAPI.generateMotifAggregateAllPlot(
                    params.fastaFile,
                    params.motif,
                    params.upstream,
                    params.downstream,
                    maxReadsPerMotif,
                    normalization,
                    theme
                );
            } else if (state.pythonBackend) {
                // Subprocess backend not yet implemented for motif aggregate all
                throw new Error(
                    'Motif aggregate plots are only available with Positron runtime. Please use Positron IDE.'
                );
            } else {
                throw new Error('No backend available');
            }
        },
        ErrorContext.MOTIF_PLOT,
        `Generating aggregate plot for all ${params.motif} matches (-${params.upstream}bp to +${params.downstream}bp)...`
    );
}

/**
 * Plot signal overlay comparison between multiple samples
 * Phase 1 - Default multi-sample comparison visualization
 */
async function plotSignalOverlayComparison(
    sampleNames: string[],
    state: ExtensionState,
    maxReads?: number | null
): Promise<void> {
    await safeExecuteWithProgress(
        async () => {
            if (!state.squiggyAPI) {
                throw new Error('SquiggyAPI not initialized');
            }

            // Get plot options
            const options = state.plotOptionsProvider?.getOptions();
            const normalization = options?.normalization || 'ZNORM';

            // Detect theme from VSCode settings
            const isDark = vscode.window.activeColorTheme.kind === vscode.ColorThemeKind.Dark;
            const theme = isDark ? 'DARK' : 'LIGHT';

            // Generate signal overlay plot
            if (state.usePositron && state.positronClient) {
                await state.squiggyAPI.generateSignalOverlayComparison(
                    sampleNames,
                    normalization,
                    theme,
                    maxReads
                );
            } else if (state.pythonBackend) {
                // Subprocess backend not yet implemented for overlay plots
                throw new Error(
                    'Signal overlay comparison plots are only available with Positron runtime. Please use Positron IDE.'
                );
            } else {
                throw new Error('No backend available');
            }
        },
        ErrorContext.PLOT_GENERATE,
        `Comparing samples: ${sampleNames.join(', ')}...`
    );
}

/**
 * Plot delta comparison between two or more samples
 * Phase 4 - Multi-sample comparison feature (optional, 2-sample only)
 */
async function plotDeltaComparison(
    sampleNames: string[],
    state: ExtensionState,
    maxReads?: number | null
): Promise<void> {
    await safeExecuteWithProgress(
        async () => {
            if (!state.squiggyAPI) {
                throw new Error('SquiggyAPI not initialized');
            }

            // Get plot options
            const options = state.plotOptionsProvider?.getOptions();
            const normalization = options?.normalization || 'ZNORM';

            // Detect theme from VSCode settings
            const isDark = vscode.window.activeColorTheme.kind === vscode.ColorThemeKind.Dark;
            const theme = isDark ? 'DARK' : 'LIGHT';

            // Generate delta plot
            if (state.usePositron && state.positronClient) {
                await state.squiggyAPI.generateDeltaPlot(
                    sampleNames,
                    normalization,
                    theme,
                    maxReads
                );
            } else if (state.pythonBackend) {
                // Subprocess backend not yet implemented for delta plots
                throw new Error(
                    'Delta comparison plots are only available with Positron runtime. Please use Positron IDE.'
                );
            } else {
                throw new Error('No backend available');
            }
        },
        ErrorContext.PLOT_GENERATE,
        `Comparing samples: ${sampleNames.join(', ')}...`
    );
}
