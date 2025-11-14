/**
 * Plot Commands
 *
 * Handles generating plots for individual reads and aggregates.
 * Extracted from extension.ts to improve modularity.
 */

import * as vscode from 'vscode';
import { ExtensionState, ReferenceInfo } from '../state/extension-state';
import { ReadItem } from '../types/squiggy-reads-types';
import { ErrorContext, safeExecuteWithProgress } from '../utils/error-handler';
import { logger } from '../utils/logger';

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
            async (readIdOrItem?: string | ReadItem, coordinateSpace?: 'signal' | 'sequence') => {
                // Validate that at least one sample is loaded (for multi-sample mode)
                const hasSamples = state.getAllSampleNames().length > 0;
                const hasLegacyPod5 = !!state.currentPod5File;

                if (!hasSamples && !hasLegacyPod5) {
                    vscode.window.showErrorMessage(
                        'No POD5 file loaded. Use "Load Sample(s)" in the Samples panel.'
                    );
                    return;
                }

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

                await plotReads(readIds, state, coordinateSpace);
            }
        )
    );

    // Plot aggregate for reference
    context.subscriptions.push(
        vscode.commands.registerCommand('squiggy.plotAggregate', async (referenceName?: string) => {
            // Validate that at least one sample is loaded (for multi-sample mode)
            const hasSamples = state.getAllSampleNames().length > 0;
            const hasLegacyPod5 = !!state.currentPod5File;

            if (!hasSamples && !hasLegacyPod5) {
                vscode.window.showErrorMessage(
                    'No POD5 file loaded. Use "Load Sample(s)" in the Samples panel.'
                );
                return;
            }

            // For multi-sample mode: check if selected sample has BAM
            if (hasSamples) {
                const selectedSample = state.selectedReadExplorerSample;
                if (selectedSample) {
                    const sample = state.getSample(selectedSample);
                    if (sample && !sample.hasBam) {
                        vscode.window.showErrorMessage(
                            `Sample "${selectedSample}" does not have BAM alignment data. Aggregate plots require BAM files.`
                        );
                        return;
                    }
                }
            } else if (!state.currentBamFile) {
                // Legacy mode: check currentBamFile
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
            async (sampleNames?: string[], referenceName?: string, maxReads?: number | null) => {
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

                // If no reference provided, use placeholder (Python backend will handle default)
                if (!referenceName) {
                    vscode.window.showWarningMessage(
                        'No reference name provided for delta plot. Please select a reference in Plot Options.'
                    );
                    return;
                }

                await plotDeltaComparison(sampleNames, referenceName, state, maxReads);
            }
        )
    );

    // Plot aggregate comparison - Multi-sample aggregate statistics
    context.subscriptions.push(
        vscode.commands.registerCommand(
            'squiggy.plotAggregateComparison',
            async (params?: {
                sampleNames: string[];
                reference: string;
                metrics: string[];
                maxReads?: number;
            }) => {
                // If no params provided, validate and prompt user
                if (!params) {
                    const loadedSamples = state.getAllSampleNames();

                    if (loadedSamples.length < 2) {
                        vscode.window.showErrorMessage(
                            'Aggregate comparison requires at least 2 loaded samples. ' +
                                'Use "Load Sample" to add samples for comparison.'
                        );
                        return;
                    }

                    // Check that samples have BAM files
                    const samplesWithBam = loadedSamples.filter((name) => {
                        const sample = state.getSample(name);
                        return sample && sample.hasBam;
                    });

                    if (samplesWithBam.length < 2) {
                        vscode.window.showErrorMessage(
                            'Aggregate comparison requires at least 2 samples with BAM files. ' +
                                'Load BAM files for your samples.'
                        );
                        return;
                    }

                    vscode.window.showWarningMessage(
                        'Please use the Advanced Plotting pane to configure and generate aggregate comparison plots.'
                    );
                    return;
                }

                await plotAggregateComparison(params, state);
            }
        )
    );

    // Plot multi-read overlay - Overlay reads from multiple samples
    context.subscriptions.push(
        vscode.commands.registerCommand(
            'squiggy.plotMultiReadOverlay',
            async (
                sampleNames?: string[],
                maxReads?: number,
                coordinateSpace?: 'signal' | 'sequence'
            ) => {
                // If no params provided, show error
                if (!sampleNames || sampleNames.length === 0) {
                    vscode.window.showErrorMessage(
                        'Please select samples in the Plotting panel to generate multi-read overlay plots.'
                    );
                    return;
                }

                // Default to 10 reads per sample if not specified
                const maxReadsPerSample = maxReads || 10;
                const coordSpace = coordinateSpace || 'signal';

                await plotMultiReadOverlay(sampleNames, maxReadsPerSample, coordSpace, state);
            }
        )
    );

    // Plot multi-read stacked - Stack reads from multiple samples
    context.subscriptions.push(
        vscode.commands.registerCommand(
            'squiggy.plotMultiReadStacked',
            async (
                sampleNames?: string[],
                maxReads?: number,
                coordinateSpace?: 'signal' | 'sequence'
            ) => {
                // If no params provided, show error
                if (!sampleNames || sampleNames.length === 0) {
                    vscode.window.showErrorMessage(
                        'Please select samples in the Plotting panel to generate multi-read stacked plots.'
                    );
                    return;
                }

                // Default to 5 reads per sample if not specified (stacked needs fewer)
                const maxReadsPerSample = maxReads || 5;
                const coordSpace = coordinateSpace || 'signal';

                await plotMultiReadStacked(sampleNames, maxReadsPerSample, coordSpace, state);
            }
        )
    );
}

/**
 * Plot reads
 */
async function plotReads(
    readIds: string[],
    state: ExtensionState,
    coordinateSpace?: 'signal' | 'sequence'
): Promise<void> {
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

            logger.info(
                `Generating ${mode} plot for ${readIds.length} read${readIds.length !== 1 ? 's' : ''} (normalization: ${normalization}, coordinate space: ${coordinateSpace || 'default'})`
            );

            // Get modification filters
            const modFilters = state.modificationsProvider?.getFilters();
            if (!modFilters) {
                throw new Error('Modification filters not available');
            }

            // Detect VS Code theme
            const colorThemeKind = vscode.window.activeColorTheme.kind;
            const theme = colorThemeKind === vscode.ColorThemeKind.Dark ? 'DARK' : 'LIGHT';

            if (state.usePositron) {
                // Use dedicated kernel - plot appears in Plots pane automatically
                const api = await state.ensureBackgroundKernel();
                await api.generatePlot(
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
                    state.selectedReadExplorerSample || undefined, // Pass current sample for multi-sample mode
                    coordinateSpace
                );
                logger.info('Plot generated successfully (dedicated kernel)');
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

            if (state.usePositron) {
                // Use dedicated kernel - plot appears in Plots pane automatically
                const api = await state.ensureBackgroundKernel();
                await api.generateAggregatePlot(
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
                    true, // transformCoordinates
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

            if (state.usePositron) {
                // Use dedicated kernel - plot appears in Plots pane automatically
                const api = await state.ensureBackgroundKernel();
                await api.generateMotifAggregateAllPlot(
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
            // Get plot options
            const options = state.plotOptionsProvider?.getOptions();
            const normalization = options?.normalization || 'ZNORM';

            // Detect theme from VSCode settings
            const isDark = vscode.window.activeColorTheme.kind === vscode.ColorThemeKind.Dark;
            const theme = isDark ? 'DARK' : 'LIGHT';

            // Generate signal overlay plot
            if (state.usePositron && state.positronClient) {
                const api = await state.ensureBackgroundKernel();
                await api.generateSignalOverlayComparison(
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
    referenceName: string,
    state: ExtensionState,
    maxReads?: number | null
): Promise<void> {
    await safeExecuteWithProgress(
        async () => {
            // Get plot options
            const options = state.plotOptionsProvider?.getOptions();
            const normalization = options?.normalization || 'ZNORM';

            // Detect theme from VSCode settings
            const isDark = vscode.window.activeColorTheme.kind === vscode.ColorThemeKind.Dark;
            const theme = isDark ? 'DARK' : 'LIGHT';

            // Generate delta plot
            if (state.usePositron && state.positronClient) {
                const api = await state.ensureBackgroundKernel();
                await api.generateDeltaPlot(
                    sampleNames,
                    referenceName,
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
        `Comparing samples: ${sampleNames.join(', ')} on reference ${referenceName}...`
    );
}

/**
 * Check which samples have the specified reference
 */
async function checkSampleReferenceCompatibility(
    sampleNames: string[],
    referenceName: string,
    state: ExtensionState
): Promise<{ compatible: string[]; incompatible: Array<{ name: string; references: string[] }> }> {
    logger.info(
        `[Reference Check] Starting compatibility check for reference '${referenceName}' across ${sampleNames.length} samples: ${sampleNames.join(', ')}`
    );

    const compatible: string[] = [];
    const incompatible: Array<{ name: string; references: string[] }> = [];

    for (const sampleName of sampleNames) {
        const sample = state.getSample(sampleName);
        if (!sample) {
            logger.warning(
                `[Reference Check] Sample '${sampleName}' not found in state - skipping`
            );
            continue;
        }

        logger.debug(
            `[Reference Check] Sample '${sampleName}': hasBam=${sample.hasBam}, references=${sample.references ? JSON.stringify(sample.references) : 'undefined'}`
        );

        // If reference info is missing, try to fetch it on-demand
        if (!sample.references || sample.references.length === 0) {
            if (sample.hasBam && state.usePositron) {
                logger.info(
                    `[Reference Check] Sample '${sampleName}' has no cached references - fetching on-demand from Python...`
                );
                try {
                    const api = await state.ensureBackgroundKernel();
                    const sampleInfo = await api.getSampleInfo(sampleName);
                    if (sampleInfo && sampleInfo.references) {
                        // Update the sample with fetched reference info
                        sample.references = sampleInfo.references;
                        logger.info(
                            `[Reference Check] Fetched ${sampleInfo.references.length} references for '${sampleName}': ${sampleInfo.references.map((r: ReferenceInfo) => r.name).join(', ')}`
                        );
                    } else {
                        logger.warning(
                            `[Reference Check] getSampleInfo returned no references for '${sampleName}'`
                        );
                    }
                } catch (error) {
                    logger.error(
                        `[Reference Check] Failed to fetch reference info for sample '${sampleName}':`,
                        error
                    );
                }
            } else {
                logger.debug(
                    `[Reference Check] Sample '${sampleName}': hasBam=${sample.hasBam}, usePositron=${state.usePositron} - cannot fetch references`
                );
            }
        }

        // If still no reference info, assume compatible (will fail in Python if not)
        if (!sample.references || sample.references.length === 0) {
            logger.warning(
                `[Reference Check] Sample '${sampleName}' has no reference info even after fetch - assuming compatible (may fail later)`
            );
            compatible.push(sampleName);
            continue;
        }

        const hasReference = sample.references.some((ref) => ref.name === referenceName);
        if (hasReference) {
            logger.info(
                `[Reference Check] Sample '${sampleName}' HAS reference '${referenceName}' - COMPATIBLE`
            );
            compatible.push(sampleName);
        } else {
            logger.warning(
                `[Reference Check] Sample '${sampleName}' MISSING reference '${referenceName}' - INCOMPATIBLE (has: ${sample.references.map((r: ReferenceInfo) => r.name).join(', ')})`
            );
            incompatible.push({
                name: sampleName,
                references: sample.references.map((r: ReferenceInfo) => r.name),
            });
        }
    }

    logger.info(
        `[Reference Check] Result: ${compatible.length} compatible, ${incompatible.length} incompatible`
    );
    if (incompatible.length > 0) {
        logger.info(
            `[Reference Check] Incompatible samples: ${incompatible.map((s) => s.name).join(', ')}`
        );
    }

    return { compatible, incompatible };
}

/**
 * Show dialog when reference mismatch is detected
 * Informs user which samples don't have the selected reference
 */
async function showReferenceCompatibilityDialog(
    referenceName: string,
    compatibleSamples: string[],
    incompatibleSamples: Array<{ name: string; references: string[] }>
): Promise<void> {
    // Build detailed message
    const lines: string[] = [];
    lines.push(`⚠️ Reference Mismatch`);
    lines.push('');
    lines.push(
        `The selected reference '${referenceName}' is not available in the currently selected samples.`
    );
    lines.push('');
    lines.push('To fix this, either:');
    lines.push(`  1. Change the Reference dropdown to a reference that all samples have, OR`);
    lines.push(`  2. Un-check samples that don't have '${referenceName}'`);
    lines.push('');

    if (compatibleSamples.length > 0) {
        lines.push(`✓ Samples with '${referenceName}' (${compatibleSamples.length}):`);
        compatibleSamples.forEach((name) => {
            lines.push(`  • ${name}`);
        });
        lines.push('');
    }

    lines.push(`✗ Samples without '${referenceName}' (${incompatibleSamples.length}):`);
    incompatibleSamples.forEach((sample) => {
        const refList = sample.references.slice(0, 3).join(', ');
        const more =
            sample.references.length > 3 ? `, ... (${sample.references.length} total)` : '';
        lines.push(`  • ${sample.name} (has: ${refList}${more})`);
    });

    const message = lines.join('\n');

    // Show informational dialog - user must fix manually
    await vscode.window.showErrorMessage(message, { modal: true }, 'OK');
}

/**
 * Plot aggregate comparison across multiple samples
 * Compares aggregate statistics (signal, dwell time, quality) across samples
 */
async function plotAggregateComparison(
    params: {
        sampleNames: string[];
        reference: string;
        metrics: string[];
        maxReads?: number;
    },
    state: ExtensionState
): Promise<void> {
    await safeExecuteWithProgress(
        async () => {
            if (!state.squiggyAPI) {
                throw new Error('SquiggyAPI not initialized');
            }

            // Validate params
            if (params.sampleNames.length < 2) {
                throw new Error('Aggregate comparison requires at least 2 samples');
            }

            if (!params.reference) {
                throw new Error('Reference name is required for aggregate comparison');
            }

            if (!params.metrics || params.metrics.length === 0) {
                throw new Error('At least one metric must be selected for comparison');
            }

            // Check sample-reference compatibility (fetch reference info if missing)
            const { compatible, incompatible } = await checkSampleReferenceCompatibility(
                params.sampleNames,
                params.reference,
                state
            );

            // If there are incompatible samples, show dialog and abort
            if (incompatible.length > 0) {
                logger.info(
                    `[Reference Check] Found ${incompatible.length} incompatible samples - showing dialog to user`
                );

                await showReferenceCompatibilityDialog(params.reference, compatible, incompatible);

                logger.info('[Reference Check] User must fix sample selection - aborting plot');
                return; // User must manually fix selection
            }

            logger.info('[Reference Check] All samples compatible - proceeding with plot');

            // Get plot options
            const options = state.plotOptionsProvider?.getOptions();
            const normalization = options?.normalization || 'ZNORM';

            // Detect theme from VSCode settings
            const isDark = vscode.window.activeColorTheme.kind === vscode.ColorThemeKind.Dark;
            const theme = isDark ? 'DARK' : 'LIGHT';

            // Get sample colors from Sample Manager
            const sampleColors: Record<string, string> = {};
            for (const sampleName of params.sampleNames) {
                const sample = state.getSample(sampleName);
                if (sample && sample.metadata?.displayColor) {
                    sampleColors[sampleName] = sample.metadata.displayColor;
                }
            }

            // Generate aggregate comparison plot
            if (state.usePositron && state.positronClient) {
                const api = await state.ensureBackgroundKernel();
                await api.generateAggregateComparison(
                    params.sampleNames,
                    params.reference,
                    params.metrics,
                    params.maxReads || null,
                    normalization,
                    theme,
                    Object.keys(sampleColors).length > 0 ? sampleColors : undefined
                );
            } else if (state.pythonBackend) {
                // Subprocess backend not yet implemented for aggregate comparison
                throw new Error(
                    'Aggregate comparison plots are only available with Positron runtime. Please use Positron IDE.'
                );
            } else {
                throw new Error('No backend available');
            }
        },
        ErrorContext.PLOT_GENERATE,
        `Comparing aggregate statistics for samples: ${params.sampleNames.join(', ')}...`
    );
}

/**
 * Generate multi-read overlay plot from selected samples
 * Extracts N reads from each sample and overlays them with sample-based coloring
 */
async function plotMultiReadOverlay(
    sampleNames: string[],
    maxReadsPerSample: number,
    coordinateSpace: 'signal' | 'sequence',
    state: ExtensionState
): Promise<void> {
    await safeExecuteWithProgress(
        async () => {
            if (!state.squiggyAPI) {
                throw new Error('SquiggyAPI not initialized');
            }

            // Validate params
            if (sampleNames.length === 0) {
                throw new Error('At least one sample must be selected');
            }

            // Get background API
            const api = await state.ensureBackgroundKernel();

            // Extract reads from each sample and build mappings
            const allReadIds: string[] = [];
            const readSampleMap: Record<string, string> = {};
            const readColors: Record<string, string> = {};

            for (const sampleName of sampleNames) {
                const sample = state.getSample(sampleName);
                if (!sample) {
                    throw new Error(`Sample '${sampleName}' not found`);
                }

                // Get sample color (default to a color if not set)
                const sampleColor = sample.metadata?.displayColor || '#888888';

                // Get read IDs for this sample
                const readIds = await api.getReadIdsForSample(
                    sampleName,
                    0,
                    maxReadsPerSample
                );

                // Add to combined list and build mappings
                for (const readId of readIds) {
                    allReadIds.push(readId);
                    readSampleMap[readId] = sampleName;
                    readColors[readId] = sampleColor;
                }
            }

            if (allReadIds.length === 0) {
                throw new Error('No reads found in selected samples');
            }

            // Get plot options
            const options = state.plotOptionsProvider?.getOptions();
            if (!options) {
                throw new Error('Plot options not available');
            }

            const normalization = options.normalization;

            // Get modification filters
            const modFilters = state.modificationsProvider?.getFilters();
            if (!modFilters) {
                throw new Error('Modification filters not available');
            }

            // Detect theme
            const isDark = vscode.window.activeColorTheme.kind === vscode.ColorThemeKind.Dark;
            const theme = isDark ? 'DARK' : 'LIGHT';

            // Generate plot with OVERLAY mode using multi-sample support
            await api.generateMultiSamplePlot(
                allReadIds,
                readSampleMap,
                readColors,
                'OVERLAY',
                normalization,
                theme,
                options.showDwellTime,
                options.showBaseAnnotations,
                options.scaleDwellTime,
                modFilters.minProbability,
                modFilters.enabledModTypes,
                options.downsample,
                options.showSignalPoints,
                coordinateSpace
            );
        },
        ErrorContext.PLOT_GENERATE,
        `Generating overlay plot with ${maxReadsPerSample} reads per sample from ${sampleNames.length} sample(s)...`
    );
}

/**
 * Generate multi-read stacked plot from selected samples
 * Extracts N reads from each sample and stacks them vertically with sample-based coloring
 */
async function plotMultiReadStacked(
    sampleNames: string[],
    maxReadsPerSample: number,
    coordinateSpace: 'signal' | 'sequence',
    state: ExtensionState
): Promise<void> {
    await safeExecuteWithProgress(
        async () => {
            // Validate params
            if (sampleNames.length === 0) {
                throw new Error('At least one sample must be selected');
            }

            // Get background API
            const api = await state.ensureBackgroundKernel();

            // Extract reads from each sample and build mappings
            const allReadIds: string[] = [];
            const readSampleMap: Record<string, string> = {};
            const readColors: Record<string, string> = {};

            for (const sampleName of sampleNames) {
                const sample = state.getSample(sampleName);
                if (!sample) {
                    throw new Error(`Sample '${sampleName}' not found`);
                }

                // Get sample color (default to a color if not set)
                const sampleColor = sample.metadata?.displayColor || '#888888';

                // Get read IDs for this sample
                const readIds = await api.getReadIdsForSample(
                    sampleName,
                    0,
                    maxReadsPerSample
                );

                // Add to combined list and build mappings
                for (const readId of readIds) {
                    allReadIds.push(readId);
                    readSampleMap[readId] = sampleName;
                    readColors[readId] = sampleColor;
                }
            }

            if (allReadIds.length === 0) {
                throw new Error('No reads found in selected samples');
            }

            // Warn if too many reads for stacking
            const totalReads = allReadIds.length;
            if (totalReads > 20) {
                const result = await vscode.window.showWarningMessage(
                    `You're about to stack ${totalReads} reads. This may be hard to read. Continue?`,
                    'Yes',
                    'No'
                );
                if (result !== 'Yes') {
                    return;
                }
            }

            // Get plot options
            const options = state.plotOptionsProvider?.getOptions();
            if (!options) {
                throw new Error('Plot options not available');
            }

            const normalization = options.normalization;

            // Get modification filters
            const modFilters = state.modificationsProvider?.getFilters();
            if (!modFilters) {
                throw new Error('Modification filters not available');
            }

            // Detect theme
            const isDark = vscode.window.activeColorTheme.kind === vscode.ColorThemeKind.Dark;
            const theme = isDark ? 'DARK' : 'LIGHT';

            // Generate plot with STACKED mode using multi-sample support
            await api.generateMultiSamplePlot(
                allReadIds,
                readSampleMap,
                readColors,
                'STACKED',
                normalization,
                theme,
                options.showDwellTime,
                options.showBaseAnnotations,
                options.scaleDwellTime,
                modFilters.minProbability,
                modFilters.enabledModTypes,
                options.downsample,
                options.showSignalPoints,
                coordinateSpace
            );
        },
        ErrorContext.PLOT_GENERATE,
        `Generating stacked plot with ${maxReadsPerSample} reads per sample from ${sampleNames.length} sample(s)...`
    );
}
