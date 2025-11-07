/**
 * Squiggy Positron Extension
 *
 * Main entry point - handles activation, deactivation, and registration only.
 * All logic is delegated to focused modules for better maintainability.
 */

import * as vscode from 'vscode';
import { ExtensionState } from './state/extension-state';
import { ReadsViewPane } from './views/squiggy-reads-view-pane';
import { PlotOptionsViewProvider } from './views/squiggy-plot-options-view';
import { ModificationsPanelProvider } from './views/squiggy-modifications-panel';
import { MotifSearchPanelProvider } from './views/squiggy-motif-panel';
import { SamplesPanelProvider } from './views/squiggy-samples-panel';
import { SessionPanelProvider } from './views/squiggy-session-panel';
import { registerFileCommands } from './commands/file-commands';
import { registerPlotCommands } from './commands/plot-commands';
import { registerStateCommands } from './commands/state-commands';
import { registerSessionCommands } from './commands/session-commands';
import { registerKernelListeners } from './listeners/kernel-listeners';
import { logger } from './utils/logger';

// Global extension state
const state = new ExtensionState();

/**
 * Extension activation
 */
export async function activate(context: vscode.ExtensionContext) {
    // Initialize centralized logger (creates Output Channel)
    logger.initialize(context);

    // Initialize backends (Positron or subprocess fallback)
    await state.initializeBackends(context);

    // Create and register UI panel providers
    const sessionPanelProvider = new SessionPanelProvider(context.extensionUri, context, state);
    const readsViewPane = new ReadsViewPane(context.extensionUri, state);
    const plotOptionsProvider = new PlotOptionsViewProvider(context.extensionUri, state);
    const modificationsProvider = new ModificationsPanelProvider(context.extensionUri);
    const motifSearchProvider = new MotifSearchPanelProvider(context.extensionUri, state);
    const samplesProvider = new SamplesPanelProvider(context.extensionUri, state);

    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider(SamplesPanelProvider.viewType, samplesProvider),
        vscode.window.registerWebviewViewProvider(
            SessionPanelProvider.viewType,
            sessionPanelProvider
        ),
        vscode.window.registerWebviewViewProvider(ReadsViewPane.viewType, readsViewPane),
        vscode.window.registerWebviewViewProvider(
            PlotOptionsViewProvider.viewType,
            plotOptionsProvider
        ),
        vscode.window.registerWebviewViewProvider(
            ModificationsPanelProvider.viewType,
            modificationsProvider
        ),
        vscode.window.registerWebviewViewProvider(
            MotifSearchPanelProvider.viewType,
            motifSearchProvider
        )
    );

    // Initialize state with panel references
    state.initializePanels(
        readsViewPane,
        plotOptionsProvider,
        modificationsProvider,
        samplesProvider
    );

    // Set initial context for modifications panel (hidden by default)
    vscode.commands.executeCommand('setContext', 'squiggy.hasModifications', false);

    // Check if squiggy package is installed and set context for command enablement
    if (state.packageManager) {
        const isInstalled = await state.packageManager.isSquiggyInstalled();
        await vscode.commands.executeCommand('setContext', 'squiggy.packageInstalled', isInstalled);

        // Show helpful message if not installed
        if (!isInstalled) {
            const choice = await vscode.window.showWarningMessage(
                'Squiggy requires the squiggy-positron Python package. ' +
                    'Install it with: uv pip install squiggy-positron',
                'Copy Install Command',
                'Dismiss'
            );

            if (choice === 'Copy Install Command') {
                await vscode.env.clipboard.writeText('uv pip install squiggy-positron');
                vscode.window.showInformationMessage(
                    'Copied to clipboard. Paste in your terminal to install.'
                );
            }
        }
    } else {
        // No package manager available (non-Positron mode) - assume package is available
        await vscode.commands.executeCommand('setContext', 'squiggy.packageInstalled', true);
    }

    // Listen for plot option changes and refresh current plot
    context.subscriptions.push(
        plotOptionsProvider.onDidChangeOptions(() => {
            if (state.currentPlotReadIds && state.currentPlotReadIds.length > 0) {
                // Re-plot with new options
                vscode.commands.executeCommand('squiggy.plotRead', state.currentPlotReadIds[0]);
            }
        })
    );

    // Listen for modification filter changes and refresh current plot
    context.subscriptions.push(
        modificationsProvider.onDidChangeFilters(() => {
            if (state.currentPlotReadIds && state.currentPlotReadIds.length > 0) {
                // Re-plot with new modification filters
                vscode.commands.executeCommand('squiggy.plotRead', state.currentPlotReadIds[0]);
            }
        })
    );

    // Listen for theme changes and refresh current plot
    context.subscriptions.push(
        vscode.window.onDidChangeActiveColorTheme(() => {
            if (state.currentPlotReadIds && state.currentPlotReadIds.length > 0) {
                // Re-plot with new theme
                vscode.commands.executeCommand('squiggy.plotRead', state.currentPlotReadIds[0]);
            }
        })
    );

    // Listen for loaded items changes and sync samples to plot options pane
    context.subscriptions.push(
        state.onLoadedItemsChanged((items) => {
            logger.debug('[extension.ts] onLoadedItemsChanged fired, items:', items.length);

            // Filter for samples and convert to SampleItem format
            const samples = items
                .filter((item) => item.type === 'sample')
                .map((item) => ({
                    name: item.sampleName || '',
                    pod5Path: item.pod5Path || '',
                    bamPath: item.bamPath,
                    fastaPath: item.fastaPath,
                    readCount: item.readCount, // Use unified state directly (already populated)
                    hasBam: !!item.bamPath,
                    hasFasta: !!item.fastaPath,
                }));

            logger.debug('[extension.ts] Filtered samples:', samples.length, samples);

            // Sync to plot options provider
            logger.debug('[extension.ts] plotOptionsProvider exists?', !!plotOptionsProvider);
            if (plotOptionsProvider) {
                plotOptionsProvider.updateLoadedSamples(samples);
            } else {
                logger.error('[extension.ts] plotOptionsProvider is undefined!');
            }

            // Update POD5/BAM status in plot options pane based on loaded samples
            const hasPod5 = samples.length > 0; // Any samples = POD5 is loaded
            const hasBam = samples.some((s) => s.hasBam); // Any sample with BAM

            logger.debug('[extension.ts] Setting hasPod5:', hasPod5, 'hasBam:', hasBam);

            if (plotOptionsProvider) {
                plotOptionsProvider.updatePod5Status(hasPod5);
                plotOptionsProvider.updateBamStatus(hasBam);
            }

            // If we have BAM files, fetch and update references
            if (hasBam && state.squiggyAPI && plotOptionsProvider) {
                // Get references from the first sample with BAM
                const sampleWithBam = samples.find((s) => s.hasBam);
                if (sampleWithBam) {
                    logger.debug(
                        '[extension.ts] Fetching references for sample:',
                        sampleWithBam.name
                    );
                    state.squiggyAPI.getReferencesForSample(sampleWithBam.name).then((refs) => {
                        logger.debug('[extension.ts] Got references:', refs);
                        if (plotOptionsProvider) {
                            plotOptionsProvider.updateReferences(refs);
                        }
                    });
                }
            }

            // Refresh Read Explorer to update available samples dropdown
            const allSamples = state.getAllSampleNames();
            logger.debug('[extension.ts] Sample loaded. All samples:', allSamples);
            logger.debug('[extension.ts] Refreshing Read Explorer');
            readsViewPane?.refresh();
        })
    );

    // Listen for sample unload requests
    context.subscriptions.push(
        samplesProvider.onDidRequestUnload(async (sampleName) => {
            if (!state.squiggyAPI) {
                vscode.window.showErrorMessage('API not available');
                return;
            }

            try {
                // Call Python to remove sample
                await state.squiggyAPI.removeSample(sampleName);

                // Update extension state
                state.removeSample(sampleName);
                state.removeLoadedItem(`sample:${sampleName}`);

                // If this was the selected sample in Read Explorer, clear selection and reset view
                if (state.selectedReadExplorerSample === sampleName) {
                    state.selectedReadExplorerSample = null;
                    readsViewPane?.setReads([]);
                }

                // Refresh panels
                samplesProvider.refresh();
                readsViewPane?.refresh();

                vscode.window.showInformationMessage(`Sample '${sampleName}' unloaded`);
            } catch (error) {
                vscode.window.showErrorMessage(`Failed to unload sample: ${error}`);
            }
        })
    );

    // Listen for aggregate plot generation requests from plot options panel
    context.subscriptions.push(
        plotOptionsProvider.onDidRequestAggregatePlot(async (options) => {
            logger.debug('[Extension] Aggregate plot requested with samples:', options.sampleNames);

            if (!state.squiggyAPI) {
                vscode.window.showErrorMessage('API not available');
                return;
            }

            try {
                // Get current theme
                const isDarkTheme =
                    vscode.window.activeColorTheme.kind === vscode.ColorThemeKind.Dark;
                const theme = isDarkTheme ? 'DARK' : 'LIGHT';

                // Get modification filters from Modifications panel
                const modFilters = modificationsProvider.getFilters();

                // Route based on number of samples selected
                if (options.sampleNames.length === 1) {
                    // Single-sample aggregate plot
                    await state.squiggyAPI.generateAggregatePlot(
                        options.reference,
                        options.maxReads,
                        options.normalization,
                        theme,
                        options.showModifications,
                        modFilters.minProbability,
                        modFilters.enabledModTypes,
                        options.showPileup,
                        options.showDwellTime,
                        options.showSignal,
                        options.showQuality,
                        options.clipXAxisToAlignment,
                        options.transformCoordinates,
                        options.sampleNames[0]
                    );

                    vscode.window.showInformationMessage(
                        `Generated aggregate plot for ${options.reference}`
                    );
                } else if (options.sampleNames.length > 1) {
                    // Multi-sample aggregate plot
                    if (options.viewStyle === 'overlay') {
                        // Use aggregate comparison (overlays mean signals)
                        // Convert boolean flags to metrics array
                        const metrics: string[] = [];
                        if (options.showSignal) {
                            metrics.push('signal');
                        }
                        if (options.showDwellTime) {
                            metrics.push('dwell_time');
                        }
                        if (options.showQuality) {
                            metrics.push('quality');
                        }

                        await vscode.commands.executeCommand('squiggy.plotAggregateComparison', {
                            sampleNames: options.sampleNames,
                            reference: options.reference,
                            metrics: metrics,
                            maxReads: options.maxReads,
                        });

                        vscode.window.showInformationMessage(
                            `Generated aggregate comparison for ${options.sampleNames.length} samples`
                        );
                    } else {
                        // Multi-track mode (separate detailed tracks per sample)
                        vscode.window.showWarningMessage(
                            'Multi-track aggregate view not yet implemented. Use overlay mode for now.'
                        );
                    }
                } else {
                    vscode.window.showErrorMessage('No samples selected for aggregate plot');
                }
            } catch (error) {
                vscode.window.showErrorMessage(`Failed to generate aggregate plot: ${error}`);
            }
        })
    );

    // Listen for signal overlay comparison requests from plot options panel
    context.subscriptions.push(
        plotOptionsProvider.onDidRequestSignalOverlay(async (params) => {
            await vscode.commands.executeCommand(
                'squiggy.plotSignalOverlayComparison',
                params.sampleNames,
                params.maxReads
            );
        })
    );

    // Listen for signal delta comparison requests from plot options panel
    context.subscriptions.push(
        plotOptionsProvider.onDidRequestSignalDelta(async (params) => {
            await vscode.commands.executeCommand(
                'squiggy.plotDeltaComparison',
                params.sampleNames,
                params.reference,
                params.maxReads
            );
        })
    );

    // Listen for aggregate comparison requests from plot options panel
    context.subscriptions.push(
        plotOptionsProvider.onDidRequestAggregateComparison(async (params) => {
            await vscode.commands.executeCommand('squiggy.plotAggregateComparison', {
                sampleNames: params.sampleNames,
                reference: params.reference,
                metrics: params.metrics,
                maxReads: params.maxReads,
            });
        })
    );

    // Listen for multi-read overlay requests from plot options panel
    context.subscriptions.push(
        plotOptionsProvider.onDidRequestMultiReadOverlay(async (params) => {
            await vscode.commands.executeCommand(
                'squiggy.plotMultiReadOverlay',
                params.sampleNames,
                params.maxReads,
                params.coordinateSpace
            );
        })
    );

    // Listen for multi-read stacked requests from plot options panel
    context.subscriptions.push(
        plotOptionsProvider.onDidRequestMultiReadStacked(async (params) => {
            await vscode.commands.executeCommand(
                'squiggy.plotMultiReadStacked',
                params.sampleNames,
                params.maxReads,
                params.coordinateSpace
            );
        })
    );

    // Register kernel event listeners (session changes, restarts)
    registerKernelListeners(context, state);

    // Register all commands
    registerFileCommands(context, state);
    registerPlotCommands(context, state);
    registerStateCommands(context, state);
    registerSessionCommands(context, state);

    // Register command to show logs
    context.subscriptions.push(
        vscode.commands.registerCommand('squiggy.showLogs', () => {
            logger.show();
        })
    );

    // Register command to set log level
    context.subscriptions.push(
        vscode.commands.registerCommand('squiggy.setLogLevel', async () => {
            const currentLevel = logger.getMinLevel();
            const levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR'];
            const selected = await vscode.window.showQuickPick(levels, {
                placeHolder: `Select log level (current: ${currentLevel})`,
                title: 'Squiggy Log Level',
            });

            if (selected) {
                // Update both the logger and VS Code settings
                const config = vscode.workspace.getConfiguration('squiggy');
                await config.update('logLevel', selected, vscode.ConfigurationTarget.Global);
                vscode.window.showInformationMessage(
                    `Log level set to ${selected}. Logs will show ${selected} and above.`
                );
            }
        })
    );

    // Extension activated silently - no welcome message needed
}

/**
 * Extension deactivation
 */
export function deactivate() {
    if (state.pythonBackend) {
        state.pythonBackend.stop();
    }
}
