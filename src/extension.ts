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

// Global extension state
const state = new ExtensionState();

/**
 * Extension activation
 */
export async function activate(context: vscode.ExtensionContext) {
    // Initialize backends (Positron or subprocess fallback)
    await state.initializeBackends(context);

    // Create and register UI panel providers
    const sessionPanelProvider = new SessionPanelProvider(context.extensionUri, context, state);
    const readsViewPane = new ReadsViewPane(context.extensionUri, state);
    const plotOptionsProvider = new PlotOptionsViewProvider(context.extensionUri);
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
            // Filter for samples and convert to SampleItem format
            const samples = items
                .filter((item) => item.type === 'sample')
                .map((item) => ({
                    name: item.sampleName || '',
                    pod5Path: item.pod5Path || '',
                    bamPath: item.bamPath,
                    fastaPath: item.fastaPath,
                    readCount: state.getSample(item.sampleName || '')?.readCount || 0,
                    hasBam: !!item.bamPath,
                    hasFasta: !!item.fastaPath,
                }));

            // Sync to plot options provider
            plotOptionsProvider.updateLoadedSamples(samples);
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

                // Generate aggregate plot
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
                    options.clipXAxisToAlignment
                );

                vscode.window.showInformationMessage(
                    `Generated aggregate plot for ${options.reference}`
                );
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

    // Register kernel event listeners (session changes, restarts)
    registerKernelListeners(context, state);

    // Register all commands
    registerFileCommands(context, state);
    registerPlotCommands(context, state);
    registerStateCommands(context, state);
    registerSessionCommands(context, state);

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
