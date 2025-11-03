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
import { FilePanelProvider } from './views/squiggy-file-panel';
import { ModificationsPanelProvider } from './views/squiggy-modifications-panel';
import { MotifSearchPanelProvider } from './views/squiggy-motif-panel';
import { registerFileCommands } from './commands/file-commands';
import { registerPlotCommands } from './commands/plot-commands';
import { registerStateCommands } from './commands/state-commands';
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
    const filePanelProvider = new FilePanelProvider(context.extensionUri);
    const readsViewPane = new ReadsViewPane(context.extensionUri);
    const plotOptionsProvider = new PlotOptionsViewProvider(context.extensionUri);
    const modificationsProvider = new ModificationsPanelProvider(context.extensionUri);
    const motifSearchProvider = new MotifSearchPanelProvider(context.extensionUri, state);

    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider(FilePanelProvider.viewType, filePanelProvider),
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
        filePanelProvider,
        modificationsProvider
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

    // Register kernel event listeners (session changes, restarts)
    registerKernelListeners(context, state);

    // Register all commands
    registerFileCommands(context, state);
    registerPlotCommands(context, state);
    registerStateCommands(context, state);

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
