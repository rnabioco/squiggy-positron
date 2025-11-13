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
import { SquiggySetupPanelProvider } from './views/squiggy-setup-panel';
import { registerFileCommands } from './commands/file-commands';
import { registerPlotCommands } from './commands/plot-commands';
import { registerStateCommands } from './commands/state-commands';
import { registerSessionCommands } from './commands/session-commands';
import { registerKernelListeners } from './listeners/kernel-listeners';
import { logger } from './utils/logger';
import { SquiggyKernelState } from './backend/squiggy-kernel-manager';

// Global extension state
const state = new ExtensionState();

// Track if we're in setup mode (squiggy not installed)
let isSetupMode = false;

// Status bar item for dedicated kernel
let kernelStatusBarItem: vscode.StatusBarItem;

/**
 * Extension activation
 */
export async function activate(context: vscode.ExtensionContext) {
    // Initialize centralized logger (creates Output Channel)
    logger.initialize(context);

    // Initialize backends (Positron or subprocess fallback)
    await state.initializeBackends(context);

    // Initialize dedicated kernel status bar (Positron only)
    if (state.usePositron && state.kernelManager) {
        initializeKernelStatusBar(context);
    }

    // Register all panels and commands immediately
    // (The UI will be controlled by the squiggy.setupMode context key)
    await registerAllPanelsAndCommands(context);

    // Check installation status after kernel is ready
    if (state.usePositron && state.packageManager) {
        // Wait for kernel to be ready, then check installation
        waitForKernelAndCheckInstallation(context).catch((error) => {
            logger.error(`Failed to check installation: ${error}`);
            // On error, default to setup mode
            vscode.commands.executeCommand('setContext', 'squiggy.setupMode', true);
            isSetupMode = true;
        });
    } else {
        // Not using Positron - activate in normal mode immediately
        await vscode.commands.executeCommand('setContext', 'squiggy.setupMode', false);
        isSetupMode = false;
        logger.info('Normal mode activated (not using Positron)');
    }
}

/**
 * Wait for Python kernel to be ready, then check squiggy installation
 */
async function waitForKernelAndCheckInstallation(_context: vscode.ExtensionContext): Promise<void> {
    logger.info('Waiting for Python kernel to be ready before checking installation...');

    try {
        // eslint-disable-next-line @typescript-eslint/no-var-requires, @typescript-eslint/no-require-imports
        const positron = require('positron');

        // Wait for a foreground session to be available
        let session = await positron.runtime.getForegroundSession();
        let attempts = 0;
        const maxAttempts = 30; // 30 seconds max wait

        while (!session && attempts < maxAttempts) {
            await new Promise((resolve) => setTimeout(resolve, 1000));
            session = await positron.runtime.getForegroundSession();
            attempts++;
        }

        if (!session) {
            logger.warning(
                'No Python kernel session found after 30 seconds - activating in setup mode'
            );
            await vscode.commands.executeCommand('setContext', 'squiggy.setupMode', true);
            isSetupMode = true;
            return;
        }

        logger.info(`Python kernel session found: ${session.metadata.sessionId}`);

        // Check if squiggy is installed
        if (!state.packageManager) {
            logger.error('Package manager not available');
            return;
        }

        const isInstalled = await state.packageManager.isSquiggyInstalled();

        if (isInstalled) {
            logger.info('Squiggy package detected - activating in normal mode');
            await vscode.commands.executeCommand('setContext', 'squiggy.setupMode', false);
            isSetupMode = false;
        } else {
            logger.info('Squiggy package not found - activating in setup mode');
            await vscode.commands.executeCommand('setContext', 'squiggy.setupMode', true);
            isSetupMode = true;
        }
    } catch (error) {
        logger.error(`Error checking installation: ${error}`);
        throw error;
    }
}

/**
 * Register all panels and commands
 * Panels are shown/hidden based on the squiggy.setupMode context key
 */
async function registerAllPanelsAndCommands(context: vscode.ExtensionContext): Promise<void> {
    logger.info('Registering UI panels and commands...');

    // Set initial context to setup mode (will be updated after kernel check)
    await vscode.commands.executeCommand('setContext', 'squiggy.setupMode', true);

    // Create setup panel
    const setupPanelProvider = new SquiggySetupPanelProvider(context.extensionUri, state, context);

    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider(
            SquiggySetupPanelProvider.viewType,
            setupPanelProvider
        )
    );

    // Create normal mode panel providers
    const sessionPanelProvider = new SessionPanelProvider(context.extensionUri, context, state);
    const readsViewPane = new ReadsViewPane(context.extensionUri, state);
    const plotOptionsProvider = new PlotOptionsViewProvider(context.extensionUri, state);
    const modificationsProvider = new ModificationsPanelProvider(context.extensionUri);
    const motifSearchProvider = new MotifSearchPanelProvider(context.extensionUri, state);
    const samplesProvider = new SamplesPanelProvider(context.extensionUri, state);

    // Initialize state with panel references (needed for session serialization)
    state.initializePanels(
        readsViewPane,
        plotOptionsProvider,
        modificationsProvider,
        samplesProvider
    );

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

    // Set initial context for modifications panel (hidden by default)
    vscode.commands.executeCommand('setContext', 'squiggy.hasModifications', false);

    // Check if squiggy package is installed and set context for command enablement
    if (state.packageManager) {
        const isInstalled = await state.packageManager.isSquiggyInstalled();
        await vscode.commands.executeCommand('setContext', 'squiggy.packageInstalled', isInstalled);
        // Setup panel now handles all installation guidance - no popup needed
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

            // Update POD5/BAM/FASTA status in plot options pane based on loaded samples
            const hasPod5 = samples.length > 0; // Any samples = POD5 is loaded
            const hasBam = samples.some((s) => s.hasBam); // Any sample with BAM
            const hasFasta = samples.some((s) => s.hasFasta); // Any sample with FASTA

            logger.debug(
                '[extension.ts] Setting hasPod5:',
                hasPod5,
                'hasBam:',
                hasBam,
                'hasFasta:',
                hasFasta
            );

            if (plotOptionsProvider) {
                plotOptionsProvider.updatePod5Status(hasPod5);
                plotOptionsProvider.updateBamStatus(hasBam);
                plotOptionsProvider.updateFastaStatus(hasFasta);
            }

            // If we have BAM files, fetch and aggregate references from all samples
            if (hasBam && state.squiggyAPI && plotOptionsProvider) {
                // Get all samples with BAM files
                const samplesWithBam = samples.filter((s) => s.hasBam);
                if (samplesWithBam.length > 0) {
                    logger.debug(
                        '[extension.ts] Fetching references from',
                        samplesWithBam.length,
                        'samples with BAM:',
                        samplesWithBam.map((s) => s.name).join(', ')
                    );

                    // Fetch references from all samples in parallel
                    Promise.all(
                        samplesWithBam.map((sample) =>
                            state.squiggyAPI!.getReferencesForSample(sample.name)
                        )
                    ).then((allRefs) => {
                        // Aggregate unique references from all samples
                        const uniqueRefs = Array.from(
                            new Set(
                                allRefs.flat().filter((ref) => ref !== null && ref !== undefined)
                            )
                        );

                        logger.debug(
                            '[extension.ts] Aggregated',
                            uniqueRefs.length,
                            'unique references from all samples:',
                            uniqueRefs
                        );

                        if (plotOptionsProvider) {
                            plotOptionsProvider.updateReferences(uniqueRefs);
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
                const errorMessage = error instanceof Error ? error.message : String(error);
                vscode.window.showErrorMessage(
                    `Failed to generate aggregate plot: ${errorMessage}`
                );
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
    // Pass installation checker to re-check when kernel changes
    registerKernelListeners(context, state, async () => {
        // Re-check installation when kernel session changes
        if (state.packageManager) {
            logger.info('Kernel session changed - rechecking squiggy installation...');
            const isInstalled = await state.packageManager.isSquiggyInstalled();

            if (isInstalled && isSetupMode) {
                // Switched to interpreter with squiggy - activate normal mode
                logger.info('New kernel has squiggy installed - switching to normal mode');
                await vscode.commands.executeCommand('setContext', 'squiggy.setupMode', false);
                isSetupMode = false;
                vscode.window.showInformationMessage(
                    'Squiggy package detected in new Python environment - extension activated'
                );
            } else if (!isInstalled && !isSetupMode) {
                // Switched to interpreter without squiggy - activate setup mode
                logger.warning(
                    'New kernel does not have squiggy installed - switching to setup mode'
                );
                await vscode.commands.executeCommand('setContext', 'squiggy.setupMode', true);
                isSetupMode = true;
                vscode.window.showWarningMessage(
                    'Squiggy package not found in new Python environment - please install it'
                );
            }
        }
    });

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

    // Register command to restart dedicated kernel
    context.subscriptions.push(
        vscode.commands.registerCommand('squiggy.restartBackgroundKernel', async () => {
            if (!state.kernelManager) {
                vscode.window.showWarningMessage(
                    'Dedicated kernel not available (not in Positron mode)'
                );
                return;
            }

            try {
                const currentState = state.kernelManager.getState();

                if (currentState === SquiggyKernelState.Uninitialized) {
                    // Start for the first time
                    await vscode.window.withProgress(
                        {
                            location: vscode.ProgressLocation.Notification,
                            title: 'Starting Squiggy dedicated kernel...',
                            cancellable: false,
                        },
                        async () => {
                            await state.kernelManager!.start();
                        }
                    );
                    vscode.window.showInformationMessage('Squiggy dedicated kernel started');
                } else {
                    // Restart existing kernel
                    await vscode.window.withProgress(
                        {
                            location: vscode.ProgressLocation.Notification,
                            title: 'Restarting Squiggy dedicated kernel...',
                            cancellable: false,
                        },
                        async () => {
                            await state.kernelManager!.restart();
                        }
                    );
                    vscode.window.showInformationMessage('Squiggy dedicated kernel restarted');
                }
            } catch (error) {
                const errorMessage = error instanceof Error ? error.message : String(error);
                vscode.window.showErrorMessage(
                    `Failed to restart dedicated kernel: ${errorMessage}`
                );
                logger.error(`Dedicated kernel restart failed: ${errorMessage}`);
            }
        })
    );

    // Register command to check installation
    context.subscriptions.push(
        vscode.commands.registerCommand('squiggy.checkInstallation', async () => {
            await checkInstallationAndReload(context);
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

    logger.info('All panels and commands registered');
}

/**
 * Check if squiggy is installed and update UI accordingly
 * No reload required - UI automatically updates based on context key
 */
async function checkInstallationAndReload(_context: vscode.ExtensionContext): Promise<void> {
    if (!state.packageManager) {
        vscode.window.showErrorMessage('Package manager not available');
        return;
    }

    logger.info('Checking squiggy installation status...');
    const isInstalled = await state.packageManager.isSquiggyInstalled();

    if (isSetupMode && isInstalled) {
        // Was in setup mode, now installed - switch to normal mode
        logger.info('Squiggy now installed - switching to normal mode');
        await vscode.commands.executeCommand('setContext', 'squiggy.setupMode', false);
        isSetupMode = false;
        const version = await state.packageManager.getSquiggyVersion();
        vscode.window.showInformationMessage(
            `Squiggy package detected (version ${version || 'unknown'})! Extension is now active.`
        );
    } else if (!isSetupMode && !isInstalled) {
        // Was in normal mode, now uninstalled - switch to setup mode
        logger.warning('Squiggy no longer installed - switching to setup mode');
        await vscode.commands.executeCommand('setContext', 'squiggy.setupMode', true);
        isSetupMode = true;
        vscode.window.showWarningMessage(
            'Squiggy package is no longer installed. Please reinstall it using the instructions in the sidebar.'
        );
    } else if (isSetupMode && !isInstalled) {
        // Still not installed
        logger.info('Squiggy still not installed');
        vscode.window.showWarningMessage(
            'Squiggy package is still not installed. Please install it using the instructions in the sidebar.'
        );
    } else {
        // Already installed and in normal mode
        const version = await state.packageManager.getSquiggyVersion();
        logger.info(`Squiggy is installed (version ${version || 'unknown'})`);
        vscode.window.showInformationMessage(
            `Squiggy package is installed (version ${version || 'unknown'})`
        );
    }
}

/**
 * Initialize dedicated kernel status bar
 */
function initializeKernelStatusBar(context: vscode.ExtensionContext): void {
    if (!state.kernelManager) {
        return;
    }

    // Create status bar item
    kernelStatusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
    kernelStatusBarItem.command = 'squiggy.restartBackgroundKernel';
    context.subscriptions.push(kernelStatusBarItem);

    // Update status bar based on current state
    updateKernelStatusBar(state.kernelManager.getState());

    // Listen for state changes
    context.subscriptions.push(
        state.kernelManager.onDidChangeState((newState) => {
            updateKernelStatusBar(newState);
        })
    );

    // Show status bar
    kernelStatusBarItem.show();

    logger.info('Kernel status bar initialized');
}

/**
 * Update kernel status bar based on state
 */
function updateKernelStatusBar(kernelState: SquiggyKernelState): void {
    if (!kernelStatusBarItem) {
        return;
    }

    logger.info(`Updating kernel status bar: ${kernelState}`);

    switch (kernelState) {
        case SquiggyKernelState.Uninitialized:
            kernelStatusBarItem.text = '$(circle-outline) Squiggy Kernel';
            kernelStatusBarItem.tooltip = 'Squiggy dedicated kernel not started (click to start)';
            kernelStatusBarItem.backgroundColor = undefined;
            break;
        case SquiggyKernelState.Starting:
            kernelStatusBarItem.text = '$(sync~spin) Squiggy Kernel';
            kernelStatusBarItem.tooltip = 'Starting Squiggy dedicated kernel...';
            kernelStatusBarItem.backgroundColor = undefined;
            break;
        case SquiggyKernelState.Ready:
            kernelStatusBarItem.text = '$(check) Squiggy Kernel';
            kernelStatusBarItem.tooltip = 'Squiggy dedicated kernel ready (click to restart)';
            kernelStatusBarItem.backgroundColor = undefined;
            break;
        case SquiggyKernelState.Restarting:
            kernelStatusBarItem.text = '$(sync~spin) Squiggy Kernel';
            kernelStatusBarItem.tooltip = 'Restarting Squiggy dedicated kernel...';
            kernelStatusBarItem.backgroundColor = undefined;
            break;
        case SquiggyKernelState.Error:
            kernelStatusBarItem.text = '$(error) Squiggy Kernel';
            kernelStatusBarItem.tooltip = 'Squiggy dedicated kernel error (click to restart)';
            kernelStatusBarItem.backgroundColor = new vscode.ThemeColor(
                'statusBarItem.errorBackground'
            );
            break;
    }
}

/**
 * Extension deactivation
 */
export function deactivate() {
    if (state.pythonBackend) {
        state.pythonBackend.stop();
    }

    if (kernelStatusBarItem) {
        kernelStatusBarItem.dispose();
    }
}
