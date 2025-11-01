/**
 * Squiggy Positron Extension
 *
 * Main entry point for the extension. Handles activation, deactivation,
 * and registration of all commands, views, and providers.
 */

import * as vscode from 'vscode';
import { promises as fs } from 'fs';
import * as path from 'path';
import { PositronRuntime } from './backend/squiggy-positron-runtime';
import { PythonBackend } from './backend/squiggy-python-backend';
import { ReadTreeProvider, ReadItem } from './views/squiggy-read-explorer';
import { ReadSearchViewProvider } from './views/squiggy-read-search-view';
import { PlotOptionsViewProvider } from './views/squiggy-plot-options-view';
import { FilePanelProvider } from './views/squiggy-file-panel';
import { ModificationsPanelProvider } from './views/squiggy-modifications-panel';

let positronRuntime: PositronRuntime;
let pythonBackend: PythonBackend | null = null;
let readTreeProvider: ReadTreeProvider;
let readSearchProvider: ReadSearchViewProvider;
let plotOptionsProvider: PlotOptionsViewProvider;
let filePanelProvider: FilePanelProvider;
let modificationsProvider: ModificationsPanelProvider;
let usePositron = false;
let extensionContext: vscode.ExtensionContext;

// Track loaded files and current plot
let _currentPod5File: string | undefined;
let _currentBamFile: string | undefined;
let currentPlotReadIds: string[] | undefined;

/**
 * Extension activation
 */
export async function activate(context: vscode.ExtensionContext) {
    // Store context for helper functions
    extensionContext = context;

    // Try to use Positron runtime first
    positronRuntime = new PositronRuntime();
    usePositron = positronRuntime.isAvailable();

    if (usePositron) {
        // Installation check deferred until first use to avoid console clutter
    } else {
        // Fallback to subprocess JSON-RPC
        const pythonPath = getPythonPath();
        const serverPath = context.asAbsolutePath(path.join('src', 'python', 'server.py'));
        pythonBackend = new PythonBackend(pythonPath, serverPath);

        try {
            await pythonBackend.start();

            // Register cleanup on deactivation
            context.subscriptions.push({
                dispose: () => pythonBackend?.stop(),
            });
        } catch (error) {
            vscode.window.showErrorMessage(
                `Failed to start Python backend: ${error}. ` +
                    `Please ensure Python is installed and the squiggy package is available.`
            );
        }
    }

    // Create and register file panel provider
    filePanelProvider = new FilePanelProvider(context.extensionUri);
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider(FilePanelProvider.viewType, filePanelProvider)
    );

    // Create read tree provider
    readTreeProvider = new ReadTreeProvider();
    const readTreeView = vscode.window.createTreeView('squiggyReadList', {
        treeDataProvider: readTreeProvider,
        canSelectMany: true,
    });
    context.subscriptions.push(readTreeView);

    // Create and register read search provider
    readSearchProvider = new ReadSearchViewProvider(context.extensionUri);
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider(
            ReadSearchViewProvider.viewType,
            readSearchProvider
        )
    );

    // Connect search to tree provider
    context.subscriptions.push(
        readSearchProvider.onDidChangeSearchText((searchText) => {
            readTreeProvider.filterReads(searchText);
        })
    );

    // Create and register plot options provider
    plotOptionsProvider = new PlotOptionsViewProvider(context.extensionUri);
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider(
            PlotOptionsViewProvider.viewType,
            plotOptionsProvider
        )
    );

    // Listen for plot option changes and refresh current plot
    context.subscriptions.push(
        plotOptionsProvider.onDidChangeOptions(() => {
            if (currentPlotReadIds && currentPlotReadIds.length > 0) {
                // Re-plot with new options
                plotReads(currentPlotReadIds, context);
            }
        })
    );

    // Create and register modifications panel provider
    modificationsProvider = new ModificationsPanelProvider(context.extensionUri);
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider(
            ModificationsPanelProvider.viewType,
            modificationsProvider
        )
    );

    // Set initial context for modifications panel (hidden by default)
    vscode.commands.executeCommand('setContext', 'squiggy.hasModifications', false);

    // Listen for modification filter changes and refresh current plot
    context.subscriptions.push(
        modificationsProvider.onDidChangeFilters(() => {
            if (currentPlotReadIds && currentPlotReadIds.length > 0) {
                // Re-plot with new modification filters
                plotReads(currentPlotReadIds, context);
            }
        })
    );

    // Listen for theme changes and refresh current plot
    context.subscriptions.push(
        vscode.window.onDidChangeActiveColorTheme(() => {
            if (currentPlotReadIds && currentPlotReadIds.length > 0) {
                // Re-plot with new theme
                plotReads(currentPlotReadIds, context);
            }
        })
    );

    // Listen for Positron runtime session changes (kernel restarts)
    if (usePositron) {
        try {
            // eslint-disable-next-line @typescript-eslint/no-var-requires
            const positron = require('positron');

            // Helper function to clear extension state
            const clearExtensionState = (reason: string) => {
                _currentPod5File = undefined;
                _currentBamFile = undefined;
                currentPlotReadIds = undefined;

                // Reset installation check flags (new kernel won't have package installed)
                squiggyInstallChecked = false;
                squiggyInstallDeclined = false;

                // Clear UI panels
                readTreeProvider.setReads([]);
                filePanelProvider.setPOD5Info('', 0, '0 MB');
                filePanelProvider.setBAMInfo('', 0, '0 MB');
                plotOptionsProvider.updateBamStatus(false);

                console.log(`Squiggy: ${reason}, state cleared`);
            };

            // Listen for session changes (kernel switches)
            context.subscriptions.push(
                positron.runtime.onDidChangeForegroundSession((_sessionId: string | undefined) => {
                    clearExtensionState('Python session changed');
                })
            );

            // Also listen to runtime state changes on the current session
            // This catches kernel restarts within the same session
            const setupSessionListeners = async () => {
                try {
                    const session = await positron.runtime.getForegroundSession();
                    console.log(
                        'Squiggy: Setting up session listeners, session:',
                        session?.metadata.sessionId
                    );

                    if (session && session.onDidChangeRuntimeState) {
                        context.subscriptions.push(
                            session.onDidChangeRuntimeState((state: any) => {
                                console.log('Squiggy: Runtime state changed to:', state);
                                // Clear state when kernel is restarting or has exited
                                if (state === 'restarting' || state === 'exited') {
                                    clearExtensionState(`Kernel ${state}`);
                                }
                            })
                        );
                        console.log('Squiggy: Successfully attached runtime state listener');
                    } else {
                        console.log(
                            'Squiggy: No session or no onDidChangeRuntimeState event available'
                        );
                    }
                } catch (error) {
                    console.error('Squiggy: Error setting up session listeners:', error);
                }
            };
            setupSessionListeners();

            // Re-setup listeners when session changes
            context.subscriptions.push(
                positron.runtime.onDidChangeForegroundSession(async () => {
                    await setupSessionListeners();
                })
            );
        } catch (error) {
            // Positron API not available - not running in Positron
        }
    }

    // Register commands
    registerCommands(context, readTreeView);

    // Show welcome message
    const backendType = usePositron ? 'Positron kernel' : 'subprocess backend';
    vscode.window.showInformationMessage(
        `Squiggy extension loaded (using ${backendType})! Use "Open POD5 File" to get started.`
    );
}

/**
 * Extension deactivation
 */
export function deactivate() {
    if (pythonBackend) {
        pythonBackend.stop();
    }
}

/**
 * Register all extension commands
 */
function registerCommands(
    context: vscode.ExtensionContext,
    readTreeView: vscode.TreeView<ReadItem>
) {
    // Open POD5 file
    context.subscriptions.push(
        vscode.commands.registerCommand('squiggy.openPOD5', async () => {
            const fileUri = await vscode.window.showOpenDialog({
                canSelectMany: false,
                filters: { 'POD5 Files': ['pod5'] },
                title: 'Open POD5 File',
            });

            if (fileUri && fileUri[0]) {
                await openPOD5File(fileUri[0].fsPath);
            }
        })
    );

    // Open BAM file
    context.subscriptions.push(
        vscode.commands.registerCommand('squiggy.openBAM', async () => {
            const fileUri = await vscode.window.showOpenDialog({
                canSelectMany: false,
                filters: { 'BAM Files': ['bam'] },
                title: 'Open BAM File',
            });

            if (fileUri && fileUri[0]) {
                await openBAMFile(fileUri[0].fsPath);
            }
        })
    );

    // Load test data (yeast tRNA POD5 + BAM)
    context.subscriptions.push(
        vscode.commands.registerCommand('squiggy.loadTestData', async () => {
            const pod5Path = path.join(
                context.extensionPath,
                'tests',
                'data',
                'yeast_trna_reads.pod5'
            );
            const bamPath = path.join(
                context.extensionPath,
                'tests',
                'data',
                'yeast_trna_mappings.bam'
            );

            // Check if files are already loaded
            const pod5AlreadyLoaded = _currentPod5File === pod5Path;
            const bamAlreadyLoaded = _currentBamFile === bamPath;

            if (pod5AlreadyLoaded && bamAlreadyLoaded) {
                vscode.window.showInformationMessage(
                    'Test data is already loaded. Use "Refresh Read List" to update the view.'
                );
                return;
            }

            // Load files sequentially (only if not already loaded)
            if (!pod5AlreadyLoaded) {
                await openPOD5File(pod5Path);
            }
            if (!bamAlreadyLoaded) {
                await openBAMFile(bamPath);
            }

            vscode.window.showInformationMessage('Test data loaded successfully!');
        })
    );

    // Plot selected reads
    context.subscriptions.push(
        vscode.commands.registerCommand(
            'squiggy.plotRead',
            async (readIdOrItem?: string | ReadItem) => {
                let readIds: string[];

                // If called with a readId string (from TreeView click)
                if (typeof readIdOrItem === 'string') {
                    readIds = [readIdOrItem];
                }
                // If called with a ReadItem (from command palette or context menu)
                else if (readIdOrItem && 'readId' in readIdOrItem) {
                    readIds = [readIdOrItem.readId];
                }
                // Otherwise use current selection
                else {
                    const selection = readTreeView.selection;
                    if (selection.length === 0) {
                        vscode.window.showWarningMessage('Please select one or more reads to plot');
                        return;
                    }
                    readIds = selection.map((item) => item.readId);
                }

                await plotReads(readIds, context);
            }
        )
    );

    // Plot aggregate for reference
    context.subscriptions.push(
        vscode.commands.registerCommand(
            'squiggy.plotAggregate',
            async (referenceItem?: ReadItem) => {
                // Validate that both POD5 and BAM are loaded
                if (!_currentPod5File) {
                    vscode.window.showErrorMessage(
                        'No POD5 file loaded. Use "Open POD5 File" first.'
                    );
                    return;
                }
                if (!_currentBamFile) {
                    vscode.window.showErrorMessage(
                        'Aggregate plots require a BAM file. Use "Open BAM File" first.'
                    );
                    return;
                }

                // Extract reference name from the ReadItem
                let referenceName: string;
                if (referenceItem && referenceItem.itemType === 'reference') {
                    // Use readId (clean reference name) not label (which has count appended)
                    referenceName = referenceItem.readId;
                } else {
                    vscode.window.showErrorMessage('Please select a reference from the read list');
                    return;
                }

                await plotAggregate(referenceName, context);
            }
        )
    );

    // Refresh reads
    context.subscriptions.push(
        vscode.commands.registerCommand('squiggy.refreshReads', () => {
            readTreeProvider.refresh();
        })
    );

    // Clear state (useful after kernel restart)
    context.subscriptions.push(
        vscode.commands.registerCommand('squiggy.clearState', async () => {
            // Clear extension state
            _currentPod5File = undefined;
            _currentBamFile = undefined;
            currentPlotReadIds = undefined;

            // Reset installation check flags (allow re-prompting)
            squiggyInstallChecked = false;
            squiggyInstallDeclined = false;

            // Clear UI panels
            readTreeProvider.setReads([]);
            filePanelProvider.setPOD5Info('', 0, '0 MB');
            filePanelProvider.setBAMInfo('', 0, '0 MB');
            plotOptionsProvider.updateBamStatus(false);

            // Clear Python kernel state if using Positron
            if (usePositron) {
                try {
                    await positronRuntime.executeSilent(`
import squiggy
squiggy.close_pod5()
# Clear global variables
if '_squiggy_reader' in globals():
    del _squiggy_reader
if '_squiggy_read_ids' in globals():
    del _squiggy_read_ids
`);
                } catch (error) {
                    // Ignore errors if kernel is not running
                }
            }

            vscode.window.showInformationMessage(
                'Squiggy state cleared. Load new files to continue.'
            );
        })
    );
}

/**
 * State tracking for squiggy installation check
 */
let squiggyInstallChecked = false;
let squiggyInstallDeclined = false;

/**
 * Check if squiggy package is installed and prompt user to install if not
 * @returns true if squiggy is available, false otherwise
 */
async function ensureSquiggyAvailable(): Promise<boolean> {
    if (!usePositron) {
        // Non-Positron mode - assume squiggy is available via subprocess backend
        return true;
    }

    // Only check once per session
    if (squiggyInstallChecked) {
        // If user previously declined, don't prompt again
        if (squiggyInstallDeclined) {
            return false;
        }
        // Otherwise check if it's installed now
        const installed = await positronRuntime.isSquiggyInstalled();
        filePanelProvider.setSquiggyStatus(installed);
        return installed;
    }

    try {
        // Check if squiggy is installed
        const installed = await positronRuntime.isSquiggyInstalled();

        if (installed) {
            // Get version and update status badge
            const version = await positronRuntime.getSquiggyVersion();
            filePanelProvider.setSquiggyStatus(true, version || undefined);
            squiggyInstallChecked = true;
            return true;
        }

        // Not installed - prompt user
        const shouldInstall = await promptInstallSquiggy();

        if (shouldInstall) {
            // Install squiggy
            const success = await installSquiggyPackage();
            squiggyInstallChecked = true;

            if (success) {
                const version = await positronRuntime.getSquiggyVersion();
                filePanelProvider.setSquiggyStatus(true, version || undefined);
                return true;
            } else {
                filePanelProvider.setSquiggyStatus(false);
                return false;
            }
        } else {
            // User declined installation
            squiggyInstallDeclined = true;
            squiggyInstallChecked = true;
            filePanelProvider.setSquiggyStatus(false);
            return false;
        }
    } catch (error) {
        // Error during check - mark as unavailable
        filePanelProvider.setSquiggyStatus(false);
        squiggyInstallChecked = true;
        return false;
    }
}

/**
 * Prompt user to install squiggy package
 * @returns true if user wants to install, false otherwise
 */
async function promptInstallSquiggy(): Promise<boolean> {
    try {
        // Try to use Positron's modal dialog first
        // eslint-disable-next-line @typescript-eslint/no-var-requires
        const positron = require('positron');

        if (positron.window && positron.window.showSimpleModalDialogPrompt) {
            const title = 'Install Squiggy Python Package?';
            const message =
                'To visualize POD5 files, Squiggy needs to install the <code>squiggy</code> Python package ' +
                'in your active Python environment. This will install the package and its dependencies ' +
                '(bokeh, numpy, pod5, pysam).';
            const okButtonTitle = 'Install';

            return await positron.window.showSimpleModalDialogPrompt(title, message, okButtonTitle);
        }
    } catch {
        // Positron modal not available - fall through to VSCode dialog
    }

    // Fallback to VSCode information message with buttons
    const choice = await vscode.window.showInformationMessage(
        'Squiggy requires the Python package "squiggy" to be installed in your active Python environment. ' +
            'Install now?',
        'Install',
        'Cancel'
    );

    return choice === 'Install';
}

/**
 * Install squiggy package via pip
 * @returns true if installation succeeded, false otherwise
 */
async function installSquiggyPackage(): Promise<boolean> {
    try {
        return await vscode.window.withProgress(
            {
                location: vscode.ProgressLocation.Notification,
                title: 'Installing squiggy Python package...',
                cancellable: false,
            },
            async () => {
                try {
                    // Install from extension directory (editable install for development)
                    await positronRuntime.installSquiggy(extensionContext.extensionPath);

                    vscode.window.showInformationMessage(
                        'Successfully installed squiggy Python package!'
                    );
                    return true;
                } catch (error) {
                    const errorMessage = error instanceof Error ? error.message : String(error);
                    vscode.window.showErrorMessage(
                        `Failed to install squiggy package: ${errorMessage}`
                    );
                    return false;
                }
            }
        );
    } catch {
        return false;
    }
}

/**
 * Open a POD5 file
 */
async function openPOD5File(filePath: string) {
    // Ensure squiggy is available (check if installed, prompt if needed)
    const squiggyAvailable = await ensureSquiggyAvailable();

    if (!squiggyAvailable) {
        // User declined installation or installation failed
        vscode.window.showWarningMessage(
            'Cannot open POD5 file: squiggy Python package is not installed. ' +
                'Please install it manually with: pip install -e <extension-path>'
        );
        return;
    }

    try {
        await vscode.window.withProgress(
            {
                location: vscode.ProgressLocation.Notification,
                title: 'Opening POD5 file...',
                cancellable: false,
            },
            async () => {
                let numReads: number;
                let readIds: string[] = [];

                if (usePositron) {
                    // Use Positron kernel - lazy load read IDs
                    const result = await positronRuntime.loadPOD5(filePath);
                    numReads = result.numReads;

                    // Get first 1000 read IDs for tree view (lazy loading)
                    readIds = await positronRuntime.getReadIds(0, 1000);
                } else if (pythonBackend) {
                    // Use subprocess backend
                    const backendResult = await pythonBackend.call('open_pod5', {
                        file_path: filePath,
                    });
                    numReads = backendResult.num_reads;
                    readIds = backendResult.read_ids || [];
                } else {
                    throw new Error('No backend available');
                }

                // Update read tree
                if (readIds.length > 0) {
                    readTreeProvider.setReads(readIds);
                }

                // Track file and update file panel display
                _currentPod5File = filePath;

                // Get file size
                // Using imported fs.promises
                const stats = await fs.stat(filePath);
                const fileSizeMB = (stats.size / (1024 * 1024)).toFixed(2);

                filePanelProvider.setPOD5Info(filePath, numReads, `${fileSizeMB} MB`);
            }
        );
    } catch (error) {
        const errorMessage = error instanceof Error ? error.message : String(error);
        if (errorMessage.includes('No backend available') || errorMessage.includes('kernel')) {
            vscode.window.showErrorMessage(
                'Failed to open POD5 file: No Python kernel is running. ' +
                    'Please start a Python console first (use the Console pane or create a .py file).'
            );
        } else {
            vscode.window.showErrorMessage(`Failed to open POD5 file: ${error}`);
        }
    }
}

/**
 * Open a BAM file
 */
async function openBAMFile(filePath: string) {
    try {
        await vscode.window.withProgress(
            {
                location: vscode.ProgressLocation.Notification,
                title: 'Opening BAM file...',
                cancellable: false,
            },
            async () => {
                let numReads: number;
                let hasModifications: boolean;
                let modificationTypes: string[];
                let hasProbabilities: boolean;
                let referenceToReads: Record<string, string[]> = {};

                if (usePositron) {
                    // Use Positron kernel - lazy load reference mapping
                    const result = await positronRuntime.loadBAM(filePath);
                    numReads = result.numReads;
                    hasModifications = result.hasModifications;
                    modificationTypes = result.modificationTypes;
                    hasProbabilities = result.hasProbabilities;

                    // Get references and build mapping (lazy loading)
                    const references = await positronRuntime.getReferences();
                    for (const ref of references) {
                        const reads = await positronRuntime.getReadsForReference(ref);
                        referenceToReads[ref] = reads;
                    }
                } else if (pythonBackend) {
                    // Use subprocess backend
                    const backendResult = await pythonBackend.call('open_bam', {
                        file_path: filePath,
                    });
                    numReads = backendResult.num_reads;
                    referenceToReads = backendResult.reference_to_reads || {};
                    hasModifications = backendResult.has_modifications || false;
                    modificationTypes = backendResult.modification_types || [];
                    hasProbabilities = backendResult.has_probabilities || false;
                } else {
                    throw new Error('No backend available');
                }

                // Track file and update file panel display
                _currentBamFile = filePath;

                // Get file size
                // Using imported fs.promises
                const stats = await fs.stat(filePath);
                const fileSizeMB = (stats.size / (1024 * 1024)).toFixed(2);

                filePanelProvider.setBAMInfo(filePath, numReads, `${fileSizeMB} MB`);

                // Update read tree to show reads grouped by reference
                if (Object.keys(referenceToReads).length > 0) {
                    const refMap = new Map<string, string[]>(Object.entries(referenceToReads));
                    readTreeProvider.setReadsGrouped(refMap);
                }

                // Update modifications panel and context
                if (hasModifications) {
                    modificationsProvider.setModificationInfo(
                        hasModifications,
                        modificationTypes,
                        hasProbabilities
                    );
                    vscode.commands.executeCommand('setContext', 'squiggy.hasModifications', true);
                } else {
                    modificationsProvider.clear();
                    vscode.commands.executeCommand('setContext', 'squiggy.hasModifications', false);
                }

                // Update plot options to show EVENTALIGN mode and set as default
                plotOptionsProvider.updateBamStatus(true);
            }
        );
    } catch (error) {
        vscode.window.showErrorMessage(`Failed to open BAM file: ${error}`);
    }
}

/**
 * Plot reads
 */
async function plotReads(readIds: string[], _context: vscode.ExtensionContext) {
    try {
        // Track current plot for refresh
        currentPlotReadIds = readIds;

        await vscode.window.withProgress(
            {
                location: vscode.ProgressLocation.Notification,
                title: `Generating plot for ${readIds.length} read(s)...`,
                cancellable: false,
            },
            async () => {
                // Get options from sidebar panel
                const options = plotOptionsProvider.getOptions();
                const mode = options.mode;
                const normalization = options.normalization;

                // Get modification filters
                const modFilters = modificationsProvider.getFilters();

                // Detect VS Code theme
                const colorThemeKind = vscode.window.activeColorTheme.kind;
                const theme = colorThemeKind === vscode.ColorThemeKind.Dark ? 'DARK' : 'LIGHT';

                if (usePositron) {
                    // Use Positron kernel - plot appears in Plots pane automatically
                    await positronRuntime.generatePlot(
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
                        options.showSignalPoints
                    );
                } else if (pythonBackend) {
                    // Use subprocess backend - still need webview fallback
                    // TODO: subprocess backend doesn't have Plots pane integration
                    vscode.window.showWarningMessage(
                        'Plot display in Plots pane requires Positron runtime. Subplot backend not yet supported.'
                    );
                } else {
                    throw new Error('No backend available');
                }
            }
        );
    } catch (error) {
        vscode.window.showErrorMessage(`Failed to generate plot: ${error}`);
    }
}

/**
 * Generate and display aggregate plot for a reference sequence
 */
async function plotAggregate(referenceName: string, _context: vscode.ExtensionContext) {
    try {
        await vscode.window.withProgress(
            {
                location: vscode.ProgressLocation.Notification,
                title: `Generating aggregate plot for ${referenceName}...`,
                cancellable: false,
            },
            async () => {
                // Get normalization from sidebar panel
                const options = plotOptionsProvider.getOptions();
                const normalization = options.normalization;

                // Detect VS Code theme
                const colorThemeKind = vscode.window.activeColorTheme.kind;
                const theme = colorThemeKind === vscode.ColorThemeKind.Dark ? 'DARK' : 'LIGHT';

                // Get max reads from config
                const config = vscode.workspace.getConfiguration('squiggy');
                const maxReads = config.get<number>('aggregateSampleSize', 100);

                if (usePositron) {
                    // Use Positron kernel - plot appears in Plots pane automatically
                    await positronRuntime.generateAggregatePlot(
                        referenceName,
                        maxReads,
                        normalization,
                        theme
                    );
                } else if (pythonBackend) {
                    // Subprocess backend not yet implemented for aggregate
                    throw new Error(
                        'Aggregate plots are only available with Positron runtime. Please use Positron IDE.'
                    );
                } else {
                    throw new Error('No backend available');
                }
            }
        );
    } catch (error) {
        vscode.window.showErrorMessage(`Failed to generate aggregate plot: ${error}`);
    }
}

/**
 * Get Python interpreter path
 */
function getPythonPath(): string {
    // Try to find Python from environment
    const config = vscode.workspace.getConfiguration('python');
    const pythonPath = config.get<string>('defaultInterpreterPath');

    if (pythonPath) {
        return pythonPath;
    }

    // Fallback to system python3
    return 'python3';
}
