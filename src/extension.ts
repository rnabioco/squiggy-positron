/**
 * Squiggy Positron Extension
 *
 * Main entry point for the extension. Handles activation, deactivation,
 * and registration of all commands, views, and providers.
 */

import * as vscode from 'vscode';
import * as path from 'path';
import { PositronRuntime } from './backend/squiggy-positron-runtime';
import { PythonBackend } from './backend/squiggy-python-backend';
import { ReadTreeProvider, ReadItem } from './views/squiggy-read-explorer';
import { ReadSearchViewProvider } from './views/squiggy-read-search-view';
import { PlotOptionsViewProvider } from './views/squiggy-plot-options-view';
import { FilePanelProvider } from './views/squiggy-file-panel';
import { ModificationsPanelProvider } from './views/squiggy-modifications-panel';
import { SquigglePlotPanel } from './webview/squiggy-plot-panel';

let positronRuntime: PositronRuntime;
let pythonBackend: PythonBackend | null = null;
let readTreeProvider: ReadTreeProvider;
let readSearchProvider: ReadSearchViewProvider;
let plotOptionsProvider: PlotOptionsViewProvider;
let filePanelProvider: FilePanelProvider;
let modificationsProvider: ModificationsPanelProvider;
let usePositron = false;

// Track loaded files and current plot
let currentPod5File: string | undefined;
let currentBamFile: string | undefined;
let currentPlotReadIds: string[] | undefined;

/**
 * Extension activation
 */
export async function activate(context: vscode.ExtensionContext) {
    // Try to use Positron runtime first
    positronRuntime = new PositronRuntime();
    usePositron = positronRuntime.isAvailable();

    if (usePositron) {

        // Check if squiggy is installed (will check when kernel is available)
        // This check happens lazily when user first tries to use the extension
        try {
            const isInstalled = await positronRuntime.isSquiggyInstalled();
            if (!isInstalled) {
                // In development mode, automatically add workspace to sys.path
                const workspaceFolder = context.extensionPath;

                await positronRuntime.executeWithOutput(
                    `import sys\nif '${workspaceFolder}' not in sys.path:\n    sys.path.insert(0, '${workspaceFolder}')`
                );

                // Check again
                const nowInstalled = await positronRuntime.isSquiggyInstalled();
                if (nowInstalled) {
                    vscode.window.showInformationMessage(
                        'Squiggy package loaded from development workspace'
                    );
                } else {
                    vscode.window.showWarningMessage(
                        'Could not load Squiggy package. Extension may not work correctly.'
                    );
                }
            }
        } catch (error) {
            // Kernel not available yet - will check later silently
        }
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

            // Load both files sequentially
            await openPOD5File(pod5Path);
            await openBAMFile(bamPath);

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

    // Export plot
    context.subscriptions.push(
        vscode.commands.registerCommand('squiggy.exportPlot', async () => {
            const panel = SquigglePlotPanel.currentPanel;
            if (!panel) {
                vscode.window.showWarningMessage('No plot is currently open');
                return;
            }

            const fileUri = await vscode.window.showSaveDialog({
                filters: {
                    HTML: ['html'],
                    PNG: ['png'],
                    SVG: ['svg'],
                },
                title: 'Export Plot',
            });

            if (fileUri) {
                await panel.exportPlot(fileUri.fsPath);
            }
        })
    );

    // Refresh reads
    context.subscriptions.push(
        vscode.commands.registerCommand('squiggy.refreshReads', () => {
            readTreeProvider.refresh();
        })
    );
}

/**
 * Open a POD5 file
 */
async function openPOD5File(filePath: string) {
    try {
        await vscode.window.withProgress(
            {
                location: vscode.ProgressLocation.Notification,
                title: 'Opening POD5 file...',
                cancellable: false,
            },
            async () => {
                let result: { readIds?: string[]; numReads: number };

                if (usePositron) {
                    // Use Positron kernel
                    result = await positronRuntime.loadPOD5(filePath);
                } else if (pythonBackend) {
                    // Use subprocess backend
                    const backendResult = await pythonBackend.call('open_pod5', {
                        file_path: filePath,
                    });
                    result = {
                        readIds: backendResult.read_ids,
                        numReads: backendResult.num_reads,
                    };
                } else {
                    throw new Error('No backend available');
                }

                // Update read tree if we got read IDs
                if (result.readIds && result.readIds.length > 0) {
                    readTreeProvider.setReads(result.readIds);
                    vscode.window.showInformationMessage(
                        `Loaded ${result.numReads} reads (showing first ${result.readIds.length}) from ${path.basename(filePath)}`
                    );
                } else {
                    vscode.window.showInformationMessage(
                        `Loaded ${result.numReads} reads from ${path.basename(filePath)}. ` +
                            `Read IDs are available in the '_squiggy_read_ids' variable in the console.`
                    );
                }

                // Track file and update file panel display
                currentPod5File = filePath;

                // Get file size
                const fs = require('fs').promises;
                const stats = await fs.stat(filePath);
                const fileSizeMB = (stats.size / (1024 * 1024)).toFixed(2);

                filePanelProvider.setPOD5Info(filePath, result.numReads, `${fileSizeMB} MB`);
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
                let result: {
                    numReads: number;
                    referenceToReads?: Record<string, string[]>;
                    hasModifications?: boolean;
                    modificationTypes?: string[];
                    hasProbabilities?: boolean;
                };

                if (usePositron) {
                    // Use Positron kernel
                    result = await positronRuntime.loadBAM(filePath);
                } else if (pythonBackend) {
                    // Use subprocess backend
                    const backendResult = await pythonBackend.call('open_bam', {
                        file_path: filePath,
                    });
                    result = {
                        numReads: backendResult.num_reads,
                        referenceToReads: backendResult.reference_to_reads,
                        hasModifications: backendResult.has_modifications,
                        modificationTypes: backendResult.modification_types,
                        hasProbabilities: backendResult.has_probabilities,
                    };
                } else {
                    throw new Error('No backend available');
                }

                // Track file and update file panel display
                currentBamFile = filePath;

                // Get file size
                const fs = require('fs').promises;
                const stats = await fs.stat(filePath);
                const fileSizeMB = (stats.size / (1024 * 1024)).toFixed(2);

                filePanelProvider.setBAMInfo(filePath, result.numReads, `${fileSizeMB} MB`);

                // Update read tree to show reads grouped by reference
                if (result.referenceToReads && Object.keys(result.referenceToReads).length > 0) {
                    const refMap = new Map<string, string[]>(
                        Object.entries(result.referenceToReads)
                    );
                    readTreeProvider.setReadsGrouped(refMap);
                }

                // Update modifications panel and context
                const hasModifications = result.hasModifications || false;
                const modificationTypes = result.modificationTypes || [];
                const hasProbabilities = result.hasProbabilities || false;

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

                vscode.window.showInformationMessage(
                    `Loaded BAM file with ${result.numReads} reads${hasModifications ? ' (contains base modifications)' : ''}`
                );
            }
        );
    } catch (error) {
        vscode.window.showErrorMessage(`Failed to open BAM file: ${error}`);
    }
}

/**
 * Plot reads
 */
async function plotReads(readIds: string[], context: vscode.ExtensionContext) {
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

                // Get config for other settings
                const config = vscode.workspace.getConfiguration('squiggy');

                let html: string;

                if (usePositron) {
                    // Use Positron kernel - generates plot and saves to temp file
                    const tempFilePath = await positronRuntime.generatePlot(
                        readIds,
                        mode,
                        normalization,
                        theme,
                        options.showDwellTime,
                        options.showBaseAnnotations,
                        options.scaleDwellTime,
                        modFilters.minProbability,
                        modFilters.enabledModTypes
                    );

                    // Read HTML from temp file
                    const fs = require('fs').promises;
                    html = await fs.readFile(tempFilePath, 'utf-8');

                    // Clean up temp file
                    await fs.unlink(tempFilePath).catch(() => {}); // Ignore errors
                } else if (pythonBackend) {
                    // Use subprocess backend
                    const result = await pythonBackend.call('generate_plot', {
                        read_ids: readIds,
                        mode: mode,
                        normalization: normalization,
                        options: {
                            theme: theme,
                            downsample: true,
                            downsample_threshold: config.get<number>('downsampleThreshold', 100000),
                            show_dwell_time: options.showDwellTime,
                            show_base_annotations: options.showBaseAnnotations,
                        },
                    });
                    html = result.html;
                } else {
                    throw new Error('No backend available');
                }

                // Show plot in webview
                const panel = SquigglePlotPanel.createOrShow(context.extensionUri);
                panel.setPlot(html, readIds);
            }
        );
    } catch (error) {
        vscode.window.showErrorMessage(`Failed to generate plot: ${error}`);
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
