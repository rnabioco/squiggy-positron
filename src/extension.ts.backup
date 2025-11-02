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
import { ReadsViewPane } from './views/squiggy-reads-view-pane';
import { PlotOptionsViewProvider } from './views/squiggy-plot-options-view';
import { FilePanelProvider } from './views/squiggy-file-panel';
import { ModificationsPanelProvider } from './views/squiggy-modifications-panel';
import { ReadItem } from './types/squiggy-reads-types';

let positronRuntime: PositronRuntime;
let pythonBackend: PythonBackend | null = null;
let readsViewPane: ReadsViewPane;
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

    // Create and register new React-based reads view pane
    readsViewPane = new ReadsViewPane(context.extensionUri);
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider(ReadsViewPane.viewType, readsViewPane)
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
            // eslint-disable-next-line @typescript-eslint/no-var-requires, @typescript-eslint/no-require-imports
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
                readsViewPane.setReads([]);
                filePanelProvider.clearPOD5();
                filePanelProvider.clearBAM();
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
                            session.onDidChangeRuntimeState((state: string) => {
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
        } catch (_error) {
            // Positron API not available - not running in Positron
        }
    }

    // Register commands
    registerCommands(context);

    // Extension activated silently - no welcome message needed
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
function registerCommands(context: vscode.ExtensionContext) {
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

    // Close POD5 file
    context.subscriptions.push(
        vscode.commands.registerCommand('squiggy.closePOD5', async () => {
            const confirm = await vscode.window.showWarningMessage(
                'Close POD5 file?',
                { modal: true },
                'Close'
            );

            if (confirm === 'Close') {
                await closePOD5File();
            }
        })
    );

    // Close BAM file
    context.subscriptions.push(
        vscode.commands.registerCommand('squiggy.closeBAM', async () => {
            const confirm = await vscode.window.showWarningMessage(
                'Close BAM file?',
                { modal: true },
                'Close'
            );

            if (confirm === 'Close') {
                await closeBAMFile();
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

                await plotReads(readIds, context);
            }
        )
    );

    // Plot aggregate for reference
    context.subscriptions.push(
        vscode.commands.registerCommand('squiggy.plotAggregate', async (referenceName?: string) => {
            // Validate that both POD5 and BAM are loaded
            if (!_currentPod5File) {
                vscode.window.showErrorMessage('No POD5 file loaded. Use "Open POD5 File" first.');
                return;
            }
            if (!_currentBamFile) {
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

            await plotAggregate(referenceName, context);
        })
    );

    // Refresh reads
    context.subscriptions.push(
        vscode.commands.registerCommand('squiggy.refreshReads', () => {
            readsViewPane.refresh();
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
            readsViewPane.setReads([]);
            filePanelProvider.clearPOD5();
            filePanelProvider.clearBAM();
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
                } catch (_error) {
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

    // Always check if squiggy is installed (user may have installed manually)
    const installed = await positronRuntime.isSquiggyInstalled();

    if (installed) {
        // Package is installed - return success
        squiggyInstallChecked = true;
        squiggyInstallDeclined = false; // Reset declined flag since it's now installed
        return true;
    }

    // Not installed - check if we should prompt
    if (squiggyInstallChecked && squiggyInstallDeclined) {
        // User already declined this session - don't prompt again
        return false;
    }

    try {
        // Prompt user to install
        const userChoice = await promptInstallSquiggy();

        if (userChoice === 'install') {
            // Install squiggy
            const success = await installSquiggyPackage();
            squiggyInstallChecked = true;
            return success;
        } else if (userChoice === 'manual') {
            // Show manual installation guide
            await showManualInstallationGuide();
            squiggyInstallDeclined = true;
            squiggyInstallChecked = true;
            return false;
        } else {
            // User canceled installation
            squiggyInstallDeclined = true;
            squiggyInstallChecked = true;
            return false;
        }
    } catch (_error) {
        // Error during check - mark as unavailable
        squiggyInstallChecked = true;
        return false;
    }
}

/**
 * Prompt user to install squiggy package
 * @returns true if user wants to install, false otherwise
 */
async function promptInstallSquiggy(): Promise<'install' | 'manual' | 'cancel'> {
    // Use VSCode information message with three options
    const choice = await vscode.window.showInformationMessage(
        'Squiggy requires the Python package "squiggy" to be installed in your active Python environment.',
        'Install Automatically',
        'Manual Instructions',
        'Cancel'
    );

    if (choice === 'Install Automatically') {
        return 'install';
    } else if (choice === 'Manual Instructions') {
        return 'manual';
    } else {
        return 'cancel';
    }
}

/**
 * Show manual installation guide with copy-able commands
 */
async function showManualInstallationGuide(): Promise<void> {
    const extensionPath =
        vscode.extensions.getExtension('rnabioco.squiggy-positron')?.extensionPath || '';

    const items = [
        {
            label: '1. Create Virtual Environment',
            detail: 'python3 -m venv .venv',
            description: 'Create a new virtual environment in your project',
        },
        {
            label: '2. Activate Virtual Environment (macOS/Linux)',
            detail: 'source .venv/bin/activate',
            description: 'Activate the virtual environment',
        },
        {
            label: '3. Activate Virtual Environment (Windows)',
            detail: '.venv\\Scripts\\activate',
            description: 'Activate the virtual environment on Windows',
        },
        {
            label: '4. Install Squiggy Package',
            detail: `pip install -e "${extensionPath}"`,
            description: 'Install squiggy in editable mode',
        },
        {
            label: '5. Select Environment in Positron',
            detail: 'Use the Interpreter selector to choose your new .venv',
            description: 'Switch to the new virtual environment',
        },
    ];

    const selected = await vscode.window.showQuickPick(items, {
        placeHolder: 'Select a command to copy to clipboard',
        title: 'Manual Installation Steps',
    });

    if (selected && selected.detail) {
        await vscode.env.clipboard.writeText(selected.detail);
        vscode.window.showInformationMessage(`Copied to clipboard: ${selected.detail}`);
    }
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

                    // Detect PEP 668 externally-managed environment errors
                    if (
                        errorMessage.includes('EXTERNALLY_MANAGED_ENVIRONMENT') ||
                        errorMessage.includes('externally-managed-environment') ||
                        errorMessage.includes('EXTERNALLY-MANAGED')
                    ) {
                        // Show detailed error with option to see manual instructions
                        const choice = await vscode.window.showErrorMessage(
                            'Cannot install squiggy: Python environment is externally managed by your ' +
                                'system package manager. Please create a virtual environment first.',
                            'Show Instructions',
                            'Dismiss'
                        );

                        if (choice === 'Show Instructions') {
                            await showManualInstallationGuide();
                        }
                    } else {
                        // Generic installation error
                        vscode.window.showErrorMessage(
                            `Failed to install squiggy package: ${errorMessage}`
                        );
                    }
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
                    const backendResult = (await pythonBackend.call('open_pod5', {
                        file_path: filePath,
                    })) as { num_reads: number; read_ids?: string[] };
                    numReads = backendResult.num_reads;
                    readIds = backendResult.read_ids || [];
                } else {
                    throw new Error('No backend available');
                }

                // Update reads view
                if (readIds.length > 0) {
                    readsViewPane.setReads(readIds);
                }

                // Track file and update file panel display
                _currentPod5File = filePath;

                // Get file size
                // Using imported fs.promises
                const stats = await fs.stat(filePath);

                filePanelProvider.setPOD5({
                    path: filePath,
                    numReads,
                    size: stats.size,
                });
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
    // Ensure squiggy is available (check if installed, prompt if needed)
    const squiggyAvailable = await ensureSquiggyAvailable();

    if (!squiggyAvailable) {
        // User declined installation or installation failed
        vscode.window.showWarningMessage(
            'Cannot open BAM file: squiggy Python package is not installed. ' +
                'Please install it manually with: pip install -e <extension-path>'
        );
        return;
    }

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
                let hasEventAlignment: boolean = false;
                let referenceToReads: Record<string, string[]> = {};

                if (usePositron) {
                    // Use Positron kernel - lazy load reference mapping
                    const result = await positronRuntime.loadBAM(filePath);
                    numReads = result.numReads;
                    hasModifications = result.hasModifications;
                    modificationTypes = result.modificationTypes;
                    hasProbabilities = result.hasProbabilities;
                    hasEventAlignment = result.hasEventAlignment || false;

                    // Get references and build mapping (lazy loading)
                    const references = await positronRuntime.getReferences();
                    for (const ref of references) {
                        const reads = await positronRuntime.getReadsForReference(ref);
                        referenceToReads[ref] = reads;
                    }
                } else if (pythonBackend) {
                    // Use subprocess backend
                    const backendResult = (await pythonBackend.call('open_bam', {
                        file_path: filePath,
                    })) as {
                        num_reads: number;
                        reference_to_reads?: Record<string, string[]>;
                        has_modifications?: boolean;
                        modification_types?: string[];
                        has_probabilities?: boolean;
                        has_event_alignment?: boolean;
                    };
                    numReads = backendResult.num_reads;
                    referenceToReads = backendResult.reference_to_reads || {};
                    hasModifications = backendResult.has_modifications || false;
                    modificationTypes = backendResult.modification_types || [];
                    hasProbabilities = backendResult.has_probabilities || false;
                    hasEventAlignment = backendResult.has_event_alignment || false;
                } else {
                    throw new Error('No backend available');
                }

                // Track file and update file panel display
                _currentBamFile = filePath;

                // Get file size
                // Using imported fs.promises
                const stats = await fs.stat(filePath);

                filePanelProvider.setBAM({
                    path: filePath,
                    numReads,
                    numRefs: Object.keys(referenceToReads).length,
                    size: stats.size,
                    hasMods: hasModifications,
                    hasEvents: hasEventAlignment,
                });

                // Update reads view to show reads grouped by reference
                if (Object.keys(referenceToReads).length > 0) {
                    const refMap = new Map<string, string[]>(Object.entries(referenceToReads));

                    // Convert to ReadItem[] with reference info
                    const readItemsMap = new Map<string, any[]>();
                    for (const [ref, reads] of refMap.entries()) {
                        readItemsMap.set(
                            ref,
                            reads.map((readId) => ({
                                type: 'read' as const,
                                readId,
                                referenceName: ref,
                                indentLevel: 1,
                            }))
                        );
                    }
                    readsViewPane.setReadsGrouped(readItemsMap);
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
 * Close POD5 file
 */
async function closePOD5File() {
    try {
        // Clear Python state
        if (usePositron) {
            await positronRuntime.executeSilent(`
import squiggy
squiggy.close_pod5()
`);
        }

        // Clear extension state
        _currentPod5File = undefined;

        // Clear UI
        filePanelProvider.clearPOD5();
        readsViewPane.setReads([]);

        vscode.window.showInformationMessage('POD5 file closed');
    } catch (error) {
        vscode.window.showErrorMessage(`Failed to close POD5 file: ${error}`);
    }
}

/**
 * Close BAM file
 */
async function closeBAMFile() {
    try {
        // Clear Python state
        if (usePositron) {
            await positronRuntime.executeSilent(`
# Clear BAM file state
_current_bam_path = None
`);
        }

        // Clear extension state
        _currentBamFile = undefined;

        // Clear UI
        filePanelProvider.clearBAM();
        modificationsProvider.clear();
        plotOptionsProvider.updateBamStatus(false);
        vscode.commands.executeCommand('setContext', 'squiggy.hasModifications', false);

        // If POD5 is still loaded, revert to flat read list
        if (_currentPod5File && usePositron) {
            const readIds = await positronRuntime.getReadIds(0, 1000);
            readsViewPane.setReads(readIds);
        }

        vscode.window.showInformationMessage('BAM file closed');
    } catch (error) {
        vscode.window.showErrorMessage(`Failed to close BAM file: ${error}`);
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
