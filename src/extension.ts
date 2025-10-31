/**
 * Squiggy Positron Extension
 *
 * Main entry point for the extension. Handles activation, deactivation,
 * and registration of all commands, views, and providers.
 */

import * as vscode from 'vscode';
import * as path from 'path';
import { PositronRuntime } from './backend/positronRuntime';
import { PythonBackend } from './backend/pythonBackend';
import { ReadTreeProvider, ReadItem } from './views/readExplorer';
import { SquigglePlotPanel } from './webview/plotPanel';

let positronRuntime: PositronRuntime;
let pythonBackend: PythonBackend | null = null;
let readTreeProvider: ReadTreeProvider;
let usePositron = false;

/**
 * Extension activation
 */
export async function activate(context: vscode.ExtensionContext) {
    console.log('Squiggy extension activating...');

    // Try to use Positron runtime first
    positronRuntime = new PositronRuntime();
    usePositron = positronRuntime.isAvailable();

    if (usePositron) {
        console.log('Using Positron runtime API');

        // Check if squiggy is installed (will check when kernel is available)
        // This check happens lazily when user first tries to use the extension
        try {
            const isInstalled = await positronRuntime.isSquiggyInstalled();
            if (!isInstalled) {
                const install = await vscode.window.showInformationMessage(
                    'Squiggy package is not installed in the Python kernel. Install it now?',
                    'Install',
                    'Cancel'
                );

                if (install === 'Install') {
                    await vscode.window.withProgress({
                        location: vscode.ProgressLocation.Notification,
                        title: 'Installing squiggy package...',
                        cancellable: false
                    }, async () => {
                        await positronRuntime.installSquiggy();
                    });
                    vscode.window.showInformationMessage('Squiggy package installed successfully!');
                } else {
                    vscode.window.showWarningMessage('Squiggy extension requires the squiggy Python package');
                }
            }
        } catch (error) {
            // Kernel not available yet - will check later
            console.log('Could not check squiggy installation (kernel may not be running yet):', error);
        }
    } else {
        console.log('Positron runtime not available, using subprocess backend');

        // Fallback to subprocess JSON-RPC
        const pythonPath = getPythonPath();
        const serverPath = context.asAbsolutePath(path.join('src', 'python', 'server.py'));
        pythonBackend = new PythonBackend(pythonPath, serverPath);

        try {
            await pythonBackend.start();

            // Register cleanup on deactivation
            context.subscriptions.push({
                dispose: () => pythonBackend?.stop()
            });
        } catch (error) {
            vscode.window.showErrorMessage(
                `Failed to start Python backend: ${error}. ` +
                `Please ensure Python is installed and the squiggy package is available.`
            );
        }
    }

    // Create read tree provider
    readTreeProvider = new ReadTreeProvider();
    const readTreeView = vscode.window.createTreeView('squiggyReadList', {
        treeDataProvider: readTreeProvider,
        canSelectMany: true
    });
    context.subscriptions.push(readTreeView);

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
function registerCommands(context: vscode.ExtensionContext, readTreeView: vscode.TreeView<ReadItem>) {
    // Open POD5 file
    context.subscriptions.push(
        vscode.commands.registerCommand('squiggy.openPOD5', async () => {
            const fileUri = await vscode.window.showOpenDialog({
                canSelectMany: false,
                filters: { 'POD5 Files': ['pod5'] },
                title: 'Open POD5 File'
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
                title: 'Open BAM File'
            });

            if (fileUri && fileUri[0]) {
                await openBAMFile(fileUri[0].fsPath);
            }
        })
    );

    // Plot selected reads
    context.subscriptions.push(
        vscode.commands.registerCommand('squiggy.plotRead', async (item?: ReadItem) => {
            const selection = readTreeView.selection;
            if (selection.length === 0) {
                vscode.window.showWarningMessage('Please select one or more reads to plot');
                return;
            }

            const readIds = selection.map(item => item.readId);
            await plotReads(readIds, context);
        })
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
                    'HTML': ['html'],
                    'PNG': ['png'],
                    'SVG': ['svg']
                },
                title: 'Export Plot'
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
        await vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: 'Opening POD5 file...',
            cancellable: false
        }, async () => {
            let result: { readIds?: string[], numReads: number };

            if (usePositron) {
                // Use Positron kernel
                result = await positronRuntime.loadPOD5(filePath);
            } else if (pythonBackend) {
                // Use subprocess backend
                const backendResult = await pythonBackend.call('open_pod5', { file_path: filePath });
                result = {
                    readIds: backendResult.read_ids,
                    numReads: backendResult.num_reads
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
        });
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
        await vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: 'Opening BAM file...',
            cancellable: false
        }, async () => {
            let result: { numReads: number };

            if (usePositron) {
                // Use Positron kernel
                result = await positronRuntime.loadBAM(filePath);
            } else if (pythonBackend) {
                // Use subprocess backend
                const backendResult = await pythonBackend.call('open_bam', { file_path: filePath });
                result = {
                    numReads: backendResult.num_reads
                };
            } else {
                throw new Error('No backend available');
            }

            vscode.window.showInformationMessage(
                `Loaded BAM file with ${result.numReads} reads`
            );
        });
    } catch (error) {
        vscode.window.showErrorMessage(`Failed to open BAM file: ${error}`);
    }
}

/**
 * Plot reads
 */
async function plotReads(readIds: string[], context: vscode.ExtensionContext) {
    try {
        await vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: `Generating plot for ${readIds.length} read(s)...`,
            cancellable: false
        }, async () => {
            // Get configuration
            const config = vscode.workspace.getConfiguration('squiggy');
            const mode = config.get<string>('defaultPlotMode', 'SINGLE');
            const normalization = config.get<string>('defaultNormalization', 'ZNORM');
            const theme = config.get<string>('theme', 'LIGHT');

            let html: string;

            if (usePositron) {
                // Use Positron kernel - generates plot and saves to temp file
                const tempFilePath = await positronRuntime.generatePlot(
                    readIds,
                    mode,
                    normalization,
                    theme
                );

                // Read HTML from temp file
                const fs = require('fs').promises;
                html = await fs.readFile(tempFilePath, 'utf-8');

                // Clean up temp file
                await fs.unlink(tempFilePath).catch(() => { }); // Ignore errors
            } else if (pythonBackend) {
                // Use subprocess backend
                const result = await pythonBackend.call('generate_plot', {
                    read_ids: readIds,
                    mode: mode,
                    normalization: normalization,
                    options: {
                        theme: theme,
                        downsample: true,
                        downsample_threshold: config.get<number>('downsampleThreshold', 100000)
                    }
                });
                html = result.html;
            } else {
                throw new Error('No backend available');
            }

            // Show plot in webview
            const panel = SquigglePlotPanel.createOrShow(context.extensionUri);
            panel.setPlot(html, readIds);
        });
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
