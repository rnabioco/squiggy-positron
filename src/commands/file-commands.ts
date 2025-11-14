/**
 * File Commands
 *
 * Handles opening/closing POD5 and BAM files, plus loading test data.
 * Extracted from extension.ts to improve modularity.
 * Uses FileLoadingService for centralized file loading and deduplication.
 */

import * as vscode from 'vscode';
import * as path from 'path';
import { promises as fs } from 'fs';
import { ExtensionState } from '../state/extension-state';
import { ErrorContext, handleError, safeExecuteWithProgress } from '../utils/error-handler';
import { FileLoadingService } from '../services/file-loading-service';
import { LoadedItem } from '../types/loaded-item';
import { POD5LoadResult, BAMLoadResult } from '../types/file-loading-types';
import { logger } from '../utils/logger';

/**
 * Register file-related commands
 */
export function registerFileCommands(
    context: vscode.ExtensionContext,
    state: ExtensionState
): void {
    // Open POD5 file
    context.subscriptions.push(
        vscode.commands.registerCommand('squiggy.openPOD5', async () => {
            const fileUri = await vscode.window.showOpenDialog({
                canSelectMany: false,
                filters: { 'POD5 Files': ['pod5'] },
                title: 'Open POD5 File',
            });

            if (fileUri && fileUri[0]) {
                await openPOD5File(fileUri[0].fsPath, state);
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
                await openBAMFile(fileUri[0].fsPath, state);
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
                await closePOD5File(state);
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
                await closeBAMFile(state);
            }
        })
    );

    // Open FASTA file
    context.subscriptions.push(
        vscode.commands.registerCommand('squiggy.openFASTA', async () => {
            const fileUri = await vscode.window.showOpenDialog({
                canSelectMany: false,
                filters: { 'FASTA Files': ['fa', 'fasta', 'fna'] },
                title: 'Open FASTA File',
            });

            if (fileUri && fileUri[0]) {
                await openFASTAFile(fileUri[0].fsPath, state);
            }
        })
    );

    // Close FASTA file
    context.subscriptions.push(
        vscode.commands.registerCommand('squiggy.closeFASTA', async () => {
            const confirm = await vscode.window.showWarningMessage(
                'Close FASTA file?',
                { modal: true },
                'Close'
            );

            if (confirm === 'Close') {
                await closeFASTAFile(state);
            }
        })
    );

    // Load test data
    // Uses squiggy.get_test_data_path() to access bundled test data from the Python package
    context.subscriptions.push(
        vscode.commands.registerCommand('squiggy.loadTestData', async () => {
            // Ensure squiggy is available
            if (!(await ensureSquiggyAvailable(state))) {
                vscode.window.showErrorMessage('Squiggy package not available');
                return;
            }

            // Get test data paths from the Python package (bundled with squiggy-positron)
            let pod5Path: string;
            let bamPath: string;
            let fastaPath: string;

            try {
                // Execute Python code to get test data paths
                const getPathsCode = `
import squiggy
paths = {
    'pod5': squiggy.get_test_data_path('yeast_trna_reads.pod5'),
    'bam': squiggy.get_test_data_path('yeast_trna_mappings.bam'),
    'fasta': squiggy.get_test_data_path('yeast_trna.fa')
}
                `.trim();

                // Use the client directly to execute and get variable
                await state.positronClient?.executeSilent(getPathsCode);
                const paths = (await state.positronClient?.getVariable('paths')) as {
                    pod5: string;
                    bam: string;
                    fasta: string;
                };
                pod5Path = paths.pod5;
                bamPath = paths.bam;
                fastaPath = paths.fasta;

                console.log('[loadTestData] Got test data paths from Python package:', paths);
            } catch (error) {
                vscode.window.showErrorMessage(
                    'Failed to get test data paths from squiggy package. ' +
                        'Make sure squiggy-positron is installed: uv pip install squiggy-positron'
                );
                console.error('[loadTestData] Error getting test data paths:', error);
                return;
            }

            // Check if files are already loaded
            const pod5AlreadyLoaded = state.currentPod5File === pod5Path;
            const bamAlreadyLoaded = state.currentBamFile === bamPath;
            const fastaAlreadyLoaded = state.currentFastaFile === fastaPath;

            if (pod5AlreadyLoaded && bamAlreadyLoaded && fastaAlreadyLoaded) {
                vscode.window.showInformationMessage(
                    'Test data is already loaded. Use "Refresh Read List" to update the view.'
                );
                return;
            }

            // Load files sequentially (only if not already loaded)
            if (!pod5AlreadyLoaded) {
                await openPOD5File(pod5Path, state);
            }
            if (!bamAlreadyLoaded) {
                await openBAMFile(bamPath, state);
            }
            if (!fastaAlreadyLoaded) {
                await openFASTAFile(fastaPath, state);
            }

            vscode.window.showInformationMessage('Test data loaded successfully!');
        })
    );

    // Load sample (for multi-sample comparison) - Phase 4
    context.subscriptions.push(
        vscode.commands.registerCommand('squiggy.loadSample', async () => {
            await loadSampleForComparison(context, state);
        })
    );

    // Load test multi-read datasets for comparison
    context.subscriptions.push(
        vscode.commands.registerCommand('squiggy.loadTestMultiReadDataset', async () => {
            await loadTestMultiReadDataset(context, state);
        })
    );

    // Set session-level FASTA file for all comparisons
    context.subscriptions.push(
        vscode.commands.registerCommand('squiggy.setSessionFasta', async () => {
            const fileUri = await vscode.window.showOpenDialog({
                canSelectMany: false,
                filters: { 'FASTA Files': ['fasta', 'fa', 'fna'] },
                title: 'Select FASTA File for Comparisons',
            });

            if (fileUri && fileUri[0]) {
                const fastaPath = fileUri[0].fsPath;
                state.setSessionFasta(fastaPath);

                // Notify samples panel of FASTA change
                if (state.samplesProvider) {
                    state.samplesProvider.updateSessionFasta(fastaPath);
                }

                vscode.window.showInformationMessage(`FASTA set: ${path.basename(fastaPath)}`);
            }
        })
    );

    // Load samples from dropped files
    context.subscriptions.push(
        vscode.commands.registerCommand(
            'squiggy.loadSamplesFromDropped',
            async (fileQueue: { pod5Path: string; bamPath?: string; sampleName: string }[]) => {
                await loadSamplesFromDropped(context, state, fileQueue);
            }
        )
    );

    // Load samples via file picker (from UI button)
    context.subscriptions.push(
        vscode.commands.registerCommand('squiggy.loadSamplesFromUI', async () => {
            logger.debug('[squiggy.loadSamplesFromUI] Command handler invoked');
            const fileUris = await vscode.window.showOpenDialog({
                canSelectMany: true,
                filters: { 'Sequence Files': ['pod5', 'bam'], 'All Files': ['*'] },
                title: 'Select POD5 and BAM files to load',
            });

            logger.debug(
                `[squiggy.loadSamplesFromUI] File picker returned ${fileUris?.length || 0} files`
            );
            if (!fileUris || fileUris.length === 0) {
                logger.debug('[squiggy.loadSamplesFromUI] No files selected, returning');
                return;
            }

            const filePaths = fileUris.map((uri) => uri.fsPath);
            logger.debug('[squiggy.loadSamplesFromUI] Selected file paths:', filePaths);

            // Delegate to the samples panel's file handling logic
            const samplesProvider = state.samplesProvider;
            if (samplesProvider) {
                logger.debug(
                    '[squiggy.loadSamplesFromUI] samplesProvider exists, calling loadSamplesFromFilePicker'
                );
                // The samples panel will handle categorizing and matching files
                // We need to access its private method via message or use the same logic
                // For now, use the setFilesForLoading approach
                // Actually, we need to call a public method on samplesProvider
                // Let's create a new public method that handles file paths directly
                await loadSamplesFromFilePicker(context, state, filePaths);
            } else {
                logger.debug('[squiggy.loadSamplesFromUI] samplesProvider does not exist');
            }
        })
    );

    // Internal commands for lazy loading
    context.subscriptions.push(
        vscode.commands.registerCommand('squiggy.internal.loadMoreReads', async () => {
            await loadMoreReads(state);
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand(
            'squiggy.internal.expandReference',
            async (referenceName: string, offset: number, limit: number) => {
                await expandReference(referenceName, offset, limit, state);
            }
        )
    );

    // Load reads for a specific sample (multi-sample support)
    context.subscriptions.push(
        vscode.commands.registerCommand(
            'squiggy.internal.loadReadsForSample',
            async (sampleName: string) => {
                await loadReadsForSample(sampleName, state);
            }
        )
    );

    // Update sample files (BAM/FASTA)
    context.subscriptions.push(
        vscode.commands.registerCommand(
            'squiggy.updateSampleFiles',
            async (sampleName: string, files: { bamPath?: string; fastaPath?: string }) => {
                await updateSampleFiles(sampleName, files, state);
            }
        )
    );
}

/**
 * Load more reads for POD5 pagination
 */
async function loadMoreReads(state: ExtensionState): Promise<void> {
    if (!state.usePositron || !state.squiggyAPI || !state.currentPod5File) {
        return;
    }

    // Get dedicated kernel API
    const api = await state.ensureBackgroundKernel();

    // Track POD5 pagination context
    if (!state.pod5LoadContext) {
        // Initialize context if not present
        const totalReads = await api.client.getVariable(
            'len(squiggy.io.squiggy_kernel._read_ids)'
        );
        state.pod5LoadContext = {
            currentOffset: 1000, // Initial load was 1000
            pageSize: 500,
            totalReads: totalReads as number,
        };
    }

    const { currentOffset, pageSize, totalReads } = state.pod5LoadContext;

    if (currentOffset >= totalReads) {
        // All reads loaded
        return;
    }

    try {
        // Show loading state
        state.readsViewPane?.setLoading(true, 'Loading more reads...');

        // Fetch next batch
        const nextBatch = await api.getReadIds(currentOffset, pageSize);

        // Send to React
        state.readsViewPane?.appendReads(nextBatch);

        // Update context
        state.pod5LoadContext.currentOffset += nextBatch.length;
    } finally {
        state.readsViewPane?.setLoading(false);
    }
}

/**
 * Expand reference and fetch reads (BAM lazy loading)
 */
async function expandReference(
    referenceName: string,
    offset: number,
    limit: number,
    state: ExtensionState
): Promise<void> {
    if (!state.usePositron || !state.squiggyAPI) {
        return;
    }

    try {
        // Show loading state
        state.readsViewPane?.setLoading(true, `Loading reads for ${referenceName}...`);

        // Check if we're in multi-sample mode or single-file mode
        if (state.selectedReadExplorerSample) {
            // Multi-sample mode: get reads for reference within a specific sample
            logger.debug(
                `[expandReference] Multi-sample mode: getting reads for ${referenceName} in sample ${state.selectedReadExplorerSample}`
            );
            const readIds = await state.squiggyAPI.getReadsForReferenceSample(
                state.selectedReadExplorerSample,
                referenceName
            );

            // Send to React
            state.readsViewPane?.setReadsForReference(referenceName, readIds, 0, readIds.length);
        } else if (state.currentBamFile) {
            // Single-file mode: get reads for reference from loaded BAM
            logger.debug(`[expandReference] Single-file mode: getting reads for ${referenceName}`);
            const result = await state.squiggyAPI.getReadsForReferencePaginated(
                referenceName,
                offset,
                limit
            );

            // Send to React
            state.readsViewPane?.setReadsForReference(
                referenceName,
                result.readIds,
                offset,
                result.totalCount
            );
        } else {
            logger.warning('[expandReference] No sample selected and no BAM file loaded');
            return;
        }
    } finally {
        state.readsViewPane?.setLoading(false);
    }
}

/**
 * Load reads for a specific sample from the multi-sample registry
 */
async function loadReadsForSample(sampleName: string, state: ExtensionState): Promise<void> {
    if (!state.usePositron || !state.squiggyAPI) {
        return;
    }

    try {
        // Show loading state
        state.readsViewPane?.setLoading(true, `Loading reads for sample '${sampleName}'...`);

        logger.debug(`[loadReadsForSample] Starting to load reads for '${sampleName}'`);

        // Get background API (sample data is in dedicated kernel)
        const api = await state.ensureBackgroundKernel();

        // Get read IDs and references in a single optimized batch query
        // (avoids two separate getVariable() calls which each add 3x kernel round-trips)
        const { readIds, references } = await Promise.race([
            api.getReadIdsAndReferencesForSample(sampleName),
            new Promise<{ readIds: string[]; references: string[] }>((_, reject) =>
                setTimeout(
                    () =>
                        reject(
                            new Error(
                                `Timeout loading reads and references for sample '${sampleName}' after 30 seconds`
                            )
                        ),
                    30000
                )
            ),
        ]);

        logger.debug(
            `[loadReadsForSample] Got ${readIds.length} reads and ${references.length} references for sample '${sampleName}'`
        );

        if (readIds.length === 0) {
            state.readsViewPane?.setReads([]);
            vscode.window.showWarningMessage(`No reads found in sample '${sampleName}'`);
            return;
        }

        if (references && references.length > 0) {
            // Sample has BAM - show references only (lazy load mode)
            // Fetch all reference read counts in a single optimized batch query
            logger.debug(
                `[loadReadsForSample] Loading read counts for ${references.length} references...`
            );
            const refCounts: { referenceName: string; readCount: number }[] = [];

            const readCounts = await Promise.race([
                api.getReadsCountForAllReferencesSample(sampleName),
                new Promise<{ [ref: string]: number }>((_, reject) =>
                    setTimeout(
                        () =>
                            reject(
                                new Error(
                                    `Timeout loading reference read counts for sample '${sampleName}' after 30 seconds`
                                )
                            ),
                        30000
                    )
                ),
            ]);

            for (const refName of references) {
                const count = readCounts[refName] || 0;
                logger.debug(`[loadReadsForSample] Reference '${refName}' has ${count} reads`);
                refCounts.push({
                    referenceName: refName,
                    readCount: count,
                });
            }

            state.readsViewPane?.setReferencesOnly(refCounts);
        } else {
            // Sample has only POD5 - show flat list of reads
            state.readsViewPane?.setReads(readIds);
        }
    } catch (error) {
        logger.error(`Failed to load reads for sample '${sampleName}':`, error);
        vscode.window.showErrorMessage(`Failed to load reads for sample: ${error}`);
    } finally {
        state.readsViewPane?.setLoading(false);
    }
}

/**
 * Ensure squiggy package is available (check if installed, show guidance if not)
 */
async function ensureSquiggyAvailable(state: ExtensionState): Promise<boolean> {
    if (!state.usePositron) {
        // Non-Positron mode - assume squiggy is available via subprocess backend
        return true;
    }

    const packageManager = state.packageManager;
    if (!packageManager) {
        return false;
    }

    // Don't prompt repeatedly in the same session
    if (state.squiggyInstallChecked && state.squiggyInstallDeclined) {
        return false;
    }

    try {
        // Check if package is installed and compatible
        const available = await packageManager.verifyPackage();

        if (available) {
            state.squiggyInstallChecked = true;
            state.squiggyInstallDeclined = false;
            return true;
        } else {
            // verifyPackage() already showed appropriate error message
            state.squiggyInstallChecked = true;
            state.squiggyInstallDeclined = true;
            return false;
        }
    } catch (_error) {
        // Error during check - mark as unavailable
        state.squiggyInstallChecked = true;
        state.squiggyInstallDeclined = true;
        return false;
    }
}

/**
 * Open a POD5 file
 * Uses FileLoadingService for centralized loading and unified state integration
 */
async function openPOD5File(filePath: string, state: ExtensionState): Promise<void> {
    logger.info(`Loading POD5 file: ${path.basename(filePath)}`);

    // Ensure squiggy is available (check if installed, prompt if needed)
    const squiggyAvailable = await ensureSquiggyAvailable(state);

    if (!squiggyAvailable) {
        vscode.window.showWarningMessage(
            'Cannot open POD5 file: squiggy-positron Python package is not installed. ' +
                'Please install it with: uv pip install squiggy-positron'
        );
        return;
    }

    await safeExecuteWithProgress(
        async () => {
            // Use FileLoadingService for consistent file loading
            const service = new FileLoadingService(state);
            const result = await service.loadFile(filePath, 'pod5');

            if (!result.success) {
                throw new Error(result.error || 'Failed to load POD5 file');
            }

            const pod5Result = result as POD5LoadResult;

            // Create LoadedItem for unified state
            const item: LoadedItem = {
                id: `pod5:${filePath}`,
                type: 'pod5',
                pod5Path: filePath,
                readCount: pod5Result.readCount,
                fileSize: pod5Result.fileSize,
                fileSizeFormatted: pod5Result.fileSizeFormatted,
                hasAlignments: false,
                hasReference: false,
                hasMods: false,
                hasEvents: false,
            };

            // Add to unified state (triggers onLoadedItemsChanged event)
            state.addLoadedItem(item);

            // Maintain legacy state for backward compatibility
            state.currentPod5File = filePath;

            // Get and display read IDs from dedicated kernel
            if (state.usePositron && state.squiggyAPI) {
                const api = await state.ensureBackgroundKernel();
                const readIds = await api.getReadIds(0, 1000);
                if (readIds.length > 0) {
                    state.readsViewPane?.setReads(readIds);
                }
            }

            // Update plot options to enable controls
            state.plotOptionsProvider?.updatePod5Status(true);

            logger.info(
                `Successfully loaded POD5 file: ${path.basename(filePath)} (${pod5Result.readCount.toLocaleString()} reads)`
            );
        },
        ErrorContext.POD5_LOAD,
        'Opening POD5 file...'
    );
}

/**
 * Open a BAM file
 * Uses FileLoadingService for centralized loading and unified state integration
 */
async function openBAMFile(filePath: string, state: ExtensionState): Promise<void> {
    logger.info(`Loading BAM file: ${path.basename(filePath)}`);

    // Ensure squiggy is available (check if installed, prompt if needed)
    const squiggyAvailable = await ensureSquiggyAvailable(state);

    if (!squiggyAvailable) {
        vscode.window.showWarningMessage(
            'Cannot open BAM file: squiggy-positron Python package is not installed. ' +
                'Please install it with: uv pip install squiggy-positron'
        );
        return;
    }

    await safeExecuteWithProgress(
        async () => {
            // Use FileLoadingService for consistent file loading
            const service = new FileLoadingService(state);
            const result = await service.loadFile(filePath, 'bam');

            if (!result.success) {
                throw new Error(result.error || 'Failed to load BAM file');
            }

            const bamResult = result as BAMLoadResult;

            // Update unified state with BAM file info
            // First, get the current POD5 item if it exists
            const currentPod5 = state.currentPod5File;
            if (currentPod5) {
                const pod5Item = state.getLoadedItem(`pod5:${currentPod5}`);
                if (pod5Item) {
                    // Update existing POD5 item with BAM info
                    const updatedItem: LoadedItem = {
                        ...pod5Item,
                        bamPath: filePath,
                        hasAlignments: true,
                        hasMods: bamResult.hasModifications,
                        hasEvents: bamResult.hasEventAlignment,
                    };
                    state.addLoadedItem(updatedItem);
                }
            }

            // Maintain legacy state for backward compatibility
            state.currentBamFile = filePath;

            // Get references for lazy loading from dedicated kernel
            let referenceToReads: Record<string, string[]> = {};
            if (state.usePositron && state.squiggyAPI) {
                const api = await state.ensureBackgroundKernel();
                const references = await api.getReferences();
                for (const ref of references) {
                    const readCount = await api.client.getVariable(
                        `len(squiggy.io.squiggy_kernel._ref_mapping.get('${ref.replace(/'/g, "\\'")}', []))`
                    );
                    referenceToReads[ref] = new Array(readCount as number);
                }
            }

            // Update reads view to show references
            if (Object.keys(referenceToReads).length > 0) {
                const references = Object.entries(referenceToReads).map(
                    ([referenceName, reads]) => ({
                        referenceName,
                        readCount: Array.isArray(reads) ? reads.length : 0,
                    })
                );
                state.readsViewPane?.setReferencesOnly(references);
            }

            // Update modifications panel and context
            if (bamResult.hasModifications) {
                await vscode.commands.executeCommand(
                    'setContext',
                    'squiggy.hasModifications',
                    true
                );
                state.modificationsProvider?.setModificationInfo(
                    true,
                    bamResult.modificationTypes || [],
                    bamResult.hasProbabilities
                );
            } else {
                state.modificationsProvider?.clear();
                await vscode.commands.executeCommand(
                    'setContext',
                    'squiggy.hasModifications',
                    false
                );
            }

            // Update plot options
            state.plotOptionsProvider?.updateBamStatus(true);

            const references = Object.keys(referenceToReads);
            if (references.length > 0) {
                state.plotOptionsProvider?.updateReferences(references);
            }

            logger.info(
                `Successfully loaded BAM file: ${path.basename(filePath)} ` +
                    `(${references.length} reference${references.length !== 1 ? 's' : ''}${bamResult.hasModifications ? ', with modifications' : ''})`
            );
        },
        ErrorContext.BAM_LOAD,
        'Opening BAM file...'
    );
}

/**
 * Close POD5 file
 */
async function closePOD5File(state: ExtensionState): Promise<void> {
    try {
        // Clear Python state
        if (state.usePositron && state.positronClient) {
            await state.positronClient.executeSilent(`
import squiggy
squiggy.close_pod5()
`);
        }

        // Remove from unified state
        const currentPod5 = state.currentPod5File;
        if (currentPod5) {
            state.removeLoadedItem(`pod5:${currentPod5}`);
        }

        // Clear extension state (legacy)
        state.currentPod5File = undefined;

        // Clear UI
        state.readsViewPane?.setReads([]);
        state.plotOptionsProvider?.updatePod5Status(false);

        vscode.window.showInformationMessage('POD5 file closed');
    } catch (error) {
        handleError(error, ErrorContext.POD5_CLOSE);
    }
}

/**
 * Close BAM file
 */
async function closeBAMFile(state: ExtensionState): Promise<void> {
    try {
        // Clear Python state using squiggy.close_bam()
        if (state.usePositron && state.positronClient) {
            await state.positronClient.executeSilent(`
import squiggy
squiggy.close_bam()
`);
        }

        // Update unified state - remove BAM from POD5 item
        const currentPod5 = state.currentPod5File;
        if (currentPod5) {
            const pod5Item = state.getLoadedItem(`pod5:${currentPod5}`);
            if (pod5Item) {
                // Update to remove BAM association
                const updatedItem: LoadedItem = {
                    ...pod5Item,
                    bamPath: undefined,
                    hasAlignments: false,
                    hasMods: false,
                    hasEvents: false,
                };
                state.addLoadedItem(updatedItem);
            }
        }

        // Clear extension state (legacy)
        state.currentBamFile = undefined;

        // Clear UI
        state.modificationsProvider?.clear();
        state.plotOptionsProvider?.updateBamStatus(false);
        vscode.commands.executeCommand('setContext', 'squiggy.hasModifications', false);

        // If POD5 is still loaded, revert to flat read list
        if (state.currentPod5File && state.usePositron && state.squiggyAPI) {
            const readIds = await state.squiggyAPI.getReadIds(0, 1000);
            state.readsViewPane?.setReads(readIds);
        }

        vscode.window.showInformationMessage('BAM file closed');
    } catch (error) {
        handleError(error, ErrorContext.BAM_CLOSE);
    }
}

/**
 * Open a FASTA file
 * Uses FileLoadingService for centralized loading and unified state integration
 */
async function openFASTAFile(filePath: string, state: ExtensionState): Promise<void> {
    // Ensure squiggy is available (check if installed, prompt if needed)
    const squiggyAvailable = await ensureSquiggyAvailable(state);

    if (!squiggyAvailable) {
        vscode.window.showWarningMessage(
            'Cannot open FASTA file: squiggy-positron Python package is not installed. ' +
                'Please install it with: uv pip install squiggy-positron'
        );
        return;
    }

    await safeExecuteWithProgress(
        async () => {
            // Use FileLoadingService for consistent file loading
            const service = new FileLoadingService(state);
            const result = await service.loadFile(filePath, 'fasta');

            if (!result.success) {
                throw new Error(result.error || 'Failed to load FASTA file');
            }

            // Create LoadedItem for FASTA
            const item: LoadedItem = {
                id: `fasta:${filePath}`,
                type: 'pod5', // Store as pod5 type for now (represents sequence reference)
                pod5Path: filePath, // Use pod5Path field to store FASTA path
                readCount: 0,
                fileSize: result.fileSize,
                fileSizeFormatted: result.fileSizeFormatted,
                hasAlignments: false,
                hasReference: true,
                hasMods: false,
                hasEvents: false,
            };

            // Add to unified state
            state.addLoadedItem(item);

            // Maintain legacy state for backward compatibility
            state.currentFastaFile = filePath;

            // Update plot options to enable reference track
            state.plotOptionsProvider?.updateFastaStatus(true);

            logger.debug(`[openFASTAFile] Successfully loaded: ${path.basename(filePath)}`);
            vscode.window.showInformationMessage(`FASTA file loaded: ${path.basename(filePath)}`);
        },
        ErrorContext.FASTA_LOAD,
        'Opening FASTA file...'
    );
}

/**
 * Close FASTA file
 */
async function closeFASTAFile(state: ExtensionState): Promise<void> {
    try {
        // Clear Python state using squiggy.close_fasta()
        if (state.usePositron && state.positronClient) {
            await state.positronClient.executeSilent(`
import squiggy
squiggy.close_fasta()
`);
        }

        // Remove from unified state
        const currentFasta = state.currentFastaFile;
        if (currentFasta) {
            state.removeLoadedItem(`fasta:${currentFasta}`);
        }

        // Clear extension state (legacy)
        state.currentFastaFile = undefined;

        // Update plot options to disable reference track
        state.plotOptionsProvider?.updateFastaStatus(false);

        vscode.window.showInformationMessage('FASTA file closed');
    } catch (error) {
        handleError(error, ErrorContext.FASTA_CLOSE);
    }
}

/**
 * Load a sample (POD5 + optional BAM/FASTA) for multi-sample comparison
 * Phase 4 - Multi-sample comparison feature
 */
async function loadSampleForComparison(
    context: vscode.ExtensionContext,
    state: ExtensionState
): Promise<void> {
    // Prompt for sample name
    const sampleName = await vscode.window.showInputBox({
        placeHolder: 'e.g., model_v4.2, basecaller_v5.0',
        prompt: 'Enter a name for this sample',
        validateInput: (value) => {
            if (!value || value.trim().length === 0) {
                return 'Sample name cannot be empty';
            }
            if (state.getSample(value)) {
                return 'Sample name already exists';
            }
            return '';
        },
    });

    if (!sampleName) {
        return; // User cancelled
    }

    // Select POD5 file
    const pod5Uris = await vscode.window.showOpenDialog({
        canSelectMany: false,
        filters: { 'POD5 Files': ['pod5'] },
        title: 'Select POD5 file for sample',
    });

    if (!pod5Uris || !pod5Uris[0]) {
        return; // User cancelled
    }

    const pod5Path = pod5Uris[0].fsPath;

    // Optionally select BAM file
    const bamUris = await vscode.window.showOpenDialog({
        canSelectMany: false,
        filters: { 'BAM Files': ['bam'] },
        title: 'Select BAM file (optional)',
    });

    const bamPath = bamUris?.[0]?.fsPath;

    // Optionally select FASTA file
    const fastaUris = await vscode.window.showOpenDialog({
        canSelectMany: false,
        filters: { 'FASTA Files': ['fa', 'fasta', 'fna'] },
        title: 'Select FASTA file (optional)',
    });

    const fastaPath = fastaUris?.[0]?.fsPath;

    // Ensure squiggy is available
    if (!(await ensureSquiggyAvailable(state))) {
        vscode.window.showErrorMessage('Squiggy package not available');
        return;
    }

    // Load sample via FileLoadingService
    await safeExecuteWithProgress(
        async () => {
            if (!state.squiggyAPI) {
                throw new Error('SquiggyAPI not initialized');
            }

            // Use FileLoadingService to load sample into registry (for multi-sample support)
            const service = new FileLoadingService(state);
            const loadResult = await service.loadSampleIntoRegistry(
                sampleName,
                pod5Path,
                bamPath,
                fastaPath
            );

            // Get complete sample info including references (if BAM was loaded)
            let references = undefined;
            if (bamPath && state.squiggyAPI) {
                try {
                    const sampleInfo = await state.squiggyAPI.getSampleInfo(sampleName);
                    if (sampleInfo && sampleInfo.references) {
                        references = sampleInfo.references;
                    }
                } catch (error) {
                    console.warn(
                        `Failed to fetch reference info for sample '${sampleName}':`,
                        error
                    );
                }
            }

            // Extract metadata for UI
            const metadata = await service['extractFileMetadata'](pod5Path);
            const readCount = loadResult.numReads;

            // Create LoadedItem for unified state
            const item: LoadedItem = {
                id: `sample:${sampleName}`,
                type: 'sample',
                sampleName,
                pod5Path,
                bamPath,
                fastaPath,
                readCount,
                fileSize: metadata.fileSize,
                fileSizeFormatted: metadata.fileSizeFormatted,
                hasAlignments: !!bamPath,
                hasReference: !!fastaPath,
                hasMods: false, // Will be populated if BAM has mods
                hasEvents: false, // Will be populated if BAM has event alignment
            };

            // Add to unified state (triggers onLoadedItemsChanged event)
            state.addLoadedItem(item);

            // Maintain legacy state for backward compatibility
            state.addSample({
                sampleId: `sample:${sampleName}`,
                displayName: sampleName,
                pod5Path,
                bamPath,
                fastaPath,
                readCount,
                hasBam: !!bamPath,
                hasFasta: !!fastaPath,
                references, // Add reference information
                isLoaded: true,
                metadata: {
                    autoDetected: false,
                    sourceType: 'manual',
                },
            });

            // Auto-select newly loaded sample for visualization (Issue #124 fix)
            state.addSampleToVisualization(sampleName);

            // Reveal samples panel (subscribed to unified state, so no refresh needed)
            await vscode.commands.executeCommand('squiggyComparisonSamples.focus');
            await new Promise((resolve) => setTimeout(resolve, 100));

            vscode.window.showInformationMessage(
                `Sample '${sampleName}' loaded with ${readCount} reads`
            );
        },
        ErrorContext.POD5_LOAD,
        `Loading sample '${sampleName}'`
    );
}

/**
 * Load test multi-read datasets for comparison testing
 * Loads the same test data twice with different sample names for comparison
 *
 * Uses squiggy.get_test_data_path() to access bundled test data from the Python package.
 * This works even when the extension is packaged as .vsix since the test data is
 * included in the squiggy-positron Python package.
 */
async function loadTestMultiReadDataset(
    context: vscode.ExtensionContext,
    state: ExtensionState
): Promise<void> {
    // Ensure squiggy is available
    if (!(await ensureSquiggyAvailable(state))) {
        vscode.window.showErrorMessage('Squiggy package not available');
        return;
    }

    // Get test data paths from the Python package (bundled with squiggy-positron)
    // This is more reliable than using extension paths since test data is in the Python package
    let pod5Path: string;
    let bamPath: string;
    let fastaPath: string;

    try {
        // Execute Python code to get test data paths
        const getPathsCode = `
import squiggy
paths = {
    'pod5': squiggy.get_test_data_path('yeast_trna_reads.pod5'),
    'bam': squiggy.get_test_data_path('yeast_trna_mappings.bam'),
    'fasta': squiggy.get_test_data_path('yeast_trna.fa')
}
        `.trim();

        // Use the client directly to execute and get variable
        await state.positronClient?.executeSilent(getPathsCode);
        const paths = (await state.positronClient?.getVariable('paths')) as {
            pod5: string;
            bam: string;
            fasta: string;
        };
        pod5Path = paths.pod5;
        bamPath = paths.bam;
        fastaPath = paths.fasta;

        console.log('[loadTestMultiReadDataset] Got test data paths from Python package:', paths);
    } catch (error) {
        vscode.window.showErrorMessage(
            'Failed to get test data paths from squiggy package. ' +
                'Make sure squiggy-positron is installed: uv pip install squiggy-positron'
        );
        console.error('[loadTestMultiReadDataset] Error getting test data paths:', error);
        return;
    }

    // Load two samples for comparison (using same data with different names)
    const samples = [
        { name: 'Sample_A', pod5Path, bamPath, fastaPath },
        { name: 'Sample_B', pod5Path, bamPath, fastaPath },
    ];

    try {
        logger.debug('[loadTestMultiReadDataset] Starting to load samples...');

        for (const sample of samples) {
            // Skip if already loaded
            if (state.getSample(sample.name)) {
                logger.debug(
                    `[loadTestMultiReadDataset] Sample '${sample.name}' already loaded, skipping`
                );
                continue;
            }

            logger.debug(`[loadTestMultiReadDataset] Loading sample '${sample.name}'...`);

            await safeExecuteWithProgress(
                async () => {
                    if (!state.squiggyAPI) {
                        throw new Error('SquiggyAPI not initialized');
                    }

                    // Use FileLoadingService to load sample into registry (for multi-sample support)
                    const service = new FileLoadingService(state);
                    const loadResult = await service.loadSampleIntoRegistry(
                        sample.name,
                        sample.pod5Path,
                        sample.bamPath,
                        sample.fastaPath
                    );

                    // Get complete sample info including references (if BAM was loaded)
                    let references = undefined;
                    if (sample.bamPath && state.squiggyAPI) {
                        try {
                            const sampleInfo = await state.squiggyAPI.getSampleInfo(sample.name);
                            if (sampleInfo && sampleInfo.references) {
                                references = sampleInfo.references;
                            }
                        } catch (error) {
                            console.warn(
                                `Failed to fetch reference info for sample '${sample.name}':`,
                                error
                            );
                        }
                    }

                    // Extract metadata for UI
                    const metadata = await service['extractFileMetadata'](sample.pod5Path);
                    const readCount = loadResult.numReads;

                    // Create LoadedItem for unified state
                    const item: LoadedItem = {
                        id: `sample:${sample.name}`,
                        type: 'sample',
                        sampleName: sample.name,
                        pod5Path: sample.pod5Path,
                        bamPath: sample.bamPath,
                        fastaPath: sample.fastaPath,
                        readCount,
                        fileSize: metadata.fileSize,
                        fileSizeFormatted: metadata.fileSizeFormatted,
                        hasAlignments: !!sample.bamPath,
                        hasReference: !!sample.fastaPath,
                        hasMods: false, // Will be populated if BAM has mods
                        hasEvents: false, // Will be populated if BAM has event alignment
                    };

                    // Add to unified state (triggers onLoadedItemsChanged event)
                    state.addLoadedItem(item);

                    // Maintain legacy state for backward compatibility
                    state.addSample({
                        sampleId: `sample:${sample.name}`,
                        displayName: sample.name,
                        pod5Path: sample.pod5Path,
                        bamPath: sample.bamPath,
                        fastaPath: sample.fastaPath,
                        readCount,
                        hasBam: !!sample.bamPath,
                        hasFasta: !!sample.fastaPath,
                        references, // Add reference information
                        isLoaded: true,
                        metadata: {
                            // Note: autoDetected tracks whether files were matched by naming convention.
                            // Can be enhanced in Task 3.3 when sample loading dialog is implemented.
                            autoDetected: false,
                            sourceType: 'manual',
                        },
                    });

                    // Auto-select newly loaded sample for visualization (Issue #124 fix)
                    state.addSampleToVisualization(sample.name);

                    logger.debug(
                        `[loadTestMultiReadDataset] Sample '${sample.name}' added. Total:`,
                        state.getAllSampleNames()
                    );
                },
                ErrorContext.POD5_LOAD,
                `Loading sample '${sample.name}'`
            );
        }

        // Reveal samples panel (already subscribed to unified state)
        logger.debug('[loadTestMultiReadDataset] Focusing Sample Comparison Manager panel...');
        try {
            await vscode.commands.executeCommand('squiggyComparisonSamples.focus');
        } catch (error) {
            logger.error('[loadTestMultiReadDataset] Error focusing panel:', error);
        }

        // Brief delay for webview to initialize
        await new Promise((resolve) => setTimeout(resolve, 500));

        vscode.window.showInformationMessage(
            `Test multi-read datasets loaded: ${samples.map((s) => s.name).join(', ')}. ` +
                `Please ensure the "Sample Comparison Manager" panel is expanded in the Squiggy sidebar.`
        );
    } catch (error) {
        logger.error('[loadTestMultiReadDataset] Error:', error);
        handleError(error, ErrorContext.POD5_LOAD);
    }
}

/**
 * Load multiple samples from dropped files
 * Handles batch loading of POD5/BAM pairs with smart file matching
 */
async function loadSamplesFromDropped(
    context: vscode.ExtensionContext,
    state: ExtensionState,
    fileQueue: { pod5Path: string; bamPath?: string; sampleName: string }[]
): Promise<void> {
    logger.debug(
        `[loadSamplesFromDropped] Function called with ${fileQueue.length} samples to load`
    );
    if (fileQueue.length === 0) {
        return;
    }

    logger.debug(`[loadSamplesFromDropped] Starting to load ${fileQueue.length} samples...`);
    // Check if squiggy is available
    if (!(await ensureSquiggyAvailable(state))) {
        logger.debug(`[loadSamplesFromDropped] Squiggy not available, returning`);
        return;
    }

    const results = {
        successful: 0,
        failed: 0,
        skipped: 0,
    };

    for (let i = 0; i < fileQueue.length; i++) {
        const { pod5Path, bamPath, sampleName } = fileQueue[i];
        const progressMsg = `Loading sample ${i + 1} of ${fileQueue.length}: ${sampleName}...`;

        try {
            logger.debug(
                `[loadSamplesFromDropped] Starting load for sample: '${sampleName}' (${i + 1}/${fileQueue.length})`
            );
            await safeExecuteWithProgress(
                async () => {
                    logger.debug(
                        `[loadSamplesFromDropped] Inside safeExecuteWithProgress callback for '${sampleName}'`
                    );
                    // Validate files exist
                    try {
                        await fs.access(pod5Path);
                    } catch {
                        throw new Error(`POD5 file not found: ${pod5Path}`);
                    }

                    if (bamPath) {
                        try {
                            await fs.access(bamPath);
                        } catch {
                            throw new Error(`BAM file not found: ${bamPath}`);
                        }
                    }

                    // Use FileLoadingService to load sample into multi-sample registry
                    // This enables multi-sample comparisons by storing samples in the Python registry
                    logger.debug(
                        `[loadSamplesFromDropped] About to create FileLoadingService for sample '${sampleName}'`
                    );
                    const service = new FileLoadingService(state);
                    logger.debug(
                        `[loadSamplesFromDropped] FileLoadingService created, calling loadSampleIntoRegistry...`
                    );
                    const sampleResult = await service.loadSampleIntoRegistry(
                        sampleName,
                        pod5Path,
                        bamPath,
                        state.sessionFastaPath || undefined
                    );

                    // Create LoadedItem for unified state
                    const item: LoadedItem = {
                        id: `sample:${sampleName}`,
                        type: 'sample',
                        sampleName,
                        pod5Path,
                        bamPath,
                        fastaPath: state.sessionFastaPath || undefined,
                        readCount: sampleResult.numReads,
                        fileSize: 0, // File size metadata not available from registry
                        fileSizeFormatted: 'Unknown',
                        hasAlignments: sampleResult.hasBAM ?? false,
                        hasReference: sampleResult.hasFASTA ?? false,
                        hasMods: sampleResult.bamInfo?.hasModifications ?? false,
                        hasEvents: sampleResult.bamInfo?.hasEventAlignment ?? false,
                    };

                    // Add to unified state (triggers onLoadedItemsChanged event)
                    state.addLoadedItem(item);

                    // Maintain legacy state for backward compatibility
                    state.addSample({
                        sampleId: `sample:${sampleName}`,
                        displayName: sampleName,
                        pod5Path,
                        bamPath,
                        fastaPath: state.sessionFastaPath || undefined,
                        readCount: sampleResult.numReads,
                        hasBam: !!bamPath,
                        hasFasta: !!state.sessionFastaPath,
                        isLoaded: true,
                        metadata: {
                            autoDetected: false,
                            sourceType: 'manual',
                        },
                    });

                    // Auto-select newly loaded sample for visualization (Issue #124 fix)
                    state.addSampleToVisualization(sampleName);

                    // Auto-select and load reads in Read Explorer for first sample
                    // or on user's first interaction (state.selectedReadExplorerSample will be set later)
                    if (!state.selectedReadExplorerSample) {
                        state.selectedReadExplorerSample = sampleName;
                        // Delay to ensure sample is fully registered in Python registry
                        // (the loadSample() call is async and may not complete immediately)
                        setTimeout(() => {
                            logger.debug(
                                `[loadSamplesFromDropped] Auto-loading reads for first sample: '${sampleName}'`
                            );
                            Promise.resolve(
                                vscode.commands.executeCommand(
                                    'squiggy.internal.loadReadsForSample',
                                    sampleName
                                )
                            ).catch((err: unknown) => {
                                logger.error(
                                    `Failed to auto-load reads for sample '${sampleName}':`,
                                    err
                                );
                            });
                        }, 1500); // Wait 1.5s to ensure sample is registered
                    }

                    results.successful++;
                },
                ErrorContext.POD5_LOAD,
                progressMsg
            );
        } catch (error) {
            logger.error(`[loadSamplesFromDropped] Error loading ${sampleName}:`, error);
            results.failed++;
        }
    }

    // Samples panel already subscribed to unified state, so no manual refresh needed
    // But for safety during transition, we can still call refresh if it exists
    if (state.samplesProvider) {
        await state.samplesProvider.refresh();
    }

    // Refresh Read Explorer to update available samples dropdown
    logger.debug('[loadSamplesFromDropped] Refreshing Read Explorer with all samples');
    if (state.readsViewPane) {
        state.readsViewPane.refresh();
    }

    // Show summary
    let message = `Loaded ${results.successful} sample(s)`;
    if (results.failed > 0) {
        message += `, ${results.failed} failed`;
    }
    if (results.skipped > 0) {
        message += `, ${results.skipped} skipped`;
    }

    if (results.successful > 0) {
        vscode.window.showInformationMessage(message);
    } else {
        vscode.window.showErrorMessage(`Failed to load samples: ${message}`);
    }
}

/**
 * Load samples from file picker (reuses the same logic as drag-and-drop)
 */
async function loadSamplesFromFilePicker(
    context: vscode.ExtensionContext,
    state: ExtensionState,
    filePaths: string[]
): Promise<void> {
    logger.debug(`[loadSamplesFromFilePicker] Called with ${filePaths.length} file(s):`, filePaths);
    // Categorize files by extension (reuse same logic as drag-and-drop handler)
    const pod5Files: string[] = [];
    const bamFiles: string[] = [];

    for (const filePath of filePaths) {
        const ext = path.extname(filePath).toLowerCase();
        if (ext === '.pod5') {
            pod5Files.push(filePath);
        } else if (ext === '.bam') {
            bamFiles.push(filePath);
        }
    }

    if (pod5Files.length === 0) {
        vscode.window.showWarningMessage(
            'No POD5 files selected. POD5 files are required to load samples. ' +
                'Please select at least one POD5 file. You can select BAM files at the same time for auto-matching.'
        );
        return;
    }

    // Auto-match POD5 files to BAM files using stem matching
    const fileQueue: { pod5Path: string; bamPath?: string; sampleName: string }[] = [];

    for (const pod5Path of pod5Files) {
        const pod5Basename = path.basename(pod5Path, '.pod5');
        const pod5Stem = extractStem(pod5Basename);

        // Try to find matching BAM file
        let matchedBam: string | undefined;

        // First try: exact basename match
        for (const bamPath of bamFiles) {
            const bamBasename = path.basename(bamPath);
            if (bamBasename.startsWith(pod5Basename)) {
                matchedBam = bamPath;
                break;
            }
        }

        // Second try: stem match
        if (!matchedBam) {
            for (const bamPath of bamFiles) {
                const bamBasename = path.basename(bamPath, '.bam');
                if (extractStem(bamBasename) === pod5Stem) {
                    matchedBam = bamPath;
                    break;
                }
            }
        }

        fileQueue.push({
            pod5Path,
            bamPath: matchedBam,
            sampleName: pod5Stem,
        });
    }

    // Load samples using the same logic as dropped files
    // Sample naming will be handled in the Sample Manager UI, not during file loading
    logger.debug(
        `[loadSamplesFromFilePicker] Calling loadSamplesFromDropped with ${fileQueue.length} samples`
    );
    await loadSamplesFromDropped(context, state, fileQueue);
    logger.debug(`[loadSamplesFromFilePicker] loadSamplesFromDropped completed`);
}

/**
 * Extract stem from filename (everything before first non-word character)
 */
function extractStem(filename: string): string {
    // Remove extension first
    const withoutExt = filename.replace(/\.[^.]*$/, '');
    // Extract stem: everything up to first non-alphanumeric non-underscore
    const match = withoutExt.match(/^([a-zA-Z0-9_]+)/);
    return match ? match[1] : withoutExt;
}

/**
 * Update BAM/FASTA files for a sample
 */
async function updateSampleFiles(
    sampleName: string,
    files: { bamPath?: string; fastaPath?: string },
    state: ExtensionState
): Promise<void> {
    logger.debug(`[updateSampleFiles] Updating files for sample '${sampleName}':`, files);

    // Get sample from state
    const sample = state.getSample(sampleName);
    if (!sample) {
        vscode.window.showErrorMessage(`Sample "${sampleName}" not found`);
        return;
    }

    try {
        // Update Python backend via squiggy API
        if (!(await ensureSquiggyAvailable(state))) {
            return;
        }

        const service = new FileLoadingService(state);

        // Reload the sample with new files
        // This will update the multi-sample registry in Python
        await service.loadSampleIntoRegistry(
            sampleName,
            sample.pod5Path,
            files.bamPath || sample.bamPath,
            files.fastaPath || sample.fastaPath
        );

        vscode.window.showInformationMessage(`Updated files for sample "${sampleName}"`);

        logger.debug(`[updateSampleFiles] Successfully updated files for '${sampleName}'`);
    } catch (error) {
        logger.error(`[updateSampleFiles] Error updating files for '${sampleName}':`, error);
        vscode.window.showErrorMessage(
            `Failed to update files for sample "${sampleName}": ${error instanceof Error ? error.message : String(error)}`
        );
    }
}
