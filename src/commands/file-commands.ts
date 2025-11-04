/**
 * File Commands
 *
 * Handles opening/closing POD5 and BAM files, plus loading test data.
 * Extracted from extension.ts to improve modularity.
 */

import * as vscode from 'vscode';
import * as path from 'path';
import { promises as fs } from 'fs';
import { ExtensionState } from '../state/extension-state';
import { ErrorContext, handleError, safeExecuteWithProgress } from '../utils/error-handler';

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
            const fastaPath = path.join(context.extensionPath, 'tests', 'data', 'yeast_trna.fa');

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
            const fileUris = await vscode.window.showOpenDialog({
                canSelectMany: true,
                filters: { 'Sequence Files': ['pod5', 'bam'], 'All Files': ['*'] },
                title: 'Select POD5 and BAM files to load',
            });

            if (!fileUris || fileUris.length === 0) {
                return;
            }

            const filePaths = fileUris.map((uri) => uri.fsPath);

            // Delegate to the samples panel's file handling logic
            const samplesProvider = state.samplesProvider;
            if (samplesProvider) {
                // The samples panel will handle categorizing and matching files
                // We need to access its private method via message or use the same logic
                // For now, use the setFilesForLoading approach
                // Actually, we need to call a public method on samplesProvider
                // Let's create a new public method that handles file paths directly
                await loadSamplesFromFilePicker(context, state, filePaths);
            }
        })
    );
}

/**
 * Ensure squiggy package is available (check if installed, prompt if needed)
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

    // Always check if squiggy is installed (user may have installed manually)
    const installed = await packageManager.isSquiggyInstalled();

    if (installed) {
        // Package is installed - return success
        state.squiggyInstallChecked = true;
        state.squiggyInstallDeclined = false; // Reset declined flag since it's now installed
        return true;
    }

    // Not installed - check if we should prompt
    if (state.squiggyInstallChecked && state.squiggyInstallDeclined) {
        // User already declined this session - don't prompt again
        return false;
    }

    try {
        // Prompt user to install
        const userChoice = await packageManager.promptInstallSquiggy();

        if (userChoice === 'install') {
            // Install squiggy
            const extensionPath = state.extensionContext?.extensionPath || '';
            const success = await packageManager.installSquiggyWithProgress(extensionPath);
            state.squiggyInstallChecked = true;
            return success;
        } else if (userChoice === 'manual') {
            // Show manual installation guide
            const extensionPath = state.extensionContext?.extensionPath || '';
            await packageManager.showManualInstallationGuide(extensionPath);
            state.squiggyInstallDeclined = true;
            state.squiggyInstallChecked = true;
            return false;
        } else {
            // User canceled installation
            state.squiggyInstallDeclined = true;
            state.squiggyInstallChecked = true;
            return false;
        }
    } catch (_error) {
        // Error during check - mark as unavailable
        state.squiggyInstallChecked = true;
        return false;
    }
}

/**
 * Open a POD5 file
 */
async function openPOD5File(filePath: string, state: ExtensionState): Promise<void> {
    // Ensure squiggy is available (check if installed, prompt if needed)
    const squiggyAvailable = await ensureSquiggyAvailable(state);

    if (!squiggyAvailable) {
        vscode.window.showWarningMessage(
            'Cannot open POD5 file: squiggy Python package is not installed. ' +
                'Please install it manually with: pip install -e <extension-path>'
        );
        return;
    }

    await safeExecuteWithProgress(
        async () => {
            let numReads: number;
            let readIds: string[] = [];

            if (state.usePositron && state.squiggyAPI) {
                // Use Positron kernel - lazy load read IDs
                const result = await state.squiggyAPI.loadPOD5(filePath);
                numReads = result.numReads;

                // Get first 1000 read IDs for tree view (lazy loading)
                readIds = await state.squiggyAPI.getReadIds(0, 1000);
            } else if (state.pythonBackend) {
                // Use subprocess backend
                const backendResult = (await state.pythonBackend.call('open_pod5', {
                    file_path: filePath,
                })) as { num_reads: number; read_ids?: string[] };
                numReads = backendResult.num_reads;
                readIds = backendResult.read_ids || [];
            } else {
                throw new Error('No backend available');
            }

            // Update reads view
            if (readIds.length > 0) {
                state.readsViewPane?.setReads(readIds);
            }

            // Track file and update file panel display
            state.currentPod5File = filePath;

            // Get file size
            const stats = await fs.stat(filePath);

            state.filePanelProvider?.setPOD5({
                path: filePath,
                numReads,
                size: stats.size,
            });
        },
        ErrorContext.POD5_LOAD,
        'Opening POD5 file...'
    );
}

/**
 * Open a BAM file
 */
async function openBAMFile(filePath: string, state: ExtensionState): Promise<void> {
    // Ensure squiggy is available (check if installed, prompt if needed)
    const squiggyAvailable = await ensureSquiggyAvailable(state);

    if (!squiggyAvailable) {
        vscode.window.showWarningMessage(
            'Cannot open BAM file: squiggy Python package is not installed. ' +
                'Please install it manually with: pip install -e <extension-path>'
        );
        return;
    }

    await safeExecuteWithProgress(
        async () => {
            let numReads: number;
            let hasModifications: boolean;
            let modificationTypes: string[];
            let hasProbabilities: boolean;
            let hasEventAlignment: boolean = false;
            let referenceToReads: Record<string, string[]> = {};

            if (state.usePositron && state.squiggyAPI) {
                // Use Positron kernel - lazy load reference mapping
                const result = await state.squiggyAPI.loadBAM(filePath);
                numReads = result.numReads;
                hasModifications = result.hasModifications;
                modificationTypes = result.modificationTypes;
                hasProbabilities = result.hasProbabilities;
                hasEventAlignment = result.hasEventAlignment || false;

                // Get references and build mapping (lazy loading)
                const references = await state.squiggyAPI.getReferences();
                for (const ref of references) {
                    const reads = await state.squiggyAPI.getReadsForReference(ref);
                    referenceToReads[ref] = reads;
                }
            } else if (state.pythonBackend) {
                // Use subprocess backend
                const backendResult = (await state.pythonBackend.call('open_bam', {
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
            state.currentBamFile = filePath;

            // Get file size
            const stats = await fs.stat(filePath);

            state.filePanelProvider?.setBAM({
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
                state.readsViewPane?.setReadsGrouped(readItemsMap);
            }

            // Update modifications panel and context
            if (hasModifications) {
                // Set context FIRST to make panel visible
                await vscode.commands.executeCommand(
                    'setContext',
                    'squiggy.hasModifications',
                    true
                );
                // Then update the panel data (panel is now visible)
                state.modificationsProvider?.setModificationInfo(
                    hasModifications,
                    modificationTypes,
                    hasProbabilities
                );
            } else {
                state.modificationsProvider?.clear();
                await vscode.commands.executeCommand(
                    'setContext',
                    'squiggy.hasModifications',
                    false
                );
            }

            // Update plot options to show EVENTALIGN mode and set as default
            state.plotOptionsProvider?.updateBamStatus(true);
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

        // Clear extension state
        state.currentPod5File = undefined;

        // Clear UI
        state.filePanelProvider?.clearPOD5();
        state.readsViewPane?.setReads([]);

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

        // Clear extension state
        state.currentBamFile = undefined;

        // Clear UI
        state.filePanelProvider?.clearBAM();
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
 */
async function openFASTAFile(filePath: string, state: ExtensionState): Promise<void> {
    // Ensure squiggy is available (check if installed, prompt if needed)
    const squiggyAvailable = await ensureSquiggyAvailable(state);

    if (!squiggyAvailable) {
        vscode.window.showWarningMessage(
            'Cannot open FASTA file: squiggy Python package is not installed. ' +
                'Please install it manually with: pip install -e <extension-path>'
        );
        return;
    }

    await safeExecuteWithProgress(
        async () => {
            if (state.usePositron && state.squiggyAPI) {
                // Use Positron kernel - validate FASTA and index
                await state.squiggyAPI.loadFASTA(filePath);
            } else if (state.pythonBackend) {
                // Use subprocess backend - validate FASTA file
                await state.pythonBackend.call('load_fasta', {
                    file_path: filePath,
                });
            } else {
                throw new Error('No backend available');
            }

            // Track file in state
            state.currentFastaFile = filePath;

            // Get file size
            const stats = await fs.stat(filePath);

            // Update file panel to show FASTA file
            state.filePanelProvider?.setFASTA?.({
                path: filePath,
                size: stats.size,
            });

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

        // Clear extension state
        state.currentFastaFile = undefined;

        // Clear UI
        state.filePanelProvider?.clearFASTA?.();

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

    // Load sample via API
    await safeExecuteWithProgress(
        async () => {
            if (!state.squiggyAPI) {
                throw new Error('SquiggyAPI not initialized');
            }

            const result = await state.squiggyAPI.loadSample(
                sampleName,
                pod5Path,
                bamPath,
                fastaPath
            );

            // Update extension state
            state.addSample({
                name: sampleName,
                pod5Path,
                bamPath,
                fastaPath,
                readCount: result.numReads,
                hasBam: !!bamPath,
                hasFasta: !!fastaPath,
            });

            // Reveal and refresh samples panel
            await vscode.commands.executeCommand('squiggyComparisonSamples.focus');
            await new Promise((resolve) => setTimeout(resolve, 100));
            state.samplesProvider?.refresh();

            vscode.window.showInformationMessage(
                `Sample '${sampleName}' loaded with ${result.numReads} reads`
            );
        },
        ErrorContext.POD5_LOAD,
        `Loading sample '${sampleName}'`
    );
}

/**
 * Load test multi-read datasets for comparison testing
 * Loads the same test data twice with different sample names for comparison
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

    const pod5Path = path.join(context.extensionPath, 'tests', 'data', 'yeast_trna_reads.pod5');
    const bamPath = path.join(context.extensionPath, 'tests', 'data', 'yeast_trna_mappings.bam');
    const fastaPath = path.join(context.extensionPath, 'tests', 'data', 'yeast_trna.fa');

    // Load two samples for comparison (using same data with different names)
    const samples = [
        { name: 'Sample_A', pod5Path, bamPath, fastaPath },
        { name: 'Sample_B', pod5Path, bamPath, fastaPath },
    ];

    try {
        console.log('[loadTestMultiReadDataset] Starting to load samples...');
        console.log('[loadTestMultiReadDataset] state.squiggyAPI:', state.squiggyAPI);
        console.log('[loadTestMultiReadDataset] state.samplesProvider:', state.samplesProvider);

        for (const sample of samples) {
            // Skip if already loaded
            if (state.getSample(sample.name)) {
                console.log(
                    `[loadTestMultiReadDataset] Sample '${sample.name}' already loaded, skipping`
                );
                continue;
            }

            console.log(`[loadTestMultiReadDataset] Loading sample '${sample.name}'...`);

            await safeExecuteWithProgress(
                async () => {
                    if (!state.squiggyAPI) {
                        throw new Error('SquiggyAPI not initialized');
                    }

                    console.log(
                        `[loadTestMultiReadDataset] Calling loadSample for '${sample.name}'...`
                    );
                    const result = await state.squiggyAPI.loadSample(
                        sample.name,
                        sample.pod5Path,
                        sample.bamPath,
                        sample.fastaPath
                    );
                    console.log(`[loadTestMultiReadDataset] loadSample result:`, result);

                    // Update extension state
                    const sampleInfo = {
                        name: sample.name,
                        pod5Path: sample.pod5Path,
                        bamPath: sample.bamPath,
                        fastaPath: sample.fastaPath,
                        readCount: result.numReads,
                        hasBam: true,
                        hasFasta: true,
                    };
                    console.log(`[loadTestMultiReadDataset] Adding sample to state:`, sampleInfo);
                    state.addSample(sampleInfo);
                    console.log(
                        `[loadTestMultiReadDataset] Sample added. Total samples:`,
                        state.getAllSampleNames()
                    );
                },
                ErrorContext.POD5_LOAD,
                `Loading sample '${sample.name}'`
            );
        }

        // Reveal and refresh samples panel
        console.log('[loadTestMultiReadDataset] Refreshing samples panel...');
        console.log('[loadTestMultiReadDataset] All samples in state:', state.getAllSampleNames());

        // Show the samples panel in the sidebar (this creates the webview if not already visible)
        console.log('[loadTestMultiReadDataset] Focusing Sample Comparison Manager panel...');
        try {
            await vscode.commands.executeCommand('squiggyComparisonSamples.focus');
            console.log('[loadTestMultiReadDataset] Focus command completed');
        } catch (error) {
            console.error('[loadTestMultiReadDataset] Error focusing panel:', error);
        }

        // Longer delay to ensure webview is created before refreshing
        console.log('[loadTestMultiReadDataset] Waiting for panel to initialize...');
        await new Promise((resolve) => setTimeout(resolve, 500));

        console.log('[loadTestMultiReadDataset] Calling refresh...');
        state.samplesProvider?.refresh();

        vscode.window.showInformationMessage(
            `Test multi-read datasets loaded: ${samples.map((s) => s.name).join(', ')}. ` +
                `Please ensure the "Sample Comparison Manager" panel is expanded in the Squiggy sidebar.`
        );
    } catch (error) {
        console.error('[loadTestMultiReadDataset] Error:', error);
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
    if (fileQueue.length === 0) {
        return;
    }

    // Check if squiggy is available
    if (!(await ensureSquiggyAvailable(state))) {
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
            await safeExecuteWithProgress(
                async () => {
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

                    // Load via Python API
                    const result = await state.squiggyAPI!.loadSample(
                        sampleName,
                        pod5Path,
                        bamPath,
                        state.sessionFastaPath || undefined
                    );

                    // Update extension state
                    state.addSample({
                        name: sampleName,
                        pod5Path,
                        bamPath,
                        fastaPath: state.sessionFastaPath || undefined,
                        readCount: result.numReads,
                        hasBam: !!bamPath,
                        hasFasta: !!state.sessionFastaPath,
                    });

                    results.successful++;
                },
                ErrorContext.POD5_LOAD,
                progressMsg
            );
        } catch (error) {
            console.error(`[loadSamplesFromDropped] Error loading ${sampleName}:`, error);
            results.failed++;
        }
    }

    // Refresh samples panel with all loaded samples
    if (state.samplesProvider) {
        state.samplesProvider.refresh();
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
        vscode.window.showWarningMessage('No POD5 files selected');
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

    // Prompt user to confirm and customize sample names
    const confirmed: { pod5Path: string; bamPath?: string; sampleName: string }[] = [];

    for (let i = 0; i < fileQueue.length; i++) {
        const item = fileQueue[i];
        const suggestion = item.sampleName;

        const customName = await vscode.window.showInputBox({
            prompt: `Sample ${i + 1}/${fileQueue.length}: Enter name for ${path.basename(item.pod5Path)}`,
            value: suggestion,
            validateInput: (value) => {
                if (!value.trim()) {
                    return 'Sample name cannot be empty';
                }
                if (
                    confirmed.filter((c) => c.sampleName === value).length > 0 ||
                    fileQueue.slice(i + 1).filter((f) => f.sampleName === value).length > 0
                ) {
                    return 'Sample name already used';
                }
                return null;
            },
        });

        if (customName === undefined) {
            // User cancelled
            return;
        }

        confirmed.push({
            ...item,
            sampleName: customName.trim(),
        });
    }

    // Load samples using the same logic as dropped files
    await loadSamplesFromDropped(context, state, confirmed);
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
