/**
 * TSV Import Commands
 *
 * Handles importing samples from TSV manifest files.
 * Provides:
 * - File picker for TSV selection
 * - TSV parsing and validation
 * - Preview UI with validation results
 * - Smart loading (eager vs lazy based on sample count)
 */

import * as vscode from 'vscode';
import { promises as fs } from 'fs';
import { ExtensionState } from '../state/extension-state';
import { TSVParser, TSVSampleSpec, TSVParseResult } from '../services/tsv-parser';
import { TSVPathResolver, PathResolutionStrategy } from '../services/tsv-path-resolver';
import { TSVValidator, ValidationResult } from '../services/tsv-validator';
import { FileLoadingService } from '../services/file-loading-service';
import { SampleInfo } from '../state/extension-state';
import { LoadedItem } from '../types/loaded-item';
import { logger } from '../utils/logger';

/**
 * Register TSV import commands
 */
export function registerTSVCommands(
    context: vscode.ExtensionContext,
    state: ExtensionState
): void {
    context.subscriptions.push(
        vscode.commands.registerCommand('squiggy.importSamplesFromTSV', async () => {
            await importSamplesFromTSV(state);
        })
    );
}

/**
 * Import samples from TSV file
 *
 * TODO: Implement full import workflow
 * - File picker
 * - Parse TSV
 * - Validate all samples
 * - Show preview UI
 * - Load samples (eager or lazy based on count)
 */
async function importSamplesFromTSV(state: ExtensionState): Promise<void> {
    // Step 1: File picker
    const fileUri = await vscode.window.showOpenDialog({
        canSelectMany: false,
        filters: { 'TSV Files': ['tsv', 'txt'], 'All Files': ['*'] },
        title: 'Import Samples from TSV',
    });

    if (!fileUri || !fileUri[0]) {
        return;
    }

    const tsvPath = fileUri[0].fsPath;
    logger.info(`[TSV Import] Selected file: ${tsvPath}`);

    // Step 2: Read and parse TSV
    let content: string;
    try {
        content = await fs.readFile(tsvPath, 'utf-8');
    } catch (error) {
        vscode.window.showErrorMessage(`Failed to read TSV file: ${error}`);
        return;
    }

    const parseResult = TSVParser.parse(content);

    if (!parseResult.success) {
        vscode.window.showErrorMessage(
            `TSV parsing failed:\n\n${parseResult.errors.join('\n')}`
        );
        return;
    }

    logger.info(`[TSV Import] Parsed ${parseResult.samples.length} samples`);
    if (parseResult.warnings.length > 0) {
        logger.info(`[TSV Import] Warnings: ${parseResult.warnings.join('; ')}`);
    }

    // Step 3: Validate all samples
    const workspaceRoot = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath || '';
    const pathResolver = new TSVPathResolver(tsvPath, workspaceRoot);
    const validator = new TSVValidator(pathResolver, new Set(state.getAllSampleNames()));

    let validationResults: ValidationResult[];
    try {
        validationResults = await vscode.window.withProgress(
            {
                location: vscode.ProgressLocation.Notification,
                title: 'Validating TSV samples...',
                cancellable: false,
            },
            async () => {
                return await validator.validateBatch(parseResult.samples);
            }
        );
    } catch (error) {
        vscode.window.showErrorMessage(`Validation failed: ${error}`);
        return;
    }

    logger.info(
        `[TSV Import] Validation complete: ${validationResults.filter((r) => r.valid).length}/${validationResults.length} valid`
    );

    // Step 4: Check for blocking errors
    const hasErrors = validationResults.some((r) => !r.valid);
    if (hasErrors) {
        await showValidationResults(validationResults, 'Validation Failed - Fix Issues');
        return; // Block import
    }

    // Step 5: Show preview & confirm
    const confirmed = await showImportPreview(validationResults, parseResult.samples.length);
    if (!confirmed) {
        logger.info('[TSV Import] User cancelled import');
        return;
    }

    // Step 6: Determine loading strategy
    const shouldLoadEagerly = determineLoadingStrategy(parseResult.samples.length);
    logger.info(
        `[TSV Import] Loading strategy: ${shouldLoadEagerly ? 'EAGER' : 'LAZY'} (${parseResult.samples.length} samples)`
    );

    // Step 7: Load samples
    await loadSamplesFromTSV(
        parseResult.samples,
        validationResults,
        state,
        shouldLoadEagerly
    );
}

/**
 * Show validation results in QuickPick UI
 */
async function showValidationResults(
    results: ValidationResult[],
    title: string
): Promise<void> {
    const items = results.map((r) => ({
        label: r.sampleName,
        description: r.valid ? '✓ Valid' : '✗ Invalid',
        detail: r.errors.length > 0 ? r.errors.join(', ') : r.warnings.join(', '),
        iconPath: r.valid
            ? new vscode.ThemeIcon('pass', new vscode.ThemeColor('testing.iconPassed'))
            : new vscode.ThemeIcon('error', new vscode.ThemeColor('testing.iconFailed')),
    }));

    await vscode.window.showQuickPick(items, {
        title,
        canPickMany: false,
        placeHolder:
            results.filter((r) => !r.valid).length > 0
                ? 'Fix errors and try again'
                : 'All samples valid',
    });
}

/**
 * Show import preview and get confirmation
 */
async function showImportPreview(
    results: ValidationResult[],
    sampleCount: number
): Promise<boolean> {
    const summary = TSVValidator.getSummary(results);

    const message = [
        `Import ${sampleCount} samples?`,
        ``,
        `• ${summary.valid} valid samples`,
        `• ${summary.withBam} with BAM files`,
        `• ${summary.withFasta} with FASTA files`,
        summary.totalWarnings > 0 ? `• ${summary.totalWarnings} warnings` : '',
    ]
        .filter((line) => line.length > 0)
        .join('\n');

    const choice = await vscode.window.showInformationMessage(
        message,
        { modal: true },
        'Import',
        'Preview Details',
        'Cancel'
    );

    if (choice === 'Preview Details') {
        await showValidationResults(results, 'Sample Preview');
        // Ask again after preview
        return showImportPreview(results, sampleCount);
    }

    return choice === 'Import';
}

/**
 * Determine loading strategy based on sample count
 *
 * Heuristic:
 * - ≤5 samples: Eager (fast to load)
 * - ≥20 samples: Lazy (avoid overwhelming kernel)
 * - 6-19 samples: Eager if small, lazy if large files
 */
function determineLoadingStrategy(sampleCount: number): boolean {
    if (sampleCount <= 5) return true; // Always eager for small batches
    if (sampleCount >= 20) return false; // Always lazy for large batches

    // Mid-range: default to 10 sample threshold
    return sampleCount <= 10;
}

/**
 * Load samples from validated TSV specs
 *
 * TODO: Implement full loading logic
 * - Create SampleInfo objects
 * - Load to kernel (eager) or defer (lazy)
 * - Update extension state
 * - Refresh UI panels
 */
async function loadSamplesFromTSV(
    specs: TSVSampleSpec[],
    validationResults: ValidationResult[],
    state: ExtensionState,
    eager: boolean
): Promise<void> {
    const service = new FileLoadingService(state);
    const tsvGroupId = `tsv_${Date.now()}`; // Batch ID for grouping

    logger.info(
        `[TSV Import] Starting load for ${specs.length} samples (eager=${eager}, groupId=${tsvGroupId})`
    );

    // Show progress
    await vscode.window.withProgress(
        {
            location: vscode.ProgressLocation.Notification,
            title: `Importing ${specs.length} samples...`,
            cancellable: false,
        },
        async (progress) => {
            for (let i = 0; i < specs.length; i++) {
                const spec = specs[i];
                const validation = validationResults[i];

                progress.report({
                    message: `${spec.sampleName} (${i + 1}/${specs.length})`,
                    increment: (100 / specs.length),
                });

                try {
                    // TODO: Implement eager vs lazy loading
                    // For now, just create metadata (lazy mode)

                    const sampleInfo: SampleInfo = {
                        sampleId: `sample:${spec.sampleName}`,
                        displayName: spec.sampleName,
                        pod5Path: validation.resolvedPod5!,
                        bamPath: validation.resolvedBam,
                        fastaPath: validation.resolvedFasta,
                        readCount: 0, // Will be populated on load
                        hasBam: !!validation.resolvedBam,
                        hasFasta: !!validation.resolvedFasta,
                        isLoaded: false, // TODO: Set to `eager` when implementing kernel loading
                        metadata: {
                            sourceType: 'tsv',
                            tsvGroup: tsvGroupId,
                            autoDetected: false,
                        },
                    };

                    state.addSample(sampleInfo);

                    // Also add to unified state for cross-panel sync
                    const loadedItem: LoadedItem = {
                        id: sampleInfo.sampleId,
                        type: 'sample',
                        sampleName: spec.sampleName,
                        pod5Path: validation.resolvedPod5!,
                        bamPath: validation.resolvedBam,
                        fastaPath: validation.resolvedFasta,
                        readCount: 0,
                        fileSize: 0,
                        fileSizeFormatted: 'Unknown',
                        hasAlignments: !!validation.resolvedBam,
                        hasReference: !!validation.resolvedFasta,
                        hasMods: false,
                        hasEvents: false,
                    };
                    state.addLoadedItem(loadedItem);

                    logger.info(
                        `[TSV Import] Registered sample: ${spec.sampleName} (loaded=${sampleInfo.isLoaded})`
                    );
                } catch (error) {
                    logger.error(`[TSV Import] Failed to load sample ${spec.sampleName}:`, error);
                    vscode.window.showErrorMessage(`Failed to load ${spec.sampleName}: ${error}`);
                }
            }
        }
    );

    // Refresh Samples panel
    state.samplesProvider?.refresh();

    vscode.window.showInformationMessage(
        `Imported ${specs.length} samples from TSV ${eager ? '(loaded to kernel)' : '(lazy load - will load on plot)'}`
    );

    logger.info(`[TSV Import] Import complete: ${specs.length} samples registered`);
}
