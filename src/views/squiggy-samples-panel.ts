/**
 * Sample Comparison Panel Webview View
 *
 * Manages loaded samples for multi-sample comparison with selection,
 * comparison workflow, and sample management (unload).
 * Subscribes to unified extension state for cross-panel synchronization.
 */

import * as vscode from 'vscode';
import * as path from 'path';
import { BaseWebviewProvider } from './base-webview-provider';
import {
    SamplesIncomingMessage,
    UpdateSamplesMessage,
    SampleItem,
    ClearSamplesMessage,
    UpdateSessionFastaMessage,
} from '../types/messages';
import { ExtensionState } from '../state/extension-state';
import { LoadedItem } from '../types/loaded-item';
// SampleInfo and formatFileSize unused - reserved for future features

export class SamplesPanelProvider extends BaseWebviewProvider {
    public static readonly viewType = 'squiggyComparisonSamples';

    private _state: ExtensionState;
    private _samples: SampleItem[] = [];
    private _selectedSamples: Set<string> = new Set();
    private _disposables: vscode.Disposable[] = [];

    // Event emitter for comparison requests
    private _onDidRequestComparison = new vscode.EventEmitter<string[]>();
    public readonly onDidRequestComparison = this._onDidRequestComparison.event;

    // Event emitter for sample unload requests
    private _onDidRequestUnload = new vscode.EventEmitter<string>();
    public readonly onDidRequestUnload = this._onDidRequestUnload.event;

    // Store pending maxReads value for comparison
    private _pendingMaxReads: number | null = null;

    getPendingMaxReads(): number | null {
        const value = this._pendingMaxReads;
        this._pendingMaxReads = null; // Clear after reading
        return value;
    }

    constructor(extensionUri: vscode.Uri, state: ExtensionState) {
        super(extensionUri);
        this._state = state;

        // Subscribe to unified state changes
        const itemsDisposable = this._state.onLoadedItemsChanged((items: LoadedItem[]) => {
            this._handleLoadedItemsChanged(items);
        });
        this._disposables.push(itemsDisposable);

        // Subscribe to comparison selection changes
        const comparisonDisposable = this._state.onComparisonChanged((ids: string[]) => {
            this._handleComparisonChanged(ids);
        });
        this._disposables.push(comparisonDisposable);
    }

    /**
     * Handle unified state changes - filter samples and update display
     * @private
     */
    private _handleLoadedItemsChanged(items: LoadedItem[]): void {
        // Filter for samples only (type: 'sample')
        this._samples = items
            .filter((item) => item.type === 'sample')
            .map((item) => ({
                name: item.sampleName!,
                pod5Path: item.pod5Path,
                bamPath: item.bamPath,
                fastaPath: item.fastaPath,
                readCount: item.readCount,
                hasBam: item.hasAlignments,
                hasFasta: item.hasReference,
            }));

        console.log(
            '[SamplesPanelProvider] Unified state changed, now showing',
            this._samples.length,
            'samples'
        );
        this.updateView();
    }

    /**
     * Handle comparison selection changes from unified state
     * @private
     */
    private _handleComparisonChanged(ids: string[]): void {
        // Filter for sample IDs and extract sample names
        this._selectedSamples = new Set(
            ids.filter((id) => id.startsWith('sample:')).map((id) => id.substring(7))
        );

        console.log('[SamplesPanelProvider] Comparison selection changed:', Array.from(this._selectedSamples));
        this.updateView();
    }

    /**
     * Dispose method to clean up subscriptions
     */
    public dispose(): void {
        // Clean up event subscriptions
        for (const disposable of this._disposables) {
            disposable.dispose();
        }
        this._disposables = [];
    }

    protected getTitle(): string {
        return 'Sample Comparison Manager';
    }

    protected async handleMessage(message: SamplesIncomingMessage): Promise<void> {
        switch (message.type) {
            case 'ready':
                // Webview is ready, send initial state
                this.updateView();
                break;

            case 'selectSample':
                if (message.selected) {
                    this._selectedSamples.add(message.sampleName);
                } else {
                    this._selectedSamples.delete(message.sampleName);
                }
                break;

            case 'startComparison':
                if (message.sampleNames.length >= 2) {
                    // Store maxReads for the command handler to access
                    this._pendingMaxReads = message.maxReads || null;
                    this._onDidRequestComparison.fire(message.sampleNames);
                } else {
                    vscode.window.showWarningMessage(
                        'Please select at least 2 samples for comparison'
                    );
                }
                break;

            case 'unloadSample': {
                // Ask for confirmation
                const confirm = await vscode.window.showWarningMessage(
                    `Unload sample "${message.sampleName}"?`,
                    { modal: true },
                    'Yes',
                    'Cancel'
                );

                if (confirm === 'Yes') {
                    this._onDidRequestUnload.fire(message.sampleName);
                    this._selectedSamples.delete(message.sampleName);
                }
                break;
            }

            case 'filesDropped':
                await this.handleFilesDropped(message.filePaths);
                break;

            case 'requestSetSessionFasta':
                await vscode.commands.executeCommand('squiggy.setSessionFasta');
                break;

            case 'requestLoadSamples':
                await vscode.commands.executeCommand('squiggy.loadSamplesFromUI');
                break;

            case 'setSessionFasta':
                this._state.setSessionFasta(message.fastaPath);
                break;
        }
    }

    protected updateView(): void {
        console.log('[SamplesPanelProvider] updateView called');
        console.log('[SamplesPanelProvider] _view exists:', !!this._view);

        if (!this._view) {
            console.log('[SamplesPanelProvider] No view to update');
            return;
        }

        // Rebuild samples list from extension state
        const sampleNames = this._state.getAllSampleNames();
        console.log('[SamplesPanelProvider] Sample names from state:', sampleNames);

        this._samples = Array.from(sampleNames)
            .map((name) => {
                const sampleInfo = this._state.getSample(name);
                console.log(`[SamplesPanelProvider] Sample '${name}' info:`, sampleInfo);
                if (!sampleInfo) {
                    return null;
                }

                const sampleItem: SampleItem = {
                    name: sampleInfo.name,
                    pod5Path: sampleInfo.pod5Path,
                    bamPath: sampleInfo.bamPath,
                    fastaPath: sampleInfo.fastaPath,
                    readCount: sampleInfo.readCount,
                    hasBam: sampleInfo.hasBam,
                    hasFasta: sampleInfo.hasFasta,
                };
                return sampleItem;
            })
            .filter((item): item is SampleItem => item !== null);

        console.log('[SamplesPanelProvider] Built samples array:', this._samples);

        if (this._samples.length === 0) {
            console.log('[SamplesPanelProvider] Sending clearSamples message');
            const message: ClearSamplesMessage = {
                type: 'clearSamples',
            };
            this.postMessage(message);
        } else {
            console.log(
                '[SamplesPanelProvider] Sending updateSamples message with',
                this._samples.length,
                'samples'
            );
            const message: UpdateSamplesMessage = {
                type: 'updateSamples',
                samples: this._samples,
            };
            this.postMessage(message);
        }
    }

    /**
     * Update samples display when samples are added
     */
    public async refresh(): Promise<void> {
        // If view doesn't exist yet, try to show it
        if (!this._view) {
            console.log('[SamplesPanelProvider] View not yet created, showing panel...');
            try {
                await vscode.commands.executeCommand('squiggyComparisonSamples.focus');
                // Wait a bit for view to be created
                await new Promise((resolve) => setTimeout(resolve, 500));
            } catch (error) {
                console.error('[SamplesPanelProvider] Error showing panel:', error);
            }
        }

        this.updateView();
    }

    /**
     * Get currently selected sample names
     */
    public getSelectedSamples(): string[] {
        return Array.from(this._selectedSamples);
    }

    /**
     * Clear selection
     */
    public clearSelection(): void {
        this._selectedSamples.clear();
    }

    /**
     * Update session FASTA and notify webview
     */
    public updateSessionFasta(fastaPath: string | null): void {
        const message: UpdateSessionFastaMessage = {
            type: 'updateSessionFasta',
            fastaPath,
        };
        this.postMessage(message);
    }

    /**
     * Handle dropped files - parse and auto-match POD5/BAM pairs
     */
    private async handleFilesDropped(filePaths: string[]): Promise<void> {
        console.log('🎯 DEBUG: handleFilesDropped called with paths:', filePaths);
        if (filePaths.length === 0) {
            console.log('🎯 DEBUG: No file paths provided');
            return;
        }

        try {
            // Separate files by extension
            const filesByExt = this.categorizeFiles(filePaths);

            if (filesByExt.pod5Files.length === 0) {
                vscode.window.showWarningMessage('No POD5 files found in dropped files');
                return;
            }

            // Auto-match POD5 files to BAM files
            const fileQueue = this.matchFilePairs(filesByExt.pod5Files, filesByExt.bamFiles);

            if (fileQueue.length === 0) {
                vscode.window.showWarningMessage('No valid file pairs to load');
                return;
            }

            // If FASTA files found and no session FASTA set, offer to set one
            if (filesByExt.fastaFiles.length > 0 && !this._state.sessionFastaPath) {
                const setFasta = await vscode.window.showQuickPick(
                    ['Yes, set default FASTA', 'No, skip for now'],
                    {
                        placeHolder: `Found FASTA file: ${path.basename(filesByExt.fastaFiles[0])}`,
                        canPickMany: false,
                    }
                );

                if (setFasta === 'Yes, set default FASTA') {
                    this._state.setSessionFasta(filesByExt.fastaFiles[0]);
                    this.updateSessionFasta(filesByExt.fastaFiles[0]);
                }
            }

            // Prompt user to confirm sample names before loading
            const confirmed = await this.confirmSampleNames(fileQueue);
            if (confirmed.length === 0) {
                return; // User cancelled
            }

            // Load samples via command
            await vscode.commands.executeCommand('squiggy.loadSamplesFromDropped', confirmed);
        } catch (error) {
            vscode.window.showErrorMessage(`Error handling dropped files: ${error}`);
        }
    }

    /**
     * Categorize files by extension
     */
    private categorizeFiles(filePaths: string[]): {
        pod5Files: string[];
        bamFiles: string[];
        fastaFiles: string[];
    } {
        const pod5Files: string[] = [];
        const bamFiles: string[] = [];
        const fastaFiles: string[] = [];

        for (const filePath of filePaths) {
            const ext = path.extname(filePath).toLowerCase();

            if (ext === '.pod5') {
                pod5Files.push(filePath);
            } else if (ext === '.bam') {
                bamFiles.push(filePath);
            } else if (['.fasta', '.fa', '.fna'].includes(ext)) {
                fastaFiles.push(filePath);
            }
        }

        return { pod5Files, bamFiles, fastaFiles };
    }

    /**
     * Match POD5 files to BAM files using intelligent pattern matching
     */
    private matchFilePairs(
        pod5Files: string[],
        bamFiles: string[]
    ): { pod5Path: string; bamPath?: string; sampleName: string }[] {
        const pairs = [];

        for (const pod5Path of pod5Files) {
            const pod5Basename = path.basename(pod5Path, '.pod5');
            const pod5Stem = this.extractStem(pod5Basename);

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

            // Second try: stem match (e.g., "cys_subset" matches "cys_subset.aln.bam")
            if (!matchedBam) {
                for (const bamPath of bamFiles) {
                    const bamBasename = path.basename(bamPath, '.bam');
                    if (this.extractStem(bamBasename) === pod5Stem) {
                        matchedBam = bamPath;
                        break;
                    }
                }
            }

            pairs.push({
                pod5Path,
                bamPath: matchedBam,
                sampleName: pod5Stem,
            });
        }

        return pairs;
    }

    /**
     * Extract stem from filename (everything before first non-word character)
     * e.g., "cys_subset" from "cys_subset.pod5" or "cys_subset.aln.bam"
     */
    private extractStem(filename: string): string {
        // Remove extension first
        const withoutExt = filename.replace(/\.[^.]*$/, '');
        // Extract stem: everything up to first non-alphanumeric non-underscore
        const match = withoutExt.match(/^([a-zA-Z0-9_]+)/);
        return match ? match[1] : withoutExt;
    }

    /**
     * Prompt user to confirm and customize sample names
     */
    private async confirmSampleNames(
        fileQueue: { pod5Path: string; bamPath?: string; sampleName: string }[]
    ): Promise<{ pod5Path: string; bamPath?: string; sampleName: string }[]> {
        const confirmed: { pod5Path: string; bamPath?: string; sampleName: string }[] = [];

        for (let i = 0; i < fileQueue.length; i++) {
            const item = fileQueue[i];
            const suggestion = item.sampleName;

            // Show input box with suggested name
            const customName = await vscode.window.showInputBox({
                prompt: `Sample ${i + 1}/${fileQueue.length}: Enter name for ${path.basename(item.pod5Path)}`,
                value: suggestion,
                validateInput: (value) => {
                    if (!value.trim()) {
                        return 'Sample name cannot be empty';
                    }
                    // Check for duplicate names in current queue
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
                return [];
            }

            confirmed.push({
                ...item,
                sampleName: customName.trim(),
            });
        }

        return confirmed;
    }
}
