/**
 * Centralized Extension State
 *
 * Manages all extension state including loaded files, UI panel references,
 * backend instances, and installation flags. Provides a single source of truth
 * for extension state across all modules.
 */

import * as vscode from 'vscode';
import { PositronRuntimeClient } from '../backend/positron-runtime-client';
import { SquiggyRuntimeAPI } from '../backend/squiggy-runtime-api';
import { PackageManager } from '../backend/package-manager';
import { PythonBackend } from '../backend/squiggy-python-backend';
import { ReadsViewPane } from '../views/squiggy-reads-view-pane';
import { PlotOptionsViewProvider } from '../views/squiggy-plot-options-view';
import { FilePanelProvider } from '../views/squiggy-file-panel';
import { ModificationsPanelProvider } from '../views/squiggy-modifications-panel';
import { SamplesPanelProvider } from '../views/squiggy-samples-panel';

/**
 * Information about a loaded sample (POD5 + optional BAM/FASTA)
 */
export interface SampleInfo {
    name: string;
    pod5Path: string;
    bamPath?: string;
    fastaPath?: string;
    readCount: number;
    hasBam: boolean;
    hasFasta: boolean;
}

/**
 * Centralized state manager for the extension
 */
export class ExtensionState {
    // Backend instances
    private _positronClient?: PositronRuntimeClient;
    private _squiggyAPI?: SquiggyRuntimeAPI;
    private _packageManager?: PackageManager;
    private _pythonBackend?: PythonBackend | null;
    private _usePositron: boolean = false;

    // UI panel providers
    private _readsViewPane?: ReadsViewPane;
    private _plotOptionsProvider?: PlotOptionsViewProvider;
    private _filePanelProvider?: FilePanelProvider;
    private _modificationsProvider?: ModificationsPanelProvider;
    private _samplesProvider?: SamplesPanelProvider;

    // File state
    private _currentPod5File?: string;
    private _currentBamFile?: string;
    private _currentFastaFile?: string;
    private _currentPlotReadIds?: string[];

    // Multi-sample state (Phase 4)
    private _loadedSamples: Map<string, SampleInfo> = new Map();
    private _selectedSamplesForComparison: string[] = [];

    // Installation state
    private _squiggyInstallChecked: boolean = false;
    private _squiggyInstallDeclined: boolean = false;

    // VSCode context
    private _extensionContext?: vscode.ExtensionContext;

    /**
     * Initialize backends (Positron or subprocess fallback)
     */
    async initializeBackends(context: vscode.ExtensionContext): Promise<void> {
        this._extensionContext = context;

        // Try Positron runtime first
        this._positronClient = new PositronRuntimeClient();
        this._usePositron = this._positronClient.isAvailable();

        if (this._usePositron) {
            // Use Positron runtime
            this._squiggyAPI = new SquiggyRuntimeAPI(this._positronClient);
            this._packageManager = new PackageManager(this._positronClient);
        } else {
            // Fallback to subprocess JSON-RPC
            const pythonPath = this.getPythonPath();
            const serverPath = context.asAbsolutePath('src/python/server.py');
            this._pythonBackend = new PythonBackend(pythonPath, serverPath);

            try {
                await this._pythonBackend.start();

                // Register cleanup on deactivation
                context.subscriptions.push({
                    dispose: () => this._pythonBackend?.stop(),
                });
            } catch (error) {
                vscode.window.showErrorMessage(
                    `Failed to start Python backend: ${error}. ` +
                        `Please ensure Python is installed and the squiggy package is available.`
                );
            }
        }
    }

    /**
     * Initialize UI panel providers
     */
    initializePanels(
        readsViewPane: ReadsViewPane,
        plotOptionsProvider: PlotOptionsViewProvider,
        filePanelProvider: FilePanelProvider,
        modificationsProvider: ModificationsPanelProvider,
        samplesProvider?: SamplesPanelProvider
    ): void {
        this._readsViewPane = readsViewPane;
        this._plotOptionsProvider = plotOptionsProvider;
        this._filePanelProvider = filePanelProvider;
        this._modificationsProvider = modificationsProvider;
        this._samplesProvider = samplesProvider;
    }

    /**
     * Clear all extension state (files, UI, kernel variables)
     */
    async clearAll(): Promise<void> {
        // Clear file state
        this._currentPod5File = undefined;
        this._currentBamFile = undefined;
        this._currentFastaFile = undefined;
        this._currentPlotReadIds = undefined;

        // Reset installation check flags
        this._squiggyInstallChecked = false;
        this._squiggyInstallDeclined = false;

        // Clear UI panels
        this._readsViewPane?.setReads([]);
        this._filePanelProvider?.clearPOD5();
        this._filePanelProvider?.clearBAM();
        this._filePanelProvider?.clearFASTA?.();
        this._plotOptionsProvider?.updateBamStatus(false);

        // Clear Python kernel state if using Positron
        if (this._usePositron && this._positronClient) {
            try {
                await this._positronClient.executeSilent(`
import squiggy
from squiggy.io import _squiggy_session
# Close all resources via session
_squiggy_session.close_all()
# Also call module-level cleanup functions
squiggy.close_pod5()
squiggy.close_bam()
squiggy.close_fasta()
`);
            } catch (_error) {
                // Ignore errors if kernel is not running
            }
        }
    }

    /**
     * Get Python interpreter path from VSCode settings
     */
    private getPythonPath(): string {
        const config = vscode.workspace.getConfiguration('python');
        const pythonPath = config.get<string>('defaultInterpreterPath');
        return pythonPath || 'python3';
    }

    // ========== Getters ==========

    get positronClient(): PositronRuntimeClient | undefined {
        return this._positronClient;
    }

    get squiggyAPI(): SquiggyRuntimeAPI | undefined {
        return this._squiggyAPI;
    }

    get packageManager(): PackageManager | undefined {
        return this._packageManager;
    }

    get pythonBackend(): PythonBackend | null | undefined {
        return this._pythonBackend;
    }

    get usePositron(): boolean {
        return this._usePositron;
    }

    get readsViewPane(): ReadsViewPane | undefined {
        return this._readsViewPane;
    }

    get plotOptionsProvider(): PlotOptionsViewProvider | undefined {
        return this._plotOptionsProvider;
    }

    get filePanelProvider(): FilePanelProvider | undefined {
        return this._filePanelProvider;
    }

    get modificationsProvider(): ModificationsPanelProvider | undefined {
        return this._modificationsProvider;
    }

    get samplesProvider(): SamplesPanelProvider | undefined {
        return this._samplesProvider;
    }

    get currentPod5File(): string | undefined {
        return this._currentPod5File;
    }

    set currentPod5File(value: string | undefined) {
        this._currentPod5File = value;
    }

    get currentBamFile(): string | undefined {
        return this._currentBamFile;
    }

    set currentBamFile(value: string | undefined) {
        this._currentBamFile = value;
    }

    get currentFastaFile(): string | undefined {
        return this._currentFastaFile;
    }

    set currentFastaFile(value: string | undefined) {
        this._currentFastaFile = value;
    }

    get currentPlotReadIds(): string[] | undefined {
        return this._currentPlotReadIds;
    }

    set currentPlotReadIds(value: string[] | undefined) {
        this._currentPlotReadIds = value;
    }

    get squiggyInstallChecked(): boolean {
        return this._squiggyInstallChecked;
    }

    set squiggyInstallChecked(value: boolean) {
        this._squiggyInstallChecked = value;
    }

    get squiggyInstallDeclined(): boolean {
        return this._squiggyInstallDeclined;
    }

    set squiggyInstallDeclined(value: boolean) {
        this._squiggyInstallDeclined = value;
    }

    get extensionContext(): vscode.ExtensionContext | undefined {
        return this._extensionContext;
    }

    // ========== Multi-Sample Management (Phase 4) ==========

    get loadedSamples(): Map<string, SampleInfo> {
        return this._loadedSamples;
    }

    getSample(name: string): SampleInfo | undefined {
        return this._loadedSamples.get(name);
    }

    addSample(sample: SampleInfo): void {
        this._loadedSamples.set(sample.name, sample);
    }

    removeSample(name: string): void {
        this._loadedSamples.delete(name);
    }

    getAllSampleNames(): string[] {
        return Array.from(this._loadedSamples.keys());
    }

    get selectedSamplesForComparison(): string[] {
        return this._selectedSamplesForComparison;
    }

    set selectedSamplesForComparison(value: string[]) {
        this._selectedSamplesForComparison = value;
    }

    /**
     * Add sample to comparison selection
     */
    addSampleToComparison(sampleName: string): void {
        if (!this._selectedSamplesForComparison.includes(sampleName)) {
            this._selectedSamplesForComparison.push(sampleName);
        }
    }

    /**
     * Remove sample from comparison selection
     */
    removeSampleFromComparison(sampleName: string): void {
        this._selectedSamplesForComparison = this._selectedSamplesForComparison.filter(
            (name) => name !== sampleName
        );
    }

    /**
     * Clear comparison selection
     */
    clearComparisonSelection(): void {
        this._selectedSamplesForComparison = [];
    }
}
