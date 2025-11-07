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
import { ModificationsPanelProvider } from '../views/squiggy-modifications-panel';
import { SamplesPanelProvider } from '../views/squiggy-samples-panel';
import { SessionState, SampleSessionState } from '../types/squiggy-session-types';
import { LoadedItem } from '../types/loaded-item';
import { SessionStateManager } from './session-state-manager';
import { PathResolver } from './path-resolver';
import { FileResolver } from './file-resolver';
import { logger } from '../utils/logger';

/**
 * Information about a loaded sample (POD5 + optional BAM/FASTA)
 *
 * Phase 3 refactor: Designed to support future TSV import and sample management.
 * - `sampleId`: Unique identifier (can be UUID or file-based)
 * - `displayName`: User-facing name (editable), separate from POD5 filename
 * - `pod5Path`: Single POD5 file (Phase 3), extensible to array for multi-POD5 future
 * - `isLoaded`: Tracks kernel state (prepares for lazy loading in #79 TSV import)
 * - `metadata`: Extensible object for future attributes without interface changes
 */
export interface SampleInfo {
    // Core identifiers
    sampleId: string; // Unique ID (can be UUID or derived from pod5 path)
    displayName: string; // User-facing name (editable in Sample Manager)

    // File associations
    pod5Path: string; // Single POD5 per sample (Phase 3)
    bamPath?: string;
    fastaPath?: string;

    // File metadata
    readCount: number;
    hasBam: boolean;
    hasFasta: boolean;

    // Kernel state (for lazy loading)
    isLoaded: boolean; // Whether files are loaded into kernel

    // Extensible metadata (for future features without refactoring)
    metadata?: {
        // Sample identification
        autoDetected?: boolean; // Was sample auto-detected from file names?

        // UI preferences
        displayColor?: string; // Hex or CSS color for plot rendering

        // TSV import tracking (future #79)
        sourceType?: 'manual' | 'tsv'; // Origin of sample (manual UI or TSV import)
        tsvGroup?: string; // Batch grouping if loaded from same TSV

        // Additional notes or tags (extensible)
        tags?: string[];
    };
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
    private _modificationsProvider?: ModificationsPanelProvider;
    private _samplesProvider?: SamplesPanelProvider;

    // File state
    private _currentPod5File?: string;
    private _currentBamFile?: string;
    private _currentFastaFile?: string;
    private _currentPlotReadIds?: string[];

    // Lazy loading context
    private _pod5LoadContext?: {
        currentOffset: number;
        pageSize: number;
        totalReads: number;
    };

    // Multi-sample state (Phase 4)
    private _loadedSamples: Map<string, SampleInfo> = new Map();
    private _selectedSamplesForComparison: string[] = [];
    private _samplesForVisualization: Set<string> = new Set(); // Samples selected for plotting
    private _sessionFastaPath: string | null = null; // Session-level FASTA for all comparisons
    private _selectedReadExplorerSample: string | null = null; // Currently selected sample in Read Explorer

    // ========== UNIFIED STATE (Issue #92) ==========
    // Consolidated registry replacing fragmented state silos
    private _loadedItems: Map<string, LoadedItem> = new Map();
    private _selectedItemIds: Set<string> = new Set();
    private _itemsForComparison: Set<string> = new Set();

    // Event emitters for cross-panel synchronization
    private _onLoadedItemsChanged: vscode.EventEmitter<LoadedItem[]> = new vscode.EventEmitter();
    private _onSelectionChanged: vscode.EventEmitter<string[]> = new vscode.EventEmitter();
    private _onComparisonChanged: vscode.EventEmitter<string[]> = new vscode.EventEmitter();
    private _onVisualizationSelectionChanged: vscode.EventEmitter<string[]> =
        new vscode.EventEmitter();

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
        modificationsProvider: ModificationsPanelProvider,
        samplesProvider?: SamplesPanelProvider
    ): void {
        this._readsViewPane = readsViewPane;
        this._plotOptionsProvider = plotOptionsProvider;
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
        this._plotOptionsProvider?.updatePod5Status(false);
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

    get selectedReadExplorerSample(): string | null {
        return this._selectedReadExplorerSample;
    }

    set selectedReadExplorerSample(value: string | null) {
        this._selectedReadExplorerSample = value;
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

    get pod5LoadContext():
        | { currentOffset: number; pageSize: number; totalReads: number }
        | undefined {
        return this._pod5LoadContext;
    }

    set pod5LoadContext(
        value: { currentOffset: number; pageSize: number; totalReads: number } | undefined
    ) {
        this._pod5LoadContext = value;
    }

    // ========== UNIFIED STATE EVENTS (Issue #92) ==========

    /**
     * Event fired when loaded items change (add/remove)
     * Listened to by: File Panel, Samples Panel, Reads View, etc.
     */
    get onLoadedItemsChanged(): vscode.Event<LoadedItem[]> {
        return this._onLoadedItemsChanged.event;
    }

    /**
     * Event fired when selection changes
     * Listened to by: UI panels for local selection state
     */
    get onSelectionChanged(): vscode.Event<string[]> {
        return this._onSelectionChanged.event;
    }

    /**
     * Event fired when comparison selection changes
     * Listened to by: Samples Panel for comparison mode
     */
    get onComparisonChanged(): vscode.Event<string[]> {
        return this._onComparisonChanged.event;
    }

    /**
     * Event fired when visualization selection changes (samples selected for plotting)
     * Listened to by: Samples Panel and Plotting Panel for UI synchronization
     */
    get onVisualizationSelectionChanged(): vscode.Event<string[]> {
        return this._onVisualizationSelectionChanged.event;
    }

    // ========== UNIFIED ITEM MANAGEMENT (Issue #92) ==========

    /**
     * Add or update a loaded item in the unified registry
     * Fires onLoadedItemsChanged event to notify all listeners
     *
     * @param item - LoadedItem to add or update
     */
    addLoadedItem(item: LoadedItem): void {
        this._loadedItems.set(item.id, item);
        this._notifyLoadedItemsChanged();
    }

    /**
     * Remove a loaded item from the unified registry
     * Also removes from selection sets and fires notifications
     *
     * @param id - Item ID to remove
     */
    removeLoadedItem(id: string): void {
        this._loadedItems.delete(id);
        // Also remove from selections
        this._selectedItemIds.delete(id);
        this._itemsForComparison.delete(id);
        this._notifyLoadedItemsChanged();
        this._notifySelectionChanged();
        this._notifyComparisonChanged();
    }

    /**
     * Get all loaded items
     * @returns Array of all LoadedItem objects
     */
    getLoadedItems(): LoadedItem[] {
        return Array.from(this._loadedItems.values());
    }

    /**
     * Get a specific item by ID
     * @param id - Item ID
     * @returns LoadedItem if found, undefined otherwise
     */
    getLoadedItem(id: string): LoadedItem | undefined {
        return this._loadedItems.get(id);
    }

    /**
     * Clear all loaded items (e.g., on reset)
     */
    clearLoadedItems(): void {
        this._loadedItems.clear();
        this._selectedItemIds.clear();
        this._itemsForComparison.clear();
        this._notifyLoadedItemsChanged();
        this._notifySelectionChanged();
        this._notifyComparisonChanged();
    }

    // ========== SELECTION MANAGEMENT (Issue #92) ==========

    /**
     * Update selection in UI (e.g., checkbox clicked)
     * @param ids - Array of selected item IDs
     */
    setSelectedItems(ids: string[]): void {
        this._selectedItemIds = new Set(ids);
        this._notifySelectionChanged();
    }

    /**
     * Get currently selected items
     * @returns Array of selected item IDs
     */
    getSelectedItems(): string[] {
        return Array.from(this._selectedItemIds);
    }

    /**
     * Toggle selection of an item
     * @param id - Item ID to toggle
     */
    toggleItemSelection(id: string): void {
        if (this._selectedItemIds.has(id)) {
            this._selectedItemIds.delete(id);
        } else {
            this._selectedItemIds.add(id);
        }
        this._notifySelectionChanged();
    }

    /**
     * Check if an item is selected
     * @param id - Item ID to check
     * @returns true if selected, false otherwise
     */
    isItemSelected(id: string): boolean {
        return this._selectedItemIds.has(id);
    }

    // ========== COMPARISON MANAGEMENT (Issue #92) ==========

    /**
     * Update items selected for comparison
     * @param ids - Array of item IDs for comparison
     */
    setComparisonItems(ids: string[]): void {
        this._itemsForComparison = new Set(ids);
        this._notifyComparisonChanged();
    }

    /**
     * Get items selected for comparison
     * @returns Array of item IDs for comparison
     */
    getComparisonItems(): string[] {
        return Array.from(this._itemsForComparison);
    }

    /**
     * Add item to comparison selection
     * @param id - Item ID to add
     */
    addToComparison(id: string): void {
        this._itemsForComparison.add(id);
        this._notifyComparisonChanged();
    }

    /**
     * Remove item from comparison selection
     * @param id - Item ID to remove
     */
    removeFromComparison(id: string): void {
        this._itemsForComparison.delete(id);
        this._notifyComparisonChanged();
    }

    /**
     * Clear comparison selection
     */
    clearComparison(): void {
        this._itemsForComparison.clear();
        this._notifyComparisonChanged();
    }

    // ========== PRIVATE NOTIFICATION HELPERS ==========

    /**
     * Notify all listeners that loaded items changed
     * @private
     */
    private _notifyLoadedItemsChanged(): void {
        this._onLoadedItemsChanged.fire(this.getLoadedItems());
    }

    /**
     * Notify all listeners that selection changed
     * @private
     */
    private _notifySelectionChanged(): void {
        this._onSelectionChanged.fire(this.getSelectedItems());
    }

    /**
     * Notify all listeners that comparison selection changed
     * @private
     */
    private _notifyComparisonChanged(): void {
        this._onComparisonChanged.fire(this.getComparisonItems());
    }

    /**
     * Notify all listeners that visualization selection changed
     * @private
     */
    private _notifyVisualizationSelectionChanged(): void {
        this._onVisualizationSelectionChanged.fire(this.getSamplesForVisualization());
    }

    // ========== Multi-Sample Management (Phase 4) ==========

    get loadedSamples(): Map<string, SampleInfo> {
        return this._loadedSamples;
    }

    getSample(nameOrId: string): SampleInfo | undefined {
        // First try direct lookup by sampleId
        const bySampleId = this._loadedSamples.get(nameOrId);
        if (bySampleId) {
            return bySampleId;
        }

        // Fall back to search by displayName
        for (const sample of this._loadedSamples.values()) {
            if (sample.displayName === nameOrId) {
                return sample;
            }
        }

        return undefined;
    }

    addSample(sample: SampleInfo): void {
        // Use sampleId as the key to preserve insertion order during renames
        // displayName can be edited in Sample Manager without moving the sample in the list
        this._loadedSamples.set(sample.sampleId, sample);
    }

    removeSample(name: string): void {
        // Support removal by displayName (for backward compatibility) or by sampleId
        // First try direct lookup by sampleId
        if (this._loadedSamples.has(name)) {
            this._loadedSamples.delete(name);
            return;
        }

        // Fall back to search by displayName
        for (const [sampleId, sample] of this._loadedSamples) {
            if (sample.displayName === name) {
                this._loadedSamples.delete(sampleId);
                return;
            }
        }
    }

    getAllSampleNames(): string[] {
        // Return displayNames (user-facing names) in insertion order
        return Array.from(this._loadedSamples.values()).map((sample) => sample.displayName);
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

    /**
     * Add sample to visualization selection (for plotting)
     */
    addSampleToVisualization(sampleName: string): void {
        this._samplesForVisualization.add(sampleName);
        this._notifyVisualizationSelectionChanged();
    }

    /**
     * Remove sample from visualization selection
     */
    removeSampleFromVisualization(sampleName: string): void {
        this._samplesForVisualization.delete(sampleName);
        this._notifyVisualizationSelectionChanged();
    }

    /**
     * Set visualization selection (bulk update)
     */
    setVisualizationSelection(sampleNames: string[]): void {
        this._samplesForVisualization = new Set(sampleNames);
        this._notifyVisualizationSelectionChanged();
    }

    /**
     * Check if sample is selected for visualization
     */
    isSampleSelectedForVisualization(sampleName: string): boolean {
        return this._samplesForVisualization.has(sampleName);
    }

    /**
     * Get all samples selected for visualization
     */
    getSamplesForVisualization(): string[] {
        return Array.from(this._samplesForVisualization);
    }

    /**
     * Get session-level FASTA file path
     */
    get sessionFastaPath(): string | null {
        return this._sessionFastaPath;
    }

    /**
     * Set session-level FASTA file path (applies to all samples)
     */
    setSessionFasta(fastaPath: string | null): void {
        this._sessionFastaPath = fastaPath;
    }

    /**
     * Clear session-level FASTA file
     */
    clearSessionFasta(): void {
        this._sessionFastaPath = null;
    }

    // ========== Session State Serialization ==========

    /**
     * Serialize current extension state to SessionState format
     */
    toSessionState(): SessionState {
        // Build samples object from unified state first, fall back to legacy state
        const samples: { [sampleName: string]: SampleSessionState } = {};

        // Use unified state (_loadedItems) if available
        if (this._loadedItems.size > 0) {
            // Process unified state items
            for (const [id, item] of this._loadedItems.entries()) {
                if (item.type === 'sample') {
                    // Sample from unified state
                    const sampleName = item.sampleName || id.substring(7); // Extract name after "sample:"
                    samples[sampleName] = {
                        pod5Paths: [item.pod5Path],
                        bamPath: item.bamPath,
                        fastaPath: item.fastaPath,
                    };
                } else if (item.type === 'pod5' && id.startsWith('pod5:')) {
                    // Standalone POD5 file from unified state
                    const sampleName = 'Default';
                    if (!samples[sampleName]) {
                        samples[sampleName] = {
                            pod5Paths: [item.pod5Path],
                            bamPath: item.bamPath,
                            fastaPath: item.fastaPath,
                        };
                    }
                }
            }
        } else if (this._loadedSamples.size > 0) {
            // Fall back to legacy multi-sample mode
            for (const [sampleName, sampleInfo] of this._loadedSamples.entries()) {
                samples[sampleName] = {
                    pod5Paths: [sampleInfo.pod5Path],
                    bamPath: sampleInfo.bamPath,
                    fastaPath: sampleInfo.fastaPath,
                };
            }
        } else {
            // Fall back to legacy single-file mode
            if (this._currentPod5File) {
                samples['Default'] = {
                    pod5Paths: [this._currentPod5File],
                    bamPath: this._currentBamFile,
                    fastaPath: this._currentFastaFile,
                };
            }
        }

        // Get plot options from provider with fallback defaults
        const providerOptions = this._plotOptionsProvider?.getOptions();

        // Debug logging if provider is missing
        if (!providerOptions) {
            console.warn(
                '[ExtensionState] plotOptionsProvider returned undefined, using defaults'
            );
        }

        const plotOptions = providerOptions
            ? {
                  mode: providerOptions.mode || 'SINGLE',
                  normalization: providerOptions.normalization || 'ZNORM',
                  showDwellTime: providerOptions.showDwellTime ?? false,
                  showBaseAnnotations: providerOptions.showBaseAnnotations ?? true,
                  scaleDwellTime: providerOptions.scaleDwellTime ?? false,
                  downsample: providerOptions.downsample ?? 5,
                  showSignalPoints: providerOptions.showSignalPoints ?? false,
              }
            : {
                  mode: 'SINGLE',
                  normalization: 'ZNORM',
                  showDwellTime: false,
                  showBaseAnnotations: true,
                  scaleDwellTime: false,
                  downsample: 5,
                  showSignalPoints: false,
              };

        // Get modification filters if available
        const modFilters = this._modificationsProvider?.getFilters?.();
        const modificationFilters = modFilters
            ? {
                  minProbability: modFilters.minProbability,
                  enabledModTypes: Array.from(modFilters.enabledModTypes),
              }
            : undefined;

        // Build UI state
        const ui = {
            expandedSamples: this.getAllSampleNames(),
            selectedSamplesForComparison: this._selectedSamplesForComparison,
        };

        return {
            version: '1.0.0',
            timestamp: new Date().toISOString(),
            samples,
            plotOptions,
            modificationFilters,
            ui,
        };
    }

    /**
     * Restore extension state from SessionState
     */
    async fromSessionState(session: SessionState, context: vscode.ExtensionContext): Promise<void> {
        if (!this._extensionContext) {
            this._extensionContext = context;
        }

        const isDemo = session.isDemo || false;
        const extensionUri = context.extensionUri;

        // Clear current state first
        await this.clearAll();

        // Track any errors during restoration
        const errors: string[] = [];

        // Restore each sample
        for (const [sampleName, sampleData] of Object.entries(session.samples)) {
            try {
                await this.restoreSample(sampleName, sampleData, isDemo, extensionUri);

                // Also add to unified state for cross-panel synchronization
                // Use sampleName if it's not "Default", otherwise use POD5 path
                const pod5Path = sampleData.pod5Paths?.[0];
                if (pod5Path) {
                    const itemId =
                        sampleName === 'Default' ? `pod5:${pod5Path}` : `sample:${sampleName}`;
                    const unifiedItem: LoadedItem = {
                        id: itemId,
                        type: sampleName === 'Default' ? 'pod5' : 'sample',
                        pod5Path: pod5Path,
                        bamPath: sampleData.bamPath,
                        fastaPath: sampleData.fastaPath,
                        sampleName: sampleName === 'Default' ? undefined : sampleName,
                        readCount: 0, // Will be populated by restoreSample
                        fileSize: 0, // Will be populated by restoreSample
                        fileSizeFormatted: 'Unknown',
                        hasAlignments: !!sampleData.bamPath,
                        hasReference: !!sampleData.fastaPath,
                        hasMods: false, // Will be determined when loading BAM
                        hasEvents: false, // Will be determined when loading BAM
                    };
                    this.addLoadedItem(unifiedItem);
                }
            } catch (error) {
                errors.push(`Failed to restore sample ${sampleName}: ${error}`);
            }
        }

        // Restore plot options
        if (session.plotOptions && this._plotOptionsProvider) {
            // Update provider state directly (this will trigger view update)
            const provider = this._plotOptionsProvider as any;
            provider._plotMode = session.plotOptions.mode;
            provider._normalization = session.plotOptions.normalization;
            provider._showDwellTime = session.plotOptions.showDwellTime;
            provider._showBaseAnnotations = session.plotOptions.showBaseAnnotations;
            provider._scaleDwellTime = session.plotOptions.scaleDwellTime;
            provider._downsample = session.plotOptions.downsample;
            provider._showSignalPoints = session.plotOptions.showSignalPoints;
            provider.updateView();
        }

        // Restore modification filters
        if (session.modificationFilters && this._modificationsProvider) {
            const provider = this._modificationsProvider as any;
            provider._minProbability = session.modificationFilters.minProbability;
            provider._enabledModTypes = new Set(session.modificationFilters.enabledModTypes);
            provider.updateView();
        }

        // Restore UI state
        if (session.ui) {
            this._selectedSamplesForComparison = session.ui.selectedSamplesForComparison || [];

            // Also populate unified state comparison items
            const comparisonIds = (session.ui.selectedSamplesForComparison || []).map(
                (sampleName) => `sample:${sampleName}`
            );
            if (comparisonIds.length > 0) {
                this.setComparisonItems(comparisonIds);
            }

            // Restore selected Read Explorer sample and auto-load its reads
            if (session.ui.selectedReadExplorerSample) {
                this._selectedReadExplorerSample = session.ui.selectedReadExplorerSample;
                const sampleName = session.ui.selectedReadExplorerSample;

                // Trigger reads view to load for the selected sample
                // Delay to ensure sample is fully registered in Python registry
                setTimeout(() => {
                    logger.debug(
                        `[fromSessionState] Auto-loading reads for selected sample: '${sampleName}'`
                    );
                    Promise.resolve(
                        vscode.commands.executeCommand(
                            'squiggy.internal.loadReadsForSample',
                            sampleName
                        )
                    ).catch((err: unknown) => {
                        logger.error(
                            `[fromSessionState] Failed to auto-load reads for sample '${sampleName}'`,
                            err
                        );
                    });
                }, 1500); // Wait 1.5s to ensure sample is registered
            }
        }

        // Show errors if any
        if (errors.length > 0) {
            vscode.window.showWarningMessage(`Session restored with errors:\n${errors.join('\n')}`);
        } else {
            vscode.window.showInformationMessage('Session restored successfully');
        }
    }

    /**
     * Restore a single sample from session data
     */
    private async restoreSample(
        sampleName: string,
        sampleData: SampleSessionState,
        isDemo: boolean,
        extensionUri: vscode.Uri
    ): Promise<void> {
        // Resolve POD5 paths
        const resolvedPod5Paths: string[] = [];
        for (const pod5Path of sampleData.pod5Paths) {
            let resolvedPath = pod5Path;

            // First try to resolve Python package paths (for demo)
            if (pod5Path.startsWith('<package:')) {
                resolvedPath = await PathResolver.resolvePythonPackagePath(
                    pod5Path,
                    this._positronClient
                );
            }
            // Otherwise, resolve extension-relative paths
            else if (pod5Path.includes('${extensionPath}')) {
                resolvedPath = PathResolver.resolveExtensionPath(pod5Path, extensionUri);
            }

            // Check if file exists, prompt if missing
            const resolution = await FileResolver.resolveFilePath(
                resolvedPath,
                'POD5',
                isDemo,
                extensionUri
            );

            if (resolution.resolved && resolution.newPath) {
                resolvedPod5Paths.push(resolution.newPath);
            } else {
                throw new Error(resolution.error || 'Failed to resolve POD5 file');
            }
        }

        // Resolve BAM path if present
        let resolvedBamPath: string | undefined;
        if (sampleData.bamPath) {
            let resolvedPath = sampleData.bamPath;

            // Resolve Python package paths first
            if (sampleData.bamPath.startsWith('<package:')) {
                resolvedPath = await PathResolver.resolvePythonPackagePath(
                    sampleData.bamPath,
                    this._positronClient
                );
            }
            // Otherwise, resolve extension-relative paths
            else if (sampleData.bamPath.includes('${extensionPath}')) {
                resolvedPath = PathResolver.resolveExtensionPath(sampleData.bamPath, extensionUri);
            }

            const resolution = await FileResolver.resolveFilePath(
                resolvedPath,
                'BAM',
                isDemo,
                extensionUri
            );

            if (resolution.resolved && resolution.newPath) {
                resolvedBamPath = resolution.newPath;
            } else if (!isDemo) {
                // For user sessions, BAM is optional
                vscode.window.showWarningMessage(`BAM file not found: ${sampleData.bamPath}`);
            }
        }

        // Resolve FASTA path if present
        let resolvedFastaPath: string | undefined;
        if (sampleData.fastaPath) {
            let resolvedPath = sampleData.fastaPath;

            // Resolve Python package paths first
            if (sampleData.fastaPath.startsWith('<package:')) {
                resolvedPath = await PathResolver.resolvePythonPackagePath(
                    sampleData.fastaPath,
                    this._positronClient
                );
            }
            // Otherwise, resolve extension-relative paths
            else if (sampleData.fastaPath.includes('${extensionPath}')) {
                resolvedPath = PathResolver.resolveExtensionPath(
                    sampleData.fastaPath,
                    extensionUri
                );
            }

            const resolution = await FileResolver.resolveFilePath(
                resolvedPath,
                'FASTA',
                isDemo,
                extensionUri
            );

            if (resolution.resolved && resolution.newPath) {
                resolvedFastaPath = resolution.newPath;
            }
        }

        // Load files via API
        if (!this._squiggyAPI) {
            throw new Error('Squiggy API not initialized');
        }

        // Load POD5 (assuming single POD5 for now)
        if (resolvedPod5Paths.length > 0) {
            const pod5Path = resolvedPod5Paths[0];
            const _pod5Result = await this._squiggyAPI.loadPOD5(pod5Path);
            this._currentPod5File = pod5Path;

            // Get first 1000 read IDs for reads view (lazy loading)
            const readIds = await this._squiggyAPI.getReadIds(0, 1000);
            if (readIds.length > 0) {
                this._readsViewPane?.setReads(readIds);
            }

            // Update plot options
            this._plotOptionsProvider?.updatePod5Status(true);
        }

        // Load BAM if present
        if (resolvedBamPath) {
            const _bamResult = await this._squiggyAPI.loadBAM(resolvedBamPath);
            this._currentBamFile = resolvedBamPath;

            // Get references only (lazy loading - don't fetch reads yet)
            const references = await this._squiggyAPI.getReferences();
            const referenceToReads: Record<string, string[]> = {};

            // Build reference count map by getting length for each reference
            for (const ref of references) {
                const readCount = (await this._squiggyAPI.client.getVariable(
                    `len(squiggy.io._squiggy_session.ref_mapping.get('${ref.replace(/'/g, "\\'")}', []))`
                )) as number;
                referenceToReads[ref] = new Array(readCount); // Placeholder
            }

            // Update reads view with references if POD5 was loaded
            if (resolvedPod5Paths.length > 0 && Object.keys(referenceToReads).length > 0) {
                const referenceList = Object.entries(referenceToReads).map(
                    ([referenceName, reads]) => ({
                        referenceName,
                        readCount: Array.isArray(reads) ? reads.length : 0,
                    })
                );
                this._readsViewPane?.setReferencesOnly(referenceList);
            }

            // Update plot options BAM status
            if (this._plotOptionsProvider) {
                this._plotOptionsProvider.updateBamStatus(true);
            }

            // Update plot options with available references for aggregate plots
            if (references.length > 0) {
                this._plotOptionsProvider?.updateReferences(references);
            }

            // Update modifications panel
            if (_bamResult.hasModifications && this._modificationsProvider) {
                this._modificationsProvider.setModificationInfo(
                    _bamResult.hasModifications,
                    _bamResult.modificationTypes,
                    _bamResult.hasProbabilities
                );
            } else if (this._modificationsProvider) {
                this._modificationsProvider.clear();
            }
        }

        // Load FASTA if present
        if (resolvedFastaPath) {
            await this._squiggyAPI.loadFASTA?.(resolvedFastaPath);
            this._currentFastaFile = resolvedFastaPath;
        }

        // Add to loaded samples if multi-sample mode
        if (resolvedPod5Paths.length > 0) {
            // CRITICAL: Load sample into Python registry so TypeScript queries work
            try {
                logger.debug(
                    `[restoreSample] Loading sample '${sampleName}' into Python registry...`
                );
                await this._squiggyAPI.loadSample(
                    sampleName,
                    resolvedPod5Paths[0],
                    resolvedBamPath,
                    resolvedFastaPath
                );
                logger.debug(
                    `[restoreSample] Sample '${sampleName}' successfully loaded into Python registry`
                );
            } catch (error) {
                logger.error(`[restoreSample] Failed to load sample into Python registry`, error);
                // Continue anyway - sample is in TypeScript state
            }

            const sampleInfo: SampleInfo = {
                // Core identifiers
                sampleId: `sample:${sampleName}`, // Consistent with LoadedItem ID format
                displayName: sampleName, // Can be edited later in Sample Manager
                // File associations
                pod5Path: resolvedPod5Paths[0],
                bamPath: resolvedBamPath,
                fastaPath: resolvedFastaPath,
                // File metadata
                readCount: 0, // Will be populated by loadPOD5
                hasBam: !!resolvedBamPath,
                hasFasta: !!resolvedFastaPath,
                // Kernel state
                isLoaded: true, // Files have been loaded to kernel
                // Extensible metadata
                metadata: {
                    autoDetected: false, // Manual sample creation for now
                    sourceType: 'manual', // Manual UI creation (not from TSV)
                },
            };
            this._loadedSamples.set(sampleName, sampleInfo);
        }
    }

    /**
     * Load demo session with packaged test data
     */
    async loadDemoSession(context: vscode.ExtensionContext): Promise<void> {
        // Get demo session from manager
        const demoSession = SessionStateManager.getDemoSession(context.extensionUri);

        // Restore from demo session
        await this.fromSessionState(demoSession, context);

        vscode.window.showInformationMessage(
            'Demo session loaded! Explore 180 yeast tRNA reads with base annotations.'
        );
    }
}
