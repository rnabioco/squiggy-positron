/**
 * Centralized Extension State
 *
 * Manages all extension state including loaded files, UI panel references,
 * backend instances, and installation flags. Provides a single source of truth
 * for extension state across all modules.
 */

import * as vscode from 'vscode';
import { SquiggyRuntimeAPI } from '../backend/squiggy-runtime-api';
import { SquiggyKernelManager, SquiggyKernelState } from '../backend/squiggy-kernel-manager';
import { ReadsViewPane } from '../views/squiggy-reads-panel';
import { PlotOptionsViewProvider } from '../views/squiggy-plot-options-panel';
import { ModificationsPanelProvider } from '../views/squiggy-modifications-panel';
import { SamplesPanelProvider } from '../views/squiggy-samples-panel';
import { SessionState, SampleSessionState } from '../types/squiggy-session-types';
import { LoadedItem } from '../types/loaded-item';
import { SessionStateManager } from './session-state-manager';
import { PathResolver } from './path-resolver';
import { FileResolver } from './file-resolver';
import { logger } from '../utils/logger';
import { statusBarMessenger } from '../utils/status-bar-messenger';

/**
 * Information about a loaded sample (POD5 + optional BAM/FASTA)
 *
 * - `sampleId`: Unique identifier (can be UUID or file-based)
 * - `displayName`: User-facing name (editable), separate from POD5 filename
 * - `pod5Path`: Single POD5 file, extensible to array for multi-POD5 future
 * - `isLoaded`: Tracks kernel state (prepares for lazy loading)
 * - `metadata`: Extensible object for future attributes without interface changes
 */
export interface ReferenceInfo {
    name: string;
    readCount: number;
    length?: number;
}

export interface SampleInfo {
    // Core identifiers
    sampleId: string; // Unique ID (can be UUID or derived from pod5 path)
    displayName: string; // User-facing name (editable in Sample Manager)

    // File associations
    pod5Path: string;
    bamPath?: string;
    fastaPath?: string;

    // File metadata
    readCount: number;
    hasBam: boolean;
    hasFasta: boolean;
    hasMods?: boolean; // BAM has MM/ML tags for base modifications
    hasEvents?: boolean; // BAM has mv tag for signal-to-base mapping

    // Reference information (from BAM alignment)
    references?: ReferenceInfo[]; // List of references this sample aligns to

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
    // Backend instances - uses dedicated "Squiggy Kernel" only (no foreground fallback)
    private _squiggyAPI?: SquiggyRuntimeAPI;
    private _kernelManager?: SquiggyKernelManager;

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

    // Multi-sample state
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
     * Initialize backends (Positron only)
     * Uses a dedicated "Squiggy Kernel" session - no foreground kernel fallback
     */
    async initializeBackends(context: vscode.ExtensionContext): Promise<void> {
        this._extensionContext = context;

        // Initialize dedicated kernel manager for all extension operations
        // Uses getPreferredRuntime() which returns the squiggy venv (Positron auto-discovers it)
        this._kernelManager = new SquiggyKernelManager(context.extensionPath, context);
        context.subscriptions.push(this._kernelManager);

        logger.info('Squiggy kernel manager initialized, starting kernel...');

        // Auto-start the kernel (don't await - let it start in background)
        this._kernelManager.start().catch((error) => {
            logger.error(`Failed to auto-start kernel: ${error}`);
        });
    }

    /**
     * Ensure the Squiggy kernel is started and return the API
     * Lazily starts the kernel on first call
     *
     * NO FALLBACK: If kernel fails to start, throws error (no foreground fallback)
     */
    async ensureKernel(): Promise<SquiggyRuntimeAPI> {
        if (!this._kernelManager) {
            throw new Error('Squiggy kernel manager not initialized. Is Positron available?');
        }

        // Start kernel if not already started
        const currentState = this._kernelManager.getState();
        if (currentState === SquiggyKernelState.Uninitialized) {
            logger.info('Starting Squiggy kernel (first use)...');
            await this._kernelManager.start();
        } else if (currentState === SquiggyKernelState.Error) {
            logger.info('Restarting Squiggy kernel (was in error state)...');
            await this._kernelManager.restart();
        }

        // Create kernel API if not already created
        if (!this._squiggyAPI) {
            this._squiggyAPI = new SquiggyRuntimeAPI(this._kernelManager);
            logger.info('Squiggy kernel API created');
        }

        return this._squiggyAPI;
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
        this._plotOptionsProvider?.updateBamStatus(null);

        // Clear Python kernel state in the dedicated Squiggy kernel
        if (this._kernelManager) {
            try {
                const api = await this.ensureKernel();
                await api.client.executeSilent(`
import squiggy
from squiggy.io import squiggy_kernel
# Close all resources via session
squiggy_kernel.close_all()
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

    // ========== Getters ==========

    get kernelManager(): SquiggyKernelManager | undefined {
        return this._kernelManager;
    }

    get squiggyAPI(): SquiggyRuntimeAPI | undefined {
        return this._squiggyAPI;
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
        const selected = this.getSamplesForVisualization();
        logger.info(
            `[ExtensionState] Visualization selection changed: ${selected.length} samples selected: ${selected.join(', ')}`
        );
        this._onVisualizationSelectionChanged.fire(selected);
    }

    // ========== Multi-Sample Management ==========

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
            logger.warning(
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
     *
     * Uses progressive loading for improved UX:
     * - Phase 1: Immediately shows skeleton samples with loading spinners
     * - Phase 2: Loads sample data in parallel, updating UI as each completes
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

        // ========== PHASE 1: Immediate UI Population ==========
        // Create skeleton samples with loading indicators and show them immediately
        logger.info(
            `[fromSessionState] Phase 1: Creating skeleton UI for ${Object.keys(session.samples).length} samples`
        );

        for (const [sampleName, sampleData] of Object.entries(session.samples)) {
            const pod5Path = sampleData.pod5Paths?.[0];
            if (!pod5Path) continue;

            const itemId = sampleName === 'Default' ? `pod5:${pod5Path}` : `sample:${sampleName}`;

            // Create skeleton LoadedItem with isLoading=true
            const skeletonItem: LoadedItem = {
                id: itemId,
                type: sampleName === 'Default' ? 'pod5' : 'sample',
                pod5Path: pod5Path,
                bamPath: sampleData.bamPath,
                fastaPath: sampleData.fastaPath,
                sampleName: sampleName === 'Default' ? undefined : sampleName,
                readCount: 0, // Placeholder until loaded
                fileSize: 0,
                fileSizeFormatted: 'Unknown',
                hasAlignments: !!sampleData.bamPath,
                hasReference: !!sampleData.fastaPath,
                hasMods: false,
                hasEvents: false,
                isLoading: true, // Show loading spinner
                loadingMessage: 'Loading...',
            };
            this.addLoadedItem(skeletonItem);

            // Also create a skeleton SampleInfo entry in legacy state
            const skeletonSampleInfo: SampleInfo = {
                sampleId: itemId,
                displayName: sampleName,
                pod5Path: pod5Path,
                bamPath: sampleData.bamPath,
                fastaPath: sampleData.fastaPath,
                readCount: 0,
                hasBam: !!sampleData.bamPath,
                hasFasta: !!sampleData.fastaPath,
                isLoaded: false,
                metadata: {
                    autoDetected: false,
                    sourceType: 'manual',
                },
            };
            this._loadedSamples.set(sampleName, skeletonSampleInfo);
        }

        // Update samples panel to show skeletons immediately
        this._samplesProvider?.refresh();

        // ========== PHASE 2: Parallel Background Loading ==========
        // Load sample data concurrently, updating UI as each completes
        logger.info('[fromSessionState] Phase 2: Loading sample data in parallel');

        const loadPromises = Object.entries(session.samples).map(
            async ([sampleName, sampleData]) => {
                try {
                    await this.restoreSampleWithProgress(
                        sampleName,
                        sampleData,
                        isDemo,
                        extensionUri
                    );
                } catch (error) {
                    errors.push(`Failed to restore sample ${sampleName}: ${error}`);
                    // Mark sample as failed in UI
                    const pod5Path = sampleData.pod5Paths?.[0];
                    if (pod5Path) {
                        const itemId =
                            sampleName === 'Default' ? `pod5:${pod5Path}` : `sample:${sampleName}`;
                        this.updateLoadedItemStatus(itemId, false, `Error: ${error}`);
                    }
                }
            }
        );

        // Wait for all samples to finish loading
        await Promise.all(loadPromises);

        // ========== PHASE 3: Restore UI State ==========
        // Restore plot options
        if (session.plotOptions && this._plotOptionsProvider) {
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

            const comparisonIds = (session.ui.selectedSamplesForComparison || []).map(
                (name) => `sample:${name}`
            );
            if (comparisonIds.length > 0) {
                this.setComparisonItems(comparisonIds);
            }

            // Restore selected Read Explorer sample and auto-load its reads
            if (session.ui.selectedReadExplorerSample) {
                this._selectedReadExplorerSample = session.ui.selectedReadExplorerSample;
                const sampleName = session.ui.selectedReadExplorerSample;

                // Delay to ensure sample is fully registered
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
                }, 500); // Reduced delay since samples load in parallel now
            }
        }

        // Show errors if any
        if (errors.length > 0) {
            vscode.window.showWarningMessage(`Session restored with errors:\n${errors.join('\n')}`);
        } else {
            statusBarMessenger.show('Restored', 'folder-opened');
        }
    }

    /**
     * Update the loading status of a LoadedItem and notify listeners
     * @private
     */
    private updateLoadedItemStatus(
        itemId: string,
        isLoading: boolean,
        loadingMessage?: string
    ): void {
        const item = this._loadedItems.get(itemId);
        if (item) {
            item.isLoading = isLoading;
            item.loadingMessage = loadingMessage;
            this._notifyLoadedItemsChanged();
        }
    }

    /**
     * Update a LoadedItem with new data and notify listeners
     * @private
     */
    private updateLoadedItemData(itemId: string, updates: Partial<LoadedItem>): void {
        const item = this._loadedItems.get(itemId);
        if (item) {
            Object.assign(item, updates);
            this._notifyLoadedItemsChanged();
        }
    }

    /**
     * Restore a single sample with progress updates for progressive loading
     * Updates the UI as each phase completes
     * @private
     */
    private async restoreSampleWithProgress(
        sampleName: string,
        sampleData: SampleSessionState,
        isDemo: boolean,
        extensionUri: vscode.Uri
    ): Promise<void> {
        const pod5Path = sampleData.pod5Paths?.[0];
        if (!pod5Path) return;

        const itemId = sampleName === 'Default' ? `pod5:${pod5Path}` : `sample:${sampleName}`;

        // Update status: Resolving paths
        this.updateLoadedItemStatus(itemId, true, 'Resolving paths...');

        // Get API client for path resolution
        const api = await this.ensureKernel();
        const pathResolverClient = api.client;

        // Resolve POD5 paths
        const resolvedPod5Paths: string[] = [];
        for (const pod5PathItem of sampleData.pod5Paths) {
            let resolvedPath = pod5PathItem;

            if (pod5PathItem.startsWith('<package:')) {
                resolvedPath = await PathResolver.resolvePythonPackagePath(
                    pod5PathItem,
                    pathResolverClient
                );
            } else if (pod5PathItem.includes('${extensionPath}')) {
                resolvedPath = PathResolver.resolveExtensionPath(pod5PathItem, extensionUri);
            }

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

        // Resolve BAM path
        let resolvedBamPath: string | undefined;
        if (sampleData.bamPath) {
            this.updateLoadedItemStatus(itemId, true, 'Loading BAM...');

            let resolvedPath = sampleData.bamPath;
            if (sampleData.bamPath.startsWith('<package:')) {
                resolvedPath = await PathResolver.resolvePythonPackagePath(
                    sampleData.bamPath,
                    pathResolverClient
                );
            } else if (sampleData.bamPath.includes('${extensionPath}')) {
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
                vscode.window.showWarningMessage(`BAM file not found: ${sampleData.bamPath}`);
            }
        }

        // Resolve FASTA path
        let resolvedFastaPath: string | undefined;
        if (sampleData.fastaPath) {
            let resolvedPath = sampleData.fastaPath;
            if (sampleData.fastaPath.startsWith('<package:')) {
                resolvedPath = await PathResolver.resolvePythonPackagePath(
                    sampleData.fastaPath,
                    pathResolverClient
                );
            } else if (sampleData.fastaPath.includes('${extensionPath}')) {
                resolvedPath = PathResolver.resolveExtensionPath(sampleData.fastaPath, extensionUri);
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

        // Update status: Loading POD5
        this.updateLoadedItemStatus(itemId, true, 'Loading POD5...');

        // Load POD5
        if (resolvedPod5Paths.length > 0) {
            const resolvedPod5Path = resolvedPod5Paths[0];
            await api.loadPOD5(resolvedPod5Path);
            this._currentPod5File = resolvedPod5Path;

            // Get first 1000 read IDs for reads view
            const readIds = await api.getReadIds(0, 1000);
            if (readIds.length > 0) {
                this._readsViewPane?.setReads(readIds);
            }

            this._plotOptionsProvider?.updatePod5Status(true);
        }

        // Load BAM if present
        let bamResult: any;
        if (resolvedBamPath) {
            this.updateLoadedItemStatus(itemId, true, 'Loading alignments...');
            bamResult = await api.loadBAM(resolvedBamPath);
            this._currentBamFile = resolvedBamPath;

            // Get references using batch query for performance
            const references = await api.getReferences();

            // OPTIMIZATION: Batch query for reference counts instead of N sequential calls
            if (references.length > 0) {
                const refCounts = await this.batchGetReferenceCounts(api);

                // Update reads view with references
                const referenceList = references.map((refName) => ({
                    referenceName: refName,
                    readCount: refCounts[refName] ?? 0,
                }));
                this._readsViewPane?.setReferencesOnly(referenceList);

                this._plotOptionsProvider?.updateReferences(references);
            }

            if (this._plotOptionsProvider) {
                this._plotOptionsProvider.updateBamStatus({ isRna: bamResult.isRna });
            }

            if (bamResult.hasModifications && this._modificationsProvider) {
                this._modificationsProvider.setModificationInfo(
                    bamResult.hasModifications,
                    bamResult.modificationTypes,
                    bamResult.hasProbabilities
                );
            } else if (this._modificationsProvider) {
                this._modificationsProvider.clear();
            }
        }

        // Load FASTA if present
        if (resolvedFastaPath) {
            await api.loadFASTA?.(resolvedFastaPath);
            this._currentFastaFile = resolvedFastaPath;
        }

        // Load sample into Python registry
        this.updateLoadedItemStatus(itemId, true, 'Registering sample...');
        let numReads = 0;
        try {
            logger.debug(`[restoreSampleWithProgress] Loading sample '${sampleName}' into registry`);
            const sampleResult = await api.loadSample(
                sampleName,
                resolvedPod5Paths[0],
                resolvedBamPath,
                resolvedFastaPath
            );
            numReads = sampleResult.numReads;
            logger.debug(`[restoreSampleWithProgress] Sample '${sampleName}' loaded: ${numReads} reads`);
        } catch (error) {
            logger.error(`[restoreSampleWithProgress] Failed to load sample into registry`, error);
        }

        // ========== Update State with Final Data ==========

        // Update SampleInfo in legacy state
        const sampleInfo = this._loadedSamples.get(sampleName);
        if (sampleInfo) {
            sampleInfo.pod5Path = resolvedPod5Paths[0];
            sampleInfo.bamPath = resolvedBamPath;
            sampleInfo.fastaPath = resolvedFastaPath;
            sampleInfo.readCount = numReads;
            sampleInfo.hasBam = !!resolvedBamPath;
            sampleInfo.hasFasta = !!resolvedFastaPath;
            sampleInfo.hasMods = bamResult?.hasModifications ?? false;
            sampleInfo.hasEvents = bamResult?.hasEvents ?? false;
            sampleInfo.isLoaded = true;
        }

        // Update LoadedItem with final data and mark as complete
        this.updateLoadedItemData(itemId, {
            pod5Path: resolvedPod5Paths[0],
            bamPath: resolvedBamPath,
            fastaPath: resolvedFastaPath,
            readCount: numReads,
            hasAlignments: !!resolvedBamPath,
            hasReference: !!resolvedFastaPath,
            hasMods: bamResult?.hasModifications ?? false,
            hasEvents: bamResult?.hasEvents ?? false,
            isRna: bamResult?.isRna ?? false,
            isLoading: false, // Mark as complete
            loadingMessage: undefined,
        });

        logger.info(`[restoreSampleWithProgress] Sample '${sampleName}' restore complete`);
    }

    /**
     * Batch query for all reference counts in a single Python call
     * This replaces N sequential getVariable calls with one efficient call
     * @private
     */
    private async batchGetReferenceCounts(
        api: SquiggyRuntimeAPI
    ): Promise<Record<string, number>> {
        try {
            const counts = (await api.client.getVariable(
                `{ref: len(reads) for ref, reads in squiggy.io.squiggy_kernel._ref_mapping.items()}`
            )) as Record<string, number>;
            return counts ?? {};
        } catch (error) {
            logger.error('[batchGetReferenceCounts] Failed to batch query reference counts', error);
            return {};
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

        statusBarMessenger.show('Demo loaded', 'play');
    }
}
