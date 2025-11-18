/**
 * Plot Options Webview View
 *
 * Provides controls for plot configuration in the sidebar
 */

import * as vscode from 'vscode';
import { BaseWebviewProvider } from './base-webview-provider';
import {
    PlotOptionsIncomingMessage,
    UpdatePlotOptionsMessage,
    UpdateBamStatusMessage,
    UpdatePod5StatusMessage,
    UpdateFastaStatusMessage,
    UpdateReferencesMessage,
    UpdateLoadedSamplesMessage,
    SampleItem,
} from '../types/messages';
import { ExtensionState } from '../state/extension-state';
import { logger } from '../utils/logger';

type PlotType =
    | 'MULTI_READ_OVERLAY'
    | 'MULTI_READ_STACKED'
    | 'AGGREGATE'
    | 'COMPARE_SIGNAL_DELTA'
    | 'COMPARE_AGGREGATE';

export class PlotOptionsViewProvider extends BaseWebviewProvider {
    public static readonly viewType = 'squiggyPlotOptions';

    private _state: ExtensionState;

    private _plotType: PlotType = 'AGGREGATE';
    private _plotMode: 'SINGLE' | 'EVENTALIGN' = 'SINGLE';
    private _normalization: 'ZNORM' | 'MAD' | 'MEDIAN' | 'NONE' = 'ZNORM';
    private _showDwellTime: boolean = false;
    private _showBaseAnnotations: boolean = true;
    private _scaleDwellTime: boolean = false;
    private _downsample: number = 5;
    private _showSignalPoints: boolean = false;
    private _clipXAxisToAlignment: boolean = true;
    private _transformCoordinates: boolean = true;
    private _hasPod5File: boolean = false;
    private _hasBamFile: boolean = false;

    // Aggregate-specific state
    private _aggregateReference: string = '';
    private _aggregateMaxReads: number = 100;
    private _showModifications: boolean = true;
    private _showPileup: boolean = true;
    private _showSignal: boolean = true;
    private _showQuality: boolean = true;
    private _availableReferences: string[] = [];

    // Multi-sample state
    private _loadedSamples: SampleItem[] = [];

    constructor(extensionUri: vscode.Uri, state: ExtensionState) {
        super(extensionUri);
        this._state = state;

        // Subscribe to visualization selection changes
        const selectionDisposable = this._state.onVisualizationSelectionChanged(
            (selectedSamples) => {
                this._handleVisualizationSelectionChanged(selectedSamples);
            }
        );
        this._disposables.push(selectionDisposable);
    }

    /**
     * Handle visualization selection changes from extension state
     * Updates webview to reflect current selection
     * @private
     */
    private _handleVisualizationSelectionChanged(selectedSamples: string[]): void {
        logger.debug('[PlotOptionsViewProvider] Visualization selection changed:', selectedSamples);
        // Send update to webview
        if (this._view?.webview) {
            this._view.webview.postMessage({
                type: 'updateSelectedSamples',
                selectedSamples,
            });
        }
    }

    /**
     * Dispose method to clean up subscriptions and event emitters
     */
    public override dispose(): void {
        // Dispose all EventEmitters
        this._onDidChangeOptions.dispose();
        this._onDidRequestAggregatePlot.dispose();
        this._onDidRequestSignalOverlay.dispose();
        this._onDidRequestSignalDelta.dispose();
        this._onDidRequestAggregateComparison.dispose();
        this._onDidRequestMultiReadOverlay.dispose();
        this._onDidRequestMultiReadStacked.dispose();

        // Call base class dispose to clean up all other disposables
        super.dispose();
    }

    // Event emitter for when options change that should trigger refresh
    private _onDidChangeOptions = new vscode.EventEmitter<void>();
    public readonly onDidChangeOptions = this._onDidChangeOptions.event;

    // Event emitter for when user requests aggregate plot generation
    private _onDidRequestAggregatePlot = new vscode.EventEmitter<{
        sampleNames: string[]; // Now supports 1+ samples
        reference: string;
        maxReads: number;
        viewStyle: 'overlay' | 'multi-track'; // For multi-sample: overlay or separate tracks
        normalization: string;
        showModifications: boolean;
        showPileup: boolean;
        showDwellTime: boolean;
        showSignal: boolean;
        showQuality: boolean;
        clipXAxisToAlignment: boolean;
        transformCoordinates: boolean;
    }>();
    public readonly onDidRequestAggregatePlot = this._onDidRequestAggregatePlot.event;

    // Event emitters for comparison plots
    private _onDidRequestSignalOverlay = new vscode.EventEmitter<{
        sampleNames: string[];
        maxReads: number;
        normalization: string;
    }>();
    public readonly onDidRequestSignalOverlay = this._onDidRequestSignalOverlay.event;

    private _onDidRequestSignalDelta = new vscode.EventEmitter<{
        sampleNames: [string, string];
        reference: string;
        maxReads: number;
        normalization: string;
    }>();
    public readonly onDidRequestSignalDelta = this._onDidRequestSignalDelta.event;

    private _onDidRequestAggregateComparison = new vscode.EventEmitter<{
        sampleNames: string[];
        reference: string;
        metrics: string[];
        maxReads: number;
        normalization: string;
    }>();
    public readonly onDidRequestAggregateComparison = this._onDidRequestAggregateComparison.event;

    // Event emitters for multi-read plots
    private _onDidRequestMultiReadOverlay = new vscode.EventEmitter<{
        sampleNames: string[];
        maxReads: number;
        normalization: string;
        coordinateSpace: 'signal' | 'sequence';
    }>();
    public readonly onDidRequestMultiReadOverlay = this._onDidRequestMultiReadOverlay.event;

    private _onDidRequestMultiReadStacked = new vscode.EventEmitter<{
        sampleNames: string[];
        maxReads: number;
        normalization: string;
        coordinateSpace: 'signal' | 'sequence';
    }>();
    public readonly onDidRequestMultiReadStacked = this._onDidRequestMultiReadStacked.event;

    protected getTitle(): string {
        return 'Plotting';
    }

    protected async handleMessage(message: PlotOptionsIncomingMessage): Promise<void> {
        if (message.type === 'ready') {
            this.updateView();
            return;
        }

        if (message.type === 'optionsChanged') {
            // Update internal state
            if (message.options.plotType !== undefined) {
                this._plotType = message.options.plotType;
            }
            this._plotMode = message.options.mode;
            this._normalization = message.options.normalization;
            this._showDwellTime = message.options.showDwellTime;
            this._showBaseAnnotations = message.options.showBaseAnnotations;
            this._scaleDwellTime = message.options.scaleDwellTime;
            this._downsample = message.options.downsample;
            this._showSignalPoints = message.options.showSignalPoints;
            if (message.options.clipXAxisToAlignment !== undefined) {
                this._clipXAxisToAlignment = message.options.clipXAxisToAlignment;
            }
            if (message.options.transformCoordinates !== undefined) {
                this._transformCoordinates = message.options.transformCoordinates;
            }

            // Update aggregate-specific options if present
            if (message.options.aggregateReference !== undefined) {
                this._aggregateReference = message.options.aggregateReference;
            }
            if (message.options.aggregateMaxReads !== undefined) {
                this._aggregateMaxReads = message.options.aggregateMaxReads;
            }
            if (message.options.showModifications !== undefined) {
                this._showModifications = message.options.showModifications;
            }
            if (message.options.showPileup !== undefined) {
                this._showPileup = message.options.showPileup;
            }
            if (message.options.showSignal !== undefined) {
                this._showSignal = message.options.showSignal;
            }
            if (message.options.showQuality !== undefined) {
                this._showQuality = message.options.showQuality;
            }

            // Fire change event
            this._onDidChangeOptions.fire();
        }

        if (message.type === 'requestReferences') {
            // Request will be handled by extension.ts which listens to this event
            this._onDidChangeOptions.fire();
        }

        if (message.type === 'generateAggregatePlot') {
            // Fire event for extension.ts to handle
            this._onDidRequestAggregatePlot.fire({
                sampleNames: message.sampleNames,
                reference: message.reference,
                maxReads: message.maxReads,
                viewStyle: message.viewStyle,
                normalization: message.normalization,
                showModifications: message.showModifications,
                showPileup: message.showPileup,
                showDwellTime: message.showDwellTime,
                showSignal: message.showSignal,
                showQuality: message.showQuality,
                clipXAxisToAlignment: message.clipXAxisToAlignment,
                transformCoordinates: message.transformCoordinates,
            });
        }

        if (message.type === 'generateSignalOverlayComparison') {
            this._onDidRequestSignalOverlay.fire({
                sampleNames: message.sampleNames,
                maxReads: message.maxReads,
                normalization: message.normalization,
            });
        }

        if (message.type === 'generateSignalDelta') {
            // Validate that exactly 2 samples are provided for signal delta
            if (!Array.isArray(message.sampleNames) || message.sampleNames.length !== 2) {
                logger.error('[PlotOptions] Signal delta requires exactly 2 samples');
                return;
            }
            this._onDidRequestSignalDelta.fire({
                sampleNames: message.sampleNames as [string, string],
                reference: message.reference,
                maxReads: message.maxReads,
                normalization: message.normalization,
            });
        }

        if (message.type === 'generateAggregateComparison') {
            this._onDidRequestAggregateComparison.fire({
                sampleNames: message.sampleNames,
                reference: message.reference,
                metrics: message.metrics,
                maxReads: message.maxReads,
                normalization: message.normalization,
            });
        }

        if (message.type === 'generateMultiReadOverlay') {
            this._onDidRequestMultiReadOverlay.fire({
                sampleNames: message.sampleNames,
                maxReads: message.maxReads,
                normalization: message.normalization,
                coordinateSpace: message.coordinateSpace,
            });
        }

        if (message.type === 'generateMultiReadStacked') {
            this._onDidRequestMultiReadStacked.fire({
                sampleNames: message.sampleNames,
                maxReads: message.maxReads,
                normalization: message.normalization,
                coordinateSpace: message.coordinateSpace,
            });
        }

        if (message.type === 'toggleSampleSelection') {
            // Toggle sample selection in extension state
            const isSelected = this._state.isSampleSelectedForVisualization(message.sampleName);
            if (isSelected) {
                this._state.removeSampleFromVisualization(message.sampleName);
            } else {
                this._state.addSampleToVisualization(message.sampleName);
            }
            logger.debug(
                `[PlotOptionsViewProvider] Toggled selection for ${message.sampleName}: now ${!isSelected}`
            );
        }
    }

    protected updateView(): void {
        // Don't check isVisible - if we have a view and received 'ready',
        // the webview is ready to receive messages
        if (!this._view) {
            return;
        }

        // Send all current option values to the webview
        const updateMessage: UpdatePlotOptionsMessage = {
            type: 'updatePlotOptions',
            options: {
                plotType: this._plotType,
                mode: this._plotMode,
                normalization: this._normalization,
                showDwellTime: this._showDwellTime,
                showBaseAnnotations: this._showBaseAnnotations,
                scaleDwellTime: this._scaleDwellTime,
                downsample: this._downsample,
                showSignalPoints: this._showSignalPoints,
                clipXAxisToAlignment: this._clipXAxisToAlignment,
                transformCoordinates: this._transformCoordinates,
                aggregateReference: this._aggregateReference,
                aggregateMaxReads: this._aggregateMaxReads,
                showModifications: this._showModifications,
                showPileup: this._showPileup,
                showSignal: this._showSignal,
                showQuality: this._showQuality,
            },
        };
        this.postMessage(updateMessage);

        // Always send POD5 status on init (even if false)
        const pod5StatusMessage: UpdatePod5StatusMessage = {
            type: 'updatePod5Status',
            hasPod5: this._hasPod5File,
        };
        this.postMessage(pod5StatusMessage);

        // Always send BAM status on init (even if false)
        const bamStatusMessage: UpdateBamStatusMessage = {
            type: 'updateBamStatus',
            hasBam: this._hasBamFile,
        };
        logger.debug('[PlotOptions] Sending BAM status on init:', this._hasBamFile);
        this.postMessage(bamStatusMessage);

        // Send references if available
        if (this._hasBamFile && this._availableReferences.length > 0) {
            const referencesMessage: UpdateReferencesMessage = {
                type: 'updateReferences',
                references: this._availableReferences,
            };
            this.postMessage(referencesMessage);
        }

        // Send loaded samples if available
        if (this._loadedSamples.length > 0) {
            const samplesMessage: UpdateLoadedSamplesMessage = {
                type: 'updateLoadedSamples',
                samples: this._loadedSamples,
            };
            logger.debug(
                '[PlotOptions] Sending loaded samples on init:',
                this._loadedSamples.length
            );
            this.postMessage(samplesMessage);
        }
    }

    /**
     * Get current plot options
     */
    public getOptions() {
        return {
            plotType: this._plotType,
            mode: this._plotMode,
            normalization: this._normalization,
            showDwellTime: this._showDwellTime,
            showBaseAnnotations: this._showBaseAnnotations,
            scaleDwellTime: this._scaleDwellTime,
            downsample: this._downsample,
            showSignalPoints: this._showSignalPoints,
            clipXAxisToAlignment: this._clipXAxisToAlignment,
            transformCoordinates: this._transformCoordinates,
            aggregateReference: this._aggregateReference,
            aggregateMaxReads: this._aggregateMaxReads,
            showModifications: this._showModifications,
            showPileup: this._showPileup,
            showSignal: this._showSignal,
            showQuality: this._showQuality,
        };
    }

    /**
     * Update POD5 file status
     */
    public updatePod5Status(hasPod5: boolean) {
        logger.debug('[PlotOptions] updatePod5Status called with:', hasPod5);
        this._hasPod5File = hasPod5;

        // Update webview
        const message: UpdatePod5StatusMessage = {
            type: 'updatePod5Status',
            hasPod5: this._hasPod5File,
        };
        logger.debug('[PlotOptions] Sending POD5 status:', hasPod5);
        this.postMessage(message);
    }

    /**
     * Update BAM file status and available plot modes
     */
    public updateBamStatus(hasBam: boolean) {
        logger.debug('[PlotOptions] updateBamStatus called with:', hasBam);
        this._hasBamFile = hasBam;

        // When BAM loads, switch to AGGREGATE but keep mode as SINGLE
        // (coordinate_space toggle controls signal vs sequence coordinates)
        if (hasBam) {
            this._plotType = 'AGGREGATE';
            // Keep mode as SINGLE - no longer auto-switch to EVENTALIGN
            this._plotMode = 'SINGLE';
            this._updateConfig('defaultPlotMode', 'SINGLE');
        }
        // When BAM unloads, switch to MULTI_READ_OVERLAY
        else {
            this._plotType = 'MULTI_READ_OVERLAY';
            this._plotMode = 'SINGLE';
            this._updateConfig('defaultPlotMode', 'SINGLE');
        }

        // Update webview
        const message: UpdateBamStatusMessage = {
            type: 'updateBamStatus',
            hasBam: this._hasBamFile,
        };
        logger.debug('[PlotOptions] Sending BAM status:', hasBam);
        this.postMessage(message);
    }

    /**
     * Update FASTA file status
     */
    public updateFastaStatus(hasFasta: boolean) {
        logger.debug('[PlotOptions] updateFastaStatus called with:', hasFasta);

        // Update webview
        const message: UpdateFastaStatusMessage = {
            type: 'updateFastaStatus',
            hasFasta: hasFasta,
        };
        logger.debug('[PlotOptions] Sending FASTA status:', hasFasta);
        this.postMessage(message);
    }

    /**
     * Update available references for aggregate plots
     */
    public updateReferences(references: string[]) {
        this._availableReferences = references;
        if (references.length > 0 && !this._aggregateReference) {
            this._aggregateReference = references[0];
        }

        // Update webview
        const message: UpdateReferencesMessage = {
            type: 'updateReferences',
            references: this._availableReferences,
        };
        this.postMessage(message);
    }

    /**
     * Update loaded samples for comparison plots
     */
    public updateLoadedSamples(samples: SampleItem[]) {
        logger.debug('[PlotOptions] updateLoadedSamples called with:', samples.length, 'samples');
        this._loadedSamples = samples;

        // Update webview
        const message: UpdateLoadedSamplesMessage = {
            type: 'updateLoadedSamples',
            samples: this._loadedSamples,
        };
        logger.debug('[PlotOptions] Sending loaded samples:', samples.length);
        this.postMessage(message);
    }

    /**
     * Update workspace configuration
     */
    private _updateConfig(key: string, value: any): void {
        const config = vscode.workspace.getConfiguration('squiggy');
        config.update(key, value, vscode.ConfigurationTarget.Workspace);
    }
}
