/**
 * Type-safe message definitions for webview communication
 *
 * Defines all message types exchanged between extension and webviews
 * with strict TypeScript types to eliminate 'any' usage.
 */

import { ReadListItem, ReadItem } from './squiggy-reads-types';

// ========== Base Message Types ==========

export interface BaseMessage {
    type: string;
}

// ========== File Panel Messages ==========

export interface OpenFileMessage extends BaseMessage {
    type: 'openFile';
    fileType: 'POD5' | 'BAM' | 'FASTA';
}

export interface CloseFileMessage extends BaseMessage {
    type: 'closeFile';
    fileType: 'POD5' | 'BAM' | 'FASTA';
}

export interface AddFilesMessage extends BaseMessage {
    type: 'addFiles';
    // No parameters - triggers file picker in extension
}

export interface AddReferenceMessage extends BaseMessage {
    type: 'addReference';
    // No parameters - triggers file picker in extension
}

export interface UpdateFilesMessage extends BaseMessage {
    type: 'updateFiles';
    files: FileItem[];
}

export interface ReadyMessage extends BaseMessage {
    type: 'ready';
}

export interface FileItem {
    path: string;
    filename: string;
    type: 'POD5' | 'BAM' | 'FASTA';
    size: number;
    sizeFormatted: string;
    numReads?: number;
    numRefs?: number;
    hasMods?: boolean;
    hasEvents?: boolean;
}

export type FilePanelIncomingMessage =
    | OpenFileMessage
    | CloseFileMessage
    | AddFilesMessage
    | AddReferenceMessage
    | ReadyMessage;
export type FilePanelOutgoingMessage = UpdateFilesMessage;

// ========== Reads View Messages ==========

export interface PlotReadMessage extends BaseMessage {
    type: 'plotRead';
    readId: string;
}

export interface PlotAggregateMessage extends BaseMessage {
    type: 'plotAggregate';
    referenceName: string;
}

export interface LoadMoreReadsMessage extends BaseMessage {
    type: 'loadMore';
}

export interface SelectReadExplorerSampleMessage extends BaseMessage {
    type: 'selectSample';
    sampleName: string;
}

export interface ExpandReferenceMessage extends BaseMessage {
    type: 'expandReference';
    referenceName: string;
    offset: number;
    limit: number;
}

export interface UpdateReadsMessage extends BaseMessage {
    type: 'updateReads';
    reads: ReadListItem[];
    groupedByReference: boolean;
    referenceToReads?: [string, ReadItem[]][]; // Map of reference name to read items
}

export interface SetReferencesOnlyMessage extends BaseMessage {
    type: 'setReferencesOnly';
    references: { referenceName: string; readCount: number }[];
}

export interface AppendReadsMessage extends BaseMessage {
    type: 'appendReads';
    reads: ReadItem[];
}

export interface SetReadsForReferenceMessage extends BaseMessage {
    type: 'setReadsForReference';
    referenceName: string;
    reads: ReadItem[];
    offset: number;
    totalCount: number;
}

export interface SetLoadingMessage extends BaseMessage {
    type: 'setLoading';
    isLoading: boolean;
    message?: string;
}

export interface SetAvailableSamplesMessage extends BaseMessage {
    type: 'setAvailableSamples';
    samples: string[];
    selectedSample: string | null;
}

export type ReadsViewIncomingMessage =
    | PlotReadMessage
    | PlotAggregateMessage
    | LoadMoreReadsMessage
    | ExpandReferenceMessage
    | SelectReadExplorerSampleMessage
    | ReadyMessage;
export type ReadsViewOutgoingMessage =
    | UpdateReadsMessage
    | SetReferencesOnlyMessage
    | AppendReadsMessage
    | SetReadsForReferenceMessage
    | SetLoadingMessage
    | SetAvailableSamplesMessage;

// ========== Plot Options Messages ==========

export interface UpdatePlotOptionsMessage extends BaseMessage {
    type: 'updatePlotOptions';
    options: PlotOptions;
}

export interface PlotOptionsChangedMessage extends BaseMessage {
    type: 'optionsChanged';
    options: PlotOptions;
}

export interface UpdateBamStatusMessage extends BaseMessage {
    type: 'updateBamStatus';
    hasBam: boolean;
}

export interface UpdatePod5StatusMessage extends BaseMessage {
    type: 'updatePod5Status';
    hasPod5: boolean;
}

export interface RequestReferencesMessage extends BaseMessage {
    type: 'requestReferences';
}

export interface UpdateReferencesMessage extends BaseMessage {
    type: 'updateReferences';
    references: string[];
}

export interface GenerateAggregatePlotMessage extends BaseMessage {
    type: 'generateAggregatePlot';
    reference: string;
    maxReads: number;
    normalization: string;
    showModifications: boolean;
    showPileup: boolean;
    showDwellTime: boolean;
    showSignal: boolean;
    showQuality: boolean;
    clipXAxisToAlignment: boolean;
    transformCoordinates: boolean;
}

export interface GenerateMultiReadOverlayMessage extends BaseMessage {
    type: 'generateMultiReadOverlay';
    sampleNames: string[];
    maxReads: number; // Max reads per sample
    normalization: string;
}

export interface GenerateMultiReadStackedMessage extends BaseMessage {
    type: 'generateMultiReadStacked';
    sampleNames: string[];
    maxReads: number; // Max reads per sample
    normalization: string;
}

export interface GenerateSignalOverlayComparisonMessage extends BaseMessage {
    type: 'generateSignalOverlayComparison';
    sampleNames: string[];
    maxReads: number;
    normalization: string;
}

export interface GenerateSignalDeltaMessage extends BaseMessage {
    type: 'generateSignalDelta';
    sampleNames: [string, string]; // Exactly 2 samples for delta
    maxReads: number;
    normalization: string;
}

export interface GenerateAggregateComparisonMessage extends BaseMessage {
    type: 'generateAggregateComparison';
    sampleNames: string[];
    reference: string;
    metrics: string[]; // ['signal', 'dwell_time', 'quality']
    maxReads: number;
    normalization: string;
}

export interface UpdateLoadedSamplesMessage extends BaseMessage {
    type: 'updateLoadedSamples';
    samples: SampleItem[];
}

export interface UpdateSelectedSamplesMessage extends BaseMessage {
    type: 'updateSelectedSamples';
    selectedSamples: string[];
}

export interface PlotOptions {
    // Analysis Type - 5 plot types (removed SINGLE_READ - use Read Explorer instead)
    plotType:
        | 'MULTI_READ_OVERLAY'
        | 'MULTI_READ_STACKED'
        | 'AGGREGATE'
        | 'COMPARE_SIGNAL_DELTA'
        | 'COMPARE_AGGREGATE';

    // Single Read options
    mode: 'SINGLE' | 'EVENTALIGN';
    normalization: 'ZNORM' | 'MAD' | 'MEDIAN' | 'NONE';
    showDwellTime: boolean;
    showBaseAnnotations: boolean;
    scaleDwellTime: boolean;
    downsample: number;
    showSignalPoints: boolean;
    clipXAxisToAlignment?: boolean;
    transformCoordinates?: boolean;

    // Multi-Read Overlay/Stacked options
    maxReadsMulti?: number;

    // Aggregate (Single Sample) options
    aggregateReference?: string;
    aggregateMaxReads?: number;
    showModifications?: boolean;
    showPileup?: boolean;
    showSignal?: boolean;
    showQuality?: boolean;

    // Comparison options
    selectedSamples?: string[]; // For multi-sample comparisons
    comparisonReference?: string; // Reference for aggregate comparison
    comparisonMetrics?: string[]; // Metrics for aggregate comparison
    comparisonMaxReads?: number;
}

export type PlotOptionsIncomingMessage =
    | PlotOptionsChangedMessage
    | RequestReferencesMessage
    | GenerateAggregatePlotMessage
    | GenerateMultiReadOverlayMessage
    | GenerateMultiReadStackedMessage
    | GenerateSignalOverlayComparisonMessage
    | GenerateSignalDeltaMessage
    | GenerateAggregateComparisonMessage
    | ToggleSampleSelectionMessage
    | ReadyMessage;
export type PlotOptionsOutgoingMessage =
    | UpdatePlotOptionsMessage
    | UpdateBamStatusMessage
    | UpdatePod5StatusMessage
    | UpdateReferencesMessage
    | UpdateLoadedSamplesMessage
    | UpdateSelectedSamplesMessage;

// ========== Modifications Panel Messages ==========

export interface UpdateModInfoMessage extends BaseMessage {
    type: 'updateModInfo';
    hasModifications: boolean;
    modificationTypes: string[];
    hasProbabilities: boolean;
}

export interface ClearModsMessage extends BaseMessage {
    type: 'clearMods';
}

export interface ModFiltersChangedMessage extends BaseMessage {
    type: 'filtersChanged';
    minProbability: number;
    enabledModTypes: string[];
}

export interface ModificationFilters {
    minProbability: number;
    enabledModTypes: string[];
}

export type ModificationsIncomingMessage = ModFiltersChangedMessage | ReadyMessage;
export type ModificationsOutgoingMessage = UpdateModInfoMessage | ClearModsMessage;

// ========== Sample Comparison Panel Messages ==========

export interface SampleItem {
    name: string;
    pod5Path: string;
    bamPath?: string;
    fastaPath?: string;
    readCount: number;
    hasBam: boolean;
    hasFasta: boolean;
}

export interface UpdateSamplesMessage extends BaseMessage {
    type: 'updateSamples';
    samples: SampleItem[];
}

export interface SelectSampleMessage extends BaseMessage {
    type: 'selectSample';
    sampleName: string;
    selected: boolean;
}

export interface UnloadSampleMessage extends BaseMessage {
    type: 'unloadSample';
    sampleName: string;
}

export interface ClearSamplesMessage extends BaseMessage {
    type: 'clearSamples';
}

export interface FilesDroppedMessage extends BaseMessage {
    type: 'filesDropped';
    filePaths: string[];
}

export interface RequestSetSessionFastaMessage extends BaseMessage {
    type: 'requestSetSessionFasta';
}

export interface RequestLoadSamplesMessage extends BaseMessage {
    type: 'requestLoadSamples';
}

export interface SetSessionFastaMessage extends BaseMessage {
    type: 'setSessionFasta';
    fastaPath: string | null;
}

export interface UpdateSessionFastaMessage extends BaseMessage {
    type: 'updateSessionFasta';
    fastaPath: string | null;
}

export interface UpdateSampleNameMessage extends BaseMessage {
    type: 'updateSampleName';
    oldName: string;
    newName: string;
}

export interface UpdateSampleColorMessage extends BaseMessage {
    type: 'updateSampleColor';
    sampleName: string;
    color: string | null; // null to clear color
}

export interface ToggleSampleSelectionMessage extends BaseMessage {
    type: 'toggleSampleSelection';
    sampleName: string;
}

export interface RequestChangeSampleBamMessage extends BaseMessage {
    type: 'requestChangeSampleBam';
    sampleName: string;
}

export interface RequestChangeSampleFastaMessage extends BaseMessage {
    type: 'requestChangeSampleFasta';
    sampleName: string;
}

export type SamplesIncomingMessage =
    | SelectSampleMessage
    | UnloadSampleMessage
    | FilesDroppedMessage
    | RequestSetSessionFastaMessage
    | RequestLoadSamplesMessage
    | SetSessionFastaMessage
    | UpdateSampleNameMessage
    | UpdateSampleColorMessage
    | ToggleSampleSelectionMessage
    | RequestChangeSampleBamMessage
    | RequestChangeSampleFastaMessage
    | ReadyMessage;
export type SamplesOutgoingMessage =
    | UpdateSamplesMessage
    | ClearSamplesMessage
    | UpdateSessionFastaMessage;

// ========== Session Manager Messages ==========

export interface LoadDemoMessage extends BaseMessage {
    type: 'loadDemo';
}

export interface SaveSessionMessage extends BaseMessage {
    type: 'save';
}

export interface RestoreSessionMessage extends BaseMessage {
    type: 'restore';
}

export interface ExportSessionMessage extends BaseMessage {
    type: 'export';
}

export interface ImportSessionMessage extends BaseMessage {
    type: 'import';
}

export interface ClearSessionMessage extends BaseMessage {
    type: 'clear';
}

export interface UpdateSessionMessage extends BaseMessage {
    type: 'updateSession';
    hasSamples: boolean;
    hasSavedSession: boolean;
    sampleCount: number;
    sampleNames: string[];
}

export type SessionPanelIncomingMessage =
    | ReadyMessage
    | LoadDemoMessage
    | SaveSessionMessage
    | RestoreSessionMessage
    | ExportSessionMessage
    | ImportSessionMessage
    | ClearSessionMessage;

export type SessionPanelOutgoingMessage = UpdateSessionMessage;

// ========== Union Types for Message Handlers ==========

export type IncomingWebviewMessage =
    | FilePanelIncomingMessage
    | ReadsViewIncomingMessage
    | PlotOptionsIncomingMessage
    | ModificationsIncomingMessage
    | SamplesIncomingMessage
    | SessionPanelIncomingMessage;

export type OutgoingWebviewMessage =
    | FilePanelOutgoingMessage
    | ReadsViewOutgoingMessage
    | PlotOptionsOutgoingMessage
    | ModificationsOutgoingMessage
    | SamplesOutgoingMessage
    | SessionPanelOutgoingMessage;
