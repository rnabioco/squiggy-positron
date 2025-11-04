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

export type FilePanelIncomingMessage = OpenFileMessage | CloseFileMessage | ReadyMessage;
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

export type ReadsViewIncomingMessage =
    | PlotReadMessage
    | PlotAggregateMessage
    | LoadMoreReadsMessage
    | ExpandReferenceMessage
    | ReadyMessage;
export type ReadsViewOutgoingMessage =
    | UpdateReadsMessage
    | SetReferencesOnlyMessage
    | AppendReadsMessage
    | SetReadsForReferenceMessage
    | SetLoadingMessage;

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

export interface PlotOptions {
    mode: 'SINGLE' | 'EVENTALIGN';
    normalization: 'ZNORM' | 'MAD' | 'MEDIAN' | 'NONE';
    showDwellTime: boolean;
    showBaseAnnotations: boolean;
    scaleDwellTime: boolean;
    downsample: number;
    showSignalPoints: boolean;
}

export type PlotOptionsIncomingMessage = PlotOptionsChangedMessage | ReadyMessage;
export type PlotOptionsOutgoingMessage = UpdatePlotOptionsMessage | UpdateBamStatusMessage;

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

export interface StartComparisonMessage extends BaseMessage {
    type: 'startComparison';
    sampleNames: string[];
}

export interface UnloadSampleMessage extends BaseMessage {
    type: 'unloadSample';
    sampleName: string;
}

export interface ClearSamplesMessage extends BaseMessage {
    type: 'clearSamples';
}

export type SamplesIncomingMessage =
    | SelectSampleMessage
    | StartComparisonMessage
    | UnloadSampleMessage
    | ReadyMessage;
export type SamplesOutgoingMessage = UpdateSamplesMessage | ClearSamplesMessage;

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
