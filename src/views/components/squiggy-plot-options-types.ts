/**
 * Shared types for the Plot Options panel and its section components.
 */

import React from 'react';
import { SampleItem } from '../../types/messages';

export type PlotType =
    | 'MULTI_READ_OVERLAY'
    | 'MULTI_READ_STACKED'
    | 'REFERENCE_OVERLAY'
    | 'AGGREGATE' // Now supports 1+ samples with view toggle
    | 'COMPARE_SIGNAL_DELTA';

export interface PlotOptionsState {
    // Current selection
    plotType: PlotType;
    coordinateSpace: 'signal' | 'sequence'; // X-axis coordinate system

    // File status
    hasPod5: boolean;
    hasBam: boolean;
    hasFasta: boolean;
    hasEvents: boolean; // Any selected sample has mv tags (signal-to-base mapping)
    hasMods: boolean; // Any selected sample has MM/ML tags (base modifications)
    hasPrimers: boolean; // Any selected sample has PT/pt tag (primer/adapter trim)

    // Common options
    normalization: 'NONE' | 'ZNORM' | 'MEDIAN' | 'MAD';

    // Single Read options
    plotMode: 'SINGLE' | 'EVENTALIGN';
    showDwellTime: boolean;
    showBaseAnnotations: boolean;
    scaleDwellTime: boolean;
    showBaseColors: boolean;
    downsample: number;
    showSignalPoints: boolean;
    clipXAxisToAlignment: boolean;
    transformCoordinates: boolean;

    // Multi-Read options (Overlay/Stacked)
    maxReadsMulti: number;

    // Aggregate options (now supports 1+ samples)
    aggregateReference: string;
    aggregateMaxReads: number;
    aggregateViewStyle: 'overlay' | 'multi-track'; // For multi-sample: overlay or separate tracks
    showModifications: boolean;
    showPileup: boolean;
    showSignal: boolean;
    showQuality: boolean;
    showCoverage: boolean;
    rnaMode: boolean;
    trimPrimers: boolean;
    primer5p: string;
    adapter3p: string;
    showAdvanced: boolean;
    availableReferences: string[];

    // Comparison options
    loadedSamples: SampleItem[];
    selectedSamples: string[];
    comparisonReference: string;
    comparisonMetrics: string[]; // ['signal', 'dwell_time', 'quality']
    comparisonMaxReads: number;
}

export type SetPlotOptions = React.Dispatch<React.SetStateAction<PlotOptionsState>>;
