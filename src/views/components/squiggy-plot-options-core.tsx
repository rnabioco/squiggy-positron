/**
 * Plot Options Core Component
 *
 * React-based UI for plot configuration options
 * Supports all 7 plot types with dynamic UI
 */

import React, { useEffect, useState } from 'react';
import { vscode } from './vscode-api';
import { SampleItem } from '../../types/messages';
import './squiggy-plot-options-core.css';

type PlotType =
    | 'MULTI_READ_OVERLAY'
    | 'MULTI_READ_STACKED'
    | 'AGGREGATE' // Now supports 1+ samples with view toggle
    | 'COMPARE_SIGNAL_DELTA';

interface PlotOptionsState {
    // Current selection
    plotType: PlotType;
    coordinateSpace: 'signal' | 'sequence'; // X-axis coordinate system

    // File status
    hasPod5: boolean;
    hasBam: boolean;
    hasFasta: boolean;

    // Common options
    normalization: 'NONE' | 'ZNORM' | 'MEDIAN' | 'MAD';

    // Single Read options
    plotMode: 'SINGLE' | 'EVENTALIGN';
    showDwellTime: boolean;
    showBaseAnnotations: boolean;
    scaleDwellTime: boolean;
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
    availableReferences: string[];

    // Comparison options
    loadedSamples: SampleItem[];
    selectedSamples: string[];
    comparisonReference: string;
    comparisonMetrics: string[]; // ['signal', 'dwell_time', 'quality']
    comparisonMaxReads: number;

    // X-axis windowing options
    enableXAxisWindowing: boolean;
    xAxisMin: number | null;
    xAxisMax: number | null;
    // Reference range for slider bounds (populated from selected reference)
    referenceMinPos: number;
    referenceMaxPos: number;
}

export const PlotOptionsCore: React.FC = () => {
    const [options, setOptions] = useState<PlotOptionsState>({
        plotType: 'AGGREGATE',
        coordinateSpace: 'signal', // Default to signal space
        hasPod5: false,
        hasBam: false,
        hasFasta: false,
        normalization: 'ZNORM',
        // Single Read options (used by Read Explorer clicks, not this panel)
        plotMode: 'SINGLE',
        showDwellTime: true,
        showBaseAnnotations: true,
        scaleDwellTime: false,
        downsample: 5,
        showSignalPoints: false,
        clipXAxisToAlignment: true,
        transformCoordinates: true,
        // Multi-Read
        maxReadsMulti: 50,
        // Aggregate defaults (now supports 1+ samples)
        aggregateReference: '',
        aggregateMaxReads: 100,
        aggregateViewStyle: 'overlay', // Default to overlay for multi-sample
        showModifications: true,
        showPileup: true,
        showSignal: true,
        showQuality: true,
        availableReferences: [],
        // Comparison
        loadedSamples: [],
        selectedSamples: [],
        comparisonReference: '',
        comparisonMetrics: ['signal', 'dwell_time', 'quality'],
        comparisonMaxReads: 100,
        // X-axis windowing (disabled by default)
        enableXAxisWindowing: false,
        xAxisMin: null,
        xAxisMax: null,
        referenceMinPos: 0,
        referenceMaxPos: 1000, // Default until reference range is loaded
    });

    // Send ready message on mount
    useEffect(() => {
        vscode.postMessage({ type: 'ready' });
    }, []);

    // Listen for messages from extension
    useEffect(() => {
        const handleMessage = (event: MessageEvent) => {
            const message = event.data;
            console.log('[PlotOptions React] Received message:', message.type, message);
            switch (message.type) {
                case 'updatePlotOptions':
                    setOptions((prev) => ({
                        ...prev,
                        plotType: message.options.plotType || prev.plotType,
                        plotMode: message.options.mode || prev.plotMode,
                        normalization: message.options.normalization || prev.normalization,
                        showDwellTime: message.options.showDwellTime ?? prev.showDwellTime,
                        showBaseAnnotations:
                            message.options.showBaseAnnotations ?? prev.showBaseAnnotations,
                        scaleDwellTime: message.options.scaleDwellTime ?? prev.scaleDwellTime,
                        downsample: message.options.downsample ?? prev.downsample,
                        showSignalPoints: message.options.showSignalPoints ?? prev.showSignalPoints,
                        clipXAxisToAlignment:
                            message.options.clipXAxisToAlignment ?? prev.clipXAxisToAlignment,
                        transformCoordinates:
                            message.options.transformCoordinates ?? prev.transformCoordinates,
                        aggregateReference:
                            message.options.aggregateReference || prev.aggregateReference,
                        aggregateMaxReads:
                            message.options.aggregateMaxReads ?? prev.aggregateMaxReads,
                        showModifications:
                            message.options.showModifications ?? prev.showModifications,
                        showPileup: message.options.showPileup ?? prev.showPileup,
                        showSignal: message.options.showSignal ?? prev.showSignal,
                        showQuality: message.options.showQuality ?? prev.showQuality,
                    }));
                    break;
                case 'updatePod5Status':
                    console.log('[PlotOptions React] Updating hasPod5:', message.hasPod5);
                    setOptions((prev) => ({
                        ...prev,
                        hasPod5: message.hasPod5,
                    }));
                    break;
                case 'updateBamStatus':
                    console.log('[PlotOptions React] Updating hasBam:', message.hasBam);
                    setOptions((prev) => ({
                        ...prev,
                        hasBam: message.hasBam,
                        // Enable reference-anchored mode when BAM loads (but don't auto-switch plot type)
                        showBaseAnnotations: message.hasBam ? true : prev.showBaseAnnotations,
                    }));
                    // Request references when BAM is loaded
                    if (message.hasBam) {
                        vscode.postMessage({ type: 'requestReferences' });
                    }
                    break;
                case 'updateFastaStatus':
                    console.log('[PlotOptions React] Updating hasFasta:', message.hasFasta);
                    setOptions((prev) => ({
                        ...prev,
                        hasFasta: message.hasFasta,
                    }));
                    break;
                case 'updateReferences':
                    setOptions((prev) => ({
                        ...prev,
                        availableReferences: message.references,
                        aggregateReference: message.references[0] || '',
                        comparisonReference: message.references[0] || '',
                    }));
                    // Request range for the first reference
                    if (message.references.length > 0) {
                        vscode.postMessage({
                            type: 'requestReferenceRange',
                            referenceName: message.references[0],
                        });
                    }
                    break;
                case 'updateReferenceRange':
                    console.log(
                        '[PlotOptions React] Updating reference range:',
                        message.minPos,
                        '-',
                        message.maxPos
                    );
                    setOptions((prev) => ({
                        ...prev,
                        referenceMinPos: message.minPos,
                        referenceMaxPos: message.maxPos,
                        // Reset windowing values to null when range updates
                        xAxisMin: null,
                        xAxisMax: null,
                    }));
                    break;
                case 'updateLoadedSamples':
                    console.log(
                        '[PlotOptions React] Updating loadedSamples:',
                        message.samples.length,
                        'samples'
                    );
                    setOptions((prev) => {
                        // Don't auto-select samples here - visualization selection is managed
                        // by the Sample Manager (eye icons) and synced via updateSelectedSamples message
                        // This prevents the "only first 2 samples" bug (Issue #124)
                        return {
                            ...prev,
                            loadedSamples: message.samples,
                            // Preserve existing selectedSamples - will be updated by updateSelectedSamples
                        };
                    });
                    break;
                case 'updateSelectedSamples':
                    console.log(
                        '[PlotOptions React] Updating selectedSamples from extension:',
                        message.selectedSamples
                    );
                    setOptions((prev) => ({
                        ...prev,
                        selectedSamples: message.selectedSamples,
                    }));
                    break;
            }
        };

        window.addEventListener('message', handleMessage);
        return () => window.removeEventListener('message', handleMessage);
    }, []);

    const sendMessage = (type: string, data: any) => {
        vscode.postMessage({ type, ...data });
    };

    // Helper to determine if a plot type is available
    const isPlotTypeAvailable = (type: PlotType): boolean => {
        switch (type) {
            case 'MULTI_READ_OVERLAY':
            case 'MULTI_READ_STACKED':
                return options.hasPod5;
            case 'AGGREGATE':
                // Now supports 1+ samples, all must have BAM
                return (
                    options.loadedSamples.length >= 1 &&
                    options.loadedSamples.every((s) => s.hasBam)
                );
            case 'COMPARE_SIGNAL_DELTA':
                return (
                    options.loadedSamples.length >= 2 &&
                    options.loadedSamples.every((s) => s.hasBam)
                );
            default:
                return false;
        }
    };

    const handlePlotTypeChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
        const value = e.target.value as PlotType;
        setOptions((prev) => ({ ...prev, plotType: value }));
        sendMessage('optionsChanged', {
            options: { ...options, plotType: value },
        });
    };

    const handleNormalizationChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
        const value = e.target.value as 'NONE' | 'ZNORM' | 'MEDIAN' | 'MAD';
        setOptions((prev) => ({ ...prev, normalization: value }));
        sendMessage('optionsChanged', {
            options: { ...options, normalization: value },
        });
    };

    const handleSampleSelectionChange = (sampleName: string, checked: boolean) => {
        setOptions((prev) => {
            const newSelected = checked
                ? [...prev.selectedSamples, sampleName]
                : prev.selectedSamples.filter((s) => s !== sampleName);
            return { ...prev, selectedSamples: newSelected };
        });

        // Notify extension state to keep Samples panel in sync
        vscode.postMessage({
            type: 'toggleSampleSelection',
            sampleName: sampleName,
        });
    };

    // Generate handlers for each plot type
    const handleGenerateAggregate = () => {
        // Unified handler: works for 1+ samples
        console.log('[PlotOptions] Generating aggregate with samples:', options.selectedSamples);
        sendMessage('generateAggregatePlot', {
            sampleNames: options.selectedSamples, // Now required for all aggregate plots
            reference: options.aggregateReference,
            maxReads: options.aggregateMaxReads,
            viewStyle: options.aggregateViewStyle, // 'overlay' or 'multi-track'
            normalization: options.normalization,
            showModifications: options.showModifications,
            showPileup: options.showPileup,
            showDwellTime: options.showDwellTime,
            showSignal: options.showSignal,
            showQuality: options.showQuality,
            clipXAxisToAlignment: options.clipXAxisToAlignment,
            transformCoordinates: options.transformCoordinates,
            // X-axis windowing parameters
            enableXAxisWindowing: options.enableXAxisWindowing,
            xAxisMin: options.xAxisMin,
            xAxisMax: options.xAxisMax,
        });
    };

    const handleGenerateSignalDelta = () => {
        sendMessage('generateSignalDelta', {
            sampleNames: options.selectedSamples.slice(0, 2),
            reference: options.comparisonReference,
            maxReads: options.comparisonMaxReads,
            normalization: options.normalization,
        });
    };

    const handleGenerateMultiReadOverlay = () => {
        // Coordinate space: 'signal' if not reference-anchored, 'sequence' (reference-anchored) if enabled
        const coordinateSpace = options.showBaseAnnotations ? 'sequence' : 'signal';

        sendMessage('generateMultiReadOverlay', {
            sampleNames: options.selectedSamples,
            maxReads: options.maxReadsMulti,
            normalization: options.normalization,
            coordinateSpace: coordinateSpace,
        });
    };

    const handleGenerateMultiReadStacked = () => {
        // Coordinate space: 'signal' if not reference-anchored, 'sequence' (reference-anchored) if enabled
        const coordinateSpace = options.showBaseAnnotations ? 'sequence' : 'signal';

        sendMessage('generateMultiReadStacked', {
            sampleNames: options.selectedSamples,
            maxReads: options.maxReadsMulti,
            normalization: options.normalization,
            coordinateSpace: coordinateSpace,
        });
    };

    // Determine button state based on plot type
    const getButtonState = () => {
        if (
            options.plotType === 'MULTI_READ_OVERLAY' ||
            options.plotType === 'MULTI_READ_STACKED'
        ) {
            return {
                disabled: options.selectedSamples.length === 0 || !options.hasPod5,
                text: !options.hasPod5
                    ? 'Load POD5 to generate'
                    : options.selectedSamples.length === 0
                      ? 'Enable samples in Sample Manager'
                      : 'Generate Plot',
                handler:
                    options.plotType === 'MULTI_READ_OVERLAY'
                        ? handleGenerateMultiReadOverlay
                        : handleGenerateMultiReadStacked,
            };
        } else if (options.plotType === 'AGGREGATE') {
            return {
                disabled:
                    !options.hasBam ||
                    !options.aggregateReference ||
                    options.selectedSamples.length === 0,
                text: !options.hasBam
                    ? 'Load BAM to generate'
                    : options.selectedSamples.length === 0
                      ? 'Enable samples in Sample Manager'
                      : !options.aggregateReference
                        ? 'Select reference below'
                        : 'Generate Plot',
                handler: handleGenerateAggregate,
            };
        } else {
            // COMPARE_SIGNAL_DELTA
            return {
                disabled: options.selectedSamples.length !== 2 || !options.comparisonReference,
                text:
                    options.selectedSamples.length !== 2
                        ? 'Select exactly 2 samples in Sample Manager'
                        : !options.comparisonReference
                          ? 'Select reference below'
                          : 'Generate Plot',
                handler: handleGenerateSignalDelta,
            };
        }
    };

    const buttonState = getButtonState();

    return (
        <div className="plot-options-container">
            {/* Generate Plot Button - At Top for All Plot Types */}
            <button
                onClick={buttonState.handler}
                disabled={buttonState.disabled}
                className="plot-options-generate-button"
            >
                {buttonState.text}
            </button>

            {/* Analysis Type Section */}
            <div className="plot-options-section">
                <div className="plot-options-section-header">Analysis Type</div>
                <select
                    value={options.plotType}
                    onChange={handlePlotTypeChange}
                    disabled={!options.hasPod5}
                    className="plot-options-select"
                    style={{
                        opacity: options.hasPod5 ? 1 : 0.5,
                        cursor: options.hasPod5 ? 'default' : 'not-allowed',
                    }}
                >
                    <option
                        value="MULTI_READ_OVERLAY"
                        disabled={!isPlotTypeAvailable('MULTI_READ_OVERLAY')}
                    >
                        Per-Read Plots
                        {!isPlotTypeAvailable('MULTI_READ_OVERLAY') ? ' (requires POD5)' : ''}
                    </option>
                    <option value="AGGREGATE" disabled={!isPlotTypeAvailable('AGGREGATE')}>
                        Composite Read Plots
                        {!isPlotTypeAvailable('AGGREGATE') ? ' (requires BAM)' : ''}
                    </option>
                    <option
                        value="COMPARE_SIGNAL_DELTA"
                        disabled={!isPlotTypeAvailable('COMPARE_SIGNAL_DELTA')}
                    >
                        2-Sample Comparisons
                        {!isPlotTypeAvailable('COMPARE_SIGNAL_DELTA')
                            ? ' (requires 2 samples with BAM)'
                            : ''}
                    </option>
                </select>
                <div className="plot-options-description">
                    {!options.hasPod5 && 'Load POD5 file to enable plotting'}
                    {options.hasPod5 &&
                        options.loadedSamples.length < 2 &&
                        options.plotType.startsWith('COMPARE') &&
                        'Load 2+ samples in Sample Manager for comparisons'}
                </div>
            </div>

            {/* Coordinate Space Toggle - DEFERRED: Requires FASTA reference implementation
            <div style={{ marginBottom: '20px' }}>
                <div
                    style={{
                        fontWeight: 'bold',
                        marginBottom: '8px',
                        color: 'var(--vscode-foreground)',
                    }}
                >
                    X-Axis Coordinates
                </div>
                <select
                    value={options.coordinateSpace}
                    onChange={(e) =>
                        setOptions((prev) => ({
                            ...prev,
                            coordinateSpace: e.target.value as 'signal' | 'sequence',
                        }))
                    }
                    style={{
                        width: '100%',
                        padding: '6px',
                        background: 'var(--vscode-input-background)',
                        color: 'var(--vscode-input-foreground)',
                        border: '1px solid var(--vscode-input-border)',
                    }}
                >
                    <option value="signal">Signal Space (sample points)</option>
                    <option value="sequence" disabled={!options.hasBam}>
                        Reference Space (base positions)
                        {!options.hasBam ? ' - requires FASTA' : ''}
                    </option>
                </select>
                <div
                    style={{
                        fontSize: '0.75em',
                        color: 'var(--vscode-descriptionForeground)',
                        fontStyle: 'italic',
                        marginTop: '4px',
                    }}
                >
                    {options.coordinateSpace === 'signal'
                        ? 'X-axis shows raw sample indices'
                        : 'X-axis shows reference genome positions (requires FASTA reference)'}
                </div>
            </div>
            */}

            {/* Normalization - Common to all types */}
            <div className="plot-options-section">
                <div className="plot-options-section-header">Normalization</div>
                <select
                    value={options.normalization}
                    onChange={handleNormalizationChange}
                    className="plot-options-select"
                >
                    <option value="NONE">None (raw signal)</option>
                    <option value="ZNORM">Z-score</option>
                    <option value="MEDIAN">Median-centered</option>
                    <option value="MAD">Median Absolute Deviation</option>
                </select>
            </div>

            {/* Dynamic UI based on plot type */}
            {(options.plotType === 'MULTI_READ_OVERLAY' ||
                options.plotType === 'MULTI_READ_STACKED') && (
                <div>
                    {/* View Style: Overlay vs Stacked */}
                    <div className="plot-options-section">
                        <div className="plot-options-section-header">View Style</div>
                        <div
                            className="plot-options-radio-group"
                            style={{ flexDirection: 'row', gap: '16px' }}
                        >
                            <label className="plot-options-radio-label">
                                <input
                                    type="radio"
                                    name="perReadViewStyle"
                                    checked={options.plotType === 'MULTI_READ_OVERLAY'}
                                    onChange={() =>
                                        setOptions((prev) => ({
                                            ...prev,
                                            plotType: 'MULTI_READ_OVERLAY',
                                        }))
                                    }
                                />
                                <span>Overlay (alpha-blended)</span>
                            </label>
                            <label className="plot-options-radio-label">
                                <input
                                    type="radio"
                                    name="perReadViewStyle"
                                    checked={options.plotType === 'MULTI_READ_STACKED'}
                                    onChange={() =>
                                        setOptions((prev) => ({
                                            ...prev,
                                            plotType: 'MULTI_READ_STACKED',
                                        }))
                                    }
                                />
                                <span>Stacked (offset)</span>
                            </label>
                        </div>
                    </div>

                    {/* Display Options */}
                    <div className="plot-options-section">
                        <div className="plot-options-section-header">Display Options</div>

                        {/* Reference-anchored mode */}
                        <label
                            className="plot-options-checkbox-label"
                            style={{
                                opacity: options.hasBam ? 1 : 0.5,
                                pointerEvents: options.hasBam ? 'auto' : 'none',
                            }}
                        >
                            <input
                                type="checkbox"
                                checked={options.showBaseAnnotations}
                                disabled={!options.hasBam}
                                onChange={(e) =>
                                    setOptions((prev) => ({
                                        ...prev,
                                        showBaseAnnotations: e.target.checked,
                                    }))
                                }
                            />
                            <span>Reference-anchored</span>
                        </label>
                        <div
                            className="plot-options-helper-text"
                            style={{ marginLeft: '24px', marginBottom: '8px' }}
                        >
                            {!options.hasBam
                                ? 'Requires BAM file with alignment'
                                : 'Plot using genomic coordinates (x-axis = reference position)'}
                        </div>

                        {/* Show reference track - only when reference-anchored mode enabled and FASTA loaded */}
                        {options.showBaseAnnotations && (
                            <label
                                className="plot-options-checkbox-label"
                                style={{
                                    marginLeft: '24px',
                                    opacity: options.hasFasta ? 1 : 0.5,
                                    pointerEvents: options.hasFasta ? 'auto' : 'none',
                                }}
                            >
                                <input
                                    type="checkbox"
                                    checked={options.hasFasta}
                                    disabled={!options.hasFasta}
                                    readOnly
                                />
                                <span>Show reference track</span>
                            </label>
                        )}
                        {options.showBaseAnnotations && (
                            <div
                                className="plot-options-helper-text"
                                style={{ marginLeft: '48px', marginBottom: '8px' }}
                            >
                                {!options.hasFasta
                                    ? 'Load FASTA file to show reference sequence'
                                    : 'Reference bases with mismatch highlighting'}
                            </div>
                        )}
                    </div>

                    {/* Max Reads per Sample */}
                    <div className="plot-options-section">
                        <div className="plot-options-slider-label">
                            <span>Max reads per sample:</span>
                            <span>{options.maxReadsMulti}</span>
                        </div>
                        <input
                            type="range"
                            min="2"
                            max="100"
                            step="1"
                            value={options.maxReadsMulti}
                            onChange={(e) =>
                                setOptions((prev) => ({
                                    ...prev,
                                    maxReadsMulti: parseInt(e.target.value),
                                }))
                            }
                            className="plot-options-range-slider"
                        />
                        <div className="plot-options-helper-text">
                            Number of reads to extract from each sample
                        </div>
                    </div>

                    {/* Warning for stacked plots with too many reads */}
                    {options.plotType === 'MULTI_READ_STACKED' &&
                        options.selectedSamples.length * options.maxReadsMulti > 20 && (
                            <div className="plot-options-warning">
                                ⚠️ Stacked plots work best with ≤20 total reads (currently:{' '}
                                {options.selectedSamples.length * options.maxReadsMulti})
                            </div>
                        )}
                </div>
            )}

            {options.plotType === 'AGGREGATE' && (
                <>
                    {/* Reference Selection */}
                    <div className="plot-options-section">
                        <div className="plot-options-section-header">Reference</div>
                        <select
                            value={options.aggregateReference}
                            onChange={(e) => {
                                const newRef = e.target.value;
                                setOptions((prev) => ({
                                    ...prev,
                                    aggregateReference: newRef,
                                }));
                                // Request range for the new reference
                                if (newRef) {
                                    vscode.postMessage({
                                        type: 'requestReferenceRange',
                                        referenceName: newRef,
                                    });
                                }
                            }}
                            disabled={!options.hasBam}
                            className="plot-options-select"
                            style={{
                                opacity: options.hasBam ? 1 : 0.5,
                            }}
                        >
                            {options.availableReferences.length > 0 ? (
                                options.availableReferences.map((ref) => (
                                    <option key={ref} value={ref}>
                                        {ref}
                                    </option>
                                ))
                            ) : (
                                <option value="">No references (load BAM file)</option>
                            )}
                        </select>
                    </div>

                    {/* View Style (for multi-sample) - only show if 2+ samples selected in Samples panel */}
                    {options.selectedSamples.length > 1 && (
                        <div className="plot-options-section-large">
                            <div className="plot-options-section-header">View Style</div>
                            <select
                                value={options.aggregateViewStyle}
                                onChange={(e) =>
                                    setOptions((prev) => ({
                                        ...prev,
                                        aggregateViewStyle: e.target.value as
                                            | 'overlay'
                                            | 'multi-track',
                                    }))
                                }
                                className="plot-options-select"
                            >
                                <option value="overlay">Overlay (Mean Signals)</option>
                                <option value="multi-track">Multi-Track (Detailed)</option>
                            </select>
                            <div className="plot-options-helper-text">
                                {options.aggregateViewStyle === 'overlay'
                                    ? 'Overlays mean signals from all samples on one plot'
                                    : 'Shows detailed 5-track view for each sample'}
                            </div>
                        </div>
                    )}

                    {/* Max Reads */}
                    <div className="plot-options-section-large">
                        <div className="plot-options-slider-label">
                            <span>Maximum reads:</span>
                            <span>{options.aggregateMaxReads}</span>
                        </div>
                        <input
                            type="range"
                            min="10"
                            max="500"
                            step="10"
                            value={options.aggregateMaxReads}
                            onChange={(e) =>
                                setOptions((prev) => ({
                                    ...prev,
                                    aggregateMaxReads: parseInt(e.target.value),
                                }))
                            }
                            disabled={!options.hasBam}
                            className="plot-options-range-slider"
                        />
                    </div>

                    {/* Panel Visibility */}
                    <div className="plot-options-section">
                        <div className="plot-options-section-header">Visible Panels</div>

                        {/* Modifications Panel */}
                        <div className="plot-options-checkbox-row">
                            <input
                                type="checkbox"
                                id="showModifications"
                                checked={options.showModifications}
                                onChange={(e) =>
                                    setOptions((prev) => ({
                                        ...prev,
                                        showModifications: e.target.checked,
                                    }))
                                }
                                disabled={!options.hasBam}
                            />
                            <label
                                htmlFor="showModifications"
                                className="plot-options-checkbox-label"
                            >
                                Base modifications
                            </label>
                        </div>
                        <div
                            className="plot-options-helper-text"
                            style={{
                                marginLeft: '22px',
                                marginBottom: '12px',
                                marginTop: '-4px',
                            }}
                        >
                            Adjust filters in Modifications Explorer panel
                        </div>

                        {/* Pileup Panel */}
                        <div className="plot-options-checkbox-row">
                            <input
                                type="checkbox"
                                id="showPileup"
                                checked={options.showPileup}
                                onChange={(e) =>
                                    setOptions((prev) => ({
                                        ...prev,
                                        showPileup: e.target.checked,
                                    }))
                                }
                                disabled={!options.hasBam}
                            />
                            <label htmlFor="showPileup" className="plot-options-checkbox-label">
                                Base pileup
                            </label>
                        </div>

                        {/* Dwell Time Panel */}
                        <div className="plot-options-checkbox-row">
                            <input
                                type="checkbox"
                                id="showDwellTimeAggregate"
                                checked={options.showDwellTime}
                                onChange={(e) =>
                                    setOptions((prev) => ({
                                        ...prev,
                                        showDwellTime: e.target.checked,
                                    }))
                                }
                                disabled={!options.hasBam}
                            />
                            <label
                                htmlFor="showDwellTimeAggregate"
                                className="plot-options-checkbox-label"
                            >
                                Dwell time
                            </label>
                        </div>

                        {/* Signal Panel */}
                        <div className="plot-options-checkbox-row">
                            <input
                                type="checkbox"
                                id="showSignalAggregate"
                                checked={options.showSignal}
                                onChange={(e) =>
                                    setOptions((prev) => ({
                                        ...prev,
                                        showSignal: e.target.checked,
                                    }))
                                }
                                disabled={!options.hasBam}
                            />
                            <label
                                htmlFor="showSignalAggregate"
                                className="plot-options-checkbox-label"
                            >
                                Signal
                            </label>
                        </div>

                        {/* Quality Panel */}
                        <div className="plot-options-checkbox-row">
                            <input
                                type="checkbox"
                                id="showQualityAggregate"
                                checked={options.showQuality}
                                onChange={(e) =>
                                    setOptions((prev) => ({
                                        ...prev,
                                        showQuality: e.target.checked,
                                    }))
                                }
                                disabled={!options.hasBam}
                            />
                            <label
                                htmlFor="showQualityAggregate"
                                className="plot-options-checkbox-label"
                            >
                                Quality scores
                            </label>
                        </div>
                    </div>

                    {/* X-Axis Options */}
                    <div
                        className="plot-options-section-large"
                        style={{
                            opacity: options.hasBam ? 1 : 0.5,
                            pointerEvents: options.hasBam ? 'auto' : 'none',
                        }}
                    >
                        <div className="plot-options-section-header">X-Axis Display</div>

                        {/* Clip to Consensus */}
                        <div className="plot-options-checkbox-row">
                            <input
                                type="checkbox"
                                id="clipXAxisToAlignmentAggregate"
                                checked={options.clipXAxisToAlignment}
                                onChange={(e) =>
                                    setOptions((prev) => ({
                                        ...prev,
                                        clipXAxisToAlignment: e.target.checked,
                                    }))
                                }
                                disabled={!options.hasBam}
                            />
                            <label
                                htmlFor="clipXAxisToAlignmentAggregate"
                                className="plot-options-checkbox-label"
                            >
                                Clip x-axis to consensus region
                            </label>
                        </div>
                        <div
                            className="plot-options-description"
                            style={{
                                marginTop: '-6px',
                                marginBottom: '10px',
                            }}
                        >
                            Focus on high-coverage region (uncheck to show full reference range)
                        </div>

                        {/* Transform Coordinates */}
                        <div className="plot-options-checkbox-row">
                            <input
                                type="checkbox"
                                id="transformCoordinatesAggregate"
                                checked={options.transformCoordinates}
                                onChange={(e) =>
                                    setOptions((prev) => ({
                                        ...prev,
                                        transformCoordinates: e.target.checked,
                                    }))
                                }
                                disabled={!options.hasBam}
                            />
                            <label
                                htmlFor="transformCoordinatesAggregate"
                                className="plot-options-checkbox-label"
                            >
                                Transform to relative coordinates
                            </label>
                        </div>
                        <div
                            className="plot-options-description"
                            style={{
                                marginTop: '-6px',
                                marginBottom: '10px',
                            }}
                        >
                            Anchor position 1 to first reference base (uncheck to use genomic
                            coordinates)
                        </div>
                    </div>

                    {/* X-Axis Windowing Section */}
                    <div
                        className="plot-options-section-large"
                        style={{
                            opacity: options.hasBam ? 1 : 0.5,
                            pointerEvents: options.hasBam ? 'auto' : 'none',
                        }}
                    >
                        <div className="plot-options-section-header">X-Axis Windowing</div>

                        {/* Enable Windowing Checkbox */}
                        <div className="plot-options-checkbox-row">
                            <input
                                type="checkbox"
                                id="enableXAxisWindowing"
                                checked={options.enableXAxisWindowing}
                                onChange={(e) =>
                                    setOptions((prev) => ({
                                        ...prev,
                                        enableXAxisWindowing: e.target.checked,
                                    }))
                                }
                                disabled={!options.hasBam}
                            />
                            <label
                                htmlFor="enableXAxisWindowing"
                                className="plot-options-checkbox-label"
                            >
                                Enable x-axis windowing
                            </label>
                        </div>
                        <div
                            className="plot-options-description"
                            style={{
                                marginTop: '-6px',
                                marginBottom: '12px',
                            }}
                        >
                            Limit plot to a specific range (useful for long reads 1000+ nt)
                        </div>

                        {/* Dual-Handle Range Slider - Only shown when windowing enabled */}
                        {options.enableXAxisWindowing && (
                            <div className="plot-options-windowing-container">
                                <div className="plot-options-windowing-inputs">
                                    <div className="plot-options-windowing-input-group">
                                        <label className="plot-options-windowing-label">
                                            Min Position:
                                        </label>
                                        <input
                                            type="number"
                                            className="plot-options-windowing-input"
                                            min={options.referenceMinPos}
                                            max={options.referenceMaxPos}
                                            value={options.xAxisMin ?? options.referenceMinPos}
                                            onChange={(e) => {
                                                const val = parseInt(e.target.value);
                                                if (!isNaN(val)) {
                                                    setOptions((prev) => ({
                                                        ...prev,
                                                        xAxisMin: Math.max(
                                                            options.referenceMinPos,
                                                            Math.min(val, options.xAxisMax ?? options.referenceMaxPos)
                                                        ),
                                                    }));
                                                }
                                            }}
                                        />
                                    </div>
                                    <div className="plot-options-windowing-input-group">
                                        <label className="plot-options-windowing-label">
                                            Max Position:
                                        </label>
                                        <input
                                            type="number"
                                            className="plot-options-windowing-input"
                                            min={options.referenceMinPos}
                                            max={options.referenceMaxPos}
                                            value={options.xAxisMax ?? options.referenceMaxPos}
                                            onChange={(e) => {
                                                const val = parseInt(e.target.value);
                                                if (!isNaN(val)) {
                                                    setOptions((prev) => ({
                                                        ...prev,
                                                        xAxisMax: Math.min(
                                                            options.referenceMaxPos,
                                                            Math.max(val, options.xAxisMin ?? options.referenceMinPos)
                                                        ),
                                                    }));
                                                }
                                            }}
                                        />
                                    </div>
                                </div>

                                <button
                                    onClick={() =>
                                        setOptions((prev) => ({
                                            ...prev,
                                            xAxisMin: null,
                                            xAxisMax: null,
                                        }))
                                    }
                                    className="plot-options-reset-button"
                                >
                                    Reset to Full Range
                                </button>
                                <div className="plot-options-helper-text" style={{ marginTop: '8px' }}>
                                    Reference range: {options.referenceMinPos} - {options.referenceMaxPos} (from {options.aggregateReference || 'selected reference'})
                                </div>
                            </div>
                        )}
                    </div>
                </>
            )}

            {options.plotType === 'COMPARE_SIGNAL_DELTA' && (
                <>
                    {/* Sample Selection */}
                    <div className="plot-options-section">
                        <div className="plot-options-section-header">Samples to Compare</div>
                        {options.loadedSamples.length === 0 ? (
                            <div className="plot-options-description">
                                Load samples in Sample Manager to enable comparisons
                            </div>
                        ) : (
                            <div className="plot-options-sample-list">
                                {options.loadedSamples.map((sample) => (
                                    <div key={sample.name} className="plot-options-sample-item">
                                        <input
                                            type="checkbox"
                                            id={`sample-${sample.name}`}
                                            checked={options.selectedSamples.includes(sample.name)}
                                            onChange={(e) =>
                                                handleSampleSelectionChange(
                                                    sample.name,
                                                    e.target.checked
                                                )
                                            }
                                            disabled={
                                                options.plotType === 'COMPARE_SIGNAL_DELTA' &&
                                                options.selectedSamples.length >= 2 &&
                                                !options.selectedSamples.includes(sample.name)
                                            }
                                        />
                                        <label
                                            htmlFor={`sample-${sample.name}`}
                                            className="plot-options-checkbox-label"
                                        >
                                            {sample.name} ({sample.readCount} reads)
                                        </label>
                                    </div>
                                ))}
                            </div>
                        )}
                        {options.plotType === 'COMPARE_SIGNAL_DELTA' && (
                            <div className="plot-options-helper-text">
                                Delta plots require exactly 2 samples
                            </div>
                        )}
                    </div>

                    {/* Reference Selection */}
                    <div className="plot-options-section">
                        <div className="plot-options-section-header">Reference</div>
                        <select
                            value={options.comparisonReference}
                            onChange={(e) =>
                                setOptions((prev) => ({
                                    ...prev,
                                    comparisonReference: e.target.value,
                                }))
                            }
                            disabled={!options.hasBam}
                            className="plot-options-select"
                            style={{
                                opacity: options.hasBam ? 1 : 0.5,
                            }}
                        >
                            {options.availableReferences.length > 0 ? (
                                options.availableReferences.map((ref) => (
                                    <option key={ref} value={ref}>
                                        {ref}
                                    </option>
                                ))
                            ) : (
                                <option value="">No references (load BAM file)</option>
                            )}
                        </select>
                    </div>

                    {/* Max Reads */}
                    <div className="plot-options-section-large">
                        <div className="plot-options-slider-label">
                            <span>Maximum reads per sample:</span>
                            <span>{options.comparisonMaxReads}</span>
                        </div>
                        <input
                            type="range"
                            min="10"
                            max="500"
                            step="10"
                            value={options.comparisonMaxReads}
                            onChange={(e) =>
                                setOptions((prev) => ({
                                    ...prev,
                                    comparisonMaxReads: parseInt(e.target.value),
                                }))
                            }
                            className="plot-options-range-slider"
                        />
                    </div>
                </>
            )}
        </div>
    );
};
