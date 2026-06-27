/**
 * Plot Options Core Component
 *
 * React-based UI for plot configuration options
 * Supports all 7 plot types with dynamic UI
 *
 * This is the state container: it owns the options state, handles extension
 * messages, and exposes handlers. The per-plot-type controls are rendered by
 * the PerReadSection / AggregateSection / CompareDeltaSection components.
 */

import React, { useEffect, useState } from 'react';
import { vscode } from './vscode-api';
import { SampleItem } from '../../types/messages';
import { PlotOptionsState, PlotType } from './squiggy-plot-options-types';
import { PerReadSection } from './squiggy-plot-options-per-read';
import { AggregateSection } from './squiggy-plot-options-aggregate';
import { CompareDeltaSection } from './squiggy-plot-options-compare';
import './squiggy-plot-options-core.css';

export const PlotOptionsCore: React.FC = () => {
    const [options, setOptions] = useState<PlotOptionsState>({
        plotType: 'AGGREGATE',
        coordinateSpace: 'signal', // Default to signal space
        hasPod5: false,
        hasBam: false,
        hasFasta: false,
        hasEvents: false, // Default to false - will be updated when samples load
        hasMods: false, // Default to false - will be updated when samples load
        hasPrimers: false, // Default to false - will be updated when samples load
        normalization: 'ZNORM',
        // Single Read options (used by Read Explorer clicks, not this panel)
        plotMode: 'SINGLE',
        showDwellTime: true,
        showBaseAnnotations: true,
        scaleDwellTime: false,
        showBaseColors: true,
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
        showCoverage: false, // Off by default
        rnaMode: false,
        trimPrimers: true, // Default: trim primers (don't show adapter regions)
        primer5p: 'CCTAAGAGCAAGAAGAAGCCTGGN',
        adapter3p: 'GGCTTCTTCTTGCTCTTCC',
        showAdvanced: false,
        availableReferences: [],
        // Comparison
        loadedSamples: [],
        selectedSamples: [],
        comparisonReference: '',
        comparisonMetrics: ['signal', 'dwell_time', 'quality'],
        comparisonMaxReads: 100,
    });

    // Send ready message on mount
    useEffect(() => {
        vscode.postMessage({ type: 'ready' });
    }, []);

    // Listen for messages from extension
    useEffect(() => {
        const handleMessage = (event: MessageEvent) => {
            const message = event.data;
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
                        showBaseColors: message.options.showBaseColors ?? prev.showBaseColors,
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
                        showCoverage: message.options.showCoverage ?? prev.showCoverage,
                        rnaMode: message.options.rnaMode ?? prev.rnaMode,
                        trimPrimers: message.options.trimPrimers ?? prev.trimPrimers,
                    }));
                    break;
                case 'updatePod5Status':
                    setOptions((prev) => ({
                        ...prev,
                        hasPod5: message.hasPod5,
                    }));
                    break;
                case 'updateBamStatus':
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
                    break;
                case 'updateLoadedSamples':
                    setOptions((prev) => {
                        // Don't auto-select samples here - visualization selection is managed
                        // by the Sample Manager (eye icons) and synced via updateSelectedSamples message
                        // This prevents the "only first 2 samples" bug (Issue #124)

                        // Compute hasEvents/hasMods/hasPrimers from selected samples
                        const selectedSampleData = message.samples.filter((s: SampleItem) =>
                            prev.selectedSamples.includes(s.name)
                        );
                        const hasEvents = selectedSampleData.some(
                            (s: SampleItem) => s.hasEvents === true
                        );
                        const hasMods = selectedSampleData.some(
                            (s: SampleItem) => s.hasMods === true
                        );
                        const hasPrimers = selectedSampleData.some(
                            (s: SampleItem) => s.hasPrimers === true
                        );

                        return {
                            ...prev,
                            loadedSamples: message.samples,
                            hasEvents,
                            hasMods,
                            hasPrimers,
                            // Preserve existing selectedSamples - will be updated by updateSelectedSamples
                        };
                    });
                    break;
                case 'updateSelectedSamples':
                    setOptions((prev) => {
                        // Recompute hasEvents/hasMods/hasPrimers based on new selection
                        const selectedSampleData = prev.loadedSamples.filter((s) =>
                            message.selectedSamples.includes(s.name)
                        );
                        const hasEvents = selectedSampleData.some((s) => s.hasEvents === true);
                        const hasMods = selectedSampleData.some((s) => s.hasMods === true);
                        const hasPrimers = selectedSampleData.some((s) => s.hasPrimers === true);

                        return {
                            ...prev,
                            selectedSamples: message.selectedSamples,
                            hasEvents,
                            hasMods,
                            hasPrimers,
                        };
                    });
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
            case 'REFERENCE_OVERLAY':
                return options.hasPod5 && options.hasBam;
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
        // Send effective values - if hasEvents is false, signal/dwell are forced off
        // This triggers plot_pileup() instead of plot_aggregate() for BAMs without mv tags
        const effectiveShowDwellTime = options.showDwellTime && options.hasEvents;
        const effectiveShowSignal = options.showSignal && options.hasEvents;

        sendMessage('generateAggregatePlot', {
            sampleNames: options.selectedSamples, // Now required for all aggregate plots
            reference: options.aggregateReference,
            maxReads: options.aggregateMaxReads,
            viewStyle: options.aggregateViewStyle, // 'overlay' or 'multi-track'
            normalization: options.normalization,
            showModifications: options.showModifications,
            showPileup: options.showPileup,
            showDwellTime: effectiveShowDwellTime,
            showSignal: effectiveShowSignal,
            showQuality: options.showQuality,
            showCoverage: options.showCoverage,
            clipXAxisToAlignment: options.clipXAxisToAlignment,
            transformCoordinates: options.transformCoordinates,
            rnaMode: options.rnaMode,
            trimPrimers: options.trimPrimers,
            primer5p: options.primer5p,
            adapter3p: options.adapter3p,
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

    const handleGenerateReferenceOverlay = () => {
        sendMessage('generateReferenceOverlay', {
            sampleNames: options.selectedSamples,
            maxReads: options.maxReadsMulti,
            normalization: options.normalization,
            reference: options.aggregateReference,
        });
    };

    // Determine button state based on plot type
    const getButtonState = () => {
        if (
            options.plotType === 'MULTI_READ_OVERLAY' ||
            options.plotType === 'MULTI_READ_STACKED' ||
            options.plotType === 'REFERENCE_OVERLAY'
        ) {
            const isRefOverlay = options.plotType === 'REFERENCE_OVERLAY';
            const disabled = isRefOverlay
                ? options.selectedSamples.length === 0 ||
                  !options.hasPod5 ||
                  !options.hasBam ||
                  !options.aggregateReference
                : options.selectedSamples.length === 0 || !options.hasPod5;
            const text = !options.hasPod5
                ? 'Load POD5 to generate'
                : isRefOverlay && !options.hasBam
                  ? 'Load BAM for reference overlay'
                  : isRefOverlay && !options.aggregateReference
                    ? 'Select reference below'
                    : options.selectedSamples.length === 0
                      ? 'Enable samples in Sample Manager'
                      : 'Generate Plot';
            const handler = isRefOverlay
                ? handleGenerateReferenceOverlay
                : options.plotType === 'MULTI_READ_OVERLAY'
                  ? handleGenerateMultiReadOverlay
                  : handleGenerateMultiReadStacked;

            return { disabled, text, handler };
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

    const isPerReadPlot =
        options.plotType === 'MULTI_READ_OVERLAY' ||
        options.plotType === 'MULTI_READ_STACKED' ||
        options.plotType === 'REFERENCE_OVERLAY';

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
            {isPerReadPlot && (
                <PerReadSection
                    options={options}
                    setOptions={setOptions}
                    isPlotTypeAvailable={isPlotTypeAvailable}
                    sendMessage={sendMessage}
                />
            )}

            {options.plotType === 'AGGREGATE' && (
                <AggregateSection options={options} setOptions={setOptions} />
            )}

            {options.plotType === 'COMPARE_SIGNAL_DELTA' && (
                <CompareDeltaSection
                    options={options}
                    setOptions={setOptions}
                    handleSampleSelectionChange={handleSampleSelectionChange}
                />
            )}
        </div>
    );
};
