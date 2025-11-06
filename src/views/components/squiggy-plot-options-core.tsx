/**
 * Plot Options Core Component
 *
 * React-based UI for plot configuration options
 * Supports all 7 plot types with dynamic UI
 */

import React, { useEffect, useState } from 'react';
import { vscode } from './vscode-api';
import { SampleItem } from '../../types/messages';

type PlotType =
    | 'MULTI_READ_OVERLAY'
    | 'MULTI_READ_STACKED'
    | 'AGGREGATE'
    | 'COMPARE_SIGNAL_DELTA'
    | 'COMPARE_AGGREGATE';

interface PlotOptionsState {
    // Current selection
    plotType: PlotType;

    // File status
    hasPod5: boolean;
    hasBam: boolean;

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

    // Aggregate (Single Sample) options
    aggregateReference: string;
    aggregateMaxReads: number;
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
}

export const PlotOptionsCore: React.FC = () => {
    const [options, setOptions] = useState<PlotOptionsState>({
        plotType: 'AGGREGATE',
        hasPod5: false,
        hasBam: false,
        normalization: 'ZNORM',
        // Single Read options (used by Read Explorer clicks, not this panel)
        plotMode: 'SINGLE',
        showDwellTime: false,
        showBaseAnnotations: true,
        scaleDwellTime: false,
        downsample: 5,
        showSignalPoints: false,
        clipXAxisToAlignment: true,
        transformCoordinates: true,
        // Multi-Read
        maxReadsMulti: 50,
        // Aggregate defaults
        aggregateReference: '',
        aggregateMaxReads: 100,
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
                    setOptions((prev) => {
                        const newOptions: PlotOptionsState = {
                            ...prev,
                            hasBam: message.hasBam,
                            // When BAM loads, switch to AGGREGATE/EVENTALIGN
                            // When BAM unloads, switch to MULTI_READ_OVERLAY
                            plotMode: (message.hasBam ? 'EVENTALIGN' : 'SINGLE') as
                                | 'SINGLE'
                                | 'EVENTALIGN',
                            plotType: (message.hasBam
                                ? 'AGGREGATE'
                                : 'MULTI_READ_OVERLAY') as PlotType,
                        };
                        return newOptions;
                    });
                    // Request references when BAM is loaded
                    if (message.hasBam) {
                        vscode.postMessage({ type: 'requestReferences' });
                    }
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
                    console.log(
                        '[PlotOptions React] Updating loadedSamples:',
                        message.samples.length,
                        'samples'
                    );
                    setOptions((prev) => ({
                        ...prev,
                        loadedSamples: message.samples,
                        // Auto-select first 2 samples for comparison
                        selectedSamples: message.samples.slice(0, 2).map((s: SampleItem) => s.name),
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
                return options.hasPod5 && options.hasBam;
            case 'COMPARE_SIGNAL_DELTA':
                return options.loadedSamples.length >= 2;
            case 'COMPARE_AGGREGATE':
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
    };

    const handleMetricToggle = (metric: string, checked: boolean) => {
        setOptions((prev) => {
            const newMetrics = checked
                ? [...prev.comparisonMetrics, metric]
                : prev.comparisonMetrics.filter((m) => m !== metric);
            return { ...prev, comparisonMetrics: newMetrics };
        });
    };

    // Generate handlers for each plot type
    const handleGenerateAggregate = () => {
        sendMessage('generateAggregatePlot', {
            reference: options.aggregateReference,
            maxReads: options.aggregateMaxReads,
            normalization: options.normalization,
            showModifications: options.showModifications,
            showPileup: options.showPileup,
            showDwellTime: options.showDwellTime,
            showSignal: options.showSignal,
            showQuality: options.showQuality,
            clipXAxisToAlignment: options.clipXAxisToAlignment,
            transformCoordinates: options.transformCoordinates,
        });
    };

    const handleGenerateSignalDelta = () => {
        sendMessage('generateSignalDelta', {
            sampleNames: options.selectedSamples.slice(0, 2),
            maxReads: options.comparisonMaxReads,
            normalization: options.normalization,
        });
    };

    const handleGenerateAggregateComparison = () => {
        sendMessage('generateAggregateComparison', {
            sampleNames: options.selectedSamples,
            reference: options.comparisonReference,
            metrics: options.comparisonMetrics,
            maxReads: options.comparisonMaxReads,
            normalization: options.normalization,
        });
    };

    return (
        <div
            style={{
                padding: '10px',
                fontFamily: 'var(--vscode-font-family)',
                fontSize: 'var(--vscode-font-size)',
                color: 'var(--vscode-foreground)',
            }}
        >
            {/* Analysis Type Section */}
            <div style={{ marginBottom: '20px' }}>
                <div
                    style={{
                        fontWeight: 'bold',
                        marginBottom: '8px',
                        color: 'var(--vscode-foreground)',
                    }}
                >
                    Analysis Type
                </div>
                <select
                    value={options.plotType}
                    onChange={handlePlotTypeChange}
                    disabled={!options.hasPod5}
                    style={{
                        width: '100%',
                        padding: '4px',
                        marginBottom: '10px',
                        background: 'var(--vscode-input-background)',
                        color: 'var(--vscode-input-foreground)',
                        border: '1px solid var(--vscode-input-border)',
                        opacity: options.hasPod5 ? 1 : 0.5,
                        cursor: options.hasPod5 ? 'default' : 'not-allowed',
                    }}
                >
                    <option
                        value="MULTI_READ_OVERLAY"
                        disabled={!isPlotTypeAvailable('MULTI_READ_OVERLAY')}
                    >
                        Multi-Read Overlay
                        {!isPlotTypeAvailable('MULTI_READ_OVERLAY') ? ' (requires POD5)' : ''}
                    </option>
                    <option
                        value="MULTI_READ_STACKED"
                        disabled={!isPlotTypeAvailable('MULTI_READ_STACKED')}
                    >
                        Multi-Read Stacked
                        {!isPlotTypeAvailable('MULTI_READ_STACKED') ? ' (requires POD5)' : ''}
                    </option>
                    <option value="AGGREGATE" disabled={!isPlotTypeAvailable('AGGREGATE')}>
                        Aggregate (Single Sample)
                        {!isPlotTypeAvailable('AGGREGATE') ? ' (requires BAM)' : ''}
                    </option>
                    <option
                        value="COMPARE_AGGREGATE"
                        disabled={!isPlotTypeAvailable('COMPARE_AGGREGATE')}
                    >
                        Multi-Sample Overlay
                        {!isPlotTypeAvailable('COMPARE_AGGREGATE')
                            ? ' (requires 2+ samples with BAM)'
                            : ''}
                    </option>
                    <option
                        value="COMPARE_SIGNAL_DELTA"
                        disabled={!isPlotTypeAvailable('COMPARE_SIGNAL_DELTA')}
                    >
                        2-Sample Delta
                        {!isPlotTypeAvailable('COMPARE_SIGNAL_DELTA')
                            ? ' (requires 2 samples)'
                            : ''}
                    </option>
                </select>
                <div
                    style={{
                        fontSize: '0.85em',
                        color: 'var(--vscode-descriptionForeground)',
                        fontStyle: 'italic',
                    }}
                >
                    {!options.hasPod5 && 'Load POD5 file to enable plotting'}
                    {options.hasPod5 &&
                        options.loadedSamples.length < 2 &&
                        options.plotType.startsWith('COMPARE') &&
                        'Load 2+ samples in Sample Manager for comparisons'}
                </div>
            </div>

            {/* Normalization - Common to all types */}
            <div style={{ marginBottom: '20px' }}>
                <div
                    style={{
                        fontWeight: 'bold',
                        marginBottom: '8px',
                        color: 'var(--vscode-foreground)',
                    }}
                >
                    Normalization
                </div>
                <select
                    value={options.normalization}
                    onChange={handleNormalizationChange}
                    style={{
                        width: '100%',
                        padding: '4px',
                        marginBottom: '10px',
                        background: 'var(--vscode-input-background)',
                        color: 'var(--vscode-input-foreground)',
                        border: '1px solid var(--vscode-input-border)',
                    }}
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
                    <div style={{ marginBottom: '20px' }}>
                        <div
                            style={{
                                display: 'flex',
                                justifyContent: 'space-between',
                                marginBottom: '4px',
                                fontSize: '0.9em',
                            }}
                        >
                            <span>Maximum reads:</span>
                            <span
                                style={{
                                    fontWeight: 'bold',
                                    color: 'var(--vscode-input-foreground)',
                                }}
                            >
                                {options.maxReadsMulti}
                            </span>
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
                            style={{ width: '100%', marginBottom: '4px' }}
                        />
                        <div
                            style={{
                                fontSize: '0.85em',
                                color: 'var(--vscode-descriptionForeground)',
                                fontStyle: 'italic',
                            }}
                        >
                            Number of reads to{' '}
                            {options.plotType === 'MULTI_READ_OVERLAY' ? 'overlay' : 'stack'}
                        </div>
                    </div>
                    <div
                        style={{
                            fontSize: '0.85em',
                            color: 'var(--vscode-descriptionForeground)',
                            fontStyle: 'italic',
                        }}
                    >
                        Select multiple reads in the Reads Explorer panel, then right-click to
                        generate plot.
                    </div>
                </div>
            )}

            {options.plotType === 'AGGREGATE' && (
                <>
                    {/* Reference Selection */}
                    <div style={{ marginBottom: '20px' }}>
                        <div
                            style={{
                                fontWeight: 'bold',
                                marginBottom: '8px',
                                color: 'var(--vscode-foreground)',
                            }}
                        >
                            Reference
                        </div>
                        <select
                            value={options.aggregateReference}
                            onChange={(e) =>
                                setOptions((prev) => ({
                                    ...prev,
                                    aggregateReference: e.target.value,
                                }))
                            }
                            disabled={!options.hasBam}
                            style={{
                                width: '100%',
                                padding: '4px',
                                marginBottom: '10px',
                                background: 'var(--vscode-input-background)',
                                color: 'var(--vscode-input-foreground)',
                                border: '1px solid var(--vscode-input-border)',
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
                    <div style={{ marginBottom: '20px' }}>
                        <div
                            style={{
                                display: 'flex',
                                justifyContent: 'space-between',
                                marginBottom: '4px',
                                fontSize: '0.9em',
                            }}
                        >
                            <span>Maximum reads:</span>
                            <span
                                style={{
                                    fontWeight: 'bold',
                                    color: 'var(--vscode-input-foreground)',
                                }}
                            >
                                {options.aggregateMaxReads}
                            </span>
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
                            style={{
                                width: '100%',
                                marginBottom: '4px',
                                opacity: options.hasBam ? 1 : 0.5,
                            }}
                        />
                    </div>

                    {/* Panel Visibility */}
                    <div style={{ marginBottom: '20px' }}>
                        <div
                            style={{
                                fontWeight: 'bold',
                                marginBottom: '8px',
                                color: 'var(--vscode-foreground)',
                            }}
                        >
                            Visible Panels
                        </div>

                        {/* Modifications Panel */}
                        <div style={{ display: 'flex', alignItems: 'center', marginBottom: '8px' }}>
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
                                style={{ marginRight: '6px' }}
                            />
                            <label htmlFor="showModifications" style={{ fontSize: '0.9em' }}>
                                Base modifications
                            </label>
                        </div>
                        <div
                            style={{
                                fontSize: '0.75em',
                                color: 'var(--vscode-descriptionForeground)',
                                fontStyle: 'italic',
                                marginLeft: '22px',
                                marginBottom: '12px',
                                marginTop: '-4px',
                            }}
                        >
                            Adjust filters in Modifications Explorer panel
                        </div>

                        {/* Pileup Panel */}
                        <div style={{ display: 'flex', alignItems: 'center', marginBottom: '8px' }}>
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
                                style={{ marginRight: '6px' }}
                            />
                            <label htmlFor="showPileup" style={{ fontSize: '0.9em' }}>
                                Base pileup
                            </label>
                        </div>

                        {/* Dwell Time Panel */}
                        <div style={{ display: 'flex', alignItems: 'center', marginBottom: '8px' }}>
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
                                style={{ marginRight: '6px' }}
                            />
                            <label htmlFor="showDwellTimeAggregate" style={{ fontSize: '0.9em' }}>
                                Dwell time
                            </label>
                        </div>

                        {/* Signal Panel */}
                        <div style={{ display: 'flex', alignItems: 'center', marginBottom: '8px' }}>
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
                                style={{ marginRight: '6px' }}
                            />
                            <label htmlFor="showSignalAggregate" style={{ fontSize: '0.9em' }}>
                                Signal
                            </label>
                        </div>

                        {/* Quality Panel */}
                        <div style={{ display: 'flex', alignItems: 'center', marginBottom: '8px' }}>
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
                                style={{ marginRight: '6px' }}
                            />
                            <label htmlFor="showQualityAggregate" style={{ fontSize: '0.9em' }}>
                                Quality scores
                            </label>
                        </div>
                    </div>

                    {/* X-Axis Options */}
                    <div
                        style={{
                            marginBottom: '20px',
                            opacity: options.hasBam ? 1 : 0.5,
                            pointerEvents: options.hasBam ? 'auto' : 'none',
                        }}
                    >
                        <div
                            style={{
                                fontWeight: 'bold',
                                marginBottom: '8px',
                                color: 'var(--vscode-foreground)',
                            }}
                        >
                            X-Axis Display
                        </div>

                        {/* Clip to Consensus */}
                        <div style={{ display: 'flex', alignItems: 'center', marginBottom: '8px' }}>
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
                                style={{ marginRight: '6px' }}
                            />
                            <label
                                htmlFor="clipXAxisToAlignmentAggregate"
                                style={{ fontSize: '0.9em' }}
                            >
                                Clip x-axis to consensus region
                            </label>
                        </div>
                        <div
                            style={{
                                fontSize: '0.85em',
                                color: 'var(--vscode-descriptionForeground)',
                                fontStyle: 'italic',
                                marginTop: '-6px',
                                marginBottom: '10px',
                            }}
                        >
                            Focus on high-coverage region (uncheck to show full reference range)
                        </div>

                        {/* Transform Coordinates */}
                        <div style={{ display: 'flex', alignItems: 'center', marginBottom: '8px' }}>
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
                                style={{ marginRight: '6px' }}
                            />
                            <label
                                htmlFor="transformCoordinatesAggregate"
                                style={{ fontSize: '0.9em' }}
                            >
                                Transform to relative coordinates
                            </label>
                        </div>
                        <div
                            style={{
                                fontSize: '0.85em',
                                color: 'var(--vscode-descriptionForeground)',
                                fontStyle: 'italic',
                                marginTop: '-6px',
                                marginBottom: '10px',
                            }}
                        >
                            Anchor position 1 to first reference base (uncheck to use genomic
                            coordinates)
                        </div>
                    </div>

                    {/* Generate Button */}
                    <button
                        onClick={handleGenerateAggregate}
                        disabled={!options.hasBam || !options.aggregateReference}
                        style={{
                            width: '100%',
                            padding: '8px',
                            background: 'var(--vscode-button-background)',
                            color: 'var(--vscode-button-foreground)',
                            border: 'none',
                            cursor:
                                options.hasBam && options.aggregateReference
                                    ? 'pointer'
                                    : 'not-allowed',
                            opacity: options.hasBam && options.aggregateReference ? 1 : 0.5,
                        }}
                    >
                        {!options.hasBam ? 'Load BAM to generate' : 'Generate Aggregate Plot'}
                    </button>
                </>
            )}

            {(options.plotType === 'COMPARE_SIGNAL_DELTA' ||
                options.plotType === 'COMPARE_AGGREGATE') && (
                <>
                    {/* Sample Selection */}
                    <div style={{ marginBottom: '20px' }}>
                        <div
                            style={{
                                fontWeight: 'bold',
                                marginBottom: '8px',
                                color: 'var(--vscode-foreground)',
                            }}
                        >
                            Samples to Compare
                        </div>
                        {options.loadedSamples.length === 0 ? (
                            <div
                                style={{
                                    fontSize: '0.85em',
                                    color: 'var(--vscode-descriptionForeground)',
                                    fontStyle: 'italic',
                                }}
                            >
                                Load samples in Sample Manager to enable comparisons
                            </div>
                        ) : (
                            <div
                                style={{
                                    maxHeight: '150px',
                                    overflowY: 'auto',
                                    border: '1px solid var(--vscode-input-border)',
                                    padding: '4px',
                                }}
                            >
                                {options.loadedSamples.map((sample) => (
                                    <div
                                        key={sample.name}
                                        style={{
                                            display: 'flex',
                                            alignItems: 'center',
                                            marginBottom: '4px',
                                        }}
                                    >
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
                                            style={{ marginRight: '6px' }}
                                        />
                                        <label
                                            htmlFor={`sample-${sample.name}`}
                                            style={{ fontSize: '0.9em' }}
                                        >
                                            {sample.name} ({sample.readCount} reads)
                                        </label>
                                    </div>
                                ))}
                            </div>
                        )}
                        {options.plotType === 'COMPARE_SIGNAL_DELTA' && (
                            <div
                                style={{
                                    fontSize: '0.75em',
                                    color: 'var(--vscode-descriptionForeground)',
                                    fontStyle: 'italic',
                                    marginTop: '4px',
                                }}
                            >
                                Delta plots require exactly 2 samples
                            </div>
                        )}
                    </div>

                    {/* Reference selection for aggregate comparison */}
                    {options.plotType === 'COMPARE_AGGREGATE' && (
                        <>
                            <div style={{ marginBottom: '20px' }}>
                                <div
                                    style={{
                                        fontWeight: 'bold',
                                        marginBottom: '8px',
                                        color: 'var(--vscode-foreground)',
                                    }}
                                >
                                    Reference
                                </div>
                                <select
                                    value={options.comparisonReference}
                                    onChange={(e) =>
                                        setOptions((prev) => ({
                                            ...prev,
                                            comparisonReference: e.target.value,
                                        }))
                                    }
                                    style={{
                                        width: '100%',
                                        padding: '4px',
                                        background: 'var(--vscode-input-background)',
                                        color: 'var(--vscode-input-foreground)',
                                        border: '1px solid var(--vscode-input-border)',
                                    }}
                                >
                                    {options.availableReferences.map((ref) => (
                                        <option key={ref} value={ref}>
                                            {ref}
                                        </option>
                                    ))}
                                </select>
                            </div>

                            {/* Metrics to compare */}
                            <div style={{ marginBottom: '20px' }}>
                                <div
                                    style={{
                                        fontWeight: 'bold',
                                        marginBottom: '8px',
                                        color: 'var(--vscode-foreground)',
                                    }}
                                >
                                    Metrics to Compare
                                </div>
                                {[
                                    { key: 'signal', label: 'Signal statistics' },
                                    { key: 'dwell_time', label: 'Dwell time statistics' },
                                    { key: 'quality', label: 'Quality statistics' },
                                ].map(({ key, label }) => (
                                    <div
                                        key={key}
                                        style={{
                                            display: 'flex',
                                            alignItems: 'center',
                                            marginBottom: '8px',
                                        }}
                                    >
                                        <input
                                            type="checkbox"
                                            id={`metric-${key}`}
                                            checked={options.comparisonMetrics.includes(key)}
                                            onChange={(e) =>
                                                handleMetricToggle(key, e.target.checked)
                                            }
                                            style={{ marginRight: '6px' }}
                                        />
                                        <label
                                            htmlFor={`metric-${key}`}
                                            style={{ fontSize: '0.9em' }}
                                        >
                                            {label}
                                        </label>
                                    </div>
                                ))}
                            </div>
                        </>
                    )}

                    {/* Max Reads */}
                    <div style={{ marginBottom: '20px' }}>
                        <div
                            style={{
                                display: 'flex',
                                justifyContent: 'space-between',
                                marginBottom: '4px',
                                fontSize: '0.9em',
                            }}
                        >
                            <span>Maximum reads per sample:</span>
                            <span
                                style={{
                                    fontWeight: 'bold',
                                    color: 'var(--vscode-input-foreground)',
                                }}
                            >
                                {options.comparisonMaxReads}
                            </span>
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
                            style={{ width: '100%', marginBottom: '4px' }}
                        />
                    </div>

                    {/* Generate Button */}
                    <button
                        onClick={
                            options.plotType === 'COMPARE_SIGNAL_DELTA'
                                ? handleGenerateSignalDelta
                                : handleGenerateAggregateComparison
                        }
                        disabled={
                            options.selectedSamples.length < 2 ||
                            (options.plotType === 'COMPARE_SIGNAL_DELTA' &&
                                options.selectedSamples.length !== 2) ||
                            (options.plotType === 'COMPARE_AGGREGATE' &&
                                (!options.comparisonReference ||
                                    options.comparisonMetrics.length === 0))
                        }
                        style={{
                            width: '100%',
                            padding: '8px',
                            background: 'var(--vscode-button-background)',
                            color: 'var(--vscode-button-foreground)',
                            border: 'none',
                            cursor: 'pointer',
                            opacity: options.selectedSamples.length >= 2 ? 1 : 0.5,
                        }}
                    >
                        {options.selectedSamples.length < 2
                            ? 'Select 2+ samples'
                            : options.plotType === 'COMPARE_SIGNAL_DELTA'
                              ? 'Generate 2-Sample Delta'
                              : 'Generate Multi-Sample Overlay'}
                    </button>
                </>
            )}
        </div>
    );
};
