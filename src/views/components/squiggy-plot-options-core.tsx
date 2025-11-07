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
    | 'AGGREGATE' // Now supports 1+ samples with view toggle
    | 'COMPARE_SIGNAL_DELTA';

interface PlotOptionsState {
    // Current selection
    plotType: PlotType;
    coordinateSpace: 'signal' | 'sequence'; // X-axis coordinate system

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
}

export const PlotOptionsCore: React.FC = () => {
    const [options, setOptions] = useState<PlotOptionsState>({
        plotType: 'AGGREGATE',
        coordinateSpace: 'signal', // Default to signal space
        hasPod5: false,
        hasBam: false,
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
        sendMessage('generateMultiReadOverlay', {
            sampleNames: options.selectedSamples,
            maxReads: options.maxReadsMulti,
            normalization: options.normalization,
            coordinateSpace: options.coordinateSpace,
        });
    };

    const handleGenerateMultiReadStacked = () => {
        sendMessage('generateMultiReadStacked', {
            sampleNames: options.selectedSamples,
            maxReads: options.maxReadsMulti,
            normalization: options.normalization,
            coordinateSpace: options.coordinateSpace,
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
        <div
            style={{
                padding: '10px',
                fontFamily: 'var(--vscode-font-family)',
                fontSize: 'var(--vscode-font-size)',
                color: 'var(--vscode-foreground)',
            }}
        >
            {/* Generate Plot Button - At Top for All Plot Types */}
            <button
                onClick={buttonState.handler}
                disabled={buttonState.disabled}
                style={{
                    width: '100%',
                    padding: '10px',
                    marginBottom: '16px',
                    background: 'var(--vscode-button-background)',
                    color: 'var(--vscode-button-foreground)',
                    border: 'none',
                    cursor: buttonState.disabled ? 'not-allowed' : 'pointer',
                    opacity: buttonState.disabled ? 0.5 : 1,
                    fontSize: '1em',
                    fontWeight: 'bold',
                }}
            >
                {buttonState.text}
            </button>

            {/* Analysis Type Section */}
            <div style={{ marginBottom: '12px' }}>
                <div
                    style={{
                        fontWeight: 'bold',
                        marginBottom: '6px',
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
                        marginBottom: '6px',
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
            <div style={{ marginBottom: '12px' }}>
                <div
                    style={{
                        fontWeight: 'bold',
                        marginBottom: '6px',
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
                        marginBottom: '4px',
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
                    {/* View Style: Overlay vs Stacked */}
                    <div style={{ marginBottom: '12px' }}>
                        <div
                            style={{
                                fontWeight: 'bold',
                                marginBottom: '8px',
                                color: 'var(--vscode-foreground)',
                            }}
                        >
                            View Style
                        </div>
                        <div style={{ display: 'flex', gap: '16px' }}>
                            <label
                                style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}
                            >
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
                                    style={{ marginRight: '6px' }}
                                />
                                <span>Overlay (alpha-blended)</span>
                            </label>
                            <label
                                style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}
                            >
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
                                    style={{ marginRight: '6px' }}
                                />
                                <span>Stacked (offset)</span>
                            </label>
                        </div>
                    </div>

                    {/* Max Reads per Sample */}
                    <div style={{ marginBottom: '12px' }}>
                        <div
                            style={{
                                display: 'flex',
                                justifyContent: 'space-between',
                                marginBottom: '4px',
                                fontSize: '0.9em',
                            }}
                        >
                            <span>Max reads per sample:</span>
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
                            Number of reads to extract from each sample
                        </div>
                    </div>

                    {/* Warning for stacked plots with too many reads */}
                    {options.plotType === 'MULTI_READ_STACKED' &&
                        options.selectedSamples.length * options.maxReadsMulti > 20 && (
                            <div
                                style={{
                                    fontSize: '0.85em',
                                    color: 'var(--vscode-editorWarning-foreground)',
                                    marginBottom: '10px',
                                    padding: '6px',
                                    border: '1px solid var(--vscode-editorWarning-foreground)',
                                    borderRadius: '3px',
                                }}
                            >
                                ⚠️ Stacked plots work best with ≤20 total reads (currently:{' '}
                                {options.selectedSamples.length * options.maxReadsMulti})
                            </div>
                        )}
                </div>
            )}

            {options.plotType === 'AGGREGATE' && (
                <>
                    {/* Reference Selection */}
                    <div style={{ marginBottom: '12px' }}>
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

                    {/* View Style (for multi-sample) - only show if 2+ samples selected in Samples panel */}
                    {options.selectedSamples.length > 1 && (
                        <div style={{ marginBottom: '20px' }}>
                            <div
                                style={{
                                    fontWeight: 'bold',
                                    marginBottom: '8px',
                                    color: 'var(--vscode-foreground)',
                                }}
                            >
                                View Style
                            </div>
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
                                style={{
                                    width: '100%',
                                    padding: '4px',
                                    marginBottom: '10px',
                                    background: 'var(--vscode-input-background)',
                                    color: 'var(--vscode-input-foreground)',
                                    border: '1px solid var(--vscode-input-border)',
                                }}
                            >
                                <option value="overlay">Overlay (Mean Signals)</option>
                                <option value="multi-track">Multi-Track (Detailed)</option>
                            </select>
                            <div
                                style={{
                                    fontSize: '0.85em',
                                    color: 'var(--vscode-descriptionForeground)',
                                    marginTop: '4px',
                                }}
                            >
                                {options.aggregateViewStyle === 'overlay'
                                    ? 'Overlays mean signals from all samples on one plot'
                                    : 'Shows detailed 5-track view for each sample'}
                            </div>
                        </div>
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
                    <div style={{ marginBottom: '12px' }}>
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
                </>
            )}

            {options.plotType === 'COMPARE_SIGNAL_DELTA' && (
                <>
                    {/* Sample Selection */}
                    <div style={{ marginBottom: '12px' }}>
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

                    {/* Reference Selection */}
                    <div style={{ marginBottom: '12px' }}>
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
                </>
            )}
        </div>
    );
};
