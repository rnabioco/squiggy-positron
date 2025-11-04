/**
 * Plot Options Core Component
 *
 * React-based UI for plot configuration options
 */

import React, { useEffect, useState } from 'react';
import { vscode } from './vscode-api';

interface PlotOptionsState {
    plotType: 'SINGLE' | 'AGGREGATE';
    plotMode: 'SINGLE' | 'EVENTALIGN';
    normalization: 'NONE' | 'ZNORM' | 'MEDIAN' | 'MAD';
    showDwellTime: boolean;
    showBaseAnnotations: boolean;
    scaleDwellTime: boolean;
    downsample: number;
    showSignalPoints: boolean;
    hasPod5: boolean;
    hasBam: boolean;
    // Aggregate-specific
    aggregateReference: string;
    aggregateMaxReads: number;
    showModifications: boolean;
    showPileup: boolean;
    showSignal: boolean;
    showQuality: boolean;
    availableReferences: string[];
}

export const PlotOptionsCore: React.FC = () => {
    const [options, setOptions] = useState<PlotOptionsState>({
        plotType: 'SINGLE',
        plotMode: 'SINGLE',
        normalization: 'ZNORM',
        showDwellTime: false,
        showBaseAnnotations: true,
        scaleDwellTime: false,
        downsample: 5,
        showSignalPoints: false,
        hasPod5: false,
        hasBam: false,
        // Aggregate defaults
        aggregateReference: '',
        aggregateMaxReads: 100,
        showModifications: true,
        showPileup: true,
        showSignal: true,
        showQuality: true,
        availableReferences: [],
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
                        plotMode: message.options.mode,
                        normalization: message.options.normalization,
                        showDwellTime: message.options.showDwellTime,
                        showBaseAnnotations: message.options.showBaseAnnotations,
                        scaleDwellTime: message.options.scaleDwellTime,
                        downsample: message.options.downsample,
                        showSignalPoints: message.options.showSignalPoints,
                        aggregateReference:
                            message.options.aggregateReference || prev.aggregateReference,
                        aggregateMaxReads:
                            message.options.aggregateMaxReads || prev.aggregateMaxReads,
                        showModifications:
                            message.options.showModifications ?? prev.showModifications,
                        showPileup: message.options.showPileup ?? prev.showPileup,
                        showSignal: message.options.showSignal ?? prev.showSignal,
                        showQuality: message.options.showQuality ?? prev.showQuality,
                    }));
                    break;
                case 'updatePod5Status':
                    console.log('[PlotOptions React] Received POD5 status:', message.hasPod5);
                    setOptions((prev) => ({
                        ...prev,
                        hasPod5: message.hasPod5,
                    }));
                    break;
                case 'updateBamStatus':
                    console.log('[PlotOptions React] Received BAM status:', message.hasBam);
                    setOptions((prev) => {
                        console.log('[PlotOptions React] Previous hasBam:', prev.hasBam);
                        const newOptions: PlotOptionsState = {
                            ...prev,
                            hasBam: message.hasBam,
                            // When BAM loads, switch to AGGREGATE/EVENTALIGN
                            // When BAM unloads, switch back to SINGLE
                            plotMode: (message.hasBam ? 'EVENTALIGN' : 'SINGLE') as
                                | 'SINGLE'
                                | 'EVENTALIGN',
                            plotType: (message.hasBam ? 'AGGREGATE' : 'SINGLE') as
                                | 'SINGLE'
                                | 'AGGREGATE',
                        };
                        console.log('[PlotOptions React] New hasBam:', newOptions.hasBam);
                        console.log('[PlotOptions React] New plotType:', newOptions.plotType);
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

    const handlePlotModeChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
        const value = e.target.value as 'SINGLE' | 'EVENTALIGN';
        setOptions((prev) => ({ ...prev, plotMode: value }));
        sendMessage('optionsChanged', {
            options: { ...options, mode: value },
        });
    };

    const handleNormalizationChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
        const value = e.target.value as 'NONE' | 'ZNORM' | 'MEDIAN' | 'MAD';
        setOptions((prev) => ({ ...prev, normalization: value }));
        sendMessage('optionsChanged', {
            options: { ...options, normalization: value },
        });
    };

    const handleCheckboxChange =
        (field: keyof PlotOptionsState) => (e: React.ChangeEvent<HTMLInputElement>) => {
            const value = e.target.checked;
            let updates: Partial<PlotOptionsState> = { [field]: value };

            // Handle mutual exclusivity
            if (field === 'showDwellTime' && value) {
                updates.scaleDwellTime = false;
            } else if (field === 'scaleDwellTime' && value) {
                updates.showDwellTime = false;
            }

            setOptions((prev) => ({ ...prev, ...updates }));
            sendMessage('optionsChanged', {
                options: { ...options, ...updates },
            });
        };

    const handleDownsampleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const value = parseInt(e.target.value);
        setOptions((prev) => ({ ...prev, downsample: value }));
        sendMessage('optionsChanged', {
            options: { ...options, downsample: value },
        });
    };

    const getDownsampleLabel = (value: number) => {
        return value === 1 ? '1x (no downsampling)' : `${value}x`;
    };

    const handlePlotTypeChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
        const value = e.target.value as 'SINGLE' | 'AGGREGATE';
        setOptions((prev) => ({ ...prev, plotType: value }));
        sendMessage('optionsChanged', {
            options: { ...options, plotType: value },
        });
    };

    const handleAggregateReferenceChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
        const value = e.target.value;
        setOptions((prev) => ({ ...prev, aggregateReference: value }));
        sendMessage('optionsChanged', {
            options: { ...options, aggregateReference: value },
        });
    };

    const handleAggregateMaxReadsChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const value = parseInt(e.target.value);
        setOptions((prev) => ({ ...prev, aggregateMaxReads: value }));
        sendMessage('optionsChanged', {
            options: { ...options, aggregateMaxReads: value },
        });
    };

    const handleAggregatePanelToggle =
        (field: keyof PlotOptionsState) => (e: React.ChangeEvent<HTMLInputElement>) => {
            const value = e.target.checked;
            setOptions((prev) => ({ ...prev, [field]: value }));
            sendMessage('optionsChanged', {
                options: { ...options, [field]: value },
            });
        };

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
                    <option value="SINGLE">Single Read</option>
                    <option value="AGGREGATE" disabled={!options.hasBam}>
                        Aggregate{!options.hasBam ? ' (requires BAM)' : ''}
                    </option>
                </select>
            </div>

            {/* Aggregate Plot Controls - Only show when plotType is AGGREGATE */}
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
                            onChange={handleAggregateReferenceChange}
                            disabled={!options.hasBam}
                            style={{
                                width: '100%',
                                padding: '4px',
                                marginBottom: '10px',
                                background: 'var(--vscode-input-background)',
                                color: 'var(--vscode-input-foreground)',
                                border: '1px solid var(--vscode-input-border)',
                                opacity: options.hasBam ? 1 : 0.5,
                                cursor: options.hasBam ? 'default' : 'not-allowed',
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

                    {/* Max Reads Slider */}
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
                            id="aggregateMaxReads"
                            min="10"
                            max="500"
                            step="10"
                            value={options.aggregateMaxReads}
                            onChange={handleAggregateMaxReadsChange}
                            disabled={!options.hasBam}
                            style={{
                                width: '100%',
                                marginBottom: '4px',
                                opacity: options.hasBam ? 1 : 0.5,
                                cursor: options.hasBam ? 'pointer' : 'not-allowed',
                            }}
                        />
                        <div
                            style={{
                                fontSize: '0.85em',
                                color: 'var(--vscode-descriptionForeground)',
                                fontStyle: 'italic',
                            }}
                        >
                            Number of reads to include in aggregate plot
                        </div>
                    </div>

                    {/* Panel Visibility Toggles */}
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
                            Visible Panels
                        </div>

                        {/* Modifications Panel */}
                        <div style={{ display: 'flex', alignItems: 'center', marginBottom: '8px' }}>
                            <input
                                type="checkbox"
                                id="showModifications"
                                checked={options.showModifications}
                                onChange={handleAggregatePanelToggle('showModifications')}
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
                                onChange={handleAggregatePanelToggle('showPileup')}
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
                                onChange={handleAggregatePanelToggle('showDwellTime')}
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
                                onChange={handleAggregatePanelToggle('showSignal')}
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
                                onChange={handleAggregatePanelToggle('showQuality')}
                                disabled={!options.hasBam}
                                style={{ marginRight: '6px' }}
                            />
                            <label htmlFor="showQualityAggregate" style={{ fontSize: '0.9em' }}>
                                Quality scores
                            </label>
                        </div>
                    </div>

                    {/* Generate Button */}
                    <div style={{ marginBottom: '20px' }}>
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
                            {!options.hasBam
                                ? 'Load BAM file to generate plot'
                                : !options.aggregateReference
                                  ? 'Select reference to generate plot'
                                  : 'Generate Aggregate Plot'}
                        </button>
                    </div>
                </>
            )}

            {/* Single Read Controls - Only show when plotType is SINGLE */}
            {options.plotType === 'SINGLE' && (
                <div
                    style={{
                        opacity: options.hasPod5 ? 1 : 0.5,
                        pointerEvents: options.hasPod5 ? 'auto' : 'none',
                    }}
                >
                    {/* View Mode Section */}
                    <div style={{ marginBottom: '20px' }}>
                        <div
                            style={{
                                fontWeight: 'bold',
                                marginBottom: '8px',
                                color: 'var(--vscode-foreground)',
                            }}
                        >
                            View Mode
                        </div>
                        <select
                            value={options.plotMode}
                            onChange={handlePlotModeChange}
                            style={{
                                width: '100%',
                                padding: '4px',
                                marginBottom: '10px',
                                background: 'var(--vscode-input-background)',
                                color: 'var(--vscode-input-foreground)',
                                border: '1px solid var(--vscode-input-border)',
                            }}
                        >
                            <option value="SINGLE">Standard</option>
                            <option value="EVENTALIGN" disabled={!options.hasBam}>
                                Event-Aligned{!options.hasBam ? ' (requires BAM)' : ''}
                            </option>
                        </select>
                    </div>

                    {/* Normalization Section */}
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

                    {/* Display Options Section */}
                    <div style={{ marginBottom: '20px' }}>
                        <div
                            style={{
                                fontWeight: 'bold',
                                marginBottom: '8px',
                                color: 'var(--vscode-foreground)',
                            }}
                        >
                            Display Options
                        </div>

                        {/* Base Annotations */}
                        <div style={{ display: 'flex', alignItems: 'center', marginBottom: '8px' }}>
                            <input
                                type="checkbox"
                                id="showBaseAnnotations"
                                checked={options.showBaseAnnotations}
                                onChange={handleCheckboxChange('showBaseAnnotations')}
                                style={{ marginRight: '6px' }}
                            />
                            <label htmlFor="showBaseAnnotations" style={{ fontSize: '0.9em' }}>
                                Show base labels
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
                            Display base letters on signal (event-aligned mode)
                        </div>

                        {/* Dwell Time Color */}
                        <div style={{ display: 'flex', alignItems: 'center', marginBottom: '8px' }}>
                            <input
                                type="checkbox"
                                id="showDwellTime"
                                checked={options.showDwellTime}
                                onChange={handleCheckboxChange('showDwellTime')}
                                style={{ marginRight: '6px' }}
                            />
                            <label htmlFor="showDwellTime" style={{ fontSize: '0.9em' }}>
                                Color by dwell time
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
                            Color bases by dwell time instead of base type
                        </div>

                        {/* Scale X-Axis */}
                        <div style={{ display: 'flex', alignItems: 'center', marginBottom: '8px' }}>
                            <input
                                type="checkbox"
                                id="scaleDwellTime"
                                checked={options.scaleDwellTime}
                                onChange={handleCheckboxChange('scaleDwellTime')}
                                style={{ marginRight: '6px' }}
                            />
                            <label htmlFor="scaleDwellTime" style={{ fontSize: '0.9em' }}>
                                Scale x-axis by dwell time
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
                            X-axis shows cumulative dwell time instead of base positions
                        </div>

                        {/* Downsample Slider */}
                        <div style={{ marginBottom: '8px' }}>
                            <div
                                style={{
                                    display: 'flex',
                                    justifyContent: 'space-between',
                                    marginBottom: '4px',
                                    fontSize: '0.9em',
                                }}
                            >
                                <span>Downsample signal:</span>
                                <span
                                    style={{
                                        fontWeight: 'bold',
                                        color: 'var(--vscode-input-foreground)',
                                    }}
                                >
                                    {getDownsampleLabel(options.downsample)}
                                </span>
                            </div>
                            <input
                                type="range"
                                id="downsample"
                                min="1"
                                max="40"
                                value={options.downsample}
                                onChange={handleDownsampleChange}
                                style={{ width: '100%', marginBottom: '4px' }}
                            />
                            <div
                                style={{
                                    fontSize: '0.85em',
                                    color: 'var(--vscode-descriptionForeground)',
                                    fontStyle: 'italic',
                                }}
                            >
                                Reduce signal points for faster rendering (1 = all points)
                            </div>
                        </div>

                        {/* Signal Points */}
                        <div
                            style={{
                                display: 'flex',
                                alignItems: 'center',
                                marginBottom: '8px',
                                marginTop: '10px',
                            }}
                        >
                            <input
                                type="checkbox"
                                id="showSignalPoints"
                                checked={options.showSignalPoints}
                                onChange={handleCheckboxChange('showSignalPoints')}
                                style={{ marginRight: '6px' }}
                            />
                            <label htmlFor="showSignalPoints" style={{ fontSize: '0.9em' }}>
                                Show individual signal points
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
                            Display circles at each signal sample point
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};
