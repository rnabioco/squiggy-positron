/**
 * Plot Options Core Component
 *
 * React-based UI for plot configuration options
 */

import React, { useEffect, useState } from 'react';
import { vscode } from './vscode-api';

interface PlotOptionsState {
    plotMode: 'SINGLE' | 'EVENTALIGN';
    normalization: 'NONE' | 'ZNORM' | 'MEDIAN' | 'MAD';
    showDwellTime: boolean;
    showBaseAnnotations: boolean;
    scaleDwellTime: boolean;
    downsample: number;
    showSignalPoints: boolean;
    hasBam: boolean;
}

export const PlotOptionsCore: React.FC = () => {
    const [options, setOptions] = useState<PlotOptionsState>({
        plotMode: 'SINGLE',
        normalization: 'ZNORM',
        showDwellTime: false,
        showBaseAnnotations: true,
        scaleDwellTime: false,
        downsample: 1,
        showSignalPoints: false,
        hasBam: false,
    });

    // Listen for messages from extension
    useEffect(() => {
        const handleMessage = (event: MessageEvent) => {
            const message = event.data;
            switch (message.type) {
                case 'updatePlotOptions':
                    setOptions((prev) => ({
                        ...prev,
                        plotMode: message.options.mode,
                        normalization: message.options.normalization,
                        showDwellTime: message.options.showDwellTime,
                        showBaseAnnotations: message.options.showBaseAnnotations,
                        scaleDwellTime: message.options.scaleDwellTime,
                        downsample: message.options.downsample,
                        showSignalPoints: message.options.showSignalPoints,
                    }));
                    break;
                case 'updateBamStatus':
                    setOptions((prev) => ({
                        ...prev,
                        hasBam: message.hasBam,
                        // Switch to SINGLE if EVENTALIGN becomes unavailable
                        plotMode:
                            !message.hasBam && prev.plotMode === 'EVENTALIGN'
                                ? 'SINGLE'
                                : prev.plotMode,
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

    return (
        <div
            style={{
                padding: '10px',
                fontFamily: 'var(--vscode-font-family)',
                fontSize: 'var(--vscode-font-size)',
                color: 'var(--vscode-foreground)',
            }}
        >
            {/* Plot Mode Section */}
            <div style={{ marginBottom: '20px' }}>
                <div
                    style={{
                        fontWeight: 'bold',
                        marginBottom: '8px',
                        color: 'var(--vscode-foreground)',
                    }}
                >
                    Plot Mode
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
                    <option value="SINGLE">Single Read</option>
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
    );
};
