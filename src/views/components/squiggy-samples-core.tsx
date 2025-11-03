/**
 * Sample Comparison Panel Core Component
 *
 * React-based UI for managing samples and initiating multi-sample comparisons
 */

import React, { useEffect, useState } from 'react';
import { vscode } from './vscode-api';
import { SampleItem } from '../../types/messages';

interface SamplesState {
    samples: SampleItem[];
    selectedSamples: Set<string>;
}

export const SamplesCore: React.FC = () => {
    const [state, setState] = useState<SamplesState>({
        samples: [],
        selectedSamples: new Set(),
    });

    // Listen for messages from extension
    useEffect(() => {
        const handleMessage = (event: MessageEvent) => {
            const message = event.data;
            console.log('SamplesCore received message:', message);

            switch (message.type) {
                case 'updateSamples':
                    console.log('Updating samples:', message.samples);
                    setState((prev) => ({
                        ...prev,
                        samples: message.samples,
                    }));
                    break;

                case 'clearSamples':
                    console.log('Clearing samples');
                    setState({
                        samples: [],
                        selectedSamples: new Set(),
                    });
                    break;

                default:
                    console.log('Unknown message type:', message.type);
            }
        };

        window.addEventListener('message', handleMessage);
        // Request initial state
        console.log('SamplesCore sending ready message');
        vscode.postMessage({ type: 'ready' });

        return () => window.removeEventListener('message', handleMessage);
    }, []);

    const handleSampleToggle = (sampleName: string) => {
        setState((prev) => {
            const newSelected = new Set(prev.selectedSamples);
            if (newSelected.has(sampleName)) {
                newSelected.delete(sampleName);
            } else {
                newSelected.add(sampleName);
            }

            // Send update to extension
            vscode.postMessage({
                type: 'selectSample',
                sampleName,
                selected: newSelected.has(sampleName),
            });

            return { ...prev, selectedSamples: newSelected };
        });
    };

    const handleStartComparison = () => {
        const selectedNames = Array.from(state.selectedSamples);
        if (selectedNames.length < 2) {
            alert('Please select at least 2 samples for comparison');
            return;
        }

        console.log('Starting comparison with samples:', selectedNames);
        vscode.postMessage({
            type: 'startComparison',
            sampleNames: selectedNames,
        });
    };

    const handleUnloadSample = (sampleName: string) => {
        console.log('Requesting unload of sample:', sampleName);
        vscode.postMessage({
            type: 'unloadSample',
            sampleName,
        });
    };

    // Unused utility functions - reserved for future file size display feature
    // const formatFileSize = (bytes: number): string => {
    //     if (bytes === 0) return '0 B';
    //     const k = 1024;
    //     const sizes = ['B', 'KB', 'MB', 'GB'];
    //     const i = Math.floor(Math.log(bytes) / Math.log(k));
    //     return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
    // };

    // const getFileSize = (_filePath: string): string => {
    //     // In a real implementation, we'd pass file size from extension
    //     // For now, just show a placeholder
    //     return 'unknown';
    // };

    if (state.samples.length === 0) {
        return (
            <div
                style={{
                    padding: '10px',
                    fontFamily: 'var(--vscode-font-family)',
                    fontSize: 'var(--vscode-font-size)',
                    color: 'var(--vscode-descriptionForeground)',
                    fontStyle: 'italic',
                }}
            >
                No samples loaded. Use "Load Sample (Multi-Sample Comparison)" to add samples for
                comparison.
            </div>
        );
    }

    return (
        <div
            style={{
                padding: '10px',
                fontFamily: 'var(--vscode-font-family)',
                fontSize: 'var(--vscode-font-size)',
                color: 'var(--vscode-foreground)',
            }}
        >
            {/* Samples List */}
            <div style={{ marginBottom: '20px' }}>
                <div
                    style={{
                        fontWeight: 'bold',
                        marginBottom: '8px',
                        color: 'var(--vscode-foreground)',
                    }}
                >
                    Loaded Samples ({state.samples.length})
                </div>

                <div style={{ maxHeight: '300px', overflowY: 'auto', marginBottom: '12px' }}>
                    {state.samples.map((sample) => (
                        <div
                            key={sample.name}
                            style={{
                                padding: '8px',
                                marginBottom: '8px',
                                backgroundColor: 'var(--vscode-editor-background)',
                                border: '1px solid var(--vscode-widget-border)',
                                borderRadius: '4px',
                            }}
                        >
                            {/* Sample Name and Selection */}
                            <div
                                style={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    marginBottom: '6px',
                                }}
                            >
                                <input
                                    type="checkbox"
                                    id={`sample-${sample.name}`}
                                    checked={state.selectedSamples.has(sample.name)}
                                    onChange={() => handleSampleToggle(sample.name)}
                                    style={{ marginRight: '6px' }}
                                />
                                <label
                                    htmlFor={`sample-${sample.name}`}
                                    style={{
                                        fontWeight: 'bold',
                                        flex: 1,
                                        cursor: 'pointer',
                                    }}
                                >
                                    {sample.name}
                                </label>
                            </div>

                            {/* Sample Metadata */}
                            <div
                                style={{
                                    fontSize: '0.85em',
                                    color: 'var(--vscode-descriptionForeground)',
                                    marginBottom: '6px',
                                    paddingLeft: '22px',
                                }}
                            >
                                <div style={{ marginBottom: '2px' }}>
                                    <strong>POD5:</strong> {sample.pod5Path}
                                </div>
                                <div style={{ marginBottom: '2px' }}>
                                    <strong>Reads:</strong> {sample.readCount.toLocaleString()}
                                </div>

                                {sample.bamPath && (
                                    <div style={{ marginBottom: '2px' }}>
                                        <strong>BAM:</strong> {sample.bamPath}
                                    </div>
                                )}

                                {sample.fastaPath && (
                                    <div style={{ marginBottom: '2px' }}>
                                        <strong>FASTA:</strong> {sample.fastaPath}
                                    </div>
                                )}

                                {/* File status badges */}
                                <div style={{ marginTop: '4px', display: 'flex', gap: '4px' }}>
                                    {sample.hasBam && (
                                        <span
                                            style={{
                                                fontSize: '0.75em',
                                                backgroundColor: 'var(--vscode-badge-background)',
                                                color: 'var(--vscode-badge-foreground)',
                                                padding: '2px 6px',
                                                borderRadius: '2px',
                                            }}
                                        >
                                            BAM
                                        </span>
                                    )}
                                    {sample.hasFasta && (
                                        <span
                                            style={{
                                                fontSize: '0.75em',
                                                backgroundColor: 'var(--vscode-badge-background)',
                                                color: 'var(--vscode-badge-foreground)',
                                                padding: '2px 6px',
                                                borderRadius: '2px',
                                            }}
                                        >
                                            FASTA
                                        </span>
                                    )}
                                </div>
                            </div>

                            {/* Unload Button */}
                            <div
                                style={{
                                    paddingLeft: '22px',
                                    marginTop: '6px',
                                }}
                            >
                                <button
                                    onClick={() => handleUnloadSample(sample.name)}
                                    style={{
                                        fontSize: '0.85em',
                                        padding: '2px 8px',
                                        backgroundColor: 'var(--vscode-button-background)',
                                        color: 'var(--vscode-button-foreground)',
                                        border: 'none',
                                        borderRadius: '2px',
                                        cursor: 'pointer',
                                    }}
                                    onMouseEnter={(e) => {
                                        (e.target as HTMLButtonElement).style.backgroundColor =
                                            'var(--vscode-button-hoverBackground)';
                                    }}
                                    onMouseLeave={(e) => {
                                        (e.target as HTMLButtonElement).style.backgroundColor =
                                            'var(--vscode-button-background)';
                                    }}
                                >
                                    Unload
                                </button>
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* Comparison Controls */}
            {state.samples.length > 0 && (
                <div
                    style={{
                        borderTop: '1px solid var(--vscode-widget-border)',
                        paddingTop: '12px',
                        marginTop: '12px',
                    }}
                >
                    <div
                        style={{
                            fontSize: '0.85em',
                            color: 'var(--vscode-descriptionForeground)',
                            marginBottom: '8px',
                            fontStyle: 'italic',
                        }}
                    >
                        Selected: {state.selectedSamples.size} sample(s)
                    </div>

                    <button
                        onClick={handleStartComparison}
                        disabled={state.selectedSamples.size < 2}
                        style={{
                            width: '100%',
                            padding: '8px 12px',
                            backgroundColor:
                                state.selectedSamples.size >= 2
                                    ? 'var(--vscode-button-background)'
                                    : 'var(--vscode-button-disabledBackground)',
                            color:
                                state.selectedSamples.size >= 2
                                    ? 'var(--vscode-button-foreground)'
                                    : 'var(--vscode-input-placeholderForeground)',
                            border: 'none',
                            borderRadius: '4px',
                            fontWeight: 'bold',
                            cursor: state.selectedSamples.size >= 2 ? 'pointer' : 'not-allowed',
                            fontSize: 'var(--vscode-font-size)',
                            fontFamily: 'var(--vscode-font-family)',
                        }}
                        onMouseEnter={(e) => {
                            if (state.selectedSamples.size >= 2) {
                                (e.target as HTMLButtonElement).style.backgroundColor =
                                    'var(--vscode-button-hoverBackground)';
                            }
                        }}
                        onMouseLeave={(e) => {
                            if (state.selectedSamples.size >= 2) {
                                (e.target as HTMLButtonElement).style.backgroundColor =
                                    'var(--vscode-button-background)';
                            }
                        }}
                    >
                        Start Comparison
                    </button>

                    <div
                        style={{
                            fontSize: '0.75em',
                            color: 'var(--vscode-descriptionForeground)',
                            marginTop: '6px',
                            fontStyle: 'italic',
                        }}
                    >
                        Select at least 2 samples to compare
                    </div>
                </div>
            )}
        </div>
    );
};
