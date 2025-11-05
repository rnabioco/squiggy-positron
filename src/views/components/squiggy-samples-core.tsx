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
    sessionFastaPath: string | null; // Session-level FASTA file path
    editingSampleName: string | null; // Which sample name is being edited
    editInputValue: string; // Current value in edit input
    sampleColors: Map<string, string>; // Map of sample names to hex colors
    expandedSamples: Set<string>; // Which samples have their details expanded
}

export const SamplesCore: React.FC = () => {
    const [state, setState] = useState<SamplesState>({
        samples: [],
        selectedSamples: new Set(),
        sessionFastaPath: null,
        editingSampleName: null,
        editInputValue: '',
        sampleColors: new Map(),
        expandedSamples: new Set(), // Start with all samples collapsed
    });

    // Listen for messages from extension
    useEffect(() => {
        const handleMessage = (event: MessageEvent) => {
            const message = event.data;
            console.log('SamplesCore received message:', message);

            switch (message.type) {
                case 'updateSamples': {
                    console.log('Updating samples:', message.samples);
                    setState((prev) => {
                        // Auto-select newly added samples
                        const samples = (message as any).samples as SampleItem[];
                        const newSampleNames = new Set(samples.map((s: SampleItem) => s.name));
                        const updatedSelected = new Set(prev.selectedSamples);
                        newSampleNames.forEach((name: string) => {
                            if (!prev.selectedSamples.has(name)) {
                                updatedSelected.add(name);
                            }
                        });

                        return {
                            ...prev,
                            samples,
                            selectedSamples: updatedSelected,
                        };
                    });
                    break;
                }

                case 'clearSamples':
                    console.log('Clearing samples');
                    setState({
                        samples: [],
                        selectedSamples: new Set(),
                        sessionFastaPath: null,
                        editingSampleName: null,
                        editInputValue: '',
                        sampleColors: new Map(),
                        expandedSamples: new Set(),
                    });
                    break;

                case 'updateSessionFasta':
                    console.log('Updating session FASTA:', message.fastaPath);
                    setState((prev) => ({
                        ...prev,
                        sessionFastaPath: message.fastaPath,
                    }));
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

    const handleUnloadSample = (sampleName: string) => {
        console.log('Requesting unload of sample:', sampleName);
        vscode.postMessage({
            type: 'unloadSample',
            sampleName,
        });
    };

    const _handleSetSessionFasta = () => {
        vscode.postMessage({
            type: 'requestSetSessionFasta',
        });
    };

    const _handleClearSessionFasta = () => {
        setState((prev) => ({
            ...prev,
            sessionFastaPath: null,
        }));
        vscode.postMessage({
            type: 'setSessionFasta',
            fastaPath: null,
        });
    };

    const _handleLoadSamplesClick = () => {
        console.log('🎯 DEBUG: Load Samples button clicked');
        vscode.postMessage({
            type: 'requestLoadSamples',
        });
    };

    const handleEditSampleName = (sampleName: string) => {
        setState((prev) => ({
            ...prev,
            editingSampleName: sampleName,
            editInputValue: sampleName,
        }));
    };

    const handleSaveNameEdit = (oldName: string, newName: string) => {
        if (!newName.trim()) {
            alert('Sample name cannot be empty');
            return;
        }

        if (newName === oldName) {
            // No change, just exit edit mode
            setState((prev) => ({
                ...prev,
                editingSampleName: null,
                editInputValue: '',
            }));
            return;
        }

        // Check for duplicate names
        if (state.samples.some((s) => s.name === newName)) {
            alert('A sample with this name already exists');
            return;
        }

        // Send update to extension
        vscode.postMessage({
            type: 'updateSampleName',
            oldName: oldName,
            newName: newName.trim(),
        });

        // Exit edit mode
        setState((prev) => ({
            ...prev,
            editingSampleName: null,
            editInputValue: '',
        }));
    };

    const handleCancelNameEdit = () => {
        setState((prev) => ({
            ...prev,
            editingSampleName: null,
            editInputValue: '',
        }));
    };

    const handleSampleColorChange = (sampleName: string, color: string) => {
        setState((prev) => {
            const newColors = new Map(prev.sampleColors);
            newColors.set(sampleName, color);
            return {
                ...prev,
                sampleColors: newColors,
            };
        });

        // Send to extension
        vscode.postMessage({
            type: 'updateSampleColor',
            sampleName: sampleName,
            color: color || null,
        });
    };

    const toggleSampleExpanded = (sampleName: string) => {
        setState((prev) => {
            const newExpanded = new Set(prev.expandedSamples);
            if (newExpanded.has(sampleName)) {
                newExpanded.delete(sampleName);
            } else {
                newExpanded.add(sampleName);
            }
            return {
                ...prev,
                expandedSamples: newExpanded,
            };
        });
    };

    const handleSetFastaForAll = () => {
        vscode.postMessage({
            type: 'requestSetSessionFasta',
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
                    color: 'var(--vscode-foreground)',
                }}
            >
                <div
                    style={{
                        padding: '12px',
                        backgroundColor: 'var(--vscode-input-background)',
                        border: '1px solid var(--vscode-widget-border)',
                        borderRadius: '4px',
                        color: 'var(--vscode-descriptionForeground)',
                        fontSize: '0.9em',
                        lineHeight: '1.5',
                    }}
                >
                    <p style={{ marginTop: 0 }}>
                        <strong>No samples loaded yet.</strong>
                    </p>
                    <p style={{ margin: '8px 0' }}>
                        Use the "Load Sample Data" button in the <strong>File Explorer</strong>{' '}
                        panel to add POD5 and BAM files. Samples will appear here for organization
                        and comparison.
                    </p>
                </div>
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
            {/* "Set FASTA for All Samples" Button */}
            <div style={{ marginBottom: '12px' }}>
                <button
                    onClick={handleSetFastaForAll}
                    style={{
                        width: '100%',
                        padding: '6px 8px',
                        backgroundColor: 'var(--vscode-button-background)',
                        color: 'var(--vscode-button-foreground)',
                        border: 'none',
                        borderRadius: '2px',
                        cursor: 'pointer',
                        fontSize: 'var(--vscode-font-size)',
                        fontFamily: 'var(--vscode-font-family)',
                        marginBottom: '4px',
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
                    {state.sessionFastaPath ? '✓ Set FASTA for All' : 'Set FASTA for All'}
                </button>
                {state.sessionFastaPath && (
                    <div
                        style={{
                            fontSize: '0.75em',
                            color: 'var(--vscode-descriptionForeground)',
                            padding: '4px',
                            wordBreak: 'break-word',
                        }}
                    >
                        {state.sessionFastaPath.split('/').pop()}
                    </div>
                )}
            </div>

            {/* Samples List - Collapsible Rows */}
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

                <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                    {state.samples.map((sample) => {
                        const isExpanded = state.expandedSamples.has(sample.name);
                        return (
                            <div
                                key={sample.name}
                                style={{
                                    backgroundColor: 'var(--vscode-editor-background)',
                                    border: '1px solid var(--vscode-widget-border)',
                                    borderRadius: '4px',
                                    overflow: 'hidden',
                                }}
                            >
                                {/* Collapsed Header Row */}
                                <div
                                    style={{
                                        display: 'flex',
                                        alignItems: 'center',
                                        gap: '6px',
                                        padding: '8px',
                                        cursor: 'pointer',
                                        backgroundColor: isExpanded
                                            ? 'var(--vscode-input-background)'
                                            : 'var(--vscode-editor-background)',
                                        borderBottom: isExpanded
                                            ? '1px solid var(--vscode-widget-border)'
                                            : 'none',
                                    }}
                                    onClick={() => toggleSampleExpanded(sample.name)}
                                >
                                    {/* Expand/Collapse Toggle */}
                                    <div
                                        style={{
                                            width: '18px',
                                            height: '18px',
                                            display: 'flex',
                                            alignItems: 'center',
                                            justifyContent: 'center',
                                            color: 'var(--vscode-descriptionForeground)',
                                            fontSize: '0.9em',
                                            flexShrink: 0,
                                        }}
                                    >
                                        {isExpanded ? '▼' : '▶'}
                                    </div>

                                    {/* Selection Checkbox */}
                                    <input
                                        type="checkbox"
                                        id={`sample-${sample.name}`}
                                        checked={state.selectedSamples.has(sample.name)}
                                        onChange={(e) => {
                                            e.stopPropagation();
                                            handleSampleToggle(sample.name);
                                        }}
                                        style={{
                                            marginRight: '0px',
                                            flexShrink: 0,
                                            cursor: 'pointer',
                                        }}
                                    />

                                    {/* Color Picker */}
                                    <input
                                        type="color"
                                        value={state.sampleColors.get(sample.name) || '#808080'}
                                        onChange={(e) => {
                                            e.stopPropagation();
                                            handleSampleColorChange(sample.name, e.target.value);
                                        }}
                                        onClick={(e) => e.stopPropagation()}
                                        style={{
                                            width: '20px',
                                            height: '20px',
                                            border: 'none',
                                            borderRadius: '2px',
                                            cursor: 'pointer',
                                            flexShrink: 0,
                                        }}
                                        title="Sample color for plots"
                                    />

                                    {/* Sample Name - Editable */}
                                    {state.editingSampleName === sample.name ? (
                                        <div
                                            style={{
                                                display: 'flex',
                                                gap: '4px',
                                                flex: 1,
                                            }}
                                            onClick={(e) => e.stopPropagation()}
                                        >
                                            <input
                                                type="text"
                                                value={state.editInputValue}
                                                onChange={(e) =>
                                                    setState((prev) => ({
                                                        ...prev,
                                                        editInputValue: e.target.value,
                                                    }))
                                                }
                                                onKeyDown={(e) => {
                                                    e.stopPropagation();
                                                    if (e.key === 'Enter') {
                                                        handleSaveNameEdit(
                                                            sample.name,
                                                            state.editInputValue
                                                        );
                                                    } else if (e.key === 'Escape') {
                                                        handleCancelNameEdit();
                                                    }
                                                }}
                                                style={{
                                                    flex: 1,
                                                    padding: '4px',
                                                    backgroundColor:
                                                        'var(--vscode-input-background)',
                                                    color: 'var(--vscode-input-foreground)',
                                                    border: '1px solid var(--vscode-input-border)',
                                                    borderRadius: '2px',
                                                    fontWeight: 'bold',
                                                    fontFamily: 'var(--vscode-font-family)',
                                                    fontSize: '0.95em',
                                                }}
                                                autoFocus
                                            />
                                            <button
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    handleSaveNameEdit(
                                                        sample.name,
                                                        state.editInputValue
                                                    );
                                                }}
                                                style={{
                                                    padding: '2px 6px',
                                                    backgroundColor:
                                                        'var(--vscode-button-background)',
                                                    color: 'var(--vscode-button-foreground)',
                                                    border: 'none',
                                                    borderRadius: '2px',
                                                    cursor: 'pointer',
                                                    fontSize: '0.85em',
                                                    flexShrink: 0,
                                                }}
                                            >
                                                ✓
                                            </button>
                                            <button
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    handleCancelNameEdit();
                                                }}
                                                style={{
                                                    padding: '2px 6px',
                                                    backgroundColor:
                                                        'var(--vscode-errorForeground)',
                                                    color: 'var(--vscode-editor-background)',
                                                    border: 'none',
                                                    borderRadius: '2px',
                                                    cursor: 'pointer',
                                                    fontSize: '0.85em',
                                                    flexShrink: 0,
                                                }}
                                            >
                                                ✕
                                            </button>
                                        </div>
                                    ) : (
                                        <label
                                            htmlFor={`sample-${sample.name}`}
                                            style={{
                                                fontWeight: 'bold',
                                                flex: 1,
                                                cursor: 'pointer',
                                                userSelect: 'none',
                                            }}
                                            onDoubleClick={(e) => {
                                                e.stopPropagation();
                                                handleEditSampleName(sample.name);
                                            }}
                                        >
                                            {sample.name}
                                        </label>
                                    )}

                                    {/* Read Count Badge */}
                                    <span
                                        style={{
                                            fontSize: '0.75em',
                                            backgroundColor: 'var(--vscode-badge-background)',
                                            color: 'var(--vscode-badge-foreground)',
                                            padding: '2px 6px',
                                            borderRadius: '2px',
                                            flexShrink: 0,
                                        }}
                                    >
                                        {sample.readCount.toLocaleString()} reads
                                    </span>
                                </div>

                                {/* Expanded Details */}
                                {isExpanded && (
                                    <div
                                        style={{
                                            padding: '8px',
                                            backgroundColor: 'var(--vscode-input-background)',
                                            fontSize: '0.85em',
                                            color: 'var(--vscode-foreground)',
                                        }}
                                        onClick={(e) => e.stopPropagation()}
                                    >
                                        {/* POD5 File */}
                                        <div style={{ marginBottom: '6px' }}>
                                            <div
                                                style={{
                                                    color: 'var(--vscode-descriptionForeground)',
                                                    fontSize: '0.8em',
                                                    marginBottom: '2px',
                                                }}
                                            >
                                                POD5
                                            </div>
                                            <div
                                                style={{
                                                    padding: '4px 6px',
                                                    backgroundColor:
                                                        'var(--vscode-editor-background)',
                                                    borderRadius: '2px',
                                                    wordBreak: 'break-all',
                                                }}
                                            >
                                                {sample.pod5Path.split('/').pop()}
                                            </div>
                                        </div>

                                        {/* BAM File */}
                                        <div style={{ marginBottom: '6px' }}>
                                            <div
                                                style={{
                                                    color: 'var(--vscode-descriptionForeground)',
                                                    fontSize: '0.8em',
                                                    marginBottom: '2px',
                                                }}
                                            >
                                                BAM {sample.bamPath ? '' : '(not set)'}
                                            </div>
                                            {sample.bamPath ? (
                                                <div
                                                    style={{
                                                        display: 'flex',
                                                        gap: '4px',
                                                        alignItems: 'center',
                                                    }}
                                                >
                                                    <div
                                                        style={{
                                                            flex: 1,
                                                            padding: '4px 6px',
                                                            backgroundColor:
                                                                'var(--vscode-editor-background)',
                                                            borderRadius: '2px',
                                                            wordBreak: 'break-all',
                                                        }}
                                                    >
                                                        {sample.bamPath.split('/').pop()}
                                                    </div>
                                                    <button
                                                        style={{
                                                            padding: '2px 8px',
                                                            fontSize: '0.75em',
                                                            backgroundColor:
                                                                'var(--vscode-button-background)',
                                                            color: 'var(--vscode-button-foreground)',
                                                            border: 'none',
                                                            borderRadius: '2px',
                                                            cursor: 'pointer',
                                                            flexShrink: 0,
                                                        }}
                                                        onMouseEnter={(e) => {
                                                            (
                                                                e.target as HTMLButtonElement
                                                            ).style.backgroundColor =
                                                                'var(--vscode-button-hoverBackground)';
                                                        }}
                                                        onMouseLeave={(e) => {
                                                            (
                                                                e.target as HTMLButtonElement
                                                            ).style.backgroundColor =
                                                                'var(--vscode-button-background)';
                                                        }}
                                                    >
                                                        [Change]
                                                    </button>
                                                </div>
                                            ) : (
                                                <button
                                                    style={{
                                                        width: '100%',
                                                        padding: '4px',
                                                        fontSize: '0.85em',
                                                        backgroundColor:
                                                            'var(--vscode-button-background)',
                                                        color: 'var(--vscode-button-foreground)',
                                                        border: 'none',
                                                        borderRadius: '2px',
                                                        cursor: 'pointer',
                                                    }}
                                                    onMouseEnter={(e) => {
                                                        (
                                                            e.target as HTMLButtonElement
                                                        ).style.backgroundColor =
                                                            'var(--vscode-button-hoverBackground)';
                                                    }}
                                                    onMouseLeave={(e) => {
                                                        (
                                                            e.target as HTMLButtonElement
                                                        ).style.backgroundColor =
                                                            'var(--vscode-button-background)';
                                                    }}
                                                >
                                                    + Add BAM
                                                </button>
                                            )}
                                        </div>

                                        {/* FASTA File - Show sample FASTA or session FASTA */}
                                        <div style={{ marginBottom: '6px' }}>
                                            {(() => {
                                                const fastaPath =
                                                    sample.fastaPath || state.sessionFastaPath;
                                                const isSessionFasta =
                                                    !sample.fastaPath && state.sessionFastaPath;
                                                return (
                                                    <>
                                                        <div
                                                            style={{
                                                                color: 'var(--vscode-descriptionForeground)',
                                                                fontSize: '0.8em',
                                                                marginBottom: '2px',
                                                            }}
                                                        >
                                                            FASTA{' '}
                                                            {fastaPath
                                                                ? isSessionFasta
                                                                    ? '(session)'
                                                                    : ''
                                                                : '(not set)'}
                                                        </div>
                                                        {fastaPath ? (
                                                            <div
                                                                style={{
                                                                    display: 'flex',
                                                                    gap: '4px',
                                                                    alignItems: 'center',
                                                                }}
                                                            >
                                                                <div
                                                                    style={{
                                                                        flex: 1,
                                                                        padding: '4px 6px',
                                                                        backgroundColor:
                                                                            'var(--vscode-editor-background)',
                                                                        borderRadius: '2px',
                                                                        wordBreak: 'break-all',
                                                                    }}
                                                                >
                                                                    {fastaPath.split('/').pop()}
                                                                </div>
                                                                {!isSessionFasta && (
                                                                    <button
                                                                        style={{
                                                                            padding: '2px 8px',
                                                                            fontSize: '0.75em',
                                                                            backgroundColor:
                                                                                'var(--vscode-button-background)',
                                                                            color: 'var(--vscode-button-foreground)',
                                                                            border: 'none',
                                                                            borderRadius: '2px',
                                                                            cursor: 'pointer',
                                                                            flexShrink: 0,
                                                                        }}
                                                                        onMouseEnter={(e) => {
                                                                            (
                                                                                e.target as HTMLButtonElement
                                                                            ).style.backgroundColor =
                                                                                'var(--vscode-button-hoverBackground)';
                                                                        }}
                                                                        onMouseLeave={(e) => {
                                                                            (
                                                                                e.target as HTMLButtonElement
                                                                            ).style.backgroundColor =
                                                                                'var(--vscode-button-background)';
                                                                        }}
                                                                    >
                                                                        [Change]
                                                                    </button>
                                                                )}
                                                            </div>
                                                        ) : (
                                                            <button
                                                                style={{
                                                                    width: '100%',
                                                                    padding: '4px',
                                                                    fontSize: '0.85em',
                                                                    backgroundColor:
                                                                        'var(--vscode-button-background)',
                                                                    color: 'var(--vscode-button-foreground)',
                                                                    border: 'none',
                                                                    borderRadius: '2px',
                                                                    cursor: 'pointer',
                                                                }}
                                                                onMouseEnter={(e) => {
                                                                    (
                                                                        e.target as HTMLButtonElement
                                                                    ).style.backgroundColor =
                                                                        'var(--vscode-button-hoverBackground)';
                                                                }}
                                                                onMouseLeave={(e) => {
                                                                    (
                                                                        e.target as HTMLButtonElement
                                                                    ).style.backgroundColor =
                                                                        'var(--vscode-button-background)';
                                                                }}
                                                            >
                                                                + Add FASTA
                                                            </button>
                                                        )}
                                                    </>
                                                );
                                            })()}
                                        </div>

                                        {/* Unload Button */}
                                        <div
                                            style={{
                                                marginTop: '8px',
                                                borderTop: '1px solid var(--vscode-widget-border)',
                                                paddingTop: '8px',
                                            }}
                                        >
                                            <button
                                                onClick={() => handleUnloadSample(sample.name)}
                                                style={{
                                                    width: '100%',
                                                    padding: '4px',
                                                    fontSize: '0.85em',
                                                    backgroundColor:
                                                        'var(--vscode-button-background)',
                                                    color: 'var(--vscode-button-foreground)',
                                                    border: 'none',
                                                    borderRadius: '2px',
                                                    cursor: 'pointer',
                                                }}
                                                onMouseEnter={(e) => {
                                                    (
                                                        e.target as HTMLButtonElement
                                                    ).style.backgroundColor =
                                                        'var(--vscode-button-hoverBackground)';
                                                }}
                                                onMouseLeave={(e) => {
                                                    (
                                                        e.target as HTMLButtonElement
                                                    ).style.backgroundColor =
                                                        'var(--vscode-button-background)';
                                                }}
                                            >
                                                Unload Sample
                                            </button>
                                        </div>
                                    </div>
                                )}
                            </div>
                        );
                    })}
                </div>
            </div>

        </div>
    );
};
