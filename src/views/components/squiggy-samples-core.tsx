/**
 * Samples Panel Core Component
 *
 * React-based UI for managing samples (loading, naming, coloring, metadata)
 */

import React, { useEffect, useState, useRef } from 'react';
import { vscode } from './vscode-api';
import { SampleItem } from '../../types/messages';

// Okabe-Ito colorblind-friendly palette
// Reference: https://jfly.uni-koeln.de/color/
const OKABE_ITO_PALETTE = [
    '#E69F00', // Orange
    '#56B4E9', // Sky Blue
    '#009E73', // Bluish Green
    '#F0E442', // Yellow
    '#0072B2', // Blue
    '#D55E00', // Vermillion
    '#CC79A7', // Reddish Purple
];

interface SamplesState {
    samples: SampleItem[];
    sessionFastaPath: string | null; // Session-level FASTA file path
    editingSampleName: string | null; // Which sample name is being edited
    editInputValue: string; // Current value in edit input
    sampleColors: Map<string, string>; // Map of sample names to hex colors
    expandedSamples: Set<string>; // Which samples have their details expanded
    selectedSamplesForVisualization: Set<string>; // Which samples are selected for plotting
}

export const SamplesCore: React.FC = () => {
    const [state, setState] = useState<SamplesState>({
        samples: [],
        sessionFastaPath: null,
        editingSampleName: null,
        editInputValue: '',
        sampleColors: new Map(),
        expandedSamples: new Set(), // Start with all samples collapsed
        selectedSamplesForVisualization: new Set(), // Start with no samples selected
    });
    const editInputRef = useRef<HTMLInputElement>(null);

    // Auto-select text in rename input when entering edit mode
    useEffect(() => {
        if (state.editingSampleName && editInputRef.current) {
            // Focus and select all text
            editInputRef.current.focus();
            editInputRef.current.select();
        }
    }, [state.editingSampleName]);

    // Listen for messages from extension
    useEffect(() => {
        const handleMessage = (event: MessageEvent) => {
            const message = event.data;
            console.log('SamplesCore received message:', message);

            switch (message.type) {
                case 'updateSamples': {
                    console.log('Updating samples:', message.samples);
                    setState((prev) => {
                        const samples = (message as any).samples as SampleItem[];
                        const newColors = new Map(prev.sampleColors);
                        const newSelected = new Set(prev.selectedSamplesForVisualization);

                        // Migrate colors for renamed samples by matching file paths
                        // This preserves colors when a sample is renamed (same files, different name)
                        for (const [oldName, oldColor] of prev.sampleColors) {
                            // If the old name is not in the new samples list
                            if (!samples.some((s) => s.name === oldName)) {
                                // Find the old sample by name to get its file paths
                                const oldSample = prev.samples.find((s) => s.name === oldName);
                                if (oldSample) {
                                    // Find matching new sample by comparing file paths (unique identifiers)
                                    const matchingNewSample = samples.find(
                                        (s) =>
                                            s.pod5Path === oldSample.pod5Path &&
                                            s.bamPath === oldSample.bamPath
                                    );
                                    if (matchingNewSample) {
                                        // Migrate the color to the new sample name
                                        console.log(
                                            `[SamplesCore] Migrating color for renamed sample: ${oldName} → ${matchingNewSample.name}`
                                        );
                                        newColors.delete(oldName);
                                        newColors.set(matchingNewSample.name, oldColor);
                                    }
                                }
                            }
                        }

                        // Track newly added samples for auto-selection sync
                        const newlyAddedSamples: string[] = [];

                        // Assign default Okabe-Ito colors to new samples (that don't already have colors)
                        const newlyAssignedColors: Array<{ name: string; color: string }> = [];
                        samples.forEach((sample, index) => {
                            if (!newColors.has(sample.name)) {
                                const colorIndex = index % OKABE_ITO_PALETTE.length;
                                const color = OKABE_ITO_PALETTE[colorIndex];
                                newColors.set(sample.name, color);
                                newlyAssignedColors.push({ name: sample.name, color });
                            }
                            // Default all samples to selected for visualization
                            if (!newSelected.has(sample.name)) {
                                newSelected.add(sample.name);
                                newlyAddedSamples.push(sample.name);
                            }
                        });

                        // Persist auto-assigned colors to extension state
                        if (newlyAssignedColors.length > 0) {
                            setTimeout(() => {
                                newlyAssignedColors.forEach(({ name, color }) => {
                                    vscode.postMessage({
                                        type: 'updateSampleColor',
                                        sampleName: name,
                                        color: color,
                                    });
                                });
                            }, 0);
                        }

                        // Sync newly auto-selected samples to extension state (FIX for Issue #124)
                        if (newlyAddedSamples.length > 0) {
                            setTimeout(() => {
                                newlyAddedSamples.forEach((name) => {
                                    vscode.postMessage({
                                        type: 'toggleSampleSelection',
                                        sampleName: name,
                                    });
                                });
                            }, 0);
                        }

                        return {
                            ...prev,
                            samples,
                            sampleColors: newColors,
                            selectedSamplesForVisualization: newSelected,
                        };
                    });
                    break;
                }

                case 'clearSamples':
                    console.log('Clearing samples');
                    setState({
                        samples: [],
                        sessionFastaPath: null,
                        editingSampleName: null,
                        editInputValue: '',
                        sampleColors: new Map(),
                        expandedSamples: new Set(),
                        selectedSamplesForVisualization: new Set(),
                    });
                    break;

                case 'updateSessionFasta':
                    console.log('Updating session FASTA:', message.fastaPath);
                    setState((prev) => ({
                        ...prev,
                        sessionFastaPath: message.fastaPath,
                    }));
                    break;

                case 'updateVisualizationSelection':
                    console.log(
                        '[SamplesCore] Updating visualization selection from extension:',
                        message.selectedSamples
                    );
                    setState((prev) => ({
                        ...prev,
                        selectedSamplesForVisualization: new Set(message.selectedSamples),
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

    const handleToggleSampleSelection = (sampleName: string) => {
        setState((prev) => {
            const newSelected = new Set(prev.selectedSamplesForVisualization);
            if (newSelected.has(sampleName)) {
                newSelected.delete(sampleName);
            } else {
                newSelected.add(sampleName);
            }
            return {
                ...prev,
                selectedSamplesForVisualization: newSelected,
            };
        });

        // Send selection change to extension
        vscode.postMessage({
            type: 'toggleSampleSelection',
            sampleName: sampleName,
        });
    };

    const handleSetFastaForAll = () => {
        vscode.postMessage({
            type: 'requestSetSessionFasta',
        });
    };

    const handleAddFastaForSample = (sampleName: string) => {
        vscode.postMessage({
            type: 'requestChangeSampleFasta',
            sampleName,
        });
    };

    const handleChangeBam = (sampleName: string) => {
        vscode.postMessage({
            type: 'requestChangeSampleBam',
            sampleName,
        });
    };

    const handleChangeFasta = (sampleName: string) => {
        vscode.postMessage({
            type: 'requestChangeSampleFasta',
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

    return (
        <div
            style={{
                padding: '6px',
                fontFamily: 'var(--vscode-font-family)',
                fontSize: 'var(--vscode-font-size)',
                color: 'var(--vscode-foreground)',
            }}
        >
            {/* Toolbar Buttons */}
            <div style={{ display: 'flex', gap: '5px', marginBottom: '8px' }}>
                {/* "Load Sample" Button */}
                <button
                    onClick={_handleLoadSamplesClick}
                    style={{
                        flex: 1,
                        padding: '4px 5px',
                        backgroundColor: 'var(--vscode-button-background)',
                        color: 'var(--vscode-button-foreground)',
                        border: 'none',
                        borderRadius: '2px',
                        cursor: 'pointer',
                        fontSize: 'var(--vscode-font-size)',
                        fontFamily: 'var(--vscode-font-family)',
                    }}
                    onMouseEnter={(e) => {
                        (e.target as HTMLButtonElement).style.backgroundColor =
                            'var(--vscode-button-hoverBackground)';
                    }}
                    onMouseLeave={(e) => {
                        (e.target as HTMLButtonElement).style.backgroundColor =
                            'var(--vscode-button-background)';
                    }}
                    title="Load POD5 and optional BAM/FASTA files as samples"
                >
                    Load Sample(s)
                </button>

                {/* "Set Reference" Button */}
                <button
                    onClick={handleSetFastaForAll}
                    style={{
                        flex: 1,
                        padding: '4px 5px',
                        backgroundColor: 'var(--vscode-button-background)',
                        color: 'var(--vscode-button-foreground)',
                        border: 'none',
                        borderRadius: '2px',
                        cursor: 'pointer',
                        fontSize: 'var(--vscode-font-size)',
                        fontFamily: 'var(--vscode-font-family)',
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
                    {state.sessionFastaPath ? '✓ Set Reference' : 'Set Reference'}
                </button>
            </div>
            {state.sessionFastaPath && (
                <div
                    style={{
                        fontSize: '0.75em',
                        color: 'var(--vscode-descriptionForeground)',
                        padding: '2px',
                        wordBreak: 'break-word',
                        marginBottom: '8px',
                    }}
                >
                    {state.sessionFastaPath.split('/').pop()}
                </div>
            )}

            {/* Empty State Message */}
            {state.samples.length === 0 && (
                <div
                    style={{
                        padding: '8px',
                        backgroundColor: 'var(--vscode-input-background)',
                        border: '1px solid var(--vscode-widget-border)',
                        borderRadius: '4px',
                        color: 'var(--vscode-descriptionForeground)',
                        fontSize: '0.9em',
                        lineHeight: '1.5',
                        marginBottom: '8px',
                    }}
                >
                    <p style={{ marginTop: 0 }}>
                        <strong>No samples loaded yet.</strong>
                    </p>
                    <p style={{ margin: '5px 0' }}>
                        Click <strong>"Load Sample(s)"</strong> to add POD5 and BAM files, or{' '}
                        <strong>"Set Reference"</strong> to add a shared reference genome. Samples
                        will appear here for naming, coloring, and organization.
                    </p>
                </div>
            )}

            {/* Samples List - Collapsible Rows */}
            <div style={{ marginBottom: '12px' }}>
                <div
                    style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        marginBottom: '5px',
                    }}
                >
                    <div
                        style={{
                            fontWeight: 'bold',
                            color: 'var(--vscode-foreground)',
                        }}
                    >
                        Loaded Samples ({state.samples.length})
                    </div>

                    {/* Bulk Selection Buttons */}
                    {state.samples.length > 1 && (
                        <div style={{ display: 'flex', gap: '4px' }}>
                            <button
                                onClick={() => {
                                    // Select all samples for visualization
                                    state.samples.forEach((sample) => {
                                        if (!state.selectedSamplesForVisualization.has(sample.name)) {
                                            handleToggleSampleSelection(sample.name);
                                        }
                                    });
                                }}
                                style={{
                                    padding: '2px 6px',
                                    fontSize: '0.75em',
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
                                title="Select all samples for plotting"
                            >
                                Select All
                            </button>
                            <button
                                onClick={() => {
                                    // Deselect all samples for visualization
                                    state.samples.forEach((sample) => {
                                        if (state.selectedSamplesForVisualization.has(sample.name)) {
                                            handleToggleSampleSelection(sample.name);
                                        }
                                    });
                                }}
                                style={{
                                    padding: '2px 6px',
                                    fontSize: '0.75em',
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
                                title="Deselect all samples from plotting"
                            >
                                Deselect All
                            </button>
                        </div>
                    )}
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
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
                                        gap: '4px',
                                        padding: '5px',
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
                                    {/* Selection Checkbox with Eye Icon */}
                                    <div
                                        style={{
                                            display: 'flex',
                                            alignItems: 'center',
                                            gap: '3px',
                                            cursor: 'pointer',
                                        }}
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            handleToggleSampleSelection(sample.name);
                                        }}
                                        title="Include this sample in plot visualizations"
                                    >
                                        <span
                                            style={{
                                                fontSize: '0.85em',
                                                opacity: state.selectedSamplesForVisualization.has(
                                                    sample.name
                                                )
                                                    ? 1
                                                    : 0.3,
                                                transition: 'opacity 0.2s',
                                            }}
                                        >
                                            👁️
                                        </span>
                                        <input
                                            type="checkbox"
                                            checked={state.selectedSamplesForVisualization.has(
                                                sample.name
                                            )}
                                            onChange={(e) => {
                                                e.stopPropagation();
                                                handleToggleSampleSelection(sample.name);
                                            }}
                                            onClick={(e) => e.stopPropagation()}
                                            style={{
                                                width: '16px',
                                                height: '16px',
                                                cursor: 'pointer',
                                                flexShrink: 0,
                                                margin: 0,
                                            }}
                                        />
                                    </div>

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
                                                ref={editInputRef}
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
                                                    padding: '3px',
                                                    backgroundColor:
                                                        'var(--vscode-input-background)',
                                                    color: 'var(--vscode-input-foreground)',
                                                    border: '1px solid var(--vscode-input-border)',
                                                    borderRadius: '2px',
                                                    fontWeight: 'bold',
                                                    fontFamily: 'var(--vscode-font-family)',
                                                    fontSize: '0.95em',
                                                }}
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
                                                    padding: '2px 4px',
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
                                                    padding: '2px 4px',
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
                                            padding: '2px 4px',
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
                                            padding: '5px',
                                            backgroundColor: 'var(--vscode-input-background)',
                                            fontSize: '0.85em',
                                            color: 'var(--vscode-foreground)',
                                        }}
                                        onClick={(e) => e.stopPropagation()}
                                    >
                                        {/* POD5 File */}
                                        <div style={{ marginBottom: '4px' }}>
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
                                                    padding: '3px 4px',
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
                                        <div style={{ marginBottom: '4px' }}>
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
                                                            padding: '3px 4px',
                                                            backgroundColor:
                                                                'var(--vscode-editor-background)',
                                                            borderRadius: '2px',
                                                            wordBreak: 'break-all',
                                                        }}
                                                    >
                                                        {sample.bamPath.split('/').pop()}
                                                    </div>
                                                    <button
                                                        onClick={() => handleChangeBam(sample.name)}
                                                        style={{
                                                            padding: '2px 5px',
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
                                                        title="Change BAM file for this sample"
                                                    >
                                                        [Change]
                                                    </button>
                                                </div>
                                            ) : (
                                                <button
                                                    onClick={() => handleChangeBam(sample.name)}
                                                    style={{
                                                        width: '100%',
                                                        padding: '3px',
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
                                                    title="Add BAM file for this sample"
                                                >
                                                    + Add BAM
                                                </button>
                                            )}
                                        </div>

                                        {/* FASTA File - Show sample FASTA or session FASTA */}
                                        <div style={{ marginBottom: '4px' }}>
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
                                                                    gap: '3px',
                                                                    alignItems: 'center',
                                                                }}
                                                            >
                                                                <div
                                                                    style={{
                                                                        flex: 1,
                                                                        padding: '3px 4px',
                                                                        backgroundColor:
                                                                            'var(--vscode-editor-background)',
                                                                        borderRadius: '2px',
                                                                        wordBreak: 'break-all',
                                                                    }}
                                                                >
                                                                    {fastaPath.split('/').pop()}
                                                                    {isSessionFasta && (
                                                                        <span
                                                                            style={{
                                                                                fontSize: '0.75em',
                                                                                color: 'var(--vscode-descriptionForeground)',
                                                                                marginLeft: '3px',
                                                                            }}
                                                                        >
                                                                            (session)
                                                                        </span>
                                                                    )}
                                                                </div>
                                                                <button
                                                                    onClick={() =>
                                                                        handleChangeFasta(
                                                                            sample.name
                                                                        )
                                                                    }
                                                                    style={{
                                                                        padding: '2px 5px',
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
                                                                    title="Change FASTA file for this sample"
                                                                >
                                                                    [Change]
                                                                </button>
                                                            </div>
                                                        ) : (
                                                            <button
                                                                onClick={() =>
                                                                    handleAddFastaForSample(
                                                                        sample.name
                                                                    )
                                                                }
                                                                style={{
                                                                    width: '100%',
                                                                    padding: '3px',
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
                                                marginTop: '5px',
                                                borderTop: '1px solid var(--vscode-widget-border)',
                                                paddingTop: '5px',
                                            }}
                                        >
                                            <button
                                                onClick={() => handleUnloadSample(sample.name)}
                                                style={{
                                                    width: '100%',
                                                    padding: '3px',
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
