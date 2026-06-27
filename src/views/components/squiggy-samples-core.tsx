/**
 * Samples Panel Core Component
 *
 * React-based UI for managing samples (loading, naming, coloring, metadata)
 */

import React, { useEffect, useState, useRef, useCallback } from 'react';
import { vscode } from './vscode-api';
import { SampleItem } from '../../types/messages';
import { SampleRow } from './squiggy-sample-row';
import './squiggy-samples-core.css';

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
    nameEditError: string | null; // Inline validation error for the name edit input
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
        nameEditError: null,
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
            switch (message.type) {
                case 'updateSamples': {
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
                                        newColors.delete(oldName);
                                        newColors.set(matchingNewSample.name, oldColor);
                                    }
                                }
                            }
                        }

                        // Assign default Okabe-Ito colors to new samples (that don't already have colors)
                        const newlyAssignedColors: Array<{ name: string; color: string }> = [];
                        const newlySelectedSamples: string[] = [];
                        samples.forEach((sample, index) => {
                            if (!newColors.has(sample.name) && !sample.isDeferred) {
                                const colorIndex = index % OKABE_ITO_PALETTE.length;
                                const color = OKABE_ITO_PALETTE[colorIndex];
                                newColors.set(sample.name, color);
                                newlyAssignedColors.push({ name: sample.name, color });
                            }
                            // Default loaded (non-deferred) samples to selected for visualization
                            if (!newSelected.has(sample.name) && !sample.isDeferred) {
                                newSelected.add(sample.name);
                                newlySelectedSamples.push(sample.name);
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

                        // Sync auto-selected samples with extension state (Issue #121)
                        if (newlySelectedSamples.length > 0) {
                            setTimeout(() => {
                                newlySelectedSamples.forEach((name) => {
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
                    setState({
                        samples: [],
                        sessionFastaPath: null,
                        editingSampleName: null,
                        editInputValue: '',
                        nameEditError: null,
                        sampleColors: new Map(),
                        expandedSamples: new Set(),
                        selectedSamplesForVisualization: new Set(),
                    });
                    break;

                case 'updateSessionFasta':
                    setState((prev) => ({
                        ...prev,
                        sessionFastaPath: message.fastaPath,
                    }));
                    break;

                case 'updateVisualizationSelection':
                    setState((prev) => ({
                        ...prev,
                        selectedSamplesForVisualization: new Set(message.selectedSamples),
                    }));
                    break;

                default:
                    break;
            }
        };

        window.addEventListener('message', handleMessage);
        // Request initial state
        vscode.postMessage({ type: 'ready' });

        return () => window.removeEventListener('message', handleMessage);
    }, []);

    // Handlers passed to SampleRow are wrapped in useCallback so the memoized
    // row only re-renders when its own data changes (not when an unrelated row
    // expands or a sample is renamed elsewhere).
    const handleUnloadSample = useCallback((sampleName: string) => {
        vscode.postMessage({
            type: 'unloadSample',
            sampleName,
        });
    }, []);

    const handleUnloadAllSamples = () => {
        vscode.postMessage({
            type: 'unloadAllSamples',
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
        vscode.postMessage({
            type: 'requestLoadSamples',
        });
    };

    const handleEditSampleName = useCallback((sampleName: string) => {
        setState((prev) => ({
            ...prev,
            editingSampleName: sampleName,
            editInputValue: sampleName,
            nameEditError: null,
        }));
    }, []);

    const handleSaveNameEdit = useCallback(
        (oldName: string, newName: string) => {
            if (!newName.trim()) {
                // Inline validation error (renders below the input). A native alert()
                // would pop a non-themed OS dialog that clashes with the Positron UI.
                setState((prev) => ({ ...prev, nameEditError: 'Sample name cannot be empty' }));
                return;
            }

            if (newName === oldName) {
                // No change, just exit edit mode
                setState((prev) => ({
                    ...prev,
                    editingSampleName: null,
                    editInputValue: '',
                    nameEditError: null,
                }));
                return;
            }

            // Check for duplicate names
            if (state.samples.some((s) => s.name === newName)) {
                setState((prev) => ({
                    ...prev,
                    nameEditError: 'A sample with this name already exists',
                }));
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
                nameEditError: null,
            }));
        },
        [state.samples]
    );

    const handleCancelNameEdit = useCallback(() => {
        setState((prev) => ({
            ...prev,
            editingSampleName: null,
            editInputValue: '',
            nameEditError: null,
        }));
    }, []);

    // Update the rename input value (passed to the editing SampleRow).
    const handleEditInputChange = useCallback((value: string) => {
        setState((prev) => ({
            ...prev,
            editInputValue: value,
            nameEditError: null,
        }));
    }, []);

    const handleSampleColorChange = useCallback((sampleName: string, color: string) => {
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
    }, []);

    const toggleSampleExpanded = useCallback((sampleName: string) => {
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
    }, []);

    const handleToggleSampleSelection = useCallback((sampleName: string) => {
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
    }, []);

    const handleSetFastaForAll = () => {
        vscode.postMessage({
            type: 'requestSetSessionFasta',
        });
    };

    const handleAddFastaForSample = useCallback((sampleName: string) => {
        vscode.postMessage({
            type: 'requestChangeSampleFasta',
            sampleName,
        });
    }, []);

    const handleChangeBam = useCallback((sampleName: string) => {
        vscode.postMessage({
            type: 'requestChangeSampleBam',
            sampleName,
        });
    }, []);

    const handleChangeFasta = useCallback((sampleName: string) => {
        vscode.postMessage({
            type: 'requestChangeSampleFasta',
            sampleName,
        });
    }, []);

    return (
        <div className="samples-core-container">
            {/* Toolbar Buttons */}
            <div className="samples-toolbar">
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

                {/* "Load All" Button - shown when deferred samples exist */}
                {state.samples.some((s) => s.isDeferred) && (
                    <button
                        onClick={() => {
                            vscode.postMessage({ type: 'loadAllDeferredSamples' });
                        }}
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
                        title="Load all deferred samples sequentially"
                    >
                        Load All
                    </button>
                )}
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
            <div className="samples-list-section">
                <div className="samples-list-header">
                    <div
                        style={{
                            fontWeight: 'bold',
                            color: 'var(--vscode-foreground)',
                        }}
                    >
                        Loaded Samples ({state.samples.length})
                    </div>
                </div>

                {/* Bulk Action Toolbar */}
                {state.samples.length > 0 && (
                    <div className="samples-bulk-toolbar">
                        {state.samples.length > 1 && (
                            <>
                                <button
                                    className="samples-bulk-button"
                                    onClick={() => {
                                        state.samples.forEach((sample) => {
                                            if (
                                                !state.selectedSamplesForVisualization.has(
                                                    sample.name
                                                )
                                            ) {
                                                handleToggleSampleSelection(sample.name);
                                            }
                                        });
                                    }}
                                    title="Select all samples for plotting"
                                >
                                    Select All
                                </button>
                                <button
                                    className="samples-bulk-button"
                                    onClick={() => {
                                        state.samples.forEach((sample) => {
                                            if (
                                                state.selectedSamplesForVisualization.has(
                                                    sample.name
                                                )
                                            ) {
                                                handleToggleSampleSelection(sample.name);
                                            }
                                        });
                                    }}
                                    title="Deselect all samples from plotting"
                                >
                                    Deselect All
                                </button>
                            </>
                        )}
                        <button
                            className="samples-bulk-button danger"
                            onClick={handleUnloadAllSamples}
                            title="Unload all samples"
                        >
                            Unload All
                        </button>
                    </div>
                )}

                <div className="samples-list-container">
                    {state.samples.map((sample) => (
                        <SampleRow
                            key={sample.name}
                            sample={sample}
                            color={state.sampleColors.get(sample.name) || '#808080'}
                            isExpanded={state.expandedSamples.has(sample.name)}
                            isSelected={state.selectedSamplesForVisualization.has(sample.name)}
                            sessionFastaPath={state.sessionFastaPath}
                            isEditing={state.editingSampleName === sample.name}
                            editInputValue={state.editInputValue}
                            nameEditError={state.nameEditError}
                            editInputRef={editInputRef}
                            onToggleExpanded={toggleSampleExpanded}
                            onToggleSelection={handleToggleSampleSelection}
                            onColorChange={handleSampleColorChange}
                            onStartEdit={handleEditSampleName}
                            onSaveEdit={handleSaveNameEdit}
                            onCancelEdit={handleCancelNameEdit}
                            onEditInputChange={handleEditInputChange}
                            onChangeBam={handleChangeBam}
                            onChangeFasta={handleChangeFasta}
                            onAddFasta={handleAddFastaForSample}
                            onUnload={handleUnloadSample}
                        />
                    ))}
                </div>
            </div>
        </div>
    );
};
