/**
 * Modifications Panel Core Component
 *
 * React-based UI for base modification filtering
 */

import React, { useEffect, useState } from 'react';
import { vscode } from './vscode-api';

interface ModificationsState {
    hasModifications: boolean;
    modificationTypes: string[];
    hasProbabilities: boolean;
    minProbability: number;
    minFrequency: number;
    minModifiedReads: number;
    enabledModTypes: Set<string>;
}

// Map common modification codes to names
const modCodeToName: Record<string, string> = {
    // Single-letter codes
    m: '5-methylcytosine (5mC)',
    h: '5-hydroxymethylcytosine (5hmC)',
    a: '6-methyladenine (6mA)',
    o: '8-oxoguanine (8-oxoG)',
    // ChEBI codes (common RNA modifications)
    '17596': 'Inosine (I)',
    '28177': '1-methyladenosine (m1A)',
    '21863': '1-methylguanosine (m1G)',
    '28527': '7-methylguanosine (m7G)',
    '17802': 'Pseudouridine (Ψ)',
    '27301': '5-methyluridine (m5U)',
    '18421': 'Dihydrouridine (D)',
};

// Modification colors matching base colors from eventalign view
const modificationColors: Record<string, string> = {
    // Cytosine modifications (yellow family)
    m: '#F0E442',
    h: '#E6D835',
    f: '#DCC728',
    c: '#FFF78A',
    C: '#F0E442',
    // Adenine modifications (green family)
    a: '#009E73',
    '17596': '#00C490',
    A: '#009E73',
    // Guanine modifications (blue family)
    o: '#0072B2',
    G: '#0072B2',
    // Thymine/Uracil modifications (orange family)
    g: '#D55E00',
    e: '#FF7518',
    b: '#B34C00',
    '17802': '#FF9447',
    T: '#D55E00',
    // Default
    default: '#999999',
};

export const ModificationsCore: React.FC = () => {
    const [state, setState] = useState<ModificationsState>({
        hasModifications: false,
        modificationTypes: [],
        hasProbabilities: false,
        minProbability: 0.5,
        minFrequency: 0.2,
        minModifiedReads: 5,
        enabledModTypes: new Set(),
    });

    // Listen for messages from extension
    useEffect(() => {
        const handleMessage = (event: MessageEvent) => {
            const message = event.data;
            console.log('ModificationsCore received message:', message);
            switch (message.type) {
                case 'updateModInfo':
                    console.log('Updating mod info:', message);
                    setState({
                        hasModifications: message.hasModifications,
                        modificationTypes: message.modificationTypes,
                        hasProbabilities: message.hasProbabilities,
                        minProbability: 0.5,
                        minFrequency: 0.2,
                        minModifiedReads: 5,
                        enabledModTypes: new Set(message.modificationTypes),
                    });
                    break;
                case 'clearMods':
                    console.log('Clearing mods');
                    setState({
                        hasModifications: false,
                        modificationTypes: [],
                        hasProbabilities: false,
                        minProbability: 0.5,
                        minFrequency: 0.2,
                        minModifiedReads: 5,
                        enabledModTypes: new Set(),
                    });
                    break;
                default:
                    console.log('Unknown message type:', message.type);
            }
        };

        window.addEventListener('message', handleMessage);
        // Request initial state
        console.log('ModificationsCore sending ready message');
        vscode.postMessage({ type: 'ready' });

        return () => window.removeEventListener('message', handleMessage);
    }, []);

    const handleProbabilityChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const value = parseFloat(e.target.value);
        setState((prev) => ({ ...prev, minProbability: value }));
        vscode.postMessage({
            type: 'filtersChanged',
            minProbability: value,
            minFrequency: state.minFrequency,
            minModifiedReads: state.minModifiedReads,
            enabledModTypes: Array.from(state.enabledModTypes),
        });
    };

    const handleFrequencyChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const value = parseFloat(e.target.value) / 100; // Convert percentage to 0-1
        setState((prev) => ({ ...prev, minFrequency: value }));
        vscode.postMessage({
            type: 'filtersChanged',
            minProbability: state.minProbability,
            minFrequency: value,
            minModifiedReads: state.minModifiedReads,
            enabledModTypes: Array.from(state.enabledModTypes),
        });
    };

    const handleModifiedReadsChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const value = parseInt(e.target.value) || 5;
        setState((prev) => ({ ...prev, minModifiedReads: value }));
        vscode.postMessage({
            type: 'filtersChanged',
            minProbability: state.minProbability,
            minFrequency: state.minFrequency,
            minModifiedReads: value,
            enabledModTypes: Array.from(state.enabledModTypes),
        });
    };

    const handleModTypeToggle = (modType: string) => {
        setState((prev) => {
            const newEnabled = new Set(prev.enabledModTypes);
            if (newEnabled.has(modType)) {
                newEnabled.delete(modType);
            } else {
                newEnabled.add(modType);
            }

            // Send update to extension
            vscode.postMessage({
                type: 'filtersChanged',
                minProbability: prev.minProbability,
                minFrequency: prev.minFrequency,
                minModifiedReads: prev.minModifiedReads,
                enabledModTypes: Array.from(newEnabled),
            });

            return { ...prev, enabledModTypes: newEnabled };
        });
    };

    const getModName = (code: string): string => {
        return modCodeToName[code] || code;
    };

    const getModColor = (code: string): string => {
        return modificationColors[code] || modificationColors.default;
    };

    if (!state.hasModifications) {
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
                No modifications detected. Load a BAM file with MM/ML tags.
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
            {/* Probability Filter */}
            {state.hasProbabilities && (
                <div style={{ marginBottom: '20px' }}>
                    <div
                        style={{
                            fontWeight: 'bold',
                            marginBottom: '4px',
                            color: 'var(--vscode-foreground)',
                        }}
                    >
                        Modification Confidence Filter
                    </div>
                    <div
                        style={{
                            fontSize: '0.85em',
                            marginBottom: '8px',
                            color: 'var(--vscode-descriptionForeground)',
                        }}
                    >
                        Minimum basecaller probability to count a read as modified
                    </div>
                    <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.05"
                        value={state.minProbability}
                        onChange={handleProbabilityChange}
                        aria-label="Minimum basecaller probability threshold"
                        aria-valuetext={`${state.minProbability.toFixed(2)} probability`}
                        aria-valuenow={state.minProbability}
                        aria-valuemin={0}
                        aria-valuemax={1}
                        style={{ width: '100%', marginBottom: '4px' }}
                    />
                    <div
                        style={{
                            textAlign: 'right',
                            fontSize: '0.9em',
                            fontWeight: 'bold',
                            color: 'var(--vscode-input-foreground)',
                        }}
                    >
                        ≥ {state.minProbability.toFixed(2)}
                    </div>
                </div>
            )}

            {/* Frequency Filter */}
            {state.hasProbabilities && (
                <div style={{ marginBottom: '20px' }}>
                    <div
                        style={{
                            fontWeight: 'bold',
                            marginBottom: '4px',
                            color: 'var(--vscode-foreground)',
                        }}
                    >
                        Minimum Modification Frequency
                    </div>
                    <div
                        style={{
                            fontSize: '0.85em',
                            marginBottom: '8px',
                            color: 'var(--vscode-descriptionForeground)',
                        }}
                    >
                        Minimum % of reads modified at a position (aggregate plots only)
                    </div>
                    <input
                        type="range"
                        min="0"
                        max="100"
                        step="5"
                        value={state.minFrequency * 100}
                        onChange={handleFrequencyChange}
                        aria-label="Minimum modification frequency threshold"
                        aria-valuetext={`${(state.minFrequency * 100).toFixed(0)}% frequency`}
                        aria-valuenow={state.minFrequency * 100}
                        aria-valuemin={0}
                        aria-valuemax={100}
                        style={{ width: '100%', marginBottom: '4px' }}
                    />
                    <div
                        style={{
                            textAlign: 'right',
                            fontSize: '0.9em',
                            fontWeight: 'bold',
                            color: 'var(--vscode-input-foreground)',
                        }}
                    >
                        ≥ {(state.minFrequency * 100).toFixed(0)}%
                    </div>
                </div>
            )}

            {/* Modified Reads Count Filter */}
            {state.hasProbabilities && (
                <div style={{ marginBottom: '20px' }}>
                    <div
                        style={{
                            fontWeight: 'bold',
                            marginBottom: '4px',
                            color: 'var(--vscode-foreground)',
                        }}
                    >
                        Minimum Modified Reads
                    </div>
                    <div
                        style={{
                            fontSize: '0.85em',
                            marginBottom: '8px',
                            color: 'var(--vscode-descriptionForeground)',
                        }}
                    >
                        Minimum number of reads that must be modified (aggregate plots only)
                    </div>
                    <input
                        type="number"
                        min="1"
                        value={state.minModifiedReads}
                        onChange={handleModifiedReadsChange}
                        aria-label="Minimum modified reads count"
                        style={{
                            width: '100%',
                            padding: '4px',
                            backgroundColor: 'var(--vscode-input-background)',
                            color: 'var(--vscode-input-foreground)',
                            border: '1px solid var(--vscode-input-border)',
                            marginBottom: '4px',
                        }}
                    />
                    <div
                        style={{
                            textAlign: 'right',
                            fontSize: '0.9em',
                            fontWeight: 'bold',
                            color: 'var(--vscode-input-foreground)',
                        }}
                    >
                        ≥ {state.minModifiedReads} reads
                    </div>
                </div>
            )}

            {/* Info Box */}
            <div
                style={{
                    padding: '8px',
                    marginBottom: '20px',
                    backgroundColor: 'var(--vscode-textBlockQuote-background)',
                    borderLeft: '3px solid var(--vscode-textLink-foreground)',
                    fontSize: '0.85em',
                    color: 'var(--vscode-foreground)',
                }}
            >
                <div style={{ fontWeight: 'bold', marginBottom: '4px' }}>Aggregate plots show:</div>
                <ul style={{ marginTop: '4px', marginBottom: '0', paddingLeft: '20px' }}>
                    <li style={{ marginBottom: '4px' }}>
                        <strong>Frequency:</strong> % of reads modified at each position
                    </li>
                    <li>
                        <strong>Mean Prob:</strong> Average confidence among modified reads
                    </li>
                </ul>
            </div>

            {/* Modification Types */}
            <div style={{ marginBottom: '20px' }}>
                <div
                    style={{
                        fontWeight: 'bold',
                        marginBottom: '8px',
                        color: 'var(--vscode-foreground)',
                    }}
                >
                    Modification Types
                </div>
                {state.modificationTypes.map((modType) => (
                    <div
                        key={modType}
                        style={{
                            display: 'flex',
                            alignItems: 'center',
                            marginBottom: '8px',
                        }}
                    >
                        <input
                            type="checkbox"
                            id={`mod-${modType}`}
                            checked={state.enabledModTypes.has(modType)}
                            onChange={() => handleModTypeToggle(modType)}
                            style={{ marginRight: '6px' }}
                        />
                        <label
                            htmlFor={`mod-${modType}`}
                            style={{
                                fontSize: '0.9em',
                                display: 'flex',
                                alignItems: 'center',
                            }}
                        >
                            <span
                                style={{
                                    display: 'inline-block',
                                    width: '12px',
                                    height: '12px',
                                    backgroundColor: getModColor(modType),
                                    marginRight: '6px',
                                    border: '1px solid var(--vscode-input-border)',
                                }}
                            />
                            {getModName(modType)}
                        </label>
                    </div>
                ))}
            </div>
        </div>
    );
};
