/**
 * Sample Row
 *
 * A single collapsible sample entry in the Samples panel: header (selection,
 * color, editable name, load/badge, delete) plus expandable details
 * (POD5/BAM/FASTA paths and references).
 *
 * Extracted from SamplesCore. Wrapped in React.memo so that state changes in
 * the container that don't affect this row (e.g. expanding a different sample)
 * skip re-rendering it. For the memo to be effective the parent passes stable
 * (useCallback) handlers and a referentially-stable `sample`.
 */

import React from 'react';
import { vscode } from './vscode-api';
import { SampleItem } from '../../types/messages';

interface SampleRowProps {
    sample: SampleItem;
    color: string;
    isExpanded: boolean;
    isSelected: boolean;
    sessionFastaPath: string | null;
    isEditing: boolean;
    editInputValue: string;
    nameEditError: string | null;
    editInputRef: React.RefObject<HTMLInputElement | null>;
    onToggleExpanded: (name: string) => void;
    onToggleSelection: (name: string) => void;
    onColorChange: (name: string, color: string) => void;
    onStartEdit: (name: string) => void;
    onSaveEdit: (oldName: string, newName: string) => void;
    onCancelEdit: () => void;
    onEditInputChange: (value: string) => void;
    onChangeBam: (name: string) => void;
    onChangeFasta: (name: string) => void;
    onAddFasta: (name: string) => void;
    onUnload: (name: string) => void;
}

const SampleRowComponent: React.FC<SampleRowProps> = ({
    sample,
    color,
    isExpanded,
    isSelected,
    sessionFastaPath,
    isEditing,
    editInputValue,
    nameEditError,
    editInputRef,
    onToggleExpanded,
    onToggleSelection,
    onColorChange,
    onStartEdit,
    onSaveEdit,
    onCancelEdit,
    onEditInputChange,
    onChangeBam,
    onChangeFasta,
    onAddFasta,
    onUnload,
}) => {
    const isLoading = sample.isLoading ?? false;
    const isDeferred = sample.isDeferred ?? false;
    return (
        <div
            className={
                isLoading ? 'sample-row loading' : isDeferred ? 'sample-row deferred' : 'sample-row'
            }
            style={{
                backgroundColor: 'var(--vscode-editor-background)',
                border: '1px solid var(--vscode-widget-border)',
                borderRadius: '4px',
                overflow: 'hidden',
                opacity: isLoading ? 0.7 : 1,
                transition: 'opacity 0.3s ease',
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
                    borderBottom: isExpanded ? '1px solid var(--vscode-widget-border)' : 'none',
                }}
                onClick={() => onToggleExpanded(sample.name)}
            >
                {/* Selection Checkbox with Eye Icon (hidden for deferred) */}
                {!isDeferred && (
                    <div
                        style={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: '3px',
                            cursor: 'pointer',
                        }}
                        onClick={(e) => {
                            e.stopPropagation();
                            onToggleSelection(sample.name);
                        }}
                        title="Include this sample in plot visualizations"
                    >
                        <span
                            style={{
                                fontSize: '0.85em',
                                opacity: isSelected ? 1 : 0.3,
                                transition: 'opacity 0.2s',
                            }}
                        >
                            👁️
                        </span>
                        <input
                            type="checkbox"
                            checked={isSelected}
                            onChange={(e) => {
                                e.stopPropagation();
                                onToggleSelection(sample.name);
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
                )}

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

                {/* Color Picker (hidden for deferred) */}
                {!isDeferred && (
                    <input
                        type="color"
                        value={color}
                        onChange={(e) => {
                            e.stopPropagation();
                            onColorChange(sample.name, e.target.value);
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
                )}

                {/* Sample Name - Editable */}
                {isEditing ? (
                    <div
                        style={{
                            display: 'flex',
                            flexDirection: 'column',
                            gap: '2px',
                            flex: 1,
                        }}
                        onClick={(e) => e.stopPropagation()}
                    >
                        <div style={{ display: 'flex', gap: '4px' }}>
                            <input
                                ref={editInputRef}
                                type="text"
                                value={editInputValue}
                                onChange={(e) => onEditInputChange(e.target.value)}
                                onKeyDown={(e) => {
                                    e.stopPropagation();
                                    if (e.key === 'Enter') {
                                        onSaveEdit(sample.name, editInputValue);
                                    } else if (e.key === 'Escape') {
                                        onCancelEdit();
                                    }
                                }}
                                style={{
                                    flex: 1,
                                    padding: '3px',
                                    backgroundColor: 'var(--vscode-input-background)',
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
                                    onSaveEdit(sample.name, editInputValue);
                                }}
                                style={{
                                    padding: '2px 4px',
                                    backgroundColor: 'var(--vscode-button-background)',
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
                                    onCancelEdit();
                                }}
                                style={{
                                    padding: '2px 4px',
                                    backgroundColor: 'var(--vscode-errorForeground)',
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
                        {nameEditError && (
                            <span
                                style={{
                                    color: 'var(--vscode-errorForeground)',
                                    fontSize: '0.8em',
                                }}
                            >
                                {nameEditError}
                            </span>
                        )}
                    </div>
                ) : (
                    <label
                        htmlFor={`sample-${sample.name}`}
                        style={{
                            fontWeight: isDeferred ? 'normal' : 'bold',
                            color: isDeferred ? 'var(--vscode-descriptionForeground)' : undefined,
                            flex: 1,
                            cursor: 'pointer',
                            userSelect: 'none',
                        }}
                        onDoubleClick={(e) => {
                            e.stopPropagation();
                            onStartEdit(sample.name);
                        }}
                    >
                        {sample.name}
                    </label>
                )}

                {/* Load Button / Loading Spinner / Read Count Badge */}
                {isDeferred ? (
                    <button
                        onClick={(e) => {
                            e.stopPropagation();
                            vscode.postMessage({
                                type: 'loadDeferredSample',
                                sampleName: sample.name,
                            });
                        }}
                        style={{
                            padding: '2px 8px',
                            fontSize: '0.75em',
                            backgroundColor: 'var(--vscode-button-background)',
                            color: 'var(--vscode-button-foreground)',
                            border: 'none',
                            borderRadius: '2px',
                            cursor: 'pointer',
                            flexShrink: 0,
                        }}
                        onMouseEnter={(e) => {
                            (e.target as HTMLButtonElement).style.backgroundColor =
                                'var(--vscode-button-hoverBackground)';
                        }}
                        onMouseLeave={(e) => {
                            (e.target as HTMLButtonElement).style.backgroundColor =
                                'var(--vscode-button-background)';
                        }}
                        title="Load this sample's data from disk"
                    >
                        Load
                    </button>
                ) : isLoading ? (
                    <span
                        className="loading-spinner"
                        style={{
                            fontSize: '0.75em',
                            backgroundColor: 'var(--vscode-badge-background)',
                            color: 'var(--vscode-badge-foreground)',
                            padding: '2px 6px',
                            borderRadius: '2px',
                            flexShrink: 0,
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: '4px',
                        }}
                        title={sample.loadingMessage || 'Loading...'}
                    >
                        <span className="spinner-icon">⟳</span>
                        {sample.loadingMessage || 'Loading...'}
                    </span>
                ) : (
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
                )}

                {/* Delete/Remove Sample Button (Trash Icon) */}
                <button
                    onClick={(e) => {
                        e.stopPropagation();
                        onUnload(sample.name);
                    }}
                    style={{
                        width: '22px',
                        height: '22px',
                        padding: '2px',
                        backgroundColor: 'transparent',
                        color: 'var(--vscode-errorForeground)',
                        border: '1px solid transparent',
                        borderRadius: '2px',
                        cursor: 'pointer',
                        flexShrink: 0,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: '0.9em',
                        transition: 'all 0.2s',
                    }}
                    onMouseEnter={(e) => {
                        (e.target as HTMLButtonElement).style.backgroundColor =
                            'var(--vscode-inputValidation-errorBackground)';
                        (e.target as HTMLButtonElement).style.borderColor =
                            'var(--vscode-errorForeground)';
                    }}
                    onMouseLeave={(e) => {
                        (e.target as HTMLButtonElement).style.backgroundColor = 'transparent';
                        (e.target as HTMLButtonElement).style.borderColor = 'transparent';
                    }}
                    title="Remove this sample"
                >
                    🗑️
                </button>
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
                                backgroundColor: 'var(--vscode-editor-background)',
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
                                        backgroundColor: 'var(--vscode-editor-background)',
                                        borderRadius: '2px',
                                        wordBreak: 'break-all',
                                    }}
                                >
                                    {sample.bamPath.split('/').pop()}
                                </div>
                                <button
                                    onClick={() => onChangeBam(sample.name)}
                                    style={{
                                        padding: '2px 5px',
                                        fontSize: '0.75em',
                                        backgroundColor: 'var(--vscode-button-background)',
                                        color: 'var(--vscode-button-foreground)',
                                        border: 'none',
                                        borderRadius: '2px',
                                        cursor: 'pointer',
                                        flexShrink: 0,
                                    }}
                                    onMouseEnter={(e) => {
                                        (e.target as HTMLButtonElement).style.backgroundColor =
                                            'var(--vscode-button-hoverBackground)';
                                    }}
                                    onMouseLeave={(e) => {
                                        (e.target as HTMLButtonElement).style.backgroundColor =
                                            'var(--vscode-button-background)';
                                    }}
                                    title="Change BAM file for this sample"
                                >
                                    [Change]
                                </button>
                            </div>
                        ) : (
                            <button
                                onClick={() => onChangeBam(sample.name)}
                                style={{
                                    width: '100%',
                                    padding: '3px',
                                    fontSize: '0.85em',
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
                                title="Add BAM file for this sample"
                            >
                                + Add BAM
                            </button>
                        )}
                    </div>

                    {/* FASTA File - Show sample FASTA or session FASTA */}
                    <div style={{ marginBottom: '4px' }}>
                        {(() => {
                            const fastaPath = sample.fastaPath || sessionFastaPath;
                            const isSessionFasta = !sample.fastaPath && sessionFastaPath;
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
                                                onClick={() => onChangeFasta(sample.name)}
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
                                            onClick={() => onAddFasta(sample.name)}
                                            style={{
                                                width: '100%',
                                                padding: '3px',
                                                fontSize: '0.85em',
                                                backgroundColor: 'var(--vscode-button-background)',
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

                    {/* References (from BAM alignment) */}
                    {sample.references && sample.references.length > 0 && (
                        <div style={{ marginBottom: '4px' }}>
                            <div
                                style={{
                                    color: 'var(--vscode-descriptionForeground)',
                                    fontSize: '0.8em',
                                    marginBottom: '2px',
                                }}
                            >
                                References ({sample.references.length})
                            </div>
                            <div
                                style={{
                                    padding: '3px 4px',
                                    backgroundColor: 'var(--vscode-editor-background)',
                                    borderRadius: '2px',
                                    fontSize: '0.85em',
                                    maxHeight: '80px',
                                    overflowY: 'auto',
                                }}
                            >
                                {sample.references.map((ref, idx) => (
                                    <div
                                        key={idx}
                                        style={{
                                            padding: '1px 0',
                                            display: 'flex',
                                            justifyContent: 'space-between',
                                            gap: '4px',
                                        }}
                                    >
                                        <span
                                            style={{
                                                flex: 1,
                                                overflow: 'hidden',
                                                textOverflow: 'ellipsis',
                                                whiteSpace: 'nowrap',
                                            }}
                                            title={ref.name}
                                        >
                                            {ref.name}
                                        </span>
                                        <span
                                            style={{
                                                color: 'var(--vscode-descriptionForeground)',
                                                fontSize: '0.9em',
                                                flexShrink: 0,
                                            }}
                                        >
                                            {ref.readCount} reads
                                        </span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Unload Button */}
                    <div
                        style={{
                            marginTop: '5px',
                            borderTop: '1px solid var(--vscode-widget-border)',
                            paddingTop: '5px',
                        }}
                    >
                        <button
                            onClick={() => onUnload(sample.name)}
                            style={{
                                width: '100%',
                                padding: '3px',
                                fontSize: '0.85em',
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
                            Unload Sample
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
};

export const SampleRow = React.memo(SampleRowComponent);
