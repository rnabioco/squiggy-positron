/**
 * Per-Read plot section of the Plot Options panel
 * (Multi-Read Overlay / Multi-Read Stacked / Reference Overlay)
 *
 * Presentational section extracted from PlotOptionsCore. State lives in the
 * container; this component renders the controls for the per-read plot types.
 */

import React from 'react';
import { PlotOptionsState, PlotType, SetPlotOptions } from './squiggy-plot-options-types';

interface PerReadSectionProps {
    options: PlotOptionsState;
    setOptions: SetPlotOptions;
    isPlotTypeAvailable: (type: PlotType) => boolean;
    sendMessage: (type: string, data: unknown) => void;
}

export const PerReadSection: React.FC<PerReadSectionProps> = ({
    options,
    setOptions,
    isPlotTypeAvailable,
    sendMessage,
}) => {
    return (
        <div>
            {/* View Style: Overlay vs Stacked */}
            <div className="plot-options-section">
                <div className="plot-options-section-header">View Style</div>
                <div
                    className="plot-options-radio-group"
                    style={{ flexDirection: 'row', gap: '16px' }}
                >
                    <label className="plot-options-radio-label">
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
                        />
                        <span>Overlay (alpha-blended)</span>
                    </label>
                    <label className="plot-options-radio-label">
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
                        />
                        <span>Stacked (offset)</span>
                    </label>
                    <label
                        className="plot-options-radio-label"
                        style={{
                            opacity: isPlotTypeAvailable('REFERENCE_OVERLAY') ? 1 : 0.5,
                        }}
                    >
                        <input
                            type="radio"
                            name="perReadViewStyle"
                            checked={options.plotType === 'REFERENCE_OVERLAY'}
                            disabled={!isPlotTypeAvailable('REFERENCE_OVERLAY')}
                            onChange={() =>
                                setOptions((prev) => ({
                                    ...prev,
                                    plotType: 'REFERENCE_OVERLAY',
                                }))
                            }
                        />
                        <span>
                            Reference overlay
                            {!isPlotTypeAvailable('REFERENCE_OVERLAY') ? ' (requires BAM)' : ''}
                        </span>
                    </label>
                </div>
            </div>

            {/* Reference Selection - only for Reference Overlay */}
            {options.plotType === 'REFERENCE_OVERLAY' && (
                <div className="plot-options-section">
                    <div className="plot-options-section-header">Reference</div>
                    <select
                        value={options.aggregateReference}
                        onChange={(e) =>
                            setOptions((prev) => ({
                                ...prev,
                                aggregateReference: e.target.value,
                            }))
                        }
                        disabled={!options.hasBam}
                        className="plot-options-select"
                        style={{
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
                    <div className="plot-options-helper-text">
                        All reads will be filtered to this reference chromosome
                    </div>
                </div>
            )}

            {/* Display Options */}
            <div className="plot-options-section">
                <div className="plot-options-section-header">Display Options</div>

                {/* Reference-anchored mode */}
                <label
                    className="plot-options-checkbox-label"
                    style={{
                        opacity: options.hasBam ? 1 : 0.5,
                        pointerEvents: options.hasBam ? 'auto' : 'none',
                    }}
                >
                    <input
                        type="checkbox"
                        checked={options.showBaseAnnotations}
                        disabled={!options.hasBam}
                        onChange={(e) =>
                            setOptions((prev) => ({
                                ...prev,
                                showBaseAnnotations: e.target.checked,
                            }))
                        }
                    />
                    <span>Reference-anchored</span>
                </label>
                <div
                    className="plot-options-helper-text"
                    style={{ marginLeft: '24px', marginBottom: '8px' }}
                >
                    {!options.hasBam
                        ? 'Requires BAM file with alignment'
                        : 'Plot using genomic coordinates (x-axis = reference position)'}
                </div>

                {/* Show reference track - only when reference-anchored mode enabled and FASTA loaded */}
                {options.showBaseAnnotations && (
                    <label
                        className="plot-options-checkbox-label"
                        style={{
                            marginLeft: '24px',
                            opacity: options.hasFasta ? 1 : 0.5,
                            pointerEvents: options.hasFasta ? 'auto' : 'none',
                        }}
                    >
                        <input
                            type="checkbox"
                            checked={options.hasFasta}
                            disabled={!options.hasFasta}
                            readOnly
                        />
                        <span>Show reference track</span>
                    </label>
                )}
                {options.showBaseAnnotations && (
                    <div
                        className="plot-options-helper-text"
                        style={{ marginLeft: '48px', marginBottom: '8px' }}
                    >
                        {!options.hasFasta
                            ? 'Load FASTA file to show reference sequence'
                            : 'Reference bases with mismatch highlighting'}
                    </div>
                )}
            </div>

            {/* Max Reads per Sample */}
            <div className="plot-options-section">
                <div className="plot-options-slider-label">
                    <span>Max reads per sample:</span>
                    <span>{options.maxReadsMulti}</span>
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
                    className="plot-options-range-slider"
                />
                <div className="plot-options-helper-text">
                    Number of reads to extract from each sample
                </div>
            </div>

            {/* Warning for stacked plots with too many reads */}
            {options.plotType === 'MULTI_READ_STACKED' &&
                options.selectedSamples.length * options.maxReadsMulti > 20 && (
                    <div className="plot-options-warning">
                        ⚠️ Stacked plots work best with ≤20 total reads (currently:{' '}
                        {options.selectedSamples.length * options.maxReadsMulti})
                    </div>
                )}

            {/* Downsample factor - only for Reference Overlay */}
            {options.plotType === 'REFERENCE_OVERLAY' && (
                <div className="plot-options-section">
                    <div className="plot-options-slider-label">
                        <span>Downsample:</span>
                        <span>{options.downsample}x</span>
                    </div>
                    <input
                        type="range"
                        min="1"
                        max="20"
                        step="1"
                        value={options.downsample}
                        onChange={(e) =>
                            setOptions((prev) => ({
                                ...prev,
                                downsample: parseInt(e.target.value),
                            }))
                        }
                        className="plot-options-range-slider"
                    />
                    <div className="plot-options-helper-text">
                        Signal averaging factor (1 = no downsampling)
                    </div>
                </div>
            )}

            {/* Reference Overlay specific options */}
            {options.plotType === 'REFERENCE_OVERLAY' && (
                <div className="plot-options-section">
                    <label className="plot-options-checkbox-label">
                        <input
                            type="checkbox"
                            checked={options.scaleDwellTime}
                            onChange={(e) => {
                                const checked = e.target.checked;
                                setOptions((prev) => ({
                                    ...prev,
                                    scaleDwellTime: checked,
                                }));
                                sendMessage('optionsChanged', {
                                    options: {
                                        ...options,
                                        scaleDwellTime: checked,
                                    },
                                });
                            }}
                        />
                        <span>Scale by dwell time</span>
                    </label>
                    <div
                        className="plot-options-helper-text"
                        style={{ marginLeft: '24px', marginBottom: '8px' }}
                    >
                        Base widths proportional to pore dwell time
                    </div>

                    <label className="plot-options-checkbox-label">
                        <input
                            type="checkbox"
                            checked={options.showBaseColors}
                            onChange={(e) => {
                                const checked = e.target.checked;
                                setOptions((prev) => ({
                                    ...prev,
                                    showBaseColors: checked,
                                }));
                                sendMessage('optionsChanged', {
                                    options: {
                                        ...options,
                                        showBaseColors: checked,
                                    },
                                });
                            }}
                        />
                        <span>Show base colors</span>
                    </label>
                    <div className="plot-options-helper-text" style={{ marginLeft: '24px' }}>
                        Colored background for each base identity
                    </div>
                </div>
            )}
        </div>
    );
};
