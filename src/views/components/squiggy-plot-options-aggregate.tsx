/**
 * Aggregate (Composite Read) plot section of the Plot Options panel.
 *
 * Presentational section extracted from PlotOptionsCore. Renders reference
 * selection, view style, panel visibility toggles and x-axis options for the
 * AGGREGATE plot type. State lives in the container.
 */

import React from 'react';
import { PlotOptionsState, SetPlotOptions } from './squiggy-plot-options-types';

interface AggregateSectionProps {
    options: PlotOptionsState;
    setOptions: SetPlotOptions;
}

export const AggregateSection: React.FC<AggregateSectionProps> = ({ options, setOptions }) => {
    return (
        <>
            {/* Reference Selection */}
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
            </div>

            {/* View Style (for multi-sample) - only show if 2+ samples selected in Samples panel */}
            {options.selectedSamples.length > 1 && (
                <div className="plot-options-section-large">
                    <div className="plot-options-section-header">View Style</div>
                    <select
                        value={options.aggregateViewStyle}
                        onChange={(e) =>
                            setOptions((prev) => ({
                                ...prev,
                                aggregateViewStyle: e.target.value as 'overlay' | 'multi-track',
                            }))
                        }
                        className="plot-options-select"
                    >
                        <option value="overlay">Overlay (Mean Signals)</option>
                        <option value="multi-track">Multi-Track (Detailed)</option>
                    </select>
                    <div className="plot-options-helper-text">
                        {options.aggregateViewStyle === 'overlay'
                            ? 'Overlays mean signals from all samples on one plot'
                            : 'Shows detailed 5-track view for each sample'}
                    </div>
                </div>
            )}

            {/* Max Reads */}
            <div className="plot-options-section-large">
                <div className="plot-options-slider-label">
                    <span>Maximum reads:</span>
                    <span>{options.aggregateMaxReads}</span>
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
                    className="plot-options-range-slider"
                />
            </div>

            {/* Panel Visibility */}
            <div className="plot-options-section">
                <div className="plot-options-section-header">Visible Panels</div>

                {/* Modifications Panel */}
                <div className="plot-options-checkbox-row">
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
                    />
                    <label htmlFor="showModifications" className="plot-options-checkbox-label">
                        Base modifications
                    </label>
                </div>
                <div
                    className="plot-options-helper-text"
                    style={{
                        marginLeft: '22px',
                        marginBottom: '12px',
                        marginTop: '-4px',
                    }}
                >
                    Adjust filters in Modifications Explorer panel
                </div>

                {/* Pileup Panel */}
                <div className="plot-options-checkbox-row">
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
                    />
                    <label htmlFor="showPileup" className="plot-options-checkbox-label">
                        Base pileup
                    </label>
                </div>

                {/* RNA Mode - show U instead of T in pileup */}
                {options.showPileup && (
                    <div className="plot-options-checkbox-row" style={{ marginLeft: '22px' }}>
                        <input
                            type="checkbox"
                            id="rnaMode"
                            checked={options.rnaMode}
                            onChange={(e) =>
                                setOptions((prev) => ({
                                    ...prev,
                                    rnaMode: e.target.checked,
                                }))
                            }
                            disabled={!options.hasBam}
                        />
                        <label htmlFor="rnaMode" className="plot-options-checkbox-label">
                            Show as RNA (U instead of T)
                        </label>
                    </div>
                )}

                {/* Dwell Time Panel */}
                <div className="plot-options-checkbox-row">
                    <input
                        type="checkbox"
                        id="showDwellTimeAggregate"
                        checked={options.showDwellTime && options.hasEvents}
                        onChange={(e) =>
                            setOptions((prev) => ({
                                ...prev,
                                showDwellTime: e.target.checked,
                            }))
                        }
                        disabled={!options.hasBam || !options.hasEvents}
                    />
                    <label
                        htmlFor="showDwellTimeAggregate"
                        className="plot-options-checkbox-label"
                        style={{
                            opacity: options.hasEvents ? 1 : 0.5,
                        }}
                    >
                        Dwell time
                        {!options.hasEvents && options.hasBam && (
                            <span className="plot-options-tag-disabled">no mv tag</span>
                        )}
                    </label>
                </div>

                {/* Signal Panel */}
                <div className="plot-options-checkbox-row">
                    <input
                        type="checkbox"
                        id="showSignalAggregate"
                        checked={options.showSignal && options.hasEvents}
                        onChange={(e) =>
                            setOptions((prev) => ({
                                ...prev,
                                showSignal: e.target.checked,
                            }))
                        }
                        disabled={!options.hasBam || !options.hasEvents}
                    />
                    <label
                        htmlFor="showSignalAggregate"
                        className="plot-options-checkbox-label"
                        style={{
                            opacity: options.hasEvents ? 1 : 0.5,
                        }}
                    >
                        Signal
                        {!options.hasEvents && options.hasBam && (
                            <span className="plot-options-tag-disabled">no mv tag</span>
                        )}
                    </label>
                </div>

                {/* Quality Panel */}
                <div className="plot-options-checkbox-row">
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
                    />
                    <label htmlFor="showQualityAggregate" className="plot-options-checkbox-label">
                        Quality scores
                    </label>
                </div>

                {/* Coverage Panel */}
                <div className="plot-options-checkbox-row">
                    <input
                        type="checkbox"
                        id="showCoverageAggregate"
                        checked={options.showCoverage}
                        onChange={(e) =>
                            setOptions((prev) => ({
                                ...prev,
                                showCoverage: e.target.checked,
                            }))
                        }
                        disabled={!options.hasBam}
                    />
                    <label htmlFor="showCoverageAggregate" className="plot-options-checkbox-label">
                        Coverage depth
                    </label>
                </div>
            </div>

            {/* X-Axis Options */}
            <div
                className="plot-options-section-large"
                style={{
                    opacity: options.hasBam ? 1 : 0.5,
                    pointerEvents: options.hasBam ? 'auto' : 'none',
                }}
            >
                <div className="plot-options-section-header">X-Axis Display</div>

                {/* Clip to Consensus */}
                <div className="plot-options-checkbox-row">
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
                    />
                    <label
                        htmlFor="clipXAxisToAlignmentAggregate"
                        className="plot-options-checkbox-label"
                    >
                        Clip x-axis to consensus region
                    </label>
                </div>
                <div
                    className="plot-options-description"
                    style={{
                        marginTop: '-6px',
                        marginBottom: '10px',
                    }}
                >
                    Focus on high-coverage region (uncheck to show full reference range)
                </div>

                {/* Show Primers/Adapters - only visible when PT tag detected */}
                {options.hasPrimers && (
                    <>
                        <div className="plot-options-checkbox-row">
                            <input
                                type="checkbox"
                                id="showPrimersAggregate"
                                checked={!options.trimPrimers}
                                onChange={(e) =>
                                    setOptions((prev) => ({
                                        ...prev,
                                        trimPrimers: !e.target.checked,
                                    }))
                                }
                            />
                            <label
                                htmlFor="showPrimersAggregate"
                                className="plot-options-checkbox-label"
                            >
                                Show primers/adapters
                            </label>
                        </div>

                        {/* Advanced: Primer Sequences */}
                        <div
                            className="plot-options-advanced-toggle"
                            onClick={() =>
                                setOptions((prev) => ({
                                    ...prev,
                                    showAdvanced: !prev.showAdvanced,
                                }))
                            }
                        >
                            <span className="plot-options-advanced-arrow">
                                {options.showAdvanced ? '▾' : '▸'}
                            </span>
                            Primer sequences
                        </div>
                        {options.showAdvanced && (
                            <div className="plot-options-advanced-content">
                                <div className="plot-options-input-group">
                                    <label htmlFor="primer5p" className="plot-options-input-label">
                                        5′ primer
                                    </label>
                                    <input
                                        type="text"
                                        id="primer5p"
                                        className="plot-options-text-input"
                                        value={options.primer5p}
                                        onChange={(e) =>
                                            setOptions((prev) => ({
                                                ...prev,
                                                primer5p: e.target.value,
                                            }))
                                        }
                                        spellCheck={false}
                                    />
                                </div>
                                <div className="plot-options-input-group">
                                    <label htmlFor="adapter3p" className="plot-options-input-label">
                                        3′ adapter
                                    </label>
                                    <input
                                        type="text"
                                        id="adapter3p"
                                        className="plot-options-text-input"
                                        value={options.adapter3p}
                                        onChange={(e) =>
                                            setOptions((prev) => ({
                                                ...prev,
                                                adapter3p: e.target.value,
                                            }))
                                        }
                                        spellCheck={false}
                                    />
                                </div>
                            </div>
                        )}
                    </>
                )}

                {/* Transform Coordinates */}
                <div className="plot-options-checkbox-row">
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
                    />
                    <label
                        htmlFor="transformCoordinatesAggregate"
                        className="plot-options-checkbox-label"
                    >
                        Transform to relative coordinates
                    </label>
                </div>
                <div
                    className="plot-options-description"
                    style={{
                        marginTop: '-6px',
                        marginBottom: '10px',
                    }}
                >
                    Anchor position 1 to first reference base (uncheck to use genomic coordinates)
                </div>
            </div>
        </>
    );
};
