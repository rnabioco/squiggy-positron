/**
 * 2-Sample Comparison (signal delta) plot section of the Plot Options panel.
 *
 * Presentational section extracted from PlotOptionsCore. Renders sample
 * selection, reference selection and max-reads for the COMPARE_SIGNAL_DELTA
 * plot type. State lives in the container.
 */

import React from 'react';
import { PlotOptionsState, SetPlotOptions } from './squiggy-plot-options-types';

interface CompareDeltaSectionProps {
    options: PlotOptionsState;
    setOptions: SetPlotOptions;
    handleSampleSelectionChange: (sampleName: string, checked: boolean) => void;
}

export const CompareDeltaSection: React.FC<CompareDeltaSectionProps> = ({
    options,
    setOptions,
    handleSampleSelectionChange,
}) => {
    return (
        <>
            {/* Sample Selection */}
            <div className="plot-options-section">
                <div className="plot-options-section-header">Samples to Compare</div>
                {options.loadedSamples.length === 0 ? (
                    <div className="plot-options-description">
                        Load samples in Sample Manager to enable comparisons
                    </div>
                ) : (
                    <div className="plot-options-sample-list">
                        {options.loadedSamples.map((sample) => (
                            <div key={sample.name} className="plot-options-sample-item">
                                <input
                                    type="checkbox"
                                    id={`sample-${sample.name}`}
                                    checked={options.selectedSamples.includes(sample.name)}
                                    onChange={(e) =>
                                        handleSampleSelectionChange(sample.name, e.target.checked)
                                    }
                                    disabled={
                                        options.plotType === 'COMPARE_SIGNAL_DELTA' &&
                                        options.selectedSamples.length >= 2 &&
                                        !options.selectedSamples.includes(sample.name)
                                    }
                                />
                                <label
                                    htmlFor={`sample-${sample.name}`}
                                    className="plot-options-checkbox-label"
                                >
                                    {sample.name} ({sample.readCount} reads)
                                </label>
                            </div>
                        ))}
                    </div>
                )}
                {options.plotType === 'COMPARE_SIGNAL_DELTA' && (
                    <div className="plot-options-helper-text">
                        Delta plots require exactly 2 samples
                    </div>
                )}
            </div>

            {/* Reference Selection */}
            <div className="plot-options-section">
                <div className="plot-options-section-header">Reference</div>
                <select
                    value={options.comparisonReference}
                    onChange={(e) =>
                        setOptions((prev) => ({
                            ...prev,
                            comparisonReference: e.target.value,
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

            {/* Max Reads */}
            <div className="plot-options-section-large">
                <div className="plot-options-slider-label">
                    <span>Maximum reads per sample:</span>
                    <span>{options.comparisonMaxReads}</span>
                </div>
                <input
                    type="range"
                    min="10"
                    max="500"
                    step="10"
                    value={options.comparisonMaxReads}
                    onChange={(e) =>
                        setOptions((prev) => ({
                            ...prev,
                            comparisonMaxReads: parseInt(e.target.value),
                        }))
                    }
                    className="plot-options-range-slider"
                />
            </div>
        </>
    );
};
