/**
 * Tests for CompareDeltaSection (2-sample comparison controls)
 */

import * as React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { CompareDeltaSection } from '../squiggy-plot-options-compare';
import { PlotOptionsState } from '../squiggy-plot-options-types';
import { SampleItem } from '../../../types/messages';

const sample = (name: string, readCount: number): SampleItem => ({
    name,
    pod5Path: `/data/${name}.pod5`,
    bamPath: `/data/${name}.bam`,
    readCount,
    hasBam: true,
    hasFasta: false,
});

function makeOptions(overrides: Partial<PlotOptionsState> = {}): PlotOptionsState {
    return {
        plotType: 'COMPARE_SIGNAL_DELTA',
        coordinateSpace: 'signal',
        hasPod5: true,
        hasBam: true,
        hasFasta: false,
        hasEvents: false,
        hasMods: false,
        hasPrimers: false,
        normalization: 'ZNORM',
        plotMode: 'SINGLE',
        showDwellTime: true,
        showBaseAnnotations: false,
        scaleDwellTime: false,
        showBaseColors: true,
        downsample: 5,
        showSignalPoints: false,
        clipXAxisToAlignment: true,
        transformCoordinates: true,
        maxReadsMulti: 50,
        aggregateReference: 'chr1',
        aggregateMaxReads: 100,
        aggregateViewStyle: 'overlay',
        showModifications: true,
        showPileup: true,
        showSignal: true,
        showQuality: true,
        showCoverage: false,
        rnaMode: false,
        trimPrimers: true,
        primer5p: 'AAAA',
        adapter3p: 'TTTT',
        showAdvanced: false,
        availableReferences: ['chr1', 'chr2'],
        loadedSamples: [],
        selectedSamples: [],
        comparisonReference: 'chr1',
        comparisonMetrics: ['signal'],
        comparisonMaxReads: 100,
        ...overrides,
    };
}

describe('CompareDeltaSection', () => {
    const setOptions = jest.fn();
    const handleSampleSelectionChange = jest.fn();

    const renderSection = (options: PlotOptionsState) =>
        render(
            <CompareDeltaSection
                options={options}
                setOptions={setOptions}
                handleSampleSelectionChange={handleSampleSelectionChange}
            />
        );

    // Run `action` and return the state the section's setOptions updater would
    // produce, evaluated synchronously inside the event (before React resets
    // the controlled input's DOM value).
    const nextState = (options: PlotOptionsState, action: () => void): PlotOptionsState => {
        let result: PlotOptionsState = options;
        setOptions.mockImplementationOnce(
            (u: PlotOptionsState | ((p: PlotOptionsState) => PlotOptionsState)) => {
                result = typeof u === 'function' ? u(options) : u;
            }
        );
        action();
        return result;
    };

    beforeEach(() => jest.clearAllMocks());

    it('prompts to load samples when none are loaded', () => {
        renderSection(makeOptions({ loadedSamples: [] }));

        expect(
            screen.getByText('Load samples in Sample Manager to enable comparisons')
        ).toBeInTheDocument();
    });

    it('renders a checkbox per loaded sample with its read count', () => {
        renderSection(makeOptions({ loadedSamples: [sample('ctrl', 100), sample('treat', 250)] }));

        expect(screen.getByRole('checkbox', { name: 'ctrl (100 reads)' })).toBeInTheDocument();
        expect(screen.getByRole('checkbox', { name: 'treat (250 reads)' })).toBeInTheDocument();
    });

    it('reflects which samples are selected', () => {
        renderSection(
            makeOptions({
                loadedSamples: [sample('ctrl', 100), sample('treat', 250)],
                selectedSamples: ['ctrl'],
            })
        );

        expect(screen.getByRole('checkbox', { name: 'ctrl (100 reads)' })).toBeChecked();
        expect(screen.getByRole('checkbox', { name: 'treat (250 reads)' })).not.toBeChecked();
    });

    it('calls handleSampleSelectionChange when a sample is toggled', () => {
        renderSection(makeOptions({ loadedSamples: [sample('ctrl', 100)], selectedSamples: [] }));

        fireEvent.click(screen.getByRole('checkbox', { name: 'ctrl (100 reads)' }));
        expect(handleSampleSelectionChange).toHaveBeenCalledWith('ctrl', true);
    });

    it('disables unselected samples once two are selected', () => {
        renderSection(
            makeOptions({
                loadedSamples: [sample('a', 1), sample('b', 2), sample('c', 3)],
                selectedSamples: ['a', 'b'],
            })
        );

        // Already-selected stay enabled; the third is disabled.
        expect(screen.getByRole('checkbox', { name: 'a (1 reads)' })).toBeEnabled();
        expect(screen.getByRole('checkbox', { name: 'c (3 reads)' })).toBeDisabled();
    });

    it('shows the reference dropdown and the 2-sample hint', () => {
        renderSection(makeOptions({ loadedSamples: [sample('a', 1)] }));

        const combo = screen.getByRole('combobox');
        expect(combo).toHaveValue('chr1');
        expect(screen.getByText('Delta plots require exactly 2 samples')).toBeInTheDocument();
    });

    it('reflects and updates the maximum-reads slider', () => {
        const options = makeOptions({ comparisonMaxReads: 200 });
        renderSection(options);

        const slider = screen.getByRole('slider');
        expect(slider).toHaveValue('200');

        const next = nextState(options, () =>
            fireEvent.change(slider, { target: { value: '350' } })
        );
        expect(next).toMatchObject({ comparisonMaxReads: 350 });
    });

    it('updates the comparison reference selection', () => {
        const options = makeOptions({ loadedSamples: [sample('a', 1)] });
        renderSection(options);

        const next = nextState(options, () =>
            fireEvent.change(screen.getByRole('combobox'), { target: { value: 'chr2' } })
        );
        expect(next).toMatchObject({ comparisonReference: 'chr2' });
    });
});
