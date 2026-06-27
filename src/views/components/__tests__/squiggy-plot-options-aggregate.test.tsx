/**
 * Tests for AggregateSection (aggregate/composite plot controls)
 */

import * as React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { AggregateSection } from '../squiggy-plot-options-aggregate';
import { PlotOptionsState } from '../squiggy-plot-options-types';

function makeOptions(overrides: Partial<PlotOptionsState> = {}): PlotOptionsState {
    return {
        plotType: 'AGGREGATE',
        coordinateSpace: 'signal',
        hasPod5: true,
        hasBam: true,
        hasFasta: false,
        hasEvents: true,
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
        comparisonReference: '',
        comparisonMetrics: ['signal'],
        comparisonMaxReads: 100,
        ...overrides,
    };
}

describe('AggregateSection', () => {
    const setOptions = jest.fn();

    const renderSection = (options: PlotOptionsState) =>
        render(<AggregateSection options={options} setOptions={setOptions} />);

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

    it('renders the reference dropdown with the available references', () => {
        renderSection(makeOptions());

        const select = screen.getByRole('combobox');
        expect(select).toHaveValue('chr1');
        expect(screen.getByRole('option', { name: 'chr2' })).toBeInTheDocument();
    });

    it('only shows the view-style selector when more than one sample is selected', () => {
        const { rerender } = renderSection(makeOptions({ selectedSamples: ['a'] }));
        // Single sample → only the reference combobox
        expect(screen.getAllByRole('combobox')).toHaveLength(1);

        rerender(
            <AggregateSection
                options={makeOptions({ selectedSamples: ['a', 'b'] })}
                setOptions={setOptions}
            />
        );
        expect(screen.getAllByRole('combobox')).toHaveLength(2);
        expect(screen.getByRole('option', { name: 'Multi-Track (Detailed)' })).toBeInTheDocument();
    });

    it('reflects and updates the maximum reads slider', () => {
        const options = makeOptions({ aggregateMaxReads: 120 });
        renderSection(options);

        const slider = screen.getByRole('slider');
        expect(slider).toHaveValue('120');

        const next = nextState(options, () =>
            fireEvent.change(slider, { target: { value: '300' } })
        );
        expect(next).toMatchObject({ aggregateMaxReads: 300 });
    });

    it('toggles a panel-visibility checkbox', () => {
        const options = makeOptions({ showQuality: true });
        renderSection(options);

        const next = nextState(options, () =>
            fireEvent.click(screen.getByRole('checkbox', { name: 'Quality scores' }))
        );
        expect(next).toMatchObject({ showQuality: false });
    });

    it('shows the RNA-mode toggle only when pileup is enabled', () => {
        const { rerender } = renderSection(makeOptions({ showPileup: false }));
        expect(
            screen.queryByRole('checkbox', { name: 'Show as RNA (U instead of T)' })
        ).not.toBeInTheDocument();

        rerender(
            <AggregateSection options={makeOptions({ showPileup: true })} setOptions={setOptions} />
        );
        expect(
            screen.getByRole('checkbox', { name: 'Show as RNA (U instead of T)' })
        ).toBeInTheDocument();
    });

    it('disables dwell-time / signal toggles when the BAM has no mv tag', () => {
        renderSection(makeOptions({ hasEvents: false }));

        expect(screen.getByRole('checkbox', { name: /Dwell time/ })).toBeDisabled();
        expect(screen.getByRole('checkbox', { name: /Signal/ })).toBeDisabled();
        // "no mv tag" badges are shown for both
        expect(screen.getAllByText('no mv tag')).toHaveLength(2);
    });

    it('disables BAM-dependent controls when no BAM is loaded', () => {
        renderSection(makeOptions({ hasBam: false }));

        expect(screen.getByRole('checkbox', { name: 'Base modifications' })).toBeDisabled();
        expect(screen.getByRole('checkbox', { name: 'Base pileup' })).toBeDisabled();
    });

    it('shows the primer controls only when a PT tag is present', () => {
        const { rerender } = renderSection(makeOptions({ hasPrimers: false }));
        expect(
            screen.queryByRole('checkbox', { name: 'Show primers/adapters' })
        ).not.toBeInTheDocument();

        rerender(
            <AggregateSection options={makeOptions({ hasPrimers: true })} setOptions={setOptions} />
        );
        const showPrimers = screen.getByRole('checkbox', { name: 'Show primers/adapters' });
        // checkbox reflects !trimPrimers (trimPrimers=true -> unchecked)
        expect(showPrimers).not.toBeChecked();

        const opts = makeOptions({ hasPrimers: true });
        const next = nextState(opts, () => fireEvent.click(showPrimers));
        expect(next).toMatchObject({ trimPrimers: false });
    });
});
