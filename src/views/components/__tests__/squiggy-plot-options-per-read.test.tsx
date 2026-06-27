/**
 * Tests for PerReadSection (per-read plot controls of the Plot Options panel)
 */

import * as React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { PerReadSection } from '../squiggy-plot-options-per-read';
import { PlotOptionsState, PlotType } from '../squiggy-plot-options-types';

function makeOptions(overrides: Partial<PlotOptionsState> = {}): PlotOptionsState {
    return {
        plotType: 'MULTI_READ_OVERLAY',
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
        comparisonReference: '',
        comparisonMetrics: ['signal'],
        comparisonMaxReads: 100,
        ...overrides,
    };
}

describe('PerReadSection', () => {
    const setOptions = jest.fn();
    const sendMessage = jest.fn();
    // Reference overlay available unless a test overrides it.
    const isPlotTypeAvailable = jest.fn((_: PlotType) => true);

    const renderSection = (options: PlotOptionsState) =>
        render(
            <PerReadSection
                options={options}
                setOptions={setOptions}
                isPlotTypeAvailable={isPlotTypeAvailable}
                sendMessage={sendMessage}
            />
        );

    // Run `action` and return the state the section's setOptions updater would
    // produce. The updater is evaluated synchronously inside the event so it
    // reads the event target before React's controlled-input reconciliation
    // resets the DOM value.
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

    beforeEach(() => {
        jest.clearAllMocks();
        isPlotTypeAvailable.mockImplementation(() => true);
    });

    it('checks the radio matching the current plot type', () => {
        renderSection(makeOptions({ plotType: 'MULTI_READ_STACKED' }));

        expect(screen.getByRole('radio', { name: 'Stacked (offset)' })).toBeChecked();
        expect(screen.getByRole('radio', { name: 'Overlay (alpha-blended)' })).not.toBeChecked();
    });

    it('switches plot type when another view-style radio is selected', () => {
        const options = makeOptions({ plotType: 'MULTI_READ_OVERLAY' });
        renderSection(options);

        fireEvent.click(screen.getByRole('radio', { name: 'Stacked (offset)' }));

        const updater = setOptions.mock.calls[0][0];
        expect(updater(options)).toMatchObject({ plotType: 'MULTI_READ_STACKED' });
    });

    it('disables the reference-overlay radio when it is unavailable', () => {
        isPlotTypeAvailable.mockImplementation((t) => t !== 'REFERENCE_OVERLAY');
        renderSection(makeOptions());

        expect(screen.getByRole('radio', { name: /Reference overlay/ })).toBeDisabled();
    });

    it('only shows the reference dropdown for REFERENCE_OVERLAY', () => {
        const { rerender } = renderSection(makeOptions({ plotType: 'MULTI_READ_OVERLAY' }));
        expect(
            screen.queryByText('All reads will be filtered to this reference chromosome')
        ).not.toBeInTheDocument();

        rerender(
            <PerReadSection
                options={makeOptions({ plotType: 'REFERENCE_OVERLAY' })}
                setOptions={setOptions}
                isPlotTypeAvailable={isPlotTypeAvailable}
                sendMessage={sendMessage}
            />
        );
        expect(
            screen.getByText('All reads will be filtered to this reference chromosome')
        ).toBeInTheDocument();
    });

    it('reflects and updates max reads per sample', () => {
        const options = makeOptions({ maxReadsMulti: 42 });
        renderSection(options);

        const slider = screen.getByRole('slider');
        expect(slider).toHaveValue('42');

        const next = nextState(options, () =>
            fireEvent.change(slider, { target: { value: '88' } })
        );
        expect(next).toMatchObject({ maxReadsMulti: 88 });
    });

    it('warns about too many reads for stacked plots', () => {
        renderSection(
            makeOptions({
                plotType: 'MULTI_READ_STACKED',
                selectedSamples: ['a', 'b'],
                maxReadsMulti: 50, // 2 * 50 = 100 > 20
            })
        );

        expect(screen.getByText(/Stacked plots work best with/)).toBeInTheDocument();
    });

    it('does not warn when stacked totals are within the limit', () => {
        renderSection(
            makeOptions({
                plotType: 'MULTI_READ_STACKED',
                selectedSamples: ['a'],
                maxReadsMulti: 10, // 1 * 10 = 10 <= 20
            })
        );

        expect(screen.queryByText(/Stacked plots work best with/)).not.toBeInTheDocument();
    });

    it('updates state and notifies the extension when "Scale by dwell time" changes', () => {
        const options = makeOptions({ plotType: 'REFERENCE_OVERLAY', scaleDwellTime: false });
        renderSection(options);

        fireEvent.click(screen.getByRole('checkbox', { name: 'Scale by dwell time' }));

        const updater = setOptions.mock.calls[0][0];
        expect(updater(options)).toMatchObject({ scaleDwellTime: true });
        expect(sendMessage).toHaveBeenCalledWith(
            'optionsChanged',
            expect.objectContaining({ options: expect.objectContaining({ scaleDwellTime: true }) })
        );
    });
});
