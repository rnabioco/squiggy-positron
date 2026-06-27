/**
 * Tests for SampleRow
 *
 * Covers the collapsed header (selection, color, editable name, load/badge,
 * delete) and the expanded details (POD5/BAM/FASTA, references, unload).
 */

import * as React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';

// SampleRow imports the shared vscode API singleton, which calls
// acquireVsCodeApi() at module load — not available under jsdom.
jest.mock('../vscode-api', () => ({
    vscode: { postMessage: jest.fn() },
}));

import { SampleRow } from '../squiggy-sample-row';
import { vscode } from '../vscode-api';
import { SampleItem } from '../../../types/messages';

const mockVscode = vscode as unknown as { postMessage: jest.Mock };

describe('SampleRow', () => {
    const handlers = {
        onToggleExpanded: jest.fn(),
        onToggleSelection: jest.fn(),
        onColorChange: jest.fn(),
        onStartEdit: jest.fn(),
        onSaveEdit: jest.fn(),
        onCancelEdit: jest.fn(),
        onEditInputChange: jest.fn(),
        onChangeBam: jest.fn(),
        onChangeFasta: jest.fn(),
        onAddFasta: jest.fn(),
        onUnload: jest.fn(),
    };

    const baseSample: SampleItem = {
        name: 'sample_a',
        pod5Path: '/data/sample_a.pod5',
        bamPath: '/data/sample_a.bam',
        readCount: 1234,
        hasBam: true,
        hasFasta: false,
    };

    const baseProps = {
        sample: baseSample,
        color: '#aabbcc',
        isExpanded: false,
        isSelected: true,
        sessionFastaPath: null,
        isEditing: false,
        editInputValue: '',
        nameEditError: null,
        editInputRef: React.createRef<HTMLInputElement>(),
        ...handlers,
    };

    beforeEach(() => {
        jest.clearAllMocks();
    });

    describe('Collapsed header', () => {
        it('renders the sample name and read-count badge', () => {
            render(<SampleRow {...baseProps} />);

            expect(screen.getByText('sample_a')).toBeInTheDocument();
            expect(screen.getByText(/1,234 reads/)).toBeInTheDocument();
        });

        it('reflects the color prop in the color picker', () => {
            render(<SampleRow {...baseProps} />);

            const colorInput = screen.getByTitle('Sample color for plots');
            expect(colorInput).toHaveValue('#aabbcc');
        });

        it('reflects selection state in the checkbox', () => {
            const { rerender } = render(<SampleRow {...baseProps} isSelected={true} />);
            expect(screen.getByRole('checkbox')).toBeChecked();

            rerender(<SampleRow {...baseProps} isSelected={false} />);
            expect(screen.getByRole('checkbox')).not.toBeChecked();
        });

        it('toggles expansion when the header is clicked', () => {
            const { container } = render(<SampleRow {...baseProps} />);

            // The header row is the first child of the row container
            fireEvent.click(container.querySelector('.sample-row')!.firstElementChild!);

            expect(handlers.onToggleExpanded).toHaveBeenCalledWith('sample_a');
        });

        it('toggles selection when the checkbox is clicked, without expanding', () => {
            render(<SampleRow {...baseProps} />);

            fireEvent.click(screen.getByRole('checkbox'));

            expect(handlers.onToggleSelection).toHaveBeenCalledWith('sample_a');
            expect(handlers.onToggleExpanded).not.toHaveBeenCalled();
        });

        it('calls onColorChange when the color picker changes', () => {
            render(<SampleRow {...baseProps} />);

            fireEvent.change(screen.getByTitle('Sample color for plots'), {
                target: { value: '#112233' },
            });

            expect(handlers.onColorChange).toHaveBeenCalledWith('sample_a', '#112233');
        });

        it('starts editing on double-click of the name', () => {
            render(<SampleRow {...baseProps} />);

            fireEvent.doubleClick(screen.getByText('sample_a'));

            expect(handlers.onStartEdit).toHaveBeenCalledWith('sample_a');
        });

        it('removes the sample via the trash button', () => {
            render(<SampleRow {...baseProps} />);

            fireEvent.click(screen.getByTitle('Remove this sample'));

            expect(handlers.onUnload).toHaveBeenCalledWith('sample_a');
        });
    });

    describe('Editing mode', () => {
        const editingProps = {
            ...baseProps,
            isEditing: true,
            editInputValue: 'new_name',
        };

        it('shows the rename input with the current value', () => {
            render(<SampleRow {...editingProps} />);

            expect(screen.getByDisplayValue('new_name')).toBeInTheDocument();
        });

        it('calls onEditInputChange while typing', () => {
            render(<SampleRow {...editingProps} />);

            fireEvent.change(screen.getByDisplayValue('new_name'), {
                target: { value: 'new_name2' },
            });

            expect(handlers.onEditInputChange).toHaveBeenCalledWith('new_name2');
        });

        it('saves on the check button and on Enter', () => {
            render(<SampleRow {...editingProps} />);

            fireEvent.click(screen.getByText('✓'));
            expect(handlers.onSaveEdit).toHaveBeenCalledWith('sample_a', 'new_name');

            fireEvent.keyDown(screen.getByDisplayValue('new_name'), { key: 'Enter' });
            expect(handlers.onSaveEdit).toHaveBeenCalledTimes(2);
        });

        it('cancels on the cross button and on Escape', () => {
            render(<SampleRow {...editingProps} />);

            fireEvent.click(screen.getByText('✕'));
            expect(handlers.onCancelEdit).toHaveBeenCalledTimes(1);

            fireEvent.keyDown(screen.getByDisplayValue('new_name'), { key: 'Escape' });
            expect(handlers.onCancelEdit).toHaveBeenCalledTimes(2);
        });

        it('renders an inline validation error when provided', () => {
            render(
                <SampleRow
                    {...editingProps}
                    nameEditError="A sample with this name already exists"
                />
            );

            expect(screen.getByText('A sample with this name already exists')).toBeInTheDocument();
        });
    });

    describe('Deferred and loading states', () => {
        it('shows a Load button (and no checkbox/color) for a deferred sample', () => {
            const sample = { ...baseSample, isDeferred: true };
            render(<SampleRow {...baseProps} sample={sample} />);

            expect(screen.getByText('Load')).toBeInTheDocument();
            expect(screen.queryByRole('checkbox')).not.toBeInTheDocument();
            expect(screen.queryByTitle('Sample color for plots')).not.toBeInTheDocument();
        });

        it('posts loadDeferredSample when the Load button is clicked', () => {
            const sample = { ...baseSample, isDeferred: true };
            render(<SampleRow {...baseProps} sample={sample} />);

            fireEvent.click(screen.getByText('Load'));

            expect(mockVscode.postMessage).toHaveBeenCalledWith({
                type: 'loadDeferredSample',
                sampleName: 'sample_a',
            });
        });

        it('shows the loading message while loading', () => {
            const sample = { ...baseSample, isLoading: true, loadingMessage: 'Indexing…' };
            render(<SampleRow {...baseProps} sample={sample} />);

            expect(screen.getByText('Indexing…')).toBeInTheDocument();
        });
    });

    describe('Expanded details', () => {
        const expandedProps = { ...baseProps, isExpanded: true };

        it('shows POD5 and BAM filenames', () => {
            render(<SampleRow {...expandedProps} />);

            expect(screen.getByText('sample_a.pod5')).toBeInTheDocument();
            expect(screen.getByText('sample_a.bam')).toBeInTheDocument();
        });

        it('offers + Add BAM when no BAM is set, wired to onChangeBam', () => {
            const sample = { ...baseSample, bamPath: undefined, hasBam: false };
            render(<SampleRow {...expandedProps} sample={sample} />);

            const addBam = screen.getByText('+ Add BAM');
            fireEvent.click(addBam);
            expect(handlers.onChangeBam).toHaveBeenCalledWith('sample_a');
        });

        it('falls back to the session FASTA labelled as (session)', () => {
            render(
                <SampleRow
                    {...expandedProps}
                    sessionFastaPath="/data/ref.fa"
                    sample={{ ...baseSample, fastaPath: undefined }}
                />
            );

            expect(screen.getByText('ref.fa')).toBeInTheDocument();
            // "(session)" appears both in the header label and beside the filename
            expect(screen.getAllByText('(session)').length).toBeGreaterThan(0);
        });

        it('offers + Add FASTA when none is set, wired to onAddFasta', () => {
            render(
                <SampleRow
                    {...expandedProps}
                    sessionFastaPath={null}
                    sample={{ ...baseSample, fastaPath: undefined }}
                />
            );

            fireEvent.click(screen.getByText('+ Add FASTA'));
            expect(handlers.onAddFasta).toHaveBeenCalledWith('sample_a');
        });

        it('lists references from the BAM alignment', () => {
            const sample = {
                ...baseSample,
                references: [
                    { name: 'chr1', readCount: 500 },
                    { name: 'chr2', readCount: 100 },
                ],
            };
            render(<SampleRow {...expandedProps} sample={sample} />);

            expect(screen.getByText('References (2)')).toBeInTheDocument();
            expect(screen.getByText('chr1')).toBeInTheDocument();
            expect(screen.getByText('chr2')).toBeInTheDocument();
        });

        it('unloads from the expanded Unload button', () => {
            render(<SampleRow {...expandedProps} />);

            fireEvent.click(screen.getByText('Unload Sample'));
            expect(handlers.onUnload).toHaveBeenCalledWith('sample_a');
        });

        it('does not render details when collapsed', () => {
            render(<SampleRow {...baseProps} isExpanded={false} />);

            expect(screen.queryByText('POD5')).not.toBeInTheDocument();
            expect(screen.queryByText('Unload Sample')).not.toBeInTheDocument();
        });
    });
});
