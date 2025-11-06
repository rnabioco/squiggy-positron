/**
 * Tests for ReferenceGroupComponent
 *
 * Tests the reference group header rendering.
 * Target: >90% coverage of squiggy-reference-group.tsx
 */

import * as React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { ReferenceGroupComponent } from '../squiggy-reference-group';
import { ReferenceGroupItem } from '../../../types/squiggy-reads-types';

describe('ReferenceGroupComponent', () => {
    const mockOnToggle = jest.fn();
    const mockOnPlotAggregate = jest.fn();

    const baseProps = {
        item: {
            type: 'reference' as const,
            referenceName: 'chr1',
            readCount: 42,
            isExpanded: false,
            indentLevel: 0 as 0,
        } as ReferenceGroupItem,
        isEvenRow: true,
        nameColumnWidth: 300,
        detailsColumnWidth: 200,
        onToggle: mockOnToggle,
        onPlotAggregate: mockOnPlotAggregate,
    };

    beforeEach(() => {
        jest.clearAllMocks();
    });

    describe('Rendering', () => {
        it('should render reference name', () => {
            render(<ReferenceGroupComponent {...baseProps} />);

            expect(screen.getByText('chr1')).toBeInTheDocument();
        });

        it('should render read count', () => {
            render(<ReferenceGroupComponent {...baseProps} />);

            expect(screen.getByText('42')).toBeInTheDocument();
        });

        it('should render with even-row class for even rows', () => {
            const { container } = render(
                <ReferenceGroupComponent {...baseProps} isEvenRow={true} />
            );

            const referenceGroup = container.querySelector('.reference-group');
            expect(referenceGroup).toHaveClass('even-row');
        });

        it('should render with odd-row class for odd rows', () => {
            const { container } = render(
                <ReferenceGroupComponent {...baseProps} isEvenRow={false} />
            );

            const referenceGroup = container.querySelector('.reference-group');
            expect(referenceGroup).toHaveClass('odd-row');
        });

        it('should render chevron with collapsed state', () => {
            const { container } = render(<ReferenceGroupComponent {...baseProps} />);

            const chevron = container.querySelector('.reference-group-chevron');
            expect(chevron).not.toHaveClass('expanded');
        });

        it('should render chevron with expanded state', () => {
            const item: ReferenceGroupItem = {
                ...baseProps.item,
                isExpanded: true,
            };

            const { container } = render(<ReferenceGroupComponent {...baseProps} item={item} />);

            const chevron = container.querySelector('.reference-group-chevron');
            expect(chevron).toHaveClass('expanded');
        });
    });

    describe('Column Widths', () => {
        it('should apply name column width', () => {
            const { container } = render(
                <ReferenceGroupComponent {...baseProps} nameColumnWidth={400} />
            );

            const nameColumn = container.querySelector('.reference-group-name');
            expect(nameColumn).toHaveStyle({ width: '400px' });
        });

        it('should apply details column width', () => {
            const { container } = render(
                <ReferenceGroupComponent {...baseProps} detailsColumnWidth={250} />
            );

            const detailsColumn = container.querySelector('.reference-group-details');
            expect(detailsColumn).toHaveStyle({ width: '250px' });
        });
    });

    describe('Interactions', () => {
        it('should call onToggle when group is clicked', () => {
            const { container } = render(<ReferenceGroupComponent {...baseProps} />);

            const referenceGroup = container.querySelector('.reference-group');
            if (referenceGroup) {
                fireEvent.click(referenceGroup);
            }

            expect(mockOnToggle).toHaveBeenCalledWith('chr1');
        });

        it('should call onPlotAggregate when aggregate button clicked', () => {
            render(<ReferenceGroupComponent {...baseProps} />);

            const aggregateButton = screen.getByText('Aggregate');
            fireEvent.click(aggregateButton);

            expect(mockOnPlotAggregate).toHaveBeenCalledWith('chr1');
        });

        it('should stop propagation when aggregate button clicked', () => {
            render(<ReferenceGroupComponent {...baseProps} />);

            const aggregateButton = screen.getByText('Aggregate');
            fireEvent.click(aggregateButton);

            // onPlotAggregate should be called, but onToggle should NOT be called
            expect(mockOnPlotAggregate).toHaveBeenCalledWith('chr1');
            expect(mockOnToggle).not.toHaveBeenCalled();
        });

        it('should not call onPlotAggregate when undefined', () => {
            render(<ReferenceGroupComponent {...baseProps} onPlotAggregate={undefined} />);

            const aggregateButton = screen.getByText('Aggregate');
            fireEvent.click(aggregateButton);

            // Should not throw error
            expect(mockOnPlotAggregate).not.toHaveBeenCalled();
        });
    });

    describe('Accessibility', () => {
        it('should have title attribute with reference name and count', () => {
            const { container } = render(<ReferenceGroupComponent {...baseProps} />);

            const referenceGroup = container.querySelector('.reference-group');
            expect(referenceGroup).toHaveAttribute('title', 'chr1 (42 reads)');
        });

        it('should have descriptive button title', () => {
            render(<ReferenceGroupComponent {...baseProps} />);

            const aggregateButton = screen.getByTitle('Plot aggregate for this reference');
            expect(aggregateButton).toBeInTheDocument();
        });
    });
});
