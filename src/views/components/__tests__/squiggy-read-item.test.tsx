/**
 * Tests for ReadItemComponent
 *
 * Tests the individual read row rendering.
 * Target: >70% coverage of squiggy-read-item.tsx
 */

import * as React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { ReadItemComponent } from '../squiggy-read-item';
import { ReadItem } from '../../../types/squiggy-reads-types';

describe('ReadItemComponent', () => {
    const mockOnPlotRead = jest.fn();
    const mockOnClick = jest.fn();

    const baseProps = {
        item: {
            type: 'read' as const,
            readId: 'read_001',
            indentLevel: 0 as 0 | 1,
        } as ReadItem,
        isSelected: false,
        isFocused: false,
        isEvenRow: true,
        nameColumnWidth: 300,
        detailsColumnWidth: 200,
        onPlotRead: mockOnPlotRead,
        onClick: mockOnClick,
    };

    beforeEach(() => {
        jest.clearAllMocks();
    });

    describe('Rendering', () => {
        it('should render read ID', () => {
            render(<ReadItemComponent {...baseProps} />);

            expect(screen.getByText('read_001')).toBeInTheDocument();
        });

        it('should render with selected class when selected', () => {
            const { container } = render(<ReadItemComponent {...baseProps} isSelected={true} />);

            const readItem = container.querySelector('.read-item');
            expect(readItem).toHaveClass('selected');
        });

        it('should render with focused class when focused', () => {
            const { container } = render(<ReadItemComponent {...baseProps} isFocused={true} />);

            const readItem = container.querySelector('.read-item');
            expect(readItem).toHaveClass('focused');
        });

        it('should render with even-row class for even rows', () => {
            const { container } = render(<ReadItemComponent {...baseProps} isEvenRow={true} />);

            const readItem = container.querySelector('.read-item');
            expect(readItem).toHaveClass('even-row');
        });

        it('should render with odd-row class for odd rows', () => {
            const { container } = render(<ReadItemComponent {...baseProps} isEvenRow={false} />);

            const readItem = container.querySelector('.read-item');
            expect(readItem).toHaveClass('odd-row');
        });

        it('should render genomic position when provided', () => {
            const item: ReadItem = {
                ...baseProps.item,
                genomicPosition: 'chr1:12345-12678',
            };

            render(<ReadItemComponent {...baseProps} item={item} />);

            expect(screen.getByText('chr1:12345-12678')).toBeInTheDocument();
        });

        it('should render reference name when no genomic position', () => {
            const item: ReadItem = {
                ...baseProps.item,
                referenceName: 'chr1',
            };

            render(<ReadItemComponent {...baseProps} item={item} />);

            expect(screen.getByText('chr1')).toBeInTheDocument();
        });

        it('should render quality score when provided', () => {
            const item: ReadItem = {
                ...baseProps.item,
                quality: 15,
            };

            render(<ReadItemComponent {...baseProps} item={item} />);

            expect(screen.getByText('Q15')).toBeInTheDocument();
        });

        it('should not render reference when genomic position is present', () => {
            const item: ReadItem = {
                ...baseProps.item,
                referenceName: 'chr1',
                genomicPosition: 'chr1:12345-12678',
            };

            render(<ReadItemComponent {...baseProps} item={item} />);

            expect(screen.getByText('chr1:12345-12678')).toBeInTheDocument();
            // Reference name should not be rendered separately when position exists
            expect(screen.queryByText('chr1')).not.toBeInTheDocument();
        });
    });

    describe('Column Widths', () => {
        it('should apply name column width', () => {
            const { container } = render(
                <ReadItemComponent {...baseProps} nameColumnWidth={400} />
            );

            const nameColumn = container.querySelector('.read-item-name');
            expect(nameColumn).toHaveStyle({ width: '400px' });
        });

        it('should apply details column width', () => {
            const { container } = render(
                <ReadItemComponent {...baseProps} detailsColumnWidth={250} />
            );

            const detailsColumn = container.querySelector('.read-item-details');
            expect(detailsColumn).toHaveStyle({ width: '250px' });
        });
    });

    describe('Indentation', () => {
        it('should apply no indentation for indent level 0', () => {
            const item: ReadItem = {
                ...baseProps.item,
                indentLevel: 0,
            };

            const { container } = render(<ReadItemComponent {...baseProps} item={item} />);

            const nameColumn = container.querySelector('.read-item-name');
            // Base padding is 8px
            expect(nameColumn).toHaveStyle({ paddingLeft: '8px' });
        });

        it('should apply indentation for indent level 1', () => {
            const item: ReadItem = {
                ...baseProps.item,
                indentLevel: 1,
            };

            const { container } = render(<ReadItemComponent {...baseProps} item={item} />);

            const nameColumn = container.querySelector('.read-item-name');
            // Base padding (8) + indent (20) = 28px
            expect(nameColumn).toHaveStyle({ paddingLeft: '28px' });
        });
    });

    describe('Interactions', () => {
        it('should call onClick when row is clicked', () => {
            const { container } = render(<ReadItemComponent {...baseProps} />);

            const readItem = container.querySelector('.read-item');
            if (readItem) {
                fireEvent.click(readItem);
            }

            expect(mockOnClick).toHaveBeenCalledWith('read_001', false);
        });

        it('should call onClick with multiSelect true when ctrl key pressed', () => {
            const { container } = render(<ReadItemComponent {...baseProps} />);

            const readItem = container.querySelector('.read-item');
            if (readItem) {
                fireEvent.click(readItem, { ctrlKey: true });
            }

            expect(mockOnClick).toHaveBeenCalledWith('read_001', true);
        });

        it('should call onClick with multiSelect true when meta key pressed', () => {
            const { container } = render(<ReadItemComponent {...baseProps} />);

            const readItem = container.querySelector('.read-item');
            if (readItem) {
                fireEvent.click(readItem, { metaKey: true });
            }

            expect(mockOnClick).toHaveBeenCalledWith('read_001', true);
        });

        it('should call onPlotRead when plot button clicked', () => {
            render(<ReadItemComponent {...baseProps} />);

            const plotButton = screen.getByText('Plot');
            fireEvent.click(plotButton);

            expect(mockOnPlotRead).toHaveBeenCalledWith('read_001');
        });

        it('should stop propagation when plot button clicked', () => {
            render(<ReadItemComponent {...baseProps} />);

            const plotButton = screen.getByText('Plot');
            fireEvent.click(plotButton);

            // onPlotRead should be called, but onClick should NOT be called
            expect(mockOnPlotRead).toHaveBeenCalledWith('read_001');
            expect(mockOnClick).not.toHaveBeenCalled();
        });
    });

    describe('Accessibility', () => {
        it('should have title attribute with read ID', () => {
            const { container } = render(<ReadItemComponent {...baseProps} />);

            const readItem = container.querySelector('.read-item');
            expect(readItem).toHaveAttribute('title', 'read_001');
        });

        it('should have descriptive button title', () => {
            render(<ReadItemComponent {...baseProps} />);

            const plotButton = screen.getByTitle('Plot this read');
            expect(plotButton).toBeInTheDocument();
        });
    });
});
