/**
 * Tests for ColumnResizer Component
 *
 * Tests the draggable column resizer with mouse event handling.
 * Target: >90% coverage of column-resizer.tsx
 */

import * as React from 'react';
import { render, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { ColumnResizer } from '../column-resizer';

describe('ColumnResizer', () => {
    let mockOnResize: jest.Mock;

    beforeEach(() => {
        mockOnResize = jest.fn();
    });

    afterEach(() => {
        jest.clearAllMocks();
    });

    describe('Rendering', () => {
        it('should render with column-resizer class', () => {
            const { container } = render(<ColumnResizer onResize={mockOnResize} />);

            const resizer = container.querySelector('.column-resizer');
            expect(resizer).toBeInTheDocument();
        });

        it('should not have dragging class initially', () => {
            const { container } = render(<ColumnResizer onResize={mockOnResize} />);

            const resizer = container.querySelector('.column-resizer');
            expect(resizer).not.toHaveClass('dragging');
        });

        it('should add dragging class when mouse down', () => {
            const { container } = render(<ColumnResizer onResize={mockOnResize} />);

            const resizer = container.querySelector('.column-resizer');
            if (resizer) {
                fireEvent.mouseDown(resizer, { clientX: 100 });
            }

            expect(resizer).toHaveClass('dragging');
        });
    });

    describe('Mouse Down Event', () => {
        it('should prevent default on mouse down', () => {
            const { container } = render(<ColumnResizer onResize={mockOnResize} />);

            const resizer = container.querySelector('.column-resizer');
            const event = new MouseEvent('mousedown', { clientX: 100, bubbles: true });
            const preventDefaultSpy = jest.spyOn(event, 'preventDefault');

            if (resizer) {
                resizer.dispatchEvent(event);
            }

            expect(preventDefaultSpy).toHaveBeenCalled();
        });

        it('should capture start position on mouse down', () => {
            const { container } = render(<ColumnResizer onResize={mockOnResize} />);

            const resizer = container.querySelector('.column-resizer');
            if (resizer) {
                fireEvent.mouseDown(resizer, { clientX: 100 });

                // Move mouse - should calculate delta from start position
                fireEvent.mouseMove(document, { clientX: 150 });
            }

            expect(mockOnResize).toHaveBeenCalledWith(50);
        });
    });

    describe('Mouse Move During Drag', () => {
        it('should call onResize with deltaX during drag', () => {
            const { container } = render(<ColumnResizer onResize={mockOnResize} />);

            const resizer = container.querySelector('.column-resizer');
            if (resizer) {
                // Start dragging
                fireEvent.mouseDown(resizer, { clientX: 100 });

                // Move mouse
                fireEvent.mouseMove(document, { clientX: 150 });
            }

            expect(mockOnResize).toHaveBeenCalledWith(50);
        });

        it('should update start position after each move', () => {
            const { container } = render(<ColumnResizer onResize={mockOnResize} />);

            const resizer = container.querySelector('.column-resizer');
            if (resizer) {
                // Start dragging
                fireEvent.mouseDown(resizer, { clientX: 100 });

                // First move
                fireEvent.mouseMove(document, { clientX: 150 });

                // Second move - delta should be from new position
                fireEvent.mouseMove(document, { clientX: 175 });
            }

            expect(mockOnResize).toHaveBeenCalledTimes(2);
            expect(mockOnResize).toHaveBeenNthCalledWith(1, 50);
            expect(mockOnResize).toHaveBeenNthCalledWith(2, 25);
        });

        it('should not call onResize when not dragging', () => {
            render(<ColumnResizer onResize={mockOnResize} />);

            // Move mouse without starting drag
            fireEvent.mouseMove(document, { clientX: 150 });

            expect(mockOnResize).not.toHaveBeenCalled();
        });

        it('should handle negative deltaX', () => {
            const { container } = render(<ColumnResizer onResize={mockOnResize} />);

            const resizer = container.querySelector('.column-resizer');
            if (resizer) {
                // Start dragging
                fireEvent.mouseDown(resizer, { clientX: 100 });

                // Move mouse left
                fireEvent.mouseMove(document, { clientX: 50 });
            }

            expect(mockOnResize).toHaveBeenCalledWith(-50);
        });
    });

    describe('Mouse Up Event', () => {
        it('should stop dragging on mouse up', () => {
            const { container } = render(<ColumnResizer onResize={mockOnResize} />);

            const resizer = container.querySelector('.column-resizer');
            if (resizer) {
                // Start dragging
                fireEvent.mouseDown(resizer, { clientX: 100 });
                expect(resizer).toHaveClass('dragging');

                // Stop dragging
                fireEvent.mouseUp(document);
            }

            expect(resizer).not.toHaveClass('dragging');
        });

        it('should not call onResize after mouse up', () => {
            const { container } = render(<ColumnResizer onResize={mockOnResize} />);

            const resizer = container.querySelector('.column-resizer');
            if (resizer) {
                // Start dragging
                fireEvent.mouseDown(resizer, { clientX: 100 });

                // Move once
                fireEvent.mouseMove(document, { clientX: 150 });

                // Stop dragging
                fireEvent.mouseUp(document);

                // Try to move again
                fireEvent.mouseMove(document, { clientX: 200 });
            }

            // Should only be called once (during drag, not after)
            expect(mockOnResize).toHaveBeenCalledTimes(1);
        });
    });

    describe('Event Listener Cleanup', () => {
        it('should remove event listeners on unmount', () => {
            const addEventListenerSpy = jest.spyOn(document, 'addEventListener');
            const removeEventListenerSpy = jest.spyOn(document, 'removeEventListener');

            const { container, unmount } = render(<ColumnResizer onResize={mockOnResize} />);

            const resizer = container.querySelector('.column-resizer');
            if (resizer) {
                // Start dragging to add listeners
                fireEvent.mouseDown(resizer, { clientX: 100 });
            }

            expect(addEventListenerSpy).toHaveBeenCalledWith('mousemove', expect.any(Function));
            expect(addEventListenerSpy).toHaveBeenCalledWith('mouseup', expect.any(Function));

            // Unmount should trigger cleanup
            unmount();

            expect(removeEventListenerSpy).toHaveBeenCalledWith('mousemove', expect.any(Function));
            expect(removeEventListenerSpy).toHaveBeenCalledWith('mouseup', expect.any(Function));

            addEventListenerSpy.mockRestore();
            removeEventListenerSpy.mockRestore();
        });

        it('should remove listeners when dragging stops', () => {
            const removeEventListenerSpy = jest.spyOn(document, 'removeEventListener');

            const { container } = render(<ColumnResizer onResize={mockOnResize} />);

            const resizer = container.querySelector('.column-resizer');
            if (resizer) {
                // Start dragging
                fireEvent.mouseDown(resizer, { clientX: 100 });

                // Stop dragging
                fireEvent.mouseUp(document);
            }

            expect(removeEventListenerSpy).toHaveBeenCalledWith('mousemove', expect.any(Function));
            expect(removeEventListenerSpy).toHaveBeenCalledWith('mouseup', expect.any(Function));

            removeEventListenerSpy.mockRestore();
        });
    });
});
