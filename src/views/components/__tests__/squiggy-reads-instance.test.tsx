/**
 * Tests for ReadsInstance (virtualized reads table built on react-window 2)
 *
 * jsdom has no layout engine and no ResizeObserver, so the List never learns
 * its real size: it renders against `defaultHeight` (CONSTANTS.DEFAULT_LIST_HEIGHT).
 * ResizeObserver is stubbed as a no-op and Element.scrollTo is mocked so the
 * imperative scroll calls made by keyboard navigation can be asserted.
 */

import * as React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { ReadsInstance } from '../squiggy-reads-instance';
import {
    CONSTANTS,
    ReadItem,
    ReadListItem,
    ReadsInstanceProps,
    ReferenceGroupItem,
} from '../../../types/squiggy-reads-types';

class NoopResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
}

const makeReads = (n: number, referenceName?: string): ReadItem[] =>
    Array.from({ length: n }, (_, i) => ({
        type: 'read' as const,
        readId: `${referenceName ?? 'read'}_${i}`,
        referenceName,
        indentLevel: referenceName ? (1 as const) : (0 as const),
    }));

const makeReference = (name: string, readCount: number): ReferenceGroupItem => ({
    type: 'reference',
    referenceName: name,
    readCount,
    isExpanded: true,
    indentLevel: 0,
});

describe('ReadsInstance', () => {
    const originalResizeObserver = (global as any).ResizeObserver;
    const originalScrollTo = Element.prototype.scrollTo;
    let scrollToMock: jest.Mock;

    const baseProps = (overrides: Partial<ReadsInstanceProps> = {}): ReadsInstanceProps => ({
        items: makeReads(10),
        selectedReadIds: new Set<string>(),
        focusedIndex: null,
        nameColumnWidth: CONSTANTS.DEFAULT_NAME_WIDTH,
        detailsColumnWidth: CONSTANTS.DEFAULT_DETAILS_WIDTH,
        hasReferences: false,
        sortBy: 'name',
        sortOrder: 'asc',
        onPlotRead: jest.fn(),
        onSelectRead: jest.fn(),
        onToggleReference: jest.fn(),
        onSearch: jest.fn(),
        onLoadMore: jest.fn(),
        onUpdateColumnWidths: jest.fn(),
        onSort: jest.fn(),
        ...overrides,
    });

    const getViewport = (container: HTMLElement) =>
        container.querySelector('.reads-list-viewport') as HTMLElement;
    const getList = (container: HTMLElement) =>
        container.querySelector('[role="list"]') as HTMLElement;

    beforeAll(() => {
        (global as any).ResizeObserver = NoopResizeObserver;
    });

    afterAll(() => {
        (global as any).ResizeObserver = originalResizeObserver;
        Element.prototype.scrollTo = originalScrollTo;
    });

    beforeEach(() => {
        scrollToMock = jest.fn();
        Element.prototype.scrollTo = scrollToMock as any;
    });

    describe('Rendering', () => {
        it('shows the empty state when there are no items', () => {
            render(<ReadsInstance {...baseProps({ items: [] })} />);

            expect(
                screen.getByText('No reads to display. Load a POD5 file to get started.')
            ).toBeInTheDocument();
        });

        it('renders column headers and rows', () => {
            const { container } = render(<ReadsInstance {...baseProps()} />);

            expect(screen.getByText('Name')).toBeInTheDocument();
            expect(screen.getByText('Reads')).toBeInTheDocument();
            expect(container.querySelectorAll('.read-item')).toHaveLength(10);
            expect(screen.getByText('read_0')).toBeInTheDocument();
        });

        it('positions rows at fixed ROW_HEIGHT offsets with listitem semantics', () => {
            const { container } = render(<ReadsInstance {...baseProps()} />);

            const rows = container.querySelectorAll('[role="listitem"]');
            expect(rows).toHaveLength(10);
            const third = rows[2] as HTMLElement;
            expect(third.style.height).toBe(`${CONSTANTS.ROW_HEIGHT}px`);
            expect(third.style.transform).toBe(`translateY(${2 * CONSTANTS.ROW_HEIGHT}px)`);
            expect(third).toHaveAttribute('aria-posinset', '3');
            expect(third).toHaveAttribute('aria-setsize', '10');
        });

        it('virtualizes large lists (renders only visible rows plus overscan)', () => {
            const { container } = render(
                <ReadsInstance {...baseProps({ items: makeReads(5000) })} />
            );

            const visible = Math.ceil(CONSTANTS.DEFAULT_LIST_HEIGHT / CONSTANTS.ROW_HEIGHT);
            const rendered = container.querySelectorAll('.read-item').length;
            expect(rendered).toBeGreaterThanOrEqual(visible);
            expect(rendered).toBeLessThanOrEqual(visible + CONSTANTS.OVERSCAN_COUNT + 1);
            expect(screen.queryByText('read_4999')).not.toBeInTheDocument();
        });

        it('renders reference group rows alongside read rows', () => {
            const items: ReadListItem[] = [makeReference('chr1', 2), ...makeReads(2, 'chr1')];
            const { container } = render(
                <ReadsInstance {...baseProps({ items, hasReferences: true })} />
            );

            expect(container.querySelectorAll('.reference-group')).toHaveLength(1);
            expect(container.querySelectorAll('.read-item')).toHaveLength(2);
        });

        it('re-renders rows when rowProps change (selection)', () => {
            const props = baseProps();
            const { container, rerender } = render(<ReadsInstance {...props} />);
            expect(container.querySelectorAll('.read-item.selected')).toHaveLength(0);

            rerender(<ReadsInstance {...props} selectedReadIds={new Set(['read_3'])} />);

            const selected = container.querySelectorAll('.read-item.selected');
            expect(selected).toHaveLength(1);
            expect(selected[0]).toHaveAttribute('title', 'read_3');
        });
    });

    describe('Interaction', () => {
        it('forwards row clicks to onSelectRead', () => {
            const props = baseProps();
            render(<ReadsInstance {...props} />);

            fireEvent.click(screen.getByText('read_4'));

            expect(props.onSelectRead).toHaveBeenCalledWith('read_4', false);
        });

        it('sorts by read count from the header when references are present', () => {
            const props = baseProps({ hasReferences: true, sortBy: 'reads', sortOrder: 'desc' });
            render(<ReadsInstance {...props} />);

            fireEvent.click(screen.getByText('Reads'));

            expect(props.onSort).toHaveBeenCalledWith('reads');
            expect(screen.getByText('▼')).toBeInTheDocument();
        });
    });

    describe('Infinite scroll', () => {
        it('requests more reads once when the end of the list is rendered', () => {
            const props = baseProps({ items: makeReads(10) });
            render(<ReadsInstance {...props} />);

            expect(props.onLoadMore).toHaveBeenCalledTimes(1);
        });

        it('does not re-request when only the onLoadMore identity changes', () => {
            const props = baseProps({ items: makeReads(10) });
            const { rerender } = render(<ReadsInstance {...props} />);
            const nextOnLoadMore = jest.fn();

            rerender(<ReadsInstance {...props} onLoadMore={nextOnLoadMore} />);

            expect(props.onLoadMore).toHaveBeenCalledTimes(1);
            expect(nextOnLoadMore).not.toHaveBeenCalled();
        });

        it('does not request more reads when far from the end', () => {
            const props = baseProps({ items: makeReads(5000) });
            render(<ReadsInstance {...props} />);

            expect(props.onLoadMore).not.toHaveBeenCalled();
        });
    });

    describe('Keyboard navigation', () => {
        it('moves focus with ArrowDown/ArrowUp and scrolls the row into view', () => {
            const { container } = render(
                <ReadsInstance {...baseProps({ items: makeReads(5000) })} />
            );
            getViewport(container).focus();

            fireEvent.keyDown(document, { key: 'ArrowDown' });
            expect(container.querySelector('.read-item.focused')).toHaveAttribute(
                'title',
                'read_1'
            );
            // Row 1 is already visible, so 'auto' alignment keeps the offset at 0
            expect(scrollToMock).toHaveBeenLastCalledWith(expect.objectContaining({ top: 0 }));

            fireEvent.keyDown(document, { key: 'ArrowUp' });
            expect(container.querySelector('.read-item.focused')).toHaveAttribute(
                'title',
                'read_0'
            );
        });

        it('End scrolls to the last row aligned to the bottom; Home scrolls back', () => {
            const count = 5000;
            const { container } = render(
                <ReadsInstance {...baseProps({ items: makeReads(count) })} />
            );
            getViewport(container).focus();

            fireEvent.keyDown(document, { key: 'End' });
            expect(scrollToMock).toHaveBeenLastCalledWith(
                expect.objectContaining({
                    top: count * CONSTANTS.ROW_HEIGHT - CONSTANTS.DEFAULT_LIST_HEIGHT,
                })
            );

            fireEvent.keyDown(document, { key: 'Home' });
            expect(scrollToMock).toHaveBeenLastCalledWith(expect.objectContaining({ top: 0 }));
        });

        it('PageDown moves focus 10 rows and scrolls minimally', () => {
            const { container } = render(
                <ReadsInstance {...baseProps({ items: makeReads(5000) })} />
            );
            getViewport(container).focus();

            // Focus starts at 0; 3 x PageDown -> row 30, below the ~22-row viewport
            fireEvent.keyDown(document, { key: 'PageDown' });
            fireEvent.keyDown(document, { key: 'PageDown' });
            fireEvent.keyDown(document, { key: 'PageDown' });

            // 'auto' aligns row 30 to the bottom edge rather than centering it
            expect(scrollToMock).toHaveBeenLastCalledWith(
                expect.objectContaining({
                    top: 31 * CONSTANTS.ROW_HEIGHT - CONSTANTS.DEFAULT_LIST_HEIGHT,
                })
            );
        });

        it('Enter plots the focused read and toggles a focused reference', () => {
            const items: ReadListItem[] = [makeReference('chr1', 2), ...makeReads(2, 'chr1')];
            const props = baseProps({ items, hasReferences: true });
            const { container } = render(<ReadsInstance {...props} />);
            getViewport(container).focus();

            fireEvent.keyDown(document, { key: 'Enter' });
            expect(props.onToggleReference).toHaveBeenCalledWith('chr1');

            fireEvent.keyDown(document, { key: 'ArrowDown' });
            fireEvent.keyDown(document, { key: 'Enter' });
            expect(props.onPlotRead).toHaveBeenCalledWith('chr1_0');
        });

        it('Space toggles a focused reference', () => {
            const items: ReadListItem[] = [makeReference('chr1', 2), ...makeReads(2, 'chr1')];
            const props = baseProps({ items, hasReferences: true });
            const { container } = render(<ReadsInstance {...props} />);
            getViewport(container).focus();

            fireEvent.keyDown(document, { key: ' ' });

            expect(props.onToggleReference).toHaveBeenCalledWith('chr1');
        });

        it('ignores keys when focus is outside the component', () => {
            const props = baseProps();
            render(<ReadsInstance {...props} />);

            fireEvent.keyDown(document, { key: 'Enter' });

            expect(props.onPlotRead).not.toHaveBeenCalled();
            expect(scrollToMock).not.toHaveBeenCalled();
        });
    });

    describe('Sticky reference header', () => {
        const groupedItems = (): ReadListItem[] => [
            makeReference('chr1', 100),
            ...makeReads(100, 'chr1'),
            makeReference('chr2', 100),
            ...makeReads(100, 'chr2'),
        ];

        const scrollListTo = (container: HTMLElement, top: number) => {
            const list = getList(container);
            list.scrollTop = top;
            fireEvent.scroll(list);
        };

        it('is hidden at the top of the list', () => {
            const { container } = render(
                <ReadsInstance {...baseProps({ items: groupedItems(), hasReferences: true })} />
            );

            expect(container.querySelector('.reference-group-sticky-overlay')).toBeNull();
        });

        it('shows the owning reference once its header scrolls out of view', () => {
            const { container } = render(
                <ReadsInstance {...baseProps({ items: groupedItems(), hasReferences: true })} />
            );

            scrollListTo(container, 10 * CONSTANTS.ROW_HEIGHT);
            let overlay = container.querySelector('.reference-group-sticky-overlay');
            expect(overlay).not.toBeNull();
            expect(overlay).toHaveTextContent('chr1');

            // Past the chr2 header (index 101)
            scrollListTo(container, 120 * CONSTANTS.ROW_HEIGHT);
            overlay = container.querySelector('.reference-group-sticky-overlay');
            expect(overlay).toHaveTextContent('chr2');

            scrollListTo(container, 0);
            expect(container.querySelector('.reference-group-sticky-overlay')).toBeNull();
        });

        it('toggles the reference when the sticky header is clicked', () => {
            const props = baseProps({ items: groupedItems(), hasReferences: true });
            const { container } = render(<ReadsInstance {...props} />);

            scrollListTo(container, 10 * CONSTANTS.ROW_HEIGHT);
            const overlay = container.querySelector(
                '.reference-group-sticky-overlay .reference-group'
            ) as HTMLElement;
            fireEvent.click(overlay);

            expect(props.onToggleReference).toHaveBeenCalledWith('chr1');
        });
    });

    describe('Column resizing', () => {
        it('reports clamped widths when the header divider is dragged', () => {
            const props = baseProps({ nameColumnWidth: 200, detailsColumnWidth: 200 });
            const { container } = render(<ReadsInstance {...props} />);

            const handle = container.querySelector('.reads-header-divider > *') as HTMLElement;
            fireEvent.mouseDown(handle, { clientX: 100 });
            fireEvent.mouseMove(document, { clientX: 130 });
            fireEvent.mouseUp(document);

            expect(props.onUpdateColumnWidths).toHaveBeenCalledWith(230, 170);
        });
    });
});
