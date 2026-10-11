/**
 * ReadsInstance Component
 *
 * Virtualized list of reads using react-window.
 * Handles rendering of reference groups and individual reads.
 */

import * as React from 'react';
import { List, RowComponentProps, useListRef } from 'react-window';
import {
    ReadsInstanceProps,
    CONSTANTS,
    ReadListItem,
    ReferenceGroupItem,
} from '../../types/squiggy-reads-types';
import { ReadItemComponent } from './squiggy-read-item';
import { ReferenceGroupComponent } from './squiggy-reference-group';
import { ColumnResizer } from './column-resizer';
import './squiggy-reads-instance.css';

/**
 * Props passed to each virtualized row via react-window's `rowProps`.
 * Keeping the row renderer module-level (instead of nested in ReadsInstance)
 * means its component identity is stable across renders, so react-window does
 * not unmount/remount every visible row on each parent render (scroll, resize,
 * sort). All per-render values flow through `rowProps` instead; react-window
 * memoizes the row component and re-renders rows when these values change.
 */
interface ReadRowData {
    items: ReadListItem[];
    focusedIndex: number | null;
    selectedReadIds: Set<string>;
    nameColumnWidth: number;
    detailsColumnWidth: number;
    onPlotRead: (readId: string) => void;
    onSelectRead: (readId: string, multiSelect: boolean) => void;
    onToggleReference: (referenceName: string) => void;
}

function ReadRow({
    ariaAttributes,
    index,
    style,
    items,
    focusedIndex,
    selectedReadIds,
    nameColumnWidth,
    detailsColumnWidth,
    onPlotRead,
    onSelectRead,
    onToggleReference,
}: RowComponentProps<ReadRowData>) {
    const item = items[index];
    const isFocused = index === focusedIndex;

    if (item.type === 'reference') {
        return (
            <div style={style} {...ariaAttributes}>
                <ReferenceGroupComponent
                    item={item}
                    isEvenRow={index % 2 === 0}
                    nameColumnWidth={nameColumnWidth}
                    detailsColumnWidth={detailsColumnWidth}
                    onToggle={onToggleReference}
                />
            </div>
        );
    }
    return (
        <div style={style} {...ariaAttributes}>
            <ReadItemComponent
                item={item}
                isSelected={selectedReadIds.has(item.readId)}
                isFocused={isFocused}
                isEvenRow={index % 2 === 0}
                nameColumnWidth={nameColumnWidth}
                detailsColumnWidth={detailsColumnWidth}
                onPlotRead={onPlotRead}
                onClick={onSelectRead}
            />
        </div>
    );
}

export const ReadsInstance: React.FC<ReadsInstanceProps> = ({
    items,
    selectedReadIds,
    focusedIndex,
    nameColumnWidth,
    detailsColumnWidth,
    hasReferences,
    sortBy,
    sortOrder,
    onPlotRead,
    onSelectRead,
    onToggleReference,
    onUpdateColumnWidths,
    onLoadMore,
    onSort,
}) => {
    const containerRef = React.useRef<HTMLDivElement>(null);
    const listRef = useListRef(null);
    const [localFocusedIndex, setLocalFocusedIndex] = React.useState<number | null>(focusedIndex);

    // Track scroll position for sticky headers
    const [scrollTop, setScrollTop] = React.useState(0);
    const [stickyHeaderIndex, setStickyHeaderIndex] = React.useState<number | null>(null);

    // Track if we've already triggered load more to prevent duplicate requests
    const loadMoreTriggeredRef = React.useRef(false);

    // Reset trigger when items change
    React.useEffect(() => {
        loadMoreTriggeredRef.current = false;
    }, [items.length]);

    // Calculate which reference header should be sticky based on scroll position
    React.useEffect(() => {
        // Find the reference header that should be sticky
        let currentReferenceIndex: number | null = null;
        const scrollOffset = scrollTop;
        const firstVisibleIndex = Math.floor(scrollOffset / CONSTANTS.ROW_HEIGHT);

        // Look backwards from the first visible item to find the owning reference
        for (let i = firstVisibleIndex; i >= 0; i--) {
            const item = items[i];
            if (item?.type === 'reference') {
                // Type guard: ensure it's a ReferenceGroupItem
                const refItem = item as ReferenceGroupItem;
                // Check if this reference is expanded
                if (refItem.isExpanded) {
                    // Check if we're scrolled past this reference header
                    const referenceTop = i * CONSTANTS.ROW_HEIGHT;
                    if (scrollOffset > referenceTop) {
                        currentReferenceIndex = i;
                    }
                }
                break;
            }
        }

        setStickyHeaderIndex(currentReferenceIndex);
    }, [scrollTop, items]);

    // react-window 2 has no onScroll callback of its own; the List forwards DOM
    // props to its scroll container, so read the offset from the native event.
    const handleScroll = React.useCallback((e: React.UIEvent<HTMLDivElement>) => {
        setScrollTop(e.currentTarget.scrollTop);
    }, []);

    // Latest values for the rows-rendered callback, kept in refs so the callback
    // identity stays stable. react-window 2 re-invokes onRowsRendered whenever the
    // callback identity changes (not only when the rendered range changes, as
    // react-window 1 did), which would re-trigger load more right after each
    // append if the callback depended on items.length.
    const itemCountRef = React.useRef(items.length);
    itemCountRef.current = items.length;
    const onLoadMoreRef = React.useRef(onLoadMore);
    onLoadMoreRef.current = onLoadMore;

    // Handle infinite scroll - trigger load more when near bottom
    const handleRowsRendered = React.useCallback(
        (visibleRows: { startIndex: number; stopIndex: number }) => {
            const itemsRemaining = itemCountRef.current - visibleRows.stopIndex;

            // Trigger load more when within 50 items of the end
            if (itemsRemaining < 50 && !loadMoreTriggeredRef.current) {
                loadMoreTriggeredRef.current = true;
                onLoadMoreRef.current();
            }
        },
        []
    );

    // Handle column resizing with MIN/MAX boundaries
    const handleColumnResize = (deltaX: number) => {
        const newNameWidth = Math.min(
            CONSTANTS.MAX_COLUMN_WIDTH,
            Math.max(CONSTANTS.MIN_COLUMN_WIDTH, nameColumnWidth + deltaX)
        );
        const newDetailsWidth = Math.min(
            CONSTANTS.MAX_COLUMN_WIDTH,
            Math.max(CONSTANTS.MIN_COLUMN_WIDTH, detailsColumnWidth - deltaX)
        );
        onUpdateColumnWidths(newNameWidth, newDetailsWidth);
    };

    // Handle keyboard navigation.
    // Arrow/Page keys scroll with align 'auto' (minimal scroll to bring the row
    // into view). react-window 1's 'smart' did the same for targets within one
    // viewport; react-window 2's 'smart' centers any row that is not already
    // visible, which would re-center on every ArrowDown past the bottom edge.
    React.useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (!containerRef.current?.contains(document.activeElement)) {
                return; // Only handle keys when focused on this component
            }

            const currentIndex = localFocusedIndex ?? 0;

            switch (e.key) {
                case 'ArrowDown':
                    e.preventDefault();
                    if (currentIndex < items.length - 1) {
                        const newIndex = currentIndex + 1;
                        setLocalFocusedIndex(newIndex);
                        listRef.current?.scrollToRow({ index: newIndex, align: 'auto' });
                    }
                    break;

                case 'ArrowUp':
                    e.preventDefault();
                    if (currentIndex > 0) {
                        const newIndex = currentIndex - 1;
                        setLocalFocusedIndex(newIndex);
                        listRef.current?.scrollToRow({ index: newIndex, align: 'auto' });
                    }
                    break;

                case 'Home':
                    e.preventDefault();
                    setLocalFocusedIndex(0);
                    listRef.current?.scrollToRow({ index: 0, align: 'start' });
                    break;

                case 'End': {
                    e.preventDefault();
                    const lastIndex = items.length - 1;
                    setLocalFocusedIndex(lastIndex);
                    listRef.current?.scrollToRow({ index: lastIndex, align: 'end' });
                    break;
                }

                case 'PageDown': {
                    e.preventDefault();
                    const pageDownIndex = Math.min(items.length - 1, currentIndex + 10);
                    setLocalFocusedIndex(pageDownIndex);
                    listRef.current?.scrollToRow({ index: pageDownIndex, align: 'auto' });
                    break;
                }

                case 'PageUp': {
                    e.preventDefault();
                    const pageUpIndex = Math.max(0, currentIndex - 10);
                    setLocalFocusedIndex(pageUpIndex);
                    listRef.current?.scrollToRow({ index: pageUpIndex, align: 'auto' });
                    break;
                }

                case 'Enter':
                    e.preventDefault();
                    if (currentIndex >= 0 && currentIndex < items.length) {
                        const item = items[currentIndex];
                        if (item.type === 'read') {
                            onPlotRead(item.readId);
                        } else {
                            onToggleReference(item.referenceName);
                        }
                    }
                    break;

                case ' ':
                case 'Space':
                    e.preventDefault();
                    if (currentIndex >= 0 && currentIndex < items.length) {
                        const item = items[currentIndex];
                        if (item.type === 'reference') {
                            onToggleReference(item.referenceName);
                        }
                    }
                    break;
            }
        };

        document.addEventListener('keydown', handleKeyDown);
        return () => document.removeEventListener('keydown', handleKeyDown);
    }, [localFocusedIndex, items, onPlotRead, onToggleReference]);

    // Per-render data handed to the (stable, module-level) ReadRow renderer.
    const rowProps = React.useMemo<ReadRowData>(
        () => ({
            items,
            focusedIndex: localFocusedIndex,
            selectedReadIds,
            nameColumnWidth,
            detailsColumnWidth,
            onPlotRead,
            onSelectRead,
            onToggleReference,
        }),
        [
            items,
            localFocusedIndex,
            selectedReadIds,
            nameColumnWidth,
            detailsColumnWidth,
            onPlotRead,
            onSelectRead,
            onToggleReference,
        ]
    );

    return (
        <div className="reads-instance-container" ref={containerRef}>
            {/* Column headers - structure matches row content exactly */}
            <div className="reads-header">
                <div className="reads-header-column reads-header-name">Name</div>
                <div className="reads-header-divider">
                    <ColumnResizer onResize={handleColumnResize} />
                </div>
                <div
                    className={`reads-header-column reads-header-details ${hasReferences ? 'reads-sortable' : ''}`}
                    style={{ width: `${detailsColumnWidth}px` }}
                    onClick={() => hasReferences && onSort('reads')}
                    title={hasReferences ? 'Click to sort by read count' : ''}
                >
                    Reads
                    {hasReferences && sortBy === 'reads' && (
                        <span className={`sort-indicator ${sortOrder}`}>
                            {sortOrder === 'asc' ? '▲' : '▼'}
                        </span>
                    )}
                </div>
                <div className="reads-header-column reads-header-actions">Actions</div>
            </div>

            {/* Virtualized list */}
            {items.length > 0 ? (
                <div tabIndex={0} className="reads-list-viewport">
                    {/* List sizes itself to this flex container via ResizeObserver */}
                    <List
                        listRef={listRef}
                        rowComponent={ReadRow}
                        rowProps={rowProps}
                        rowCount={items.length}
                        rowHeight={CONSTANTS.ROW_HEIGHT}
                        defaultHeight={CONSTANTS.DEFAULT_LIST_HEIGHT}
                        overscanCount={CONSTANTS.OVERSCAN_COUNT}
                        onRowsRendered={handleRowsRendered}
                        onScroll={handleScroll}
                    />

                    {/* Sticky header overlay */}
                    {stickyHeaderIndex !== null &&
                        items[stickyHeaderIndex]?.type === 'reference' && (
                            <div className="reference-group-sticky-overlay">
                                <ReferenceGroupComponent
                                    item={items[stickyHeaderIndex] as ReferenceGroupItem}
                                    isEvenRow={stickyHeaderIndex % 2 === 0}
                                    nameColumnWidth={nameColumnWidth}
                                    detailsColumnWidth={detailsColumnWidth}
                                    onToggle={onToggleReference}
                                />
                            </div>
                        )}
                </div>
            ) : (
                <div className="reads-empty-state">
                    No reads to display. Load a POD5 file to get started.
                </div>
            )}
        </div>
    );
};
