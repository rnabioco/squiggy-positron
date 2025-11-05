/**
 * ReadsInstance Component
 *
 * Virtualized list of reads using react-window.
 * Handles rendering of reference groups and individual reads.
 */

import * as React from 'react';
import { FixedSizeList as List } from 'react-window';
import { ReadsInstanceProps, CONSTANTS } from '../../types/squiggy-reads-types';
import { ReadItemComponent } from './squiggy-read-item';
import { ReferenceGroupComponent } from './squiggy-reference-group';
import { ColumnResizer } from './column-resizer';
import './squiggy-reads-instance.css';

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
    onPlotAggregate,
    onSelectRead,
    onToggleReference,
    onUpdateColumnWidths,
    onLoadMore,
    onSort,
}) => {
    const containerRef = React.useRef<HTMLDivElement>(null);
    const listRef = React.useRef<any>(null);
    const [containerHeight, setContainerHeight] = React.useState(600);
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
            if (items[i]?.type === 'reference') {
                // Check if this reference is expanded
                if ((items[i] as any).isExpanded) {
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

    // Handle scroll events from react-window
    const handleScroll = React.useCallback(
        ({ scrollOffset }: { scrollOffset: number; scrollUpdateWasRequested: boolean }) => {
            setScrollTop(scrollOffset);
        },
        []
    );

    // Handle infinite scroll - trigger load more when near bottom
    const handleItemsRendered = React.useCallback(
        ({ visibleStopIndex }: { visibleStartIndex: number; visibleStopIndex: number }) => {
            const itemsRemaining = items.length - visibleStopIndex;

            // Trigger load more when within 50 items of the end
            if (itemsRemaining < 50 && !loadMoreTriggeredRef.current) {
                loadMoreTriggeredRef.current = true;
                onLoadMore();
            }
        },
        [items.length, onLoadMore]
    );

    // Measure container height
    React.useEffect(() => {
        const updateHeight = () => {
            if (containerRef.current) {
                const height = containerRef.current.clientHeight;
                if (height > 100) {
                    // Only update if we got a reasonable height
                    setContainerHeight(height);
                }
            }
        };

        // Try multiple times to get the height (handles async layout)
        const rafId = requestAnimationFrame(() => {
            updateHeight();
            // Try again after a short delay if height is still small
            setTimeout(() => {
                updateHeight();
            }, 100);
        });

        window.addEventListener('resize', updateHeight);
        return () => {
            cancelAnimationFrame(rafId);
            window.removeEventListener('resize', updateHeight);
        };
    }, []);

    // Items change triggers automatic re-render (no manual reset needed for FixedSizeList)
    // resetAfterIndex is only for VariableSizeList

    // Handle column resizing
    const handleColumnResize = (deltaX: number) => {
        const newNameWidth = Math.max(CONSTANTS.MIN_COLUMN_WIDTH, nameColumnWidth + deltaX);
        const newDetailsWidth = Math.max(CONSTANTS.MIN_COLUMN_WIDTH, detailsColumnWidth - deltaX);
        onUpdateColumnWidths(newNameWidth, newDetailsWidth);
    };

    // Handle keyboard navigation
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
                        listRef.current?.scrollToItem(newIndex, 'smart');
                    }
                    break;

                case 'ArrowUp':
                    e.preventDefault();
                    if (currentIndex > 0) {
                        const newIndex = currentIndex - 1;
                        setLocalFocusedIndex(newIndex);
                        listRef.current?.scrollToItem(newIndex, 'smart');
                    }
                    break;

                case 'Home':
                    e.preventDefault();
                    setLocalFocusedIndex(0);
                    listRef.current?.scrollToItem(0, 'start');
                    break;

                case 'End': {
                    e.preventDefault();
                    const lastIndex = items.length - 1;
                    setLocalFocusedIndex(lastIndex);
                    listRef.current?.scrollToItem(lastIndex, 'end');
                    break;
                }

                case 'PageDown': {
                    e.preventDefault();
                    const pageDownIndex = Math.min(items.length - 1, currentIndex + 10);
                    setLocalFocusedIndex(pageDownIndex);
                    listRef.current?.scrollToItem(pageDownIndex, 'smart');
                    break;
                }

                case 'PageUp': {
                    e.preventDefault();
                    const pageUpIndex = Math.max(0, currentIndex - 10);
                    setLocalFocusedIndex(pageUpIndex);
                    listRef.current?.scrollToItem(pageUpIndex, 'smart');
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

    // Row renderer for react-window
    const Row = ({ index, style }: { index: number; style: React.CSSProperties }) => {
        const item = items[index];
        const isFocused = index === localFocusedIndex;

        if (item.type === 'reference') {
            return (
                <div style={style}>
                    <ReferenceGroupComponent
                        item={item}
                        isEvenRow={index % 2 === 0}
                        nameColumnWidth={nameColumnWidth}
                        detailsColumnWidth={detailsColumnWidth}
                        onToggle={onToggleReference}
                        onPlotAggregate={onPlotAggregate}
                    />
                </div>
            );
        } else {
            return (
                <div style={style}>
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
    };

    return (
        <div className="reads-instance-container" ref={containerRef}>
            {/* Column headers */}
            <div className="reads-header">
                <div className="reads-header-column" style={{ width: `${nameColumnWidth}px` }}>
                    Name
                </div>
                <ColumnResizer onResize={handleColumnResize} />
                <div
                    className={`reads-header-column ${hasReferences ? 'reads-sortable' : ''}`}
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
                <div tabIndex={0} style={{ outline: 'none', flex: 1, position: 'relative' }}>
                    <List
                        ref={listRef}
                        height={Math.max(100, containerHeight - 32)} // Subtract header height
                        itemCount={items.length}
                        itemSize={CONSTANTS.ROW_HEIGHT}
                        width="100%"
                        overscanCount={CONSTANTS.OVERSCAN_COUNT}
                        onItemsRendered={handleItemsRendered}
                        onScroll={handleScroll}
                    >
                        {Row}
                    </List>

                    {/* Sticky header overlay */}
                    {stickyHeaderIndex !== null &&
                        items[stickyHeaderIndex]?.type === 'reference' && (
                            <div className="reference-group-sticky-overlay">
                                <ReferenceGroupComponent
                                    item={items[stickyHeaderIndex] as any}
                                    isEvenRow={stickyHeaderIndex % 2 === 0}
                                    nameColumnWidth={nameColumnWidth}
                                    detailsColumnWidth={detailsColumnWidth}
                                    onToggle={onToggleReference}
                                    onPlotAggregate={onPlotAggregate}
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
