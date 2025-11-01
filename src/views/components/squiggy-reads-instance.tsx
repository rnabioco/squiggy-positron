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
import './squiggy-reads-instance.css';

export const ReadsInstance: React.FC<ReadsInstanceProps> = ({
    items,
    selectedReadIds,
    focusedIndex,
    nameColumnWidth,
    detailsColumnWidth,
    onPlotRead,
    onPlotAggregate,
    onSelectRead,
    onToggleReference,
}) => {
    const containerRef = React.useRef<HTMLDivElement>(null);
    const listRef = React.useRef<any>(null);
    const [containerHeight, setContainerHeight] = React.useState(400);
    const [localFocusedIndex, setLocalFocusedIndex] = React.useState<number | null>(focusedIndex);

    // Measure container height with ResizeObserver
    React.useEffect(() => {
        if (!containerRef.current) {
            return;
        }

        const resizeObserver = new ResizeObserver((entries) => {
            for (const entry of entries) {
                setContainerHeight(entry.contentRect.height);
            }
        });

        resizeObserver.observe(containerRef.current);
        return () => resizeObserver.disconnect();
    }, []);

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
                <div className="reads-header-column" style={{ width: `${detailsColumnWidth}px` }}>
                    Reads
                </div>
                <div className="reads-header-column reads-header-actions">Actions</div>
            </div>

            {/* Virtualized list */}
            {items.length > 0 ? (
                <div tabIndex={0} style={{ outline: 'none' }}>
                    <List
                        ref={listRef}
                        height={containerHeight - 32} // Subtract header height
                        itemCount={items.length}
                        itemSize={CONSTANTS.ROW_HEIGHT}
                        width="100%"
                        overscanCount={CONSTANTS.OVERSCAN_COUNT}
                    >
                        {Row}
                    </List>
                </div>
            ) : (
                <div className="reads-empty-state">
                    No reads to display. Load a POD5 file to get started.
                </div>
            )}
        </div>
    );
};
