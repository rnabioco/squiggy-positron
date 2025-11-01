/**
 * ReadsCore Component
 *
 * Top-level React component for the reads panel.
 * Manages state and handles communication with the extension host.
 */

import * as React from 'react';
import {
    ReadListItem,
    ReadItem,
    ReferenceGroupItem,
    ReadsViewState,
    CONSTANTS,
} from '../../types/squiggy-reads-types';
import { ReadsInstance } from './squiggy-reads-instance';
import { vscode } from './vscode-api';
import './squiggy-reads-core.css';

export const ReadsCore: React.FC = () => {
    // State
    const [state, setState] = React.useState<ReadsViewState>({
        items: [],
        hasReferences: false,
        totalReadCount: 0,
        searchText: '',
        filteredItems: [],
        selectedReadIds: new Set<string>(),
        focusedIndex: null,
        expandedReferences: new Set<string>(),
        nameColumnWidth: CONSTANTS.DEFAULT_NAME_WIDTH,
        detailsColumnWidth: CONSTANTS.DEFAULT_DETAILS_WIDTH,
    });

    // Store reference to reads data (for expansion)
    const referenceToReadsRef = React.useRef<Map<string, ReadItem[]>>(new Map());

    // Handle messages from extension
    React.useEffect(() => {
        const messageHandler = (event: MessageEvent) => {
            const message = event.data;

            switch (message.type) {
                case 'setReads':
                    handleSetReads(message.items);
                    break;
                case 'setReadsGrouped':
                    handleSetReadsGrouped(message.items, message.referenceToReads);
                    break;
                case 'updateSearch':
                    handleSearch(message.searchText);
                    break;
                case 'refresh':
                    // Re-render current state
                    setState((prev) => ({ ...prev }));
                    break;
            }
        };

        window.addEventListener('message', messageHandler);
        return () => window.removeEventListener('message', messageHandler);
    }, []);

    const handleSetReads = (items: ReadListItem[]) => {
        setState((prev) => ({
            ...prev,
            items,
            hasReferences: false,
            totalReadCount: items.length,
            filteredItems: items,
            selectedReadIds: new Set(),
            focusedIndex: null,
            expandedReferences: new Set(),
        }));
    };

    const handleSetReadsGrouped = (
        items: ReadListItem[],
        referenceToReads: [string, ReadItem[]][]
    ) => {
        // Store full data for expansion
        referenceToReadsRef.current = new Map(referenceToReads);

        setState((prev) => ({
            ...prev,
            items,
            hasReferences: true,
            totalReadCount: referenceToReads.reduce((sum, [_, reads]) => sum + reads.length, 0),
            filteredItems: items,
            selectedReadIds: new Set(),
            focusedIndex: null,
            expandedReferences: new Set(),
        }));
    };

    const handleSearch = (searchText: string) => {
        setState((prev) => {
            const filtered = filterItems(prev.items, searchText, referenceToReadsRef.current);
            return {
                ...prev,
                searchText,
                filteredItems: filtered,
            };
        });
    };

    const handleToggleReference = (referenceName: string) => {
        setState((prev) => {
            const expandedReferences = new Set(prev.expandedReferences);
            const isExpanded = expandedReferences.has(referenceName);

            if (isExpanded) {
                expandedReferences.delete(referenceName);
            } else {
                expandedReferences.add(referenceName);
            }

            // Rebuild items list with expansion
            const newItems = rebuildItemsList(
                referenceToReadsRef.current,
                expandedReferences,
                prev.searchText
            );

            return {
                ...prev,
                expandedReferences,
                filteredItems: newItems,
            };
        });
    };

    const handlePlotRead = (readId: string) => {
        vscode.postMessage({ type: 'plotRead', readId });
    };

    const handlePlotAggregate = (referenceName: string) => {
        vscode.postMessage({ type: 'plotAggregate', referenceName });
    };

    const handleSelectRead = (readId: string, multiSelect: boolean) => {
        setState((prev) => {
            const selectedReadIds = new Set(prev.selectedReadIds);

            if (multiSelect) {
                if (selectedReadIds.has(readId)) {
                    selectedReadIds.delete(readId);
                } else {
                    selectedReadIds.add(readId);
                }
            } else {
                selectedReadIds.clear();
                selectedReadIds.add(readId);
            }

            return { ...prev, selectedReadIds };
        });
    };

    const handleUpdateColumnWidths = (nameWidth: number, detailsWidth: number) => {
        setState((prev) => ({
            ...prev,
            nameColumnWidth: nameWidth,
            detailsColumnWidth: detailsWidth,
        }));

        vscode.postMessage({ type: 'updateColumnWidths', nameWidth, detailsWidth });
    };

    const handleLoadMore = () => {
        vscode.postMessage({ type: 'loadMore' });
    };

    // Add keyboard handler for Esc key to clear search
    const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
        if (e.key === 'Escape') {
            handleSearch('');
        }
    };

    return (
        <div className="reads-core-container">
            {/* Search box */}
            <div className="reads-search-box">
                <input
                    type="text"
                    placeholder="Search reads..."
                    value={state.searchText}
                    onChange={(e) => handleSearch(e.target.value)}
                    onKeyDown={handleKeyDown}
                    className="reads-search-input"
                />
                {state.searchText && (
                    <button
                        className="reads-search-clear"
                        onClick={() => handleSearch('')}
                        title="Clear search (Esc)"
                    >
                        ×
                    </button>
                )}
                <div className="reads-search-count">
                    {state.filteredItems.length} / {state.totalReadCount}
                </div>
            </div>

            {/* Virtualized reads list */}
            <ReadsInstance
                items={state.filteredItems}
                selectedReadIds={state.selectedReadIds}
                focusedIndex={state.focusedIndex}
                nameColumnWidth={state.nameColumnWidth}
                detailsColumnWidth={state.detailsColumnWidth}
                onPlotRead={handlePlotRead}
                onPlotAggregate={handlePlotAggregate}
                onSelectRead={handleSelectRead}
                onToggleReference={handleToggleReference}
                onSearch={handleSearch}
                onLoadMore={handleLoadMore}
                onUpdateColumnWidths={handleUpdateColumnWidths}
            />
        </div>
    );
};

/**
 * Filter items based on search text
 * NOTE: This is only used for flat POD5-only mode
 * For BAM+POD5 grouped mode, rebuildItemsList handles filtering
 */
function filterItems(
    items: ReadListItem[],
    searchText: string,
    referenceToReads: Map<string, ReadItem[]>
): ReadListItem[] {
    if (!searchText) {
        return items;
    }

    const searchLower = searchText.toLowerCase();
    const filtered: ReadListItem[] = [];

    for (const item of items) {
        if (item.type === 'reference') {
            const refMatches = item.referenceName.toLowerCase().includes(searchLower);
            const reads = referenceToReads.get(item.referenceName) || [];

            // If reference matches, show all reads; otherwise filter reads
            const matchingReads = refMatches
                ? reads
                : reads.filter((read) => read.readId.toLowerCase().includes(searchLower));

            if (matchingReads.length > 0) {
                filtered.push({
                    ...item,
                    readCount: matchingReads.length,
                });
            }
        } else {
            // For flat list (POD5-only), just filter by read ID
            if (item.readId.toLowerCase().includes(searchLower)) {
                filtered.push(item);
            }
        }
    }

    return filtered;
}

/**
 * Rebuild items list with expansion state
 */
function rebuildItemsList(
    referenceToReads: Map<string, ReadItem[]>,
    expandedReferences: Set<string>,
    searchText: string
): ReadListItem[] {
    const items: ReadListItem[] = [];
    const searchLower = searchText.toLowerCase();

    for (const [referenceName, reads] of referenceToReads.entries()) {
        const refMatches = referenceName.toLowerCase().includes(searchLower);

        // Filter reads only if reference name doesn't match
        // If reference matches, show ALL reads under it
        const filteredReads =
            searchText && !refMatches
                ? reads.filter((read) => read.readId.toLowerCase().includes(searchLower))
                : reads;

        // Skip reference if no matches at all
        if (searchText && filteredReads.length === 0 && !refMatches) {
            continue;
        }

        // Add reference header
        const isExpanded = expandedReferences.has(referenceName);
        items.push({
            type: 'reference',
            referenceName,
            readCount: filteredReads.length,
            isExpanded,
            indentLevel: 0,
        } as ReferenceGroupItem);

        // Add reads if expanded
        if (isExpanded) {
            for (const read of filteredReads) {
                items.push({
                    ...read,
                    indentLevel: 1,
                });
            }
        }
    }

    return items;
}
