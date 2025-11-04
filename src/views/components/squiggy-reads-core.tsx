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
        searchMode: 'reference',
        filteredItems: [],
        selectedReadIds: new Set<string>(),
        focusedIndex: null,
        expandedReferences: new Set<string>(),
        sortBy: 'name',
        sortOrder: 'asc',
        nameColumnWidth: CONSTANTS.DEFAULT_NAME_WIDTH,
        detailsColumnWidth: CONSTANTS.DEFAULT_DETAILS_WIDTH,
    });

    // Loading state
    const [isLoading, setIsLoading] = React.useState(false);
    const [loadingMessage, setLoadingMessage] = React.useState('');

    // Debounced search
    const [debouncedSearchText, setDebouncedSearchText] = React.useState('');
    const searchTimeoutRef = React.useRef<number | undefined>();

    // Store reference to reads data (for expansion)
    const referenceToReadsRef = React.useRef<Map<string, ReadItem[]>>(new Map());

    // Client-side cache for expanded references
    const referenceCacheRef = React.useRef<
        Map<string, { reads: ReadItem[]; fullyLoaded: boolean; offset: number }>
    >(new Map());

    // Handle messages from extension
    React.useEffect(() => {
        const messageHandler = (event: MessageEvent) => {
            const message = event.data;

            switch (message.type) {
                case 'updateReads':
                    // Handle unified message from backend
                    if (message.groupedByReference) {
                        // For grouped reads, message.reads contains reference headers
                        // Populate the referenceToReads map for expansion logic
                        // Only update if referenceToReads has data (don't overwrite with empty)
                        if (message.referenceToReads && message.referenceToReads.length > 0) {
                            referenceToReadsRef.current = new Map(message.referenceToReads);
                        }

                        setState((prev) => {
                            let filteredItems: ReadListItem[];

                            // If we have full reference data, rebuild from scratch
                            if (referenceToReadsRef.current.size > 0) {
                                filteredItems = rebuildItemsList(
                                    referenceToReadsRef.current,
                                    prev.expandedReferences,
                                    prev.searchText,
                                    prev.sortBy,
                                    prev.sortOrder,
                                    prev.searchMode
                                );
                            } else {
                                // In lazy-loading mode, use incoming reference headers with current sort
                                filteredItems = sortReferenceHeaders(
                                    message.reads,
                                    prev.sortBy,
                                    prev.sortOrder
                                );
                            }

                            return {
                                ...prev,
                                items: message.reads,
                                hasReferences: true,
                                totalReadCount: message.referenceToReads
                                    ? message.referenceToReads.reduce(
                                          (sum: number, [_, reads]: [string, ReadItem[]]) =>
                                              sum + reads.length,
                                          0
                                      )
                                    : 0,
                                filteredItems,
                            };
                        });
                    } else {
                        // Flat list of reads
                        setState((prev) => ({
                            ...prev,
                            items: message.reads,
                            hasReferences: false,
                            totalReadCount: message.reads.length,
                            filteredItems: message.reads,
                        }));
                    }
                    break;
                case 'setReads':
                    handleSetReads(message.items);
                    break;
                case 'setReadsGrouped':
                    handleSetReadsGrouped(message.items, message.referenceToReads);
                    break;
                case 'setReferencesOnly':
                    handleSetReferencesOnly(message.references);
                    break;
                case 'appendReads':
                    handleAppendReads(message.reads);
                    break;
                case 'setReadsForReference':
                    handleSetReadsForReference(
                        message.referenceName,
                        message.reads,
                        message.offset,
                        message.totalCount
                    );
                    break;
                case 'setLoading':
                    setIsLoading(message.isLoading);
                    setLoadingMessage(message.message || '');
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

        // Send ready message to request initial data
        vscode.postMessage({ type: 'ready' });

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

        setState((prev) => {
            // Rebuild items with sorting applied
            const filteredItems = rebuildItemsList(
                referenceToReadsRef.current,
                new Set<string>(), // No references expanded initially
                '', // No search text initially
                prev.sortBy,
                prev.sortOrder,
                prev.searchMode
            );

            return {
                ...prev,
                items,
                hasReferences: true,
                totalReadCount: referenceToReads.reduce((sum, [_, reads]) => sum + reads.length, 0),
                filteredItems,
                selectedReadIds: new Set(),
                focusedIndex: null,
                expandedReferences: new Set(),
            };
        });
    };

    const handleSetReferencesOnly = (
        references: { referenceName: string; readCount: number }[]
    ) => {
        // Initialize with reference headers only (lazy loading mode)
        const items: ReadListItem[] = references.map((ref) => ({
            type: 'reference' as const,
            referenceName: ref.referenceName,
            readCount: ref.readCount,
            isExpanded: false,
            indentLevel: 0,
        }));

        // Clear cache for fresh load
        referenceCacheRef.current.clear();
        referenceToReadsRef.current.clear();

        setState((prev) => ({
            ...prev,
            items,
            hasReferences: true,
            totalReadCount: references.reduce((sum, ref) => sum + ref.readCount, 0),
            filteredItems: items,
            selectedReadIds: new Set(),
            focusedIndex: null,
            expandedReferences: new Set(),
        }));
    };

    const handleAppendReads = (newReads: ReadItem[]) => {
        // Append reads to flat list (POD5 pagination)
        setState((prev) => {
            const updatedItems = [...prev.items, ...newReads];
            return {
                ...prev,
                items: updatedItems,
                totalReadCount: prev.totalReadCount + newReads.length,
                filteredItems: filterItems(
                    updatedItems,
                    prev.searchText,
                    referenceToReadsRef.current,
                    prev.searchMode
                ),
            };
        });
    };

    const handleSetReadsForReference = (
        referenceName: string,
        reads: ReadItem[],
        offset: number,
        totalCount: number
    ) => {
        // Update cache with fetched reads
        const cached = referenceCacheRef.current.get(referenceName) || {
            reads: [],
            fullyLoaded: false,
            offset: 0,
        };

        // Merge reads (handle pagination)
        const allReads = [...cached.reads, ...reads];
        const fullyLoaded = allReads.length >= totalCount;

        referenceCacheRef.current.set(referenceName, {
            reads: allReads,
            fullyLoaded,
            offset: offset + reads.length,
        });

        // Also update referenceToReadsRef for compatibility
        referenceToReadsRef.current.set(referenceName, allReads);

        // Rebuild items list if this reference is expanded
        setState((prev) => {
            if (prev.expandedReferences.has(referenceName)) {
                const newItems = rebuildItemsList(
                    referenceToReadsRef.current,
                    prev.expandedReferences,
                    prev.searchText,
                    prev.sortBy,
                    prev.sortOrder,
                    prev.searchMode
                );
                return {
                    ...prev,
                    filteredItems: newItems,
                };
            }
            return prev;
        });
    };

    const handleSearch = (searchText: string) => {
        // Clear existing timeout
        if (searchTimeoutRef.current) {
            window.clearTimeout(searchTimeoutRef.current);
        }

        // Update search text immediately for input responsiveness
        setState((prev) => ({ ...prev, searchText }));

        // Debounce the actual filtering (300ms)
        searchTimeoutRef.current = window.setTimeout(() => {
            setDebouncedSearchText(searchText);
        }, 300);
    };

    // Effect to perform filtering when debounced search changes
    React.useEffect(() => {
        setState((prev) => {
            const filtered = filterItems(
                prev.items,
                debouncedSearchText,
                referenceToReadsRef.current,
                prev.searchMode
            );
            return {
                ...prev,
                filteredItems: filtered,
            };
        });
    }, [debouncedSearchText, state.searchMode]);

    const handleToggleReference = (referenceName: string) => {
        setState((prev) => {
            const expandedReferences = new Set(prev.expandedReferences);
            const isExpanded = expandedReferences.has(referenceName);

            if (isExpanded) {
                // Collapse - just update UI state
                expandedReferences.delete(referenceName);
            } else {
                // Expand - check cache first
                expandedReferences.add(referenceName);

                const cached = referenceCacheRef.current.get(referenceName);
                if (!cached) {
                    // Not in cache - request from backend
                    vscode.postMessage({
                        type: 'expandReference',
                        referenceName,
                        offset: 0,
                        limit: 500,
                    });
                    // Return early - will rebuild when data arrives
                    return {
                        ...prev,
                        expandedReferences,
                    };
                }
            }

            // Rebuild items list with expansion (either collapsing or using cached data)
            const newItems = rebuildItemsList(
                referenceToReadsRef.current,
                expandedReferences,
                prev.searchText,
                prev.sortBy,
                prev.sortOrder,
                prev.searchMode
            );

            return {
                ...prev,
                expandedReferences,
                filteredItems: newItems,
            };
        });
    };

    const handleSort = (column: 'name' | 'reads') => {
        setState((prev) => {
            // If clicking same column, toggle order; otherwise switch column (start with asc)
            const newSortBy = column;
            const newSortOrder =
                prev.sortBy === column && prev.sortOrder === 'asc' ? 'desc' : 'asc';

            let newItems: ReadListItem[];

            // If we have full reference data, rebuild from scratch
            if (referenceToReadsRef.current.size > 0) {
                newItems = rebuildItemsList(
                    referenceToReadsRef.current,
                    prev.expandedReferences,
                    prev.searchText,
                    newSortBy,
                    newSortOrder,
                    prev.searchMode
                );
            } else {
                // In lazy-loading mode, just re-sort the existing reference headers
                newItems = sortReferenceHeaders(prev.filteredItems, newSortBy, newSortOrder);
            }

            return {
                ...prev,
                sortBy: newSortBy,
                sortOrder: newSortOrder,
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

    const handleToggleSearchMode = () => {
        setState((prev) => ({
            ...prev,
            searchMode: prev.searchMode === 'reference' ? 'read' : 'reference',
        }));
    };

    return (
        <div className="reads-core-container">
            {/* Loading overlay */}
            {isLoading && (
                <div className="loading-overlay">
                    <div className="loading-spinner"></div>
                    {loadingMessage && <div className="loading-message">{loadingMessage}</div>}
                </div>
            )}

            {/* Search box */}
            <div className="reads-search-box">
                <button
                    className={`reads-search-mode-toggle ${state.searchMode === 'reference' ? 'active' : ''}`}
                    onClick={handleToggleSearchMode}
                    title={`Search mode: ${state.searchMode === 'reference' ? 'Reference names' : 'Read IDs'} (click to toggle)`}
                >
                    {state.searchMode === 'reference' ? 'Ref' : 'Read'}
                </button>
                <input
                    type="text"
                    placeholder={`Search ${state.searchMode === 'reference' ? 'references' : 'reads'}...`}
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
                hasReferences={state.hasReferences}
                sortBy={state.sortBy}
                sortOrder={state.sortOrder}
                onPlotRead={handlePlotRead}
                onPlotAggregate={handlePlotAggregate}
                onSelectRead={handleSelectRead}
                onToggleReference={handleToggleReference}
                onSearch={handleSearch}
                onLoadMore={handleLoadMore}
                onUpdateColumnWidths={handleUpdateColumnWidths}
                onSort={handleSort}
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
    referenceToReads: Map<string, ReadItem[]>,
    searchMode: 'reference' | 'read'
): ReadListItem[] {
    if (!searchText) {
        return items;
    }

    const searchLower = searchText.toLowerCase();
    const filtered: ReadListItem[] = [];

    for (const item of items) {
        if (item.type === 'reference') {
            const refMatches = item.referenceName.toLowerCase().includes(searchLower);

            // In reference mode, only match reference names
            if (searchMode === 'reference') {
                if (refMatches) {
                    filtered.push(item);
                }
            } else {
                // In read mode, check individual reads
                const reads = referenceToReads.get(item.referenceName) || [];
                const matchingReads = reads.filter((read) =>
                    read.readId.toLowerCase().includes(searchLower)
                );

                if (matchingReads.length > 0) {
                    filtered.push({
                        ...item,
                        readCount: matchingReads.length,
                    });
                }
            }
        } else {
            // For flat list (POD5-only)
            if (searchMode === 'reference') {
                // Search in reference name if available
                if (
                    item.referenceName &&
                    item.referenceName.toLowerCase().includes(searchLower)
                ) {
                    filtered.push(item);
                }
            } else {
                // Search in read ID
                if (item.readId.toLowerCase().includes(searchLower)) {
                    filtered.push(item);
                }
            }
        }
    }

    return filtered;
}

/**
 * Sort reference headers in place (for lazy-loading mode)
 * Preserves expanded state and child reads
 */
function sortReferenceHeaders(
    items: ReadListItem[],
    sortBy: 'name' | 'reads',
    sortOrder: 'asc' | 'desc'
): ReadListItem[] {
    const result: ReadListItem[] = [];
    const referenceBlocks: Array<{ header: ReferenceGroupItem; reads: ReadItem[] }> = [];

    // Group items into reference blocks (header + its expanded reads)
    let currentBlock: { header: ReferenceGroupItem; reads: ReadItem[] } | null = null;

    for (const item of items) {
        if (item.type === 'reference') {
            // Save previous block if exists
            if (currentBlock) {
                referenceBlocks.push(currentBlock);
            }
            // Start new block
            currentBlock = { header: item, reads: [] };
        } else {
            // Add read to current block
            if (currentBlock) {
                currentBlock.reads.push(item);
            }
        }
    }

    // Don't forget the last block
    if (currentBlock) {
        referenceBlocks.push(currentBlock);
    }

    // Sort the blocks
    referenceBlocks.sort((a, b) => {
        let comparison = 0;

        if (sortBy === 'reads') {
            // Sort by read count
            comparison = a.header.readCount - b.header.readCount;
        } else {
            // Sort by name
            comparison = a.header.referenceName.localeCompare(b.header.referenceName);
        }

        return sortOrder === 'asc' ? comparison : -comparison;
    });

    // Flatten back to list
    for (const block of referenceBlocks) {
        result.push(block.header);
        result.push(...block.reads);
    }

    return result;
}

/**
 * Rebuild items list with expansion state
 */
function rebuildItemsList(
    referenceToReads: Map<string, ReadItem[]>,
    expandedReferences: Set<string>,
    searchText: string,
    sortBy: 'name' | 'reads' = 'name',
    sortOrder: 'asc' | 'desc' = 'asc',
    searchMode: 'reference' | 'read' = 'reference'
): ReadListItem[] {
    const items: ReadListItem[] = [];
    const searchLower = searchText.toLowerCase();

    // Convert Map to array and sort
    let references = Array.from(referenceToReads.entries());

    // Apply sorting
    references.sort(([refA, readsA], [refB, readsB]) => {
        let comparison = 0;

        if (sortBy === 'reads') {
            // Sort by read count
            comparison = readsA.length - readsB.length;
        } else {
            // Sort by name (alphabetical)
            comparison = refA.localeCompare(refB);
        }

        // Apply sort order
        return sortOrder === 'asc' ? comparison : -comparison;
    });

    for (const [referenceName, reads] of references) {
        const refMatches = referenceName.toLowerCase().includes(searchLower);

        let filteredReads: ReadItem[];
        let shouldInclude: boolean;

        if (!searchText) {
            // No search - show all
            filteredReads = reads;
            shouldInclude = true;
        } else if (searchMode === 'reference') {
            // Reference mode - only match reference names
            filteredReads = reads;
            shouldInclude = refMatches;
        } else {
            // Read mode - match read IDs
            filteredReads = reads.filter((read) =>
                read.readId.toLowerCase().includes(searchLower)
            );
            shouldInclude = filteredReads.length > 0;
        }

        // Skip reference if no matches
        if (!shouldInclude) {
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
