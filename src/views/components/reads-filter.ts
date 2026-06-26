/**
 * Pure filtering helpers for the reads panel.
 *
 * Extracted from squiggy-reads-core.tsx so the search/filter logic can be unit
 * tested without the React/webview module dependencies.
 */

import { ReadListItem, ReadItem } from '../../types/squiggy-reads-types';

/**
 * Filter the reads list by search text.
 *
 * Two modes:
 * - 'reference': match reference names (grouped mode) or a read's reference (flat mode).
 * - 'read': match read IDs. In grouped mode, references with matching reads are
 *   auto-expanded and the matching reads are listed beneath their header so results
 *   are shown in reference context (#78). In grouped mode `items` only contains
 *   reference headers, so the matching reads are sourced from `referenceToReads`
 *   and never double-listed.
 */
export function filterItems(
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
                    // Auto-expand the reference and list the matching reads beneath
                    // it, so read-ID search shows results in their reference context
                    // rather than behind a collapsed header (#78). In grouped mode
                    // `items` only contains reference headers, so this does not
                    // double-list reads.
                    filtered.push({
                        ...item,
                        readCount: matchingReads.length,
                        isExpanded: true,
                    });
                    for (const read of matchingReads) {
                        filtered.push({ ...read, indentLevel: 1 });
                    }
                }
            }
        } else {
            // For flat list (POD5-only)
            if (searchMode === 'reference') {
                // Search in reference name if available
                if (item.referenceName && item.referenceName.toLowerCase().includes(searchLower)) {
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
