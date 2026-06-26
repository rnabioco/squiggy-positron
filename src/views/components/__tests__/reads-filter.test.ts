/**
 * Tests for the reads panel filter logic (filterItems).
 */

import { filterItems } from '../reads-filter';
import { ReadItem, ReferenceGroupItem, ReadListItem } from '../../../types/squiggy-reads-types';

const ref = (referenceName: string, readCount: number): ReferenceGroupItem => ({
    type: 'reference',
    referenceName,
    readCount,
    isExpanded: false,
    indentLevel: 0,
});

const read = (readId: string, referenceName?: string): ReadItem => ({
    type: 'read',
    readId,
    referenceName,
    indentLevel: 0,
});

describe('filterItems', () => {
    it('returns items unchanged when search text is empty', () => {
        const items: ReadListItem[] = [ref('chr1', 2)];
        expect(filterItems(items, '', new Map(), 'read')).toBe(items);
    });

    describe('reference mode', () => {
        it('keeps only references whose name matches', () => {
            const items: ReadListItem[] = [ref('chr1', 2), ref('chr2', 3)];
            const result = filterItems(items, 'chr2', new Map(), 'reference');
            expect(result).toEqual([ref('chr2', 3)]);
        });

        it('matches reference of flat read items', () => {
            const items: ReadListItem[] = [read('r1', 'chr1'), read('r2', 'chr2')];
            const result = filterItems(items, 'chr1', new Map(), 'reference');
            expect(result).toEqual([read('r1', 'chr1')]);
        });
    });

    describe('read mode (grouped)', () => {
        const items: ReadListItem[] = [ref('chr1', 2), ref('chr2', 1)];
        const map = new Map<string, ReadItem[]>([
            ['chr1', [read('read_aaa'), read('read_bbb')]],
            ['chr2', [read('read_ccc')]],
        ]);

        it('auto-expands matching references and lists matching reads beneath them (#78)', () => {
            const result = filterItems(items, 'read_aaa', map, 'read');

            // chr1 header (expanded, readCount = 1 match) followed by the matching read.
            expect(result).toHaveLength(2);
            const header = result[0] as ReferenceGroupItem;
            expect(header.type).toBe('reference');
            expect(header.referenceName).toBe('chr1');
            expect(header.isExpanded).toBe(true);
            expect(header.readCount).toBe(1);

            const matched = result[1] as ReadItem;
            expect(matched.type).toBe('read');
            expect(matched.readId).toBe('read_aaa');
            expect(matched.indentLevel).toBe(1);
        });

        it('drops references with no matching reads', () => {
            const result = filterItems(items, 'read_ccc', map, 'read');
            expect(result).toHaveLength(2);
            expect((result[0] as ReferenceGroupItem).referenceName).toBe('chr2');
            expect((result[1] as ReadItem).readId).toBe('read_ccc');
        });

        it('lists every matching read under a reference', () => {
            const result = filterItems(items, 'read_', map, 'read');
            // chr1: header + 2 reads; chr2: header + 1 read = 5 items
            expect(result).toHaveLength(5);
            const readIds = result
                .filter((i): i is ReadItem => i.type === 'read')
                .map((r) => r.readId);
            expect(readIds).toEqual(['read_aaa', 'read_bbb', 'read_ccc']);
        });
    });

    describe('read mode (flat POD5)', () => {
        it('matches read IDs directly', () => {
            const items: ReadListItem[] = [read('alpha'), read('beta')];
            const result = filterItems(items, 'alph', new Map(), 'read');
            expect(result).toEqual([read('alpha')]);
        });
    });
});
