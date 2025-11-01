/**
 * Tests for Read Explorer TreeView
 */

import { describe, it, expect, beforeEach } from '@jest/globals';
import { ReadTreeProvider, ReadItem } from '../squiggy-read-explorer';
import * as vscode from 'vscode';

// Use manual mock for vscode
jest.mock('vscode');

describe('ReadTreeProvider', () => {
    let provider: ReadTreeProvider;

    beforeEach(() => {
        provider = new ReadTreeProvider();
    });

    describe('setReads', () => {
        it('should set reads without reference grouping', async () => {
            const readIds = ['read1', 'read2', 'read3'];
            provider.setReads(readIds);

            const children = await provider.getChildren();
            expect(children).toHaveLength(3);
            expect(children[0].label).toBe('read1');
            expect(children[0].itemType).toBe('read');
        });

        it('should handle empty read list', async () => {
            provider.setReads([]);

            const children = await provider.getChildren();
            expect(children).toHaveLength(0);
        });
    });

    describe('setReadsGrouped', () => {
        it('should group reads by reference', async () => {
            const referenceToReads = new Map<string, string[]>([
                ['chr1', ['read1', 'read2']],
                ['chr2', ['read3', 'read4', 'read5']],
            ]);

            provider.setReadsGrouped(referenceToReads);

            const references = await provider.getChildren();
            expect(references).toHaveLength(2);
            expect(references[0].label).toBe('chr1 (2)');
            expect(references[0].itemType).toBe('reference');
            expect(references[1].label).toBe('chr2 (3)');
        });

        it('should return reads for a reference group', async () => {
            const referenceToReads = new Map<string, string[]>([['chr1', ['read1', 'read2']]]);

            provider.setReadsGrouped(referenceToReads);

            const references = await provider.getChildren();
            const chr1Item = references[0];

            const reads = await provider.getChildren(chr1Item);
            expect(reads).toHaveLength(2);
            expect(reads[0].label).toBe('read1');
            expect(reads[0].itemType).toBe('read');
            expect(reads[1].label).toBe('read2');
        });

        it('should handle empty reference groups', async () => {
            provider.setReadsGrouped(new Map());

            const children = await provider.getChildren();
            expect(children).toHaveLength(0);
        });
    });

    describe('filterReads', () => {
        beforeEach(() => {
            const readIds = ['read_abc_001', 'read_xyz_002', 'read_abc_003'];
            provider.setReads(readIds);
        });

        it('should filter reads by partial match', async () => {
            provider.filterReads('abc');

            const children = await provider.getChildren();
            expect(children).toHaveLength(2);
            expect(children[0].label).toBe('read_abc_001');
            expect(children[1].label).toBe('read_abc_003');
        });

        it('should be case-insensitive', async () => {
            provider.filterReads('ABC');

            const children = await provider.getChildren();
            expect(children).toHaveLength(2);
        });

        it('should show all reads when filter is empty', async () => {
            provider.filterReads('abc');
            provider.filterReads('');

            const children = await provider.getChildren();
            expect(children).toHaveLength(3);
        });

        it('should return empty array when no matches', async () => {
            provider.filterReads('nonexistent');

            const children = await provider.getChildren();
            expect(children).toHaveLength(0);
        });
    });

    describe('filterReads with grouped reads', () => {
        beforeEach(() => {
            const referenceToReads = new Map<string, string[]>([
                ['chr1', ['read_abc_001', 'read_xyz_002']],
                ['chr2', ['read_abc_003', 'read_def_004']],
                ['chrX', ['read_ghi_005']],
            ]);
            provider.setReadsGrouped(referenceToReads);
        });

        it('should filter by reference name and show all its reads', async () => {
            provider.filterReads('chr1');

            const references = await provider.getChildren();
            expect(references).toHaveLength(1);
            expect(references[0].label).toBe('chr1 (2)');

            const reads = await provider.getChildren(references[0]);
            expect(reads).toHaveLength(2);
        });

        it('should filter by read ID and show only matching reads', async () => {
            provider.filterReads('abc');

            const references = await provider.getChildren();
            expect(references).toHaveLength(2); // chr1 and chr2 both have abc reads

            // chr1 should show only read_abc_001
            const chr1Reads = await provider.getChildren(references[0]);
            expect(chr1Reads).toHaveLength(1);
            expect(chr1Reads[0].label).toBe('read_abc_001');

            // chr2 should show only read_abc_003
            const chr2Reads = await provider.getChildren(references[1]);
            expect(chr2Reads).toHaveLength(1);
            expect(chr2Reads[0].label).toBe('read_abc_003');
        });

        it('should filter references that match the search term', async () => {
            provider.filterReads('chrX');

            const references = await provider.getChildren();
            expect(references).toHaveLength(1);
            expect(references[0].label).toBe('chrX (1)');
        });

        it('should hide references with no matching reads', async () => {
            provider.filterReads('read_def');

            const references = await provider.getChildren();
            expect(references).toHaveLength(1);
            expect(references[0].label).toBe('chr2 (1)');
        });
    });

    describe('getTreeItem', () => {
        it('should return the same item', () => {
            const item = new ReadItem(
                'test',
                'read',
                'read123',
                vscode.TreeItemCollapsibleState.None
            );

            const result = provider.getTreeItem(item);
            expect(result).toBe(item);
        });
    });
});
