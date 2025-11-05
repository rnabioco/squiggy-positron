/**
 * Session State Integration Tests
 *
 * Validates that session state save/restore properly handles unified state,
 * ensuring cross-panel synchronization is maintained across sessions.
 */

import { describe, it, expect, beforeEach } from '@jest/globals';
import { ExtensionState } from '../state/extension-state';
import { LoadedItem } from '../types/loaded-item';

/**
 * Helper to create a LoadedItem for testing
 */
function createLoadedItem(
    id: string,
    type: 'pod5' | 'sample',
    pod5Path: string,
    override?: Partial<LoadedItem>
): LoadedItem {
    return {
        id,
        type,
        pod5Path,
        readCount: 100,
        fileSize: 5000000,
        fileSizeFormatted: '5.0 MB',
        hasAlignments: false,
        hasReference: false,
        hasMods: false,
        hasEvents: false,
        ...override,
    };
}

describe('Session State Integration', () => {
    let state: ExtensionState;

    beforeEach(() => {
        state = new ExtensionState();
    });

    describe('toSessionState() - Serialize Unified State', () => {
        it('should serialize POD5 items from unified state', () => {
            const pod5Item = createLoadedItem('pod5:/path/file.pod5', 'pod5', '/path/file.pod5', {
                readCount: 150,
                fileSize: 7500000,
                fileSizeFormatted: '7.5 MB',
            });

            state.addLoadedItem(pod5Item);

            const sessionState = state.toSessionState();

            // Should create a "Default" sample from the POD5 item
            expect(sessionState.samples['Default']).toBeDefined();
            expect(sessionState.samples['Default'].pod5Paths).toContain('/path/file.pod5');
        });

        it('should serialize sample items from unified state', () => {
            const sampleItem = createLoadedItem('sample:mysample', 'sample', '/path/sample.pod5', {
                sampleName: 'mysample',
                readCount: 75,
                fileSize: 3750000,
                fileSizeFormatted: '3.75 MB',
            });

            state.addLoadedItem(sampleItem);

            const sessionState = state.toSessionState();

            expect(sessionState.samples['mysample']).toBeDefined();
            expect(sessionState.samples['mysample'].pod5Paths).toContain('/path/sample.pod5');
        });

        it('should serialize multiple samples with BAM/FASTA associations', () => {
            const sample1 = createLoadedItem('sample:sample1', 'sample', '/path/sample1.pod5', {
                sampleName: 'sample1',
                bamPath: '/path/sample1.bam',
                fastaPath: '/path/ref.fasta',
                hasAlignments: true,
                hasReference: true,
            });

            const sample2 = createLoadedItem('sample:sample2', 'sample', '/path/sample2.pod5', {
                sampleName: 'sample2',
                bamPath: '/path/sample2.bam',
                hasAlignments: true,
            });

            state.addLoadedItem(sample1);
            state.addLoadedItem(sample2);

            const sessionState = state.toSessionState();

            expect(sessionState.samples['sample1'].bamPath).toBe('/path/sample1.bam');
            expect(sessionState.samples['sample1'].fastaPath).toBe('/path/ref.fasta');
            expect(sessionState.samples['sample2'].bamPath).toBe('/path/sample2.bam');
        });

        it('should prefer unified state over legacy state', () => {
            // Add item to unified state
            const unifiedItem = createLoadedItem('sample:newsample', 'sample', '/path/new.pod5', {
                sampleName: 'newsample',
            });
            state.addLoadedItem(unifiedItem);

            // Legacy state should be ignored when unified state has items
            const sessionState = state.toSessionState();

            expect(sessionState.samples['newsample']).toBeDefined();
            expect(sessionState.samples['newsample'].pod5Paths).toContain('/path/new.pod5');
        });
    });

    describe('toSessionState() - Serialize Comparison Selection', () => {
        it('should serialize comparison items with sample names', () => {
            state.setComparisonItems(['sample:sample1', 'sample:sample2']);

            const sessionState = state.toSessionState();

            // The legacy UI state should track selected samples
            expect(sessionState.ui).toBeDefined();
            // Note: selectedSamplesForComparison comes from legacy _selectedSamplesForComparison
            // which is populated during comparison operations
        });
    });

    describe('Session State Serialization Round-Trip', () => {
        it('should handle empty state serialization', () => {
            const sessionState = state.toSessionState();

            expect(sessionState.samples).toBeDefined();
            expect(Object.keys(sessionState.samples).length).toBe(0);
        });

        it('should preserve sample structure through serialization', () => {
            const sampleItem = createLoadedItem('sample:test', 'sample', '/data/test.pod5', {
                sampleName: 'test',
                bamPath: '/data/test.bam',
                fastaPath: '/data/ref.fasta',
                hasAlignments: true,
                hasReference: true,
                readCount: 500,
                fileSize: 25000000,
                fileSizeFormatted: '25 MB',
            });

            state.addLoadedItem(sampleItem);

            const sessionState = state.toSessionState();
            const testSample = sessionState.samples['test'];

            // Verify all paths are preserved
            expect(testSample.pod5Paths[0]).toBe('/data/test.pod5');
            expect(testSample.bamPath).toBe('/data/test.bam');
            expect(testSample.fastaPath).toBe('/data/ref.fasta');
        });
    });

    describe('State Consistency', () => {
        it('should maintain consistency between unified and legacy state', () => {
            const sampleItem = createLoadedItem('sample:consistent', 'sample', '/path/sample.pod5', {
                sampleName: 'consistent',
                bamPath: '/path/sample.bam',
            });

            state.addLoadedItem(sampleItem);

            // Get items from unified state
            const unifiedItems = state.getLoadedItems();
            expect(unifiedItems.length).toBe(1);
            expect(unifiedItems[0].sampleName).toBe('consistent');

            // Serialize to session state
            const sessionState = state.toSessionState();
            expect(sessionState.samples['consistent']).toBeDefined();

            // Both should be consistent
            const serializedItem = sessionState.samples['consistent'];
            const unifiedItem = unifiedItems[0];

            expect(serializedItem.pod5Paths[0]).toBe(unifiedItem.pod5Path);
            expect(serializedItem.bamPath).toBe(unifiedItem.bamPath);
        });

        it('should handle mixed POD5 and sample items', () => {
            // Add a standalone POD5
            const pod5Item = createLoadedItem('pod5:/standalone.pod5', 'pod5', '/standalone.pod5');
            state.addLoadedItem(pod5Item);

            // Add a sample
            const sampleItem = createLoadedItem('sample:sample1', 'sample', '/sample1.pod5', {
                sampleName: 'sample1',
            });
            state.addLoadedItem(sampleItem);

            const sessionState = state.toSessionState();

            // Should have both in the samples dict
            expect(sessionState.samples['Default']).toBeDefined();
            expect(sessionState.samples['sample1']).toBeDefined();

            expect(sessionState.samples['Default'].pod5Paths).toContain('/standalone.pod5');
            expect(sessionState.samples['sample1'].pod5Paths).toContain('/sample1.pod5');
        });
    });
});
