/**
 * Cross-Panel Synchronization Tests
 *
 * Validates that File Panel and Samples Panel respond correctly to unified state changes
 * through the event-driven architecture.
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

describe('Cross-Panel Synchronization', () => {
    let state: ExtensionState;

    beforeEach(() => {
        state = new ExtensionState();
    });

    describe('Unified State - Event Emission', () => {
        it('should emit onLoadedItemsChanged when item is added', (done) => {
            const unsub = state.onLoadedItemsChanged((items) => {
                expect(items.length).toBe(1);
                expect(items[0].type).toBe('pod5');
                unsub.dispose();
                done();
            });

            const item = createLoadedItem('pod5:/path/to/file.pod5', 'pod5', '/path/to/file.pod5');
            state.addLoadedItem(item);
        });

        it('should emit onLoadedItemsChanged with all items when new item added', (done) => {
            const item1 = createLoadedItem('pod5:/path/file1.pod5', 'pod5', '/path/file1.pod5');
            const item2 = createLoadedItem('sample:sample1', 'sample', '/path/sample1.pod5', {
                sampleName: 'sample1',
                readCount: 50,
                fileSize: 2500000,
                fileSizeFormatted: '2.5 MB',
            });

            state.addLoadedItem(item1);

            let callCount = 0;
            const unsub = state.onLoadedItemsChanged((items) => {
                callCount++;
                if (callCount === 1) {
                    // First event is after adding item2
                    expect(items.length).toBe(2);
                    expect(items.map((i) => i.id).sort()).toEqual([
                        'pod5:/path/file1.pod5',
                        'sample:sample1',
                    ]);
                    unsub.dispose();
                    done();
                }
            });

            state.addLoadedItem(item2);
        });

        it('should emit onLoadedItemsChanged when item is removed', (done) => {
            const item = createLoadedItem('pod5:/path/file.pod5', 'pod5', '/path/file.pod5');
            state.addLoadedItem(item);

            // Subscribe after item is added to only catch the removal event
            const unsub = state.onLoadedItemsChanged((items) => {
                // This should be called with empty items after removal
                expect(items.length).toBe(0);
                unsub.dispose();
                done();
            });

            // Remove the item - should trigger the event
            state.removeLoadedItem('pod5:/path/file.pod5');
        });

        it('should emit onComparisonChanged when comparison items are set', (done) => {
            const unsub = state.onComparisonChanged((ids) => {
                expect(ids).toContain('sample:sample1');
                expect(ids).toContain('sample:sample2');
                unsub.dispose();
                done();
            });

            state.setComparisonItems(['sample:sample1', 'sample:sample2']);
        });

        it('should emit onComparisonChanged when item added to comparison', (done) => {
            state.setComparisonItems(['sample:sample1']);

            // Subscribe after initial set to only catch the addition event
            const unsub = state.onComparisonChanged((ids) => {
                expect(ids).toContain('sample:sample1');
                expect(ids).toContain('sample:sample2');
                unsub.dispose();
                done();
            });

            state.addToComparison('sample:sample2');
        });
    });

    describe('File Panel - Unified State Subscription', () => {
        it('should receive notification when POD5 file added', (done) => {
            const items: LoadedItem[] = [];

            const unsub = state.onLoadedItemsChanged((updatedItems) => {
                items.push(...updatedItems);
            });

            const pod5Item = createLoadedItem(
                'pod5:/path/to/file.pod5',
                'pod5',
                '/path/to/file.pod5'
            );
            state.addLoadedItem(pod5Item);

            // Simulate panel filtering for display
            setTimeout(() => {
                const pod5Items = items.filter((item) => item.type === 'pod5');
                expect(pod5Items.length).toBe(1);
                expect(pod5Items[0].pod5Path).toBe('/path/to/file.pod5');

                unsub.dispose();
                done();
            }, 10);
        });

        it('should track both POD5 and sample items', (done) => {
            const items: LoadedItem[] = [];

            const unsub = state.onLoadedItemsChanged((updatedItems) => {
                // Replace items array with latest
                items.length = 0;
                items.push(...updatedItems);
            });

            const pod5Item = createLoadedItem('pod5:/path/file.pod5', 'pod5', '/path/file.pod5');
            const sampleItem = createLoadedItem('sample:mysample', 'sample', '/path/sample.pod5', {
                sampleName: 'mysample',
                readCount: 50,
                fileSize: 2500000,
                fileSizeFormatted: '2.5 MB',
            });

            state.addLoadedItem(pod5Item);
            setTimeout(() => {
                state.addLoadedItem(sampleItem);
            }, 10);

            setTimeout(() => {
                const allItems = items;
                expect(allItems.length).toBe(2);
                expect(allItems.some((i) => i.type === 'pod5')).toBe(true);
                expect(allItems.some((i) => i.type === 'sample')).toBe(true);

                unsub.dispose();
                done();
            }, 30);
        });

        it('should handle BAM file association with POD5', (done) => {
            const items: LoadedItem[] = [];

            const unsub = state.onLoadedItemsChanged((updatedItems) => {
                items.length = 0;
                items.push(...updatedItems);
            });

            const pod5Item = createLoadedItem('pod5:/path/file.pod5', 'pod5', '/path/file.pod5');
            state.addLoadedItem(pod5Item);

            setTimeout(() => {
                // Update POD5 item with BAM association
                const updatedItem = createLoadedItem(
                    'pod5:/path/file.pod5',
                    'pod5',
                    '/path/file.pod5',
                    {
                        bamPath: '/path/file.bam',
                        hasAlignments: true,
                    }
                );
                state.addLoadedItem(updatedItem);
            }, 10);

            setTimeout(() => {
                const pod5Item = items.find((i) => i.type === 'pod5');
                expect(pod5Item?.bamPath).toBe('/path/file.bam');
                expect(pod5Item?.hasAlignments).toBe(true);

                unsub.dispose();
                done();
            }, 30);
        });
    });

    describe('Samples Panel - Unified State Subscription', () => {
        it('should receive filtered sample items only', (done) => {
            const sampleItems: LoadedItem[] = [];

            const unsub = state.onLoadedItemsChanged((updatedItems) => {
                // Filter for samples only (as Samples Panel does)
                const samples = updatedItems.filter((item) => item.type === 'sample');
                sampleItems.length = 0;
                sampleItems.push(...samples);
            });

            const pod5Item = createLoadedItem('pod5:/path/file.pod5', 'pod5', '/path/file.pod5');
            const sampleItem = createLoadedItem('sample:mysample', 'sample', '/path/sample.pod5', {
                sampleName: 'mysample',
                readCount: 50,
                fileSize: 2500000,
                fileSizeFormatted: '2.5 MB',
            });

            state.addLoadedItem(pod5Item);
            setTimeout(() => {
                state.addLoadedItem(sampleItem);
            }, 10);

            setTimeout(() => {
                // Should only see samples
                expect(sampleItems.length).toBe(1);
                expect(sampleItems[0].sampleName).toBe('mysample');

                unsub.dispose();
                done();
            }, 30);
        });

        it('should receive comparison selection changes', (done) => {
            const sampleItem1 = createLoadedItem('sample:sample1', 'sample', '/path/sample1.pod5', {
                sampleName: 'sample1',
                readCount: 50,
                fileSize: 2500000,
                fileSizeFormatted: '2.5 MB',
            });

            const sampleItem2 = createLoadedItem('sample:sample2', 'sample', '/path/sample2.pod5', {
                sampleName: 'sample2',
                readCount: 75,
                fileSize: 3750000,
                fileSizeFormatted: '3.75 MB',
            });

            state.addLoadedItem(sampleItem1);
            state.addLoadedItem(sampleItem2);

            const selectedSampleNames: Set<string> = new Set();

            const unsub = state.onComparisonChanged((ids) => {
                // Extract sample names from "sample:" prefixed IDs
                const names = ids
                    .filter((id) => id.startsWith('sample:'))
                    .map((id) => id.substring(7));

                selectedSampleNames.clear();
                names.forEach((name) => selectedSampleNames.add(name));
            });

            setTimeout(() => {
                state.setComparisonItems(['sample:sample1', 'sample:sample2']);
            }, 10);

            setTimeout(() => {
                expect(selectedSampleNames.has('sample1')).toBe(true);
                expect(selectedSampleNames.has('sample2')).toBe(true);

                unsub.dispose();
                done();
            }, 30);
        });

        it('should handle toggling sample selection', (done) => {
            const sampleItem = createLoadedItem('sample:sample1', 'sample', '/path/sample1.pod5', {
                sampleName: 'sample1',
                readCount: 50,
                fileSize: 2500000,
                fileSizeFormatted: '2.5 MB',
            });

            state.addLoadedItem(sampleItem);

            const selectedSampleNames: Set<string> = new Set();

            const unsub = state.onComparisonChanged((ids) => {
                const names = ids
                    .filter((id) => id.startsWith('sample:'))
                    .map((id) => id.substring(7));

                selectedSampleNames.clear();
                names.forEach((name) => selectedSampleNames.add(name));
            });

            setTimeout(() => {
                state.addToComparison('sample:sample1');
            }, 10);

            setTimeout(() => {
                expect(selectedSampleNames.has('sample1')).toBe(true);
                state.removeFromComparison('sample:sample1');
            }, 20);

            setTimeout(() => {
                expect(selectedSampleNames.has('sample1')).toBe(false);
                unsub.dispose();
                done();
            }, 40);
        });
    });

    describe('Multi-Panel Coordination', () => {
        it('should keep both panels in sync when items added', (done) => {
            const fileItems: LoadedItem[] = [];
            const sampleItems: LoadedItem[] = [];

            // File Panel subscription
            const fileUnsub = state.onLoadedItemsChanged((items) => {
                fileItems.length = 0;
                fileItems.push(...items);
            });

            // Samples Panel subscription
            const sampleUnsub = state.onLoadedItemsChanged((items) => {
                const samples = items.filter((i) => i.type === 'sample');
                sampleItems.length = 0;
                sampleItems.push(...samples);
            });

            const pod5Item = createLoadedItem('pod5:/path/file.pod5', 'pod5', '/path/file.pod5');
            const sampleItem = createLoadedItem('sample:mysample', 'sample', '/path/sample.pod5', {
                sampleName: 'mysample',
                readCount: 50,
                fileSize: 2500000,
                fileSizeFormatted: '2.5 MB',
            });

            state.addLoadedItem(pod5Item);
            state.addLoadedItem(sampleItem);

            setTimeout(() => {
                // File Panel sees all items
                expect(fileItems.length).toBe(2);

                // Samples Panel sees only samples
                expect(sampleItems.length).toBe(1);
                expect(sampleItems[0].sampleName).toBe('mysample');

                fileUnsub.dispose();
                sampleUnsub.dispose();
                done();
            }, 30);
        });

        it('should coordinate removal of samples across panels', (done) => {
            const fileItems: LoadedItem[] = [];

            const unsub = state.onLoadedItemsChanged((items) => {
                fileItems.length = 0;
                fileItems.push(...items);
            });

            const sampleItem = createLoadedItem('sample:mysample', 'sample', '/path/sample.pod5', {
                sampleName: 'mysample',
                readCount: 50,
                fileSize: 2500000,
                fileSizeFormatted: '2.5 MB',
            });

            state.addLoadedItem(sampleItem);

            setTimeout(() => {
                expect(fileItems.length).toBe(1);
                state.removeLoadedItem('sample:mysample');
            }, 10);

            setTimeout(() => {
                expect(fileItems.length).toBe(0);
                unsub.dispose();
                done();
            }, 30);
        });
    });

    describe('Unified State - Query Methods', () => {
        it('should retrieve loaded items', () => {
            const pod5Item = createLoadedItem('pod5:/path/file.pod5', 'pod5', '/path/file.pod5');
            const sampleItem = createLoadedItem('sample:mysample', 'sample', '/path/sample.pod5', {
                sampleName: 'mysample',
                readCount: 50,
                fileSize: 2500000,
                fileSizeFormatted: '2.5 MB',
            });

            state.addLoadedItem(pod5Item);
            state.addLoadedItem(sampleItem);

            const items = state.getLoadedItems();
            expect(items.length).toBe(2);
            expect(items.some((i) => i.type === 'pod5')).toBe(true);
            expect(items.some((i) => i.type === 'sample')).toBe(true);
        });

        it('should retrieve specific loaded item by id', () => {
            const pod5Item = createLoadedItem('pod5:/path/file.pod5', 'pod5', '/path/file.pod5');
            state.addLoadedItem(pod5Item);

            const retrieved = state.getLoadedItem('pod5:/path/file.pod5');
            expect(retrieved).toBeDefined();
            expect(retrieved?.type).toBe('pod5');
            expect(retrieved?.pod5Path).toBe('/path/file.pod5');
        });

        it('should return undefined for non-existent item', () => {
            const retrieved = state.getLoadedItem('pod5:/nonexistent/file.pod5');
            expect(retrieved).toBeUndefined();
        });

        it('should retrieve comparison items', () => {
            state.setComparisonItems(['sample:sample1', 'sample:sample2']);

            const comparison = state.getComparisonItems();
            expect(comparison.length).toBe(2);
            expect(comparison).toContain('sample:sample1');
            expect(comparison).toContain('sample:sample2');
        });
    });

    describe('Edge Cases', () => {
        it('should handle adding duplicate items (should replace)', (done) => {
            const items: LoadedItem[] = [];

            const unsub = state.onLoadedItemsChanged((updatedItems) => {
                items.length = 0;
                items.push(...updatedItems);
            });

            const item1 = createLoadedItem('pod5:/path/file.pod5', 'pod5', '/path/file.pod5');
            const item2 = createLoadedItem('pod5:/path/file.pod5', 'pod5', '/path/file.pod5', {
                readCount: 200,
                fileSize: 10000000,
                fileSizeFormatted: '10.0 MB',
            });

            state.addLoadedItem(item1);
            setTimeout(() => {
                state.addLoadedItem(item2);
            }, 10);

            setTimeout(() => {
                expect(items.length).toBe(1);
                expect(items[0].readCount).toBe(200);

                unsub.dispose();
                done();
            }, 30);
        });

        it('should handle clearing all items', (done) => {
            const items: LoadedItem[] = [];

            const unsub = state.onLoadedItemsChanged((updatedItems) => {
                items.length = 0;
                items.push(...updatedItems);
            });

            const item1 = createLoadedItem('pod5:/path/file1.pod5', 'pod5', '/path/file1.pod5');
            const item2 = createLoadedItem('sample:sample1', 'sample', '/path/sample.pod5', {
                sampleName: 'sample1',
                readCount: 50,
                fileSize: 2500000,
                fileSizeFormatted: '2.5 MB',
            });

            state.addLoadedItem(item1);
            state.addLoadedItem(item2);

            setTimeout(() => {
                expect(items.length).toBe(2);
                state.clearLoadedItems();
            }, 10);

            setTimeout(() => {
                expect(items.length).toBe(0);
                unsub.dispose();
                done();
            }, 30);
        });

        it('should handle removing non-existent items gracefully', () => {
            const item = createLoadedItem('pod5:/path/file.pod5', 'pod5', '/path/file.pod5');
            state.addLoadedItem(item);

            // Should not throw
            state.removeLoadedItem('sample:nonexistent');
            expect(state.getLoadedItems().length).toBe(1);
        });
    });
});
