/**
 * Tests for Extension State Management
 */

import { describe, it, expect, beforeEach } from '@jest/globals';
import { ExtensionState } from '../extension-state';

describe('ExtensionState', () => {
    let state: ExtensionState;

    beforeEach(() => {
        state = new ExtensionState();
    });

    describe('initialization', () => {
        it('should initialize with default values', () => {
            expect(state.currentPod5File).toBeUndefined();
            expect(state.currentBamFile).toBeUndefined();
            expect(state.squiggyInstallChecked).toBe(false);
            expect(state.squiggyInstallDeclined).toBe(false);
        });

        it('should have undefined backend references initially', () => {
            // Kernel manager and API are undefined until initializeBackends is called
            expect(state.kernelManager).toBeUndefined();
            expect(state.squiggyAPI).toBeUndefined();
        });

        it('should have undefined panel references initially', () => {
            expect(state.readsViewPane).toBeUndefined();
            expect(state.plotOptionsProvider).toBeUndefined();
            expect(state.modificationsProvider).toBeUndefined();
        });
    });

    describe('file path tracking', () => {
        it('should track POD5 file path', () => {
            const filePath = '/path/to/file.pod5';
            state.currentPod5File = filePath;
            expect(state.currentPod5File).toBe(filePath);
        });

        it('should track BAM file path', () => {
            const filePath = '/path/to/file.bam';
            state.currentBamFile = filePath;
            expect(state.currentBamFile).toBe(filePath);
        });

        it('should allow clearing POD5 file', () => {
            state.currentPod5File = '/path/to/file.pod5';
            state.currentPod5File = undefined;
            expect(state.currentPod5File).toBeUndefined();
        });

        it('should allow clearing BAM file', () => {
            state.currentBamFile = '/path/to/file.bam';
            state.currentBamFile = undefined;
            expect(state.currentBamFile).toBeUndefined();
        });

        it('should track current plot read IDs', () => {
            const readIds = ['read1', 'read2', 'read3'];
            state.currentPlotReadIds = readIds;
            expect(state.currentPlotReadIds).toEqual(readIds);
        });
    });

    describe('installation state', () => {
        it('should track squiggy installation check status', () => {
            expect(state.squiggyInstallChecked).toBe(false);
            state.squiggyInstallChecked = true;
            expect(state.squiggyInstallChecked).toBe(true);
        });

        it('should track if user declined installation', () => {
            expect(state.squiggyInstallDeclined).toBe(false);
            state.squiggyInstallDeclined = true;
            expect(state.squiggyInstallDeclined).toBe(true);
        });

        it('should reset declined flag when needed', () => {
            state.squiggyInstallDeclined = true;
            state.squiggyInstallChecked = true;

            // Simulate successful installation
            state.squiggyInstallDeclined = false;

            expect(state.squiggyInstallDeclined).toBe(false);
            expect(state.squiggyInstallChecked).toBe(true);
        });
    });

    describe('state consistency', () => {
        it('should maintain independent file states', () => {
            state.currentPod5File = '/path/pod5.pod5';
            state.currentBamFile = '/path/bam.bam';

            expect(state.currentPod5File).toBe('/path/pod5.pod5');
            expect(state.currentBamFile).toBe('/path/bam.bam');

            state.currentPod5File = undefined;

            expect(state.currentPod5File).toBeUndefined();
            expect(state.currentBamFile).toBe('/path/bam.bam'); // Unchanged
        });

        it('should handle clearing plot read IDs', () => {
            state.currentPlotReadIds = ['read1', 'read2'];
            expect(state.currentPlotReadIds).toEqual(['read1', 'read2']);

            state.currentPlotReadIds = undefined;
            expect(state.currentPlotReadIds).toBeUndefined();
        });
    });

    describe('getter properties', () => {
        it('should expose extensionContext as readonly', () => {
            expect(state.extensionContext).toBeUndefined();
            // Set internally by initializeBackends
        });

        it('should provide getter access to backend instances', () => {
            // All backend instances start as undefined until initializeBackends is called
            expect(state.kernelManager).toBeUndefined();
            expect(state.squiggyAPI).toBeUndefined();
        });

        it('should provide getter access to panel providers', () => {
            // All panel providers start as undefined
            expect(state.readsViewPane).toBeUndefined();
            expect(state.plotOptionsProvider).toBeUndefined();
            expect(state.modificationsProvider).toBeUndefined();
        });
    });

    describe('multi-sample management (single source of truth, #187)', () => {
        const sample = (name: string, overrides = {}) => ({
            sampleId: `sample:${name}`,
            displayName: name,
            pod5Path: `/data/${name}.pod5`,
            bamPath: `/data/${name}.bam`,
            readCount: 100,
            hasBam: true,
            hasFasta: false,
            isLoaded: true,
            ...overrides,
        });

        it('adds and looks up a sample by id and by displayName', () => {
            state.addSample(sample('WT'));

            expect(state.getSample('sample:WT')?.displayName).toBe('WT');
            expect(state.getSample('WT')?.pod5Path).toBe('/data/WT.pod5');
            expect(state.getSample('missing')).toBeUndefined();
        });

        it('returns all sample displayNames in insertion order', () => {
            state.addSample(sample('WT'));
            state.addSample(sample('KO'));
            expect(state.getAllSampleNames()).toEqual(['WT', 'KO']);
        });

        it('exposes samples through the unified loadedItems registry', () => {
            state.addSample(sample('WT'));
            // The sample is a single item in the unified registry, not a second store
            const items = state.getLoadedItems().filter((i) => i.type === 'sample');
            expect(items).toHaveLength(1);
            expect(items[0].id).toBe('sample:WT');
            expect(state.loadedSamples.get('sample:WT')?.hasBam).toBe(true);
        });

        it('merges addSample onto a prior addLoadedItem without losing fields', () => {
            // Caller pattern: addLoadedItem (with file metadata) then addSample
            state.addLoadedItem({
                id: 'sample:WT',
                type: 'sample',
                sampleName: 'WT',
                pod5Path: '/data/WT.pod5',
                readCount: 100,
                hasAlignments: true,
                hasReference: false,
                hasMods: false,
                hasEvents: false,
                fileSize: 1234,
                fileSizeFormatted: '1.2 KB',
            });
            state.addSample(sample('WT', { references: [{ name: 'chr1', readCount: 50 }] }));

            const item = state.getLoadedItem('sample:WT');
            expect(item?.fileSize).toBe(1234); // preserved from addLoadedItem
            expect(item?.references).toEqual([{ name: 'chr1', readCount: 50 }]); // from addSample
            expect(state.getLoadedItems().filter((i) => i.type === 'sample')).toHaveLength(1);
        });

        it('removes a sample by id or by displayName', () => {
            state.addSample(sample('WT'));
            state.addSample(sample('KO'));

            state.removeSample('WT'); // by displayName
            expect(state.getAllSampleNames()).toEqual(['KO']);

            state.removeSample('sample:KO'); // by id
            expect(state.getAllSampleNames()).toEqual([]);
        });

        it('does not surface standalone pod5 items as samples', () => {
            state.addLoadedItem({
                id: 'pod5:/data/x.pod5',
                type: 'pod5',
                pod5Path: '/data/x.pod5',
                readCount: 10,
                hasAlignments: false,
                hasReference: false,
                hasMods: false,
                hasEvents: false,
                fileSize: 0,
                fileSizeFormatted: '',
            });
            expect(state.getAllSampleNames()).toEqual([]);
            expect(state.getSample('pod5:/data/x.pod5')).toBeUndefined();
        });
    });
});
