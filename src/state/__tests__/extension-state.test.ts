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
});
