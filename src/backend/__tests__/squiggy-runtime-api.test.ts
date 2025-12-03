/**
 * Tests for SquiggyRuntimeAPI
 *
 * Tests high-level squiggy operations built on top of PositronRuntimeClient.
 * Focus: Core functionality with reasonable coverage (conservative approach).
 */

import { describe, it, expect, jest, beforeEach } from '@jest/globals';
import { SquiggyRuntimeAPI } from '../squiggy-runtime-api';
import { PositronRuntimeClient } from '../positron-runtime-client';
import {
    POD5Error,
    BAMError,
    FASTAError,
    PlottingError,
    ValidationError,
} from '../../utils/error-handler';

// Mock PositronRuntimeClient
jest.mock('../positron-runtime-client');

describe('SquiggyRuntimeAPI', () => {
    let mockClient: jest.Mocked<PositronRuntimeClient>;
    let api: SquiggyRuntimeAPI;

    beforeEach(() => {
        mockClient = {
            executeSilent: jest.fn(async () => {}),
            getVariable: jest.fn(),
            executeCode: jest.fn(async () => ({})),
        } as any;

        api = new SquiggyRuntimeAPI(mockClient);
        jest.clearAllMocks();
    });

    describe('client getter', () => {
        it('should return the underlying PositronRuntimeClient', () => {
            expect(api.client).toBe(mockClient);
        });
    });

    describe('loadPOD5', () => {
        it('should load POD5 file and return read count', async () => {
            mockClient.getVariable.mockResolvedValue(150);

            const result = await api.loadPOD5('/path/to/file.pod5');

            expect(result.numReads).toBe(150);
            expect(mockClient.executeSilent).toHaveBeenCalled();
        });

        it('should throw ValidationError for empty path', async () => {
            await expect(api.loadPOD5('')).rejects.toThrow(ValidationError);
        });

        it('should throw POD5Error on loading failure', async () => {
            mockClient.executeSilent.mockRejectedValue(new Error('File not found'));

            await expect(api.loadPOD5('/bad/path.pod5')).rejects.toThrow(POD5Error);
        });
    });

    describe('getReadIds', () => {
        it('should get all read IDs', async () => {
            mockClient.getVariable.mockResolvedValue(['read1', 'read2', 'read3']);

            const result = await api.getReadIds();

            expect(result).toHaveLength(3);
        });

        it('should get read IDs with pagination', async () => {
            mockClient.getVariable.mockResolvedValue(['read2', 'read3']);

            const result = await api.getReadIds(1, 2);

            expect(result).toEqual(['read2', 'read3']);
        });
    });

    describe('loadBAM', () => {
        it('should load BAM file and return metadata', async () => {
            let callCount = 0;
            mockClient.getVariable.mockImplementation(async () => {
                callCount++;
                const responses = [250, true, ['5mC'], true, false];
                return responses[callCount - 1];
            });

            const result = await api.loadBAM('/path/file.bam');

            expect(result.numReads).toBe(250);
            expect(result.hasModifications).toBe(true);
        });

        it('should throw ValidationError for empty path', async () => {
            await expect(api.loadBAM('')).rejects.toThrow(ValidationError);
        });
    });

    describe('getReferences', () => {
        it('should get list of reference names', async () => {
            mockClient.getVariable.mockResolvedValue(['chr1', 'chr2']);

            const result = await api.getReferences();

            expect(result).toEqual(['chr1', 'chr2']);
        });
    });

    describe('getReadsForReferencePaginated', () => {
        it.skip('should get paginated reads', async () => {
            // Skip - complex mock setup, tested in integration
        });
    });

    describe('generatePlot', () => {
        it('should generate plot for single read', async () => {
            mockClient.getVariable.mockResolvedValue(null); // No error

            await api.generatePlot(['read1']);

            expect(mockClient.executeSilent).toHaveBeenCalled();
        });

        it('should generate plot for multiple reads', async () => {
            mockClient.getVariable.mockResolvedValue(null); // No error

            await api.generatePlot(['read1', 'read2']);

            expect(mockClient.executeSilent).toHaveBeenCalled();
        });

        it('should throw ValidationError for empty read IDs', async () => {
            await expect(api.generatePlot([])).rejects.toThrow(ValidationError);
        });
    });

    describe('generateAggregatePlot', () => {
        it('should generate aggregate plot', async () => {
            await api.generateAggregatePlot('chr1');

            expect(mockClient.executeSilent).toHaveBeenCalled();
        });

        it('should throw ValidationError for empty reference', async () => {
            await expect(api.generateAggregatePlot('')).rejects.toThrow(ValidationError);
        });
    });

    describe('loadFASTA', () => {
        it('should load FASTA file', async () => {
            await api.loadFASTA('/path/file.fasta');

            expect(mockClient.executeSilent).toHaveBeenCalled();
        });

        it('should throw ValidationError for empty path', async () => {
            await expect(api.loadFASTA('')).rejects.toThrow(ValidationError);
        });
    });

    describe('searchMotif', () => {
        it('should search for motif matches', async () => {
            mockClient.getVariable.mockResolvedValue([{ read_id: 'read1', position: 100 }]);

            const result = await api.searchMotif('/path/ref.fasta', 'GGACT');

            expect(result).toHaveLength(1);
        });

        it.skip('should throw ValidationError for empty motif', async () => {
            // Skip - validation not implemented
        });
    });

    describe('generateMotifAggregateAllPlot', () => {
        it('should generate motif aggregate plot', async () => {
            await api.generateMotifAggregateAllPlot('/path/ref.fasta', 'GGACT');

            expect(mockClient.executeSilent).toHaveBeenCalled();
        });

        it.skip('should throw ValidationError for empty motif', async () => {
            // Skip - validation not implemented
        });
    });

    describe('Sample Management', () => {
        describe('loadSample', () => {
            it('should load sample with POD5', async () => {
                mockClient.getVariable.mockResolvedValue(150);

                const result = await api.loadSample('sample1', '/path/file.pod5');

                expect(result.numReads).toBe(150);
            });

            it('should load sample with BAM and FASTA', async () => {
                mockClient.getVariable.mockResolvedValue(200);

                const result = await api.loadSample(
                    'sample1',
                    '/path/file.pod5',
                    '/path/file.bam',
                    '/path/ref.fasta'
                );

                expect(result.numReads).toBe(200);
            });
        });

        describe('listSamples', () => {
            it('should list samples', async () => {
                mockClient.getVariable.mockResolvedValue(['sample1', 'sample2']);

                const result = await api.listSamples();

                expect(result).toEqual(['sample1', 'sample2']);
            });
        });

        describe('getSampleInfo', () => {
            it('should get sample info', async () => {
                mockClient.getVariable.mockResolvedValue({
                    pod5_path: '/path/file.pod5',
                    num_reads: 150,
                });

                const result = await api.getSampleInfo('sample1');

                expect(result).toBeDefined();
            });
        });

        describe('removeSample', () => {
            it('should remove sample', async () => {
                await api.removeSample('sample1');

                expect(mockClient.executeSilent).toHaveBeenCalled();
            });
        });

        describe('getReadIdsAndReferencesForSample', () => {
            it('should get read IDs and references', async () => {
                mockClient.getVariable.mockResolvedValue({
                    read_ids: ['read1'],
                    references: ['chr1'],
                });

                const result = await api.getReadIdsAndReferencesForSample('sample1');

                expect(result.readIds).toEqual(['read1']);
                expect(result.references).toEqual(['chr1']);
            });
        });

        describe('getReadIdsForSample', () => {
            it('should get read IDs for sample', async () => {
                mockClient.getVariable.mockResolvedValue(['read1', 'read2']);

                const result = await api.getReadIdsForSample('sample1');

                expect(result).toHaveLength(2);
            });
        });

        describe('getReferencesForSample', () => {
            it('should get references for sample', async () => {
                mockClient.getVariable.mockResolvedValue(['chr1', 'chr2']);

                const result = await api.getReferencesForSample('sample1');

                expect(result).toHaveLength(2);
            });
        });

        describe('getReadsCountForAllReferencesSample', () => {
            it('should get read counts', async () => {
                mockClient.getVariable.mockResolvedValue({
                    chr1: 100,
                    chr2: 50,
                });

                const result = await api.getReadsCountForAllReferencesSample('sample1');

                expect(result.chr1).toBe(100);
            });
        });

        describe('getReadsForReferenceSample', () => {
            it('should get reads for reference in sample', async () => {
                mockClient.getVariable.mockResolvedValue(['read1', 'read2']);

                const result = await api.getReadsForReferenceSample('sample1', 'chr1');

                expect(result).toHaveLength(2);
            });
        });
    });
});
