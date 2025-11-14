/**
 * Unit tests for FileLoadingService
 *
 * Tests file loading operations, metadata extraction,
 * and error handling for File Panel and Samples Panel integration.
 */

import { FileLoadingService } from '../file-loading-service';
import { ExtensionState } from '../../state/extension-state';
import { SquiggyRuntimeAPI } from '../../backend/squiggy-runtime-api';
import * as fs from 'fs/promises';

// Mock dependencies
jest.mock('../../state/extension-state');
jest.mock('../../backend/squiggy-runtime-api');
jest.mock('fs/promises');

describe('FileLoadingService', () => {
    let service: FileLoadingService;
    let mockState: jest.Mocked<ExtensionState>;
    let mockAPI: jest.Mocked<SquiggyRuntimeAPI>;

    beforeEach(() => {
        // Setup mocks
        mockAPI = {
            loadPOD5: jest.fn(),
            loadBAM: jest.fn(),
            loadFASTA: jest.fn(),
            loadSample: jest.fn(),
        } as any;

        mockState = {
            squiggyAPI: mockAPI,
            ensureBackgroundKernel: jest.fn().mockResolvedValue(mockAPI),
        } as any;

        service = new FileLoadingService(mockState);

        // Clear all mocks before each test
        jest.clearAllMocks();

        // Re-setup ensureBackgroundKernel after clearAllMocks
        mockState.ensureBackgroundKernel = jest.fn().mockResolvedValue(mockAPI);
    });

    describe('loadFile()', () => {
        it('should dispatch to loadPOD5 when fileType is pod5', async () => {
            const filePath = '/test/file.pod5';

            // Mock fs.stat
            (fs.stat as jest.Mock).mockResolvedValue({
                size: 1024 * 1024, // 1 MB
                mtime: new Date(),
                mode: 0o644,
            });

            // Mock API result
            mockAPI.loadPOD5.mockResolvedValue({ numReads: 100 });

            const result = await service.loadFile(filePath, 'pod5');

            expect(result.fileType).toBe('pod5');
            expect(result.success).toBe(true);
            expect(mockAPI.loadPOD5).toHaveBeenCalledWith(filePath);
        });

        it('should dispatch to loadBAM when fileType is bam', async () => {
            const filePath = '/test/file.bam';

            // Mock fs.stat
            (fs.stat as jest.Mock).mockResolvedValue({
                size: 2 * 1024 * 1024, // 2 MB
                mtime: new Date(),
                mode: 0o644,
            });

            // Mock API result
            mockAPI.loadBAM.mockResolvedValue({
                numReads: 100,
                hasModifications: true,
                modificationTypes: [],
                hasProbabilities: false,
                hasEventAlignment: false,
            });

            const result = await service.loadFile(filePath, 'bam');

            expect(result.fileType).toBe('bam');
            expect(result.success).toBe(true);
            expect(mockAPI.loadBAM).toHaveBeenCalledWith(filePath);
        });

        it('should dispatch to loadFASTA when fileType is fasta', async () => {
            const filePath = '/test/file.fasta';

            // Mock fs.stat
            (fs.stat as jest.Mock).mockResolvedValue({
                size: 512 * 1024, // 512 KB
                mtime: new Date(),
                mode: 0o644,
            });

            // Mock API result (loadFASTA returns void)
            mockAPI.loadFASTA.mockResolvedValue(undefined);

            const result = await service.loadFile(filePath, 'fasta');

            expect(result.fileType).toBe('fasta');
            expect(result.success).toBe(true);
            expect(mockAPI.loadFASTA).toHaveBeenCalledWith(filePath);
        });
    });

    describe('loadPOD5()', () => {
        it('should return success result with correct metadata', async () => {
            const filePath = '/test/reads.pod5';

            // Mock fs.stat
            (fs.stat as jest.Mock).mockResolvedValue({
                size: 10 * 1024 * 1024, // 10 MB
                mtime: new Date('2025-01-01'),
                mode: 0o644,
            });

            // Mock API result
            mockAPI.loadPOD5.mockResolvedValue({ numReads: 1234 });

            const result = await service.loadFile(filePath, 'pod5');

            expect(result.success).toBe(true);
            expect(result.fileSize).toBe(10 * 1024 * 1024);
            expect(result.fileSizeFormatted).toBe('10.0 MB');
            expect((result as any).readCount).toBe(1234);
            expect(result.error).toBeNull();
        });

        it('should return error result when API fails', async () => {
            const filePath = '/test/missing.pod5';

            // Mock API failure
            mockAPI.loadPOD5.mockRejectedValue(new Error('File not found'));

            const result = await service.loadFile(filePath, 'pod5');

            expect(result.success).toBe(false);
            expect(result.error).toContain('Failed to load POD5');
            expect((result as any).readCount).toBe(0);
        });

        it('should return error when API is not initialized', async () => {
            const stateWithoutAPI = {
                squiggyAPI: undefined,
                ensureBackgroundKernel: jest
                    .fn()
                    .mockRejectedValue(new Error('API not initialized')),
            } as any;
            service = new FileLoadingService(stateWithoutAPI);

            const result = await service.loadFile('/test/file.pod5', 'pod5');

            expect(result.success).toBe(false);
            expect(result.error).toContain('API not initialized');
        });
    });

    describe('loadBAM()', () => {
        it('should return success result with alignment metadata', async () => {
            const filePath = '/test/alignments.bam';

            // Mock fs.stat
            (fs.stat as jest.Mock).mockResolvedValue({
                size: 5 * 1024 * 1024, // 5 MB
                mtime: new Date(),
                mode: 0o644,
            });

            // Mock API result
            mockAPI.loadBAM.mockResolvedValue({
                numReads: 1234,
                hasModifications: true,
                modificationTypes: ['5mC', '6mA'],
                hasProbabilities: true,
                hasEventAlignment: true,
            });

            const result = await service.loadFile(filePath, 'bam');

            expect(result.success).toBe(true);
            expect(result.fileSizeFormatted).toBe('5.0 MB');
            expect((result as any).readCount).toBe(1234);
            expect((result as any).hasModifications).toBe(true);
            expect((result as any).hasEventAlignment).toBe(true);
        });

        it('should handle missing optional BAM properties', async () => {
            const filePath = '/test/minimal.bam';

            // Mock fs.stat
            (fs.stat as jest.Mock).mockResolvedValue({
                size: 1024,
                mtime: new Date(),
                mode: 0o644,
            });

            // Mock API result with minimal data
            mockAPI.loadBAM.mockResolvedValue({
                numReads: 100,
                hasModifications: false,
                modificationTypes: [],
                hasProbabilities: false,
                hasEventAlignment: false,
            });

            const result = await service.loadFile(filePath, 'bam');

            expect(result.success).toBe(true);
            expect((result as any).numReferences).toBe(0);
            expect((result as any).hasModifications).toBe(false);
            expect((result as any).hasEventAlignment).toBe(false);
        });
    });

    describe('loadFASTA()', () => {
        it('should return success result for FASTA file', async () => {
            const filePath = '/test/reference.fasta';

            // Mock fs.stat
            (fs.stat as jest.Mock).mockResolvedValue({
                size: 3 * 1024 * 1024, // 3 MB
                mtime: new Date(),
                mode: 0o644,
            });

            // Mock API result (loadFASTA returns void)
            mockAPI.loadFASTA.mockResolvedValue(undefined);

            const result = await service.loadFile(filePath, 'fasta');

            expect(result.success).toBe(true);
            expect(result.fileSizeFormatted).toBe('3.0 MB');
            expect(result.error).toBeNull();
        });

        it('should return error when FASTA loading not supported', async () => {
            const apiWithoutFASTA = { ...mockAPI, loadFASTA: undefined } as any;
            const stateWithoutFASTA = {
                squiggyAPI: apiWithoutFASTA,
                ensureBackgroundKernel: jest.fn().mockResolvedValue(apiWithoutFASTA),
            } as any;
            service = new FileLoadingService(stateWithoutFASTA);

            const result = await service.loadFile('/test/ref.fasta', 'fasta');

            expect(result.success).toBe(false);
            expect(result.error).toContain('Failed to load FASTA');
            expect(result.error).toMatch(/is not a function|does not support FASTA/);
        });
    });

    describe('loadSample()', () => {
        it('should load complete sample with POD5, BAM, and FASTA', async () => {
            const pod5Path = '/test/reads.pod5';
            const bamPath = '/test/alignments.bam';
            const fastaPath = '/test/reference.fasta';

            // Mock fs.stat calls
            (fs.stat as jest.Mock).mockResolvedValue({
                size: 1024 * 1024,
                mtime: new Date(),
                mode: 0o644,
            });

            // Mock API calls
            mockAPI.loadPOD5.mockResolvedValue({ numReads: 100 });
            mockAPI.loadBAM.mockResolvedValue({
                numReads: 100,
                hasModifications: false,
                modificationTypes: [],
                hasProbabilities: false,
                hasEventAlignment: true,
            });
            mockAPI.loadFASTA.mockResolvedValue(undefined);

            const result = await service.loadSample(pod5Path, bamPath, fastaPath);

            expect(result.pod5Result.success).toBe(true);
            expect(result.bamResult?.success).toBe(true);
            expect(result.fastaResult?.success).toBe(true);
            expect(mockAPI.loadPOD5).toHaveBeenCalledWith(pod5Path);
            expect(mockAPI.loadBAM).toHaveBeenCalledWith(bamPath);
            expect(mockAPI.loadFASTA).toHaveBeenCalledWith(fastaPath);
        });

        it('should load sample with only POD5', async () => {
            const pod5Path = '/test/reads.pod5';

            // Mock fs.stat
            (fs.stat as jest.Mock).mockResolvedValue({
                size: 1024 * 1024,
                mtime: new Date(),
                mode: 0o644,
            });

            // Mock API
            mockAPI.loadPOD5.mockResolvedValue({ numReads: 100 });

            const result = await service.loadSample(pod5Path);

            expect(result.pod5Result.success).toBe(true);
            expect(result.bamResult).toBeUndefined();
            expect(result.fastaResult).toBeUndefined();
        });

        it('should handle BAM loading failure gracefully', async () => {
            const pod5Path = '/test/reads.pod5';
            const bamPath = '/test/missing.bam';

            // Mock fs.stat
            (fs.stat as jest.Mock).mockResolvedValue({
                size: 1024 * 1024,
                mtime: new Date(),
                mode: 0o644,
            });

            // Mock API
            mockAPI.loadPOD5.mockResolvedValue({ numReads: 100 });
            mockAPI.loadBAM.mockRejectedValue(new Error('BAM not found'));

            const result = await service.loadSample(pod5Path, bamPath);

            expect(result.pod5Result.success).toBe(true);
            expect(result.bamResult).toBeUndefined();
        });
    });

    describe('File size formatting', () => {
        it('should format bytes correctly', async () => {
            // Mock fs.stat to return specific sizes
            const testCases = [
                { bytes: 512, expected: '512 B' },
                { bytes: 1024, expected: '1.0 KB' },
                { bytes: 1024 * 1024, expected: '1.0 MB' },
                { bytes: 10 * 1024 * 1024, expected: '10.0 MB' },
                { bytes: 1024 * 1024 * 1024, expected: '1.0 GB' },
            ];

            for (const testCase of testCases) {
                (fs.stat as jest.Mock).mockResolvedValueOnce({
                    size: testCase.bytes,
                    mtime: new Date(),
                    mode: 0o644,
                });

                mockAPI.loadPOD5.mockResolvedValueOnce({ numReads: 100 });

                const result = await service.loadFile('/test/file.pod5', 'pod5');

                expect(result.fileSizeFormatted).toBe(testCase.expected);
            }
        });
    });

    describe('Error handling', () => {
        it('should catch and report fs.stat errors', async () => {
            // Mock fs.stat to fail
            (fs.stat as jest.Mock).mockRejectedValue(new Error('Permission denied'));

            mockAPI.loadPOD5.mockResolvedValue({ numReads: 100 });

            const result = await service.loadFile('/test/file.pod5', 'pod5');

            expect(result.success).toBe(true); // API load succeeded
            expect(result.fileSizeFormatted).toBe('0 B'); // But metadata extraction failed gracefully
        });

        it('should provide helpful error messages on failure', async () => {
            const apiError = new Error('Invalid POD5 file format');
            mockAPI.loadPOD5.mockRejectedValue(apiError);

            const result = await service.loadFile('/test/bad.pod5', 'pod5');

            expect(result.success).toBe(false);
            expect(result.error).toContain('Failed to load POD5');
            expect(result.error).toContain('Invalid POD5 file format');
        });
    });
});
