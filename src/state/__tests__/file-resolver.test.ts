/**
 * Tests for FileResolver
 *
 * Tests file resolution logic for missing file paths during session restoration,
 * including user prompts and demo file handling.
 * Target: >80% coverage of file-resolver.ts
 */

import { describe, it, expect, jest, beforeEach } from '@jest/globals';
import * as vscode from 'vscode';
import * as fs from 'fs/promises';
import { FileResolver, FileType } from '../file-resolver';
import { PathResolver } from '../path-resolver';

// Mock fs.promises
jest.mock('fs/promises');

// Mock PathResolver
jest.mock('../path-resolver');

describe('FileResolver', () => {
    let mockExtensionUri: vscode.Uri;

    beforeEach(() => {
        mockExtensionUri = vscode.Uri.file('/mock/extension');
        jest.clearAllMocks();
    });

    describe('resolveFilePath', () => {
        it('should return resolved=true if file exists', async () => {
            (fs.access as any).mockResolvedValue(undefined);

            const result = await FileResolver.resolveFilePath(
                '/path/to/file.pod5',
                'POD5',
                false,
                mockExtensionUri
            );

            expect(result.resolved).toBe(true);
            expect(result.newPath).toBe('/path/to/file.pod5');
            expect(result.error).toBeUndefined();
        });

        it('should show error for missing demo files', async () => {
            (fs.access as any).mockRejectedValue(new Error('File not found'));
            (PathResolver.isExtensionRelative as any).mockReturnValue(false);

            const result = await FileResolver.resolveFilePath(
                '/demo/file.pod5',
                'POD5',
                true, // isDemo
                mockExtensionUri
            );

            expect(result.resolved).toBe(false);
            expect(result.error).toContain('Demo file not found');
            expect(result.error).toContain('Extension may need to be reinstalled');
        });

        it('should show error for missing extension-relative files', async () => {
            (fs.access as any).mockRejectedValue(new Error('File not found'));
            (PathResolver.isExtensionRelative as any).mockReturnValue(true);

            const result = await FileResolver.resolveFilePath(
                '/extension/data/file.pod5',
                'POD5',
                false,
                mockExtensionUri
            );

            expect(result.resolved).toBe(false);
            expect(result.error).toContain('Demo file not found');
        });

        it('should prompt user for missing user files', async () => {
            (fs.access as any).mockRejectedValue(new Error('File not found'));
            (PathResolver.isExtensionRelative as any).mockReturnValue(false);
            (vscode.window.showWarningMessage as any).mockResolvedValue('Locate File');
            (vscode.window.showOpenDialog as any).mockResolvedValue([
                vscode.Uri.file('/new/path/file.pod5'),
            ]);

            const result = await FileResolver.resolveFilePath(
                '/old/path/file.pod5',
                'POD5',
                false,
                mockExtensionUri
            );

            expect(result.resolved).toBe(true);
            expect(result.newPath).toBe('/new/path/file.pod5');
            expect(vscode.window.showWarningMessage).toHaveBeenCalled();
            expect(vscode.window.showOpenDialog).toHaveBeenCalled();
        });

        it('should handle user skipping file location', async () => {
            (fs.access as any).mockRejectedValue(new Error('File not found'));
            (PathResolver.isExtensionRelative as any).mockReturnValue(false);
            (vscode.window.showWarningMessage as any).mockResolvedValue('Skip');

            const result = await FileResolver.resolveFilePath(
                '/path/file.pod5',
                'POD5',
                false,
                mockExtensionUri
            );

            expect(result.resolved).toBe(false);
            expect(result.error).toContain('user chose to skip');
        });

        it('should handle user cancelling file picker', async () => {
            (fs.access as any).mockRejectedValue(new Error('File not found'));
            (PathResolver.isExtensionRelative as any).mockReturnValue(false);
            (vscode.window.showWarningMessage as any).mockResolvedValue('Locate File');
            (vscode.window.showOpenDialog as any).mockResolvedValue(undefined);

            const result = await FileResolver.resolveFilePath(
                '/path/file.pod5',
                'POD5',
                false,
                mockExtensionUri
            );

            expect(result.resolved).toBe(false);
            expect(result.error).toContain('User cancelled file selection');
        });

        it('should handle empty file picker result', async () => {
            (fs.access as any).mockRejectedValue(new Error('File not found'));
            (PathResolver.isExtensionRelative as any).mockReturnValue(false);
            (vscode.window.showWarningMessage as any).mockResolvedValue('Locate File');
            (vscode.window.showOpenDialog as any).mockResolvedValue([]);

            const result = await FileResolver.resolveFilePath(
                '/path/file.pod5',
                'POD5',
                false,
                mockExtensionUri
            );

            expect(result.resolved).toBe(false);
            expect(result.error).toContain('User cancelled file selection');
        });

        it('should use correct filters for POD5 files', async () => {
            (fs.access as any).mockRejectedValue(new Error('File not found'));
            (PathResolver.isExtensionRelative as any).mockReturnValue(false);
            (vscode.window.showWarningMessage as any).mockResolvedValue('Locate File');
            (vscode.window.showOpenDialog as any).mockResolvedValue([
                vscode.Uri.file('/new/file.pod5'),
            ]);

            await FileResolver.resolveFilePath('/path/file.pod5', 'POD5', false, mockExtensionUri);

            expect(vscode.window.showOpenDialog).toHaveBeenCalledWith(
                expect.objectContaining({
                    filters: {
                        'POD5 Files': ['pod5'],
                        'All Files': ['*'],
                    },
                })
            );
        });

        it('should use correct filters for BAM files', async () => {
            (fs.access as any).mockRejectedValue(new Error('File not found'));
            (PathResolver.isExtensionRelative as any).mockReturnValue(false);
            (vscode.window.showWarningMessage as any).mockResolvedValue('Locate File');
            (vscode.window.showOpenDialog as any).mockResolvedValue([
                vscode.Uri.file('/new/file.bam'),
            ]);

            await FileResolver.resolveFilePath('/path/file.bam', 'BAM', false, mockExtensionUri);

            expect(vscode.window.showOpenDialog).toHaveBeenCalledWith(
                expect.objectContaining({
                    filters: {
                        'BAM Files': ['bam'],
                        'All Files': ['*'],
                    },
                })
            );
        });

        it('should use correct filters for FASTA files', async () => {
            (fs.access as any).mockRejectedValue(new Error('File not found'));
            (PathResolver.isExtensionRelative as any).mockReturnValue(false);
            (vscode.window.showWarningMessage as any).mockResolvedValue('Locate File');
            (vscode.window.showOpenDialog as any).mockResolvedValue([
                vscode.Uri.file('/new/file.fasta'),
            ]);

            await FileResolver.resolveFilePath(
                '/path/file.fasta',
                'FASTA',
                false,
                mockExtensionUri
            );

            expect(vscode.window.showOpenDialog).toHaveBeenCalledWith(
                expect.objectContaining({
                    filters: {
                        'FASTA Files': ['fasta', 'fa', 'fna'],
                        'All Files': ['*'],
                    },
                })
            );
        });
    });

    describe('resolvePOD5Files', () => {
        it('should resolve all POD5 files', async () => {
            (fs.access as any).mockResolvedValue(undefined);

            const results = await FileResolver.resolvePOD5Files(
                ['/path/file1.pod5', '/path/file2.pod5'],
                false,
                mockExtensionUri
            );

            expect(results).toHaveLength(2);
            expect(results[0].resolved).toBe(true);
            expect(results[0].newPath).toBe('/path/file1.pod5');
            expect(results[1].resolved).toBe(true);
            expect(results[1].newPath).toBe('/path/file2.pod5');
        });

        it('should handle mix of found and missing files', async () => {
            let callCount = 0;
            (fs.access as any).mockImplementation(() => {
                callCount++;
                if (callCount === 1) {
                    return Promise.resolve(); // First file exists
                }
                return Promise.reject(new Error('Not found')); // Second file missing
            });
            (PathResolver.isExtensionRelative as any).mockReturnValue(false);

            const results = await FileResolver.resolvePOD5Files(
                ['/path/file1.pod5', '/path/file2.pod5'],
                true,
                mockExtensionUri
            );

            expect(results).toHaveLength(2);
            expect(results[0].resolved).toBe(true);
            expect(results[1].resolved).toBe(false);
        });

        it('should handle empty array', async () => {
            const results = await FileResolver.resolvePOD5Files([], false, mockExtensionUri);

            expect(results).toHaveLength(0);
        });
    });

    describe('validateSampleFiles', () => {
        it('should return valid=true when all files exist', async () => {
            (fs.access as any).mockResolvedValue(undefined);

            const result = await FileResolver.validateSampleFiles(
                ['/path/file.pod5'],
                '/path/file.bam',
                '/path/file.fasta'
            );

            expect(result.valid).toBe(true);
            expect(result.missingFiles).toHaveLength(0);
        });

        it('should detect missing POD5 files', async () => {
            (fs.access as any).mockRejectedValue(new Error('Not found'));

            const result = await FileResolver.validateSampleFiles(['/path/file.pod5']);

            expect(result.valid).toBe(false);
            expect(result.missingFiles).toContain('/path/file.pod5');
        });

        it('should detect missing BAM file', async () => {
            let callCount = 0;
            (fs.access as any).mockImplementation((path: string) => {
                callCount++;
                if (callCount === 1) {
                    return Promise.resolve(); // POD5 exists
                }
                if (callCount === 2) {
                    return Promise.reject(new Error('Not found')); // BAM missing
                }
                return Promise.resolve(); // BAI exists
            });

            const result = await FileResolver.validateSampleFiles(
                ['/path/file.pod5'],
                '/path/file.bam'
            );

            expect(result.valid).toBe(false);
            expect(result.missingFiles).toContain('/path/file.bam');
        });

        it('should detect missing BAM index file', async () => {
            let callCount = 0;
            (fs.access as any).mockImplementation((path: string) => {
                callCount++;
                if (callCount === 1) {
                    return Promise.resolve(); // POD5 exists
                }
                if (callCount === 2) {
                    return Promise.resolve(); // BAM exists
                }
                return Promise.reject(new Error('Not found')); // BAI missing
            });

            const result = await FileResolver.validateSampleFiles(
                ['/path/file.pod5'],
                '/path/file.bam'
            );

            expect(result.valid).toBe(false);
            expect(result.missingFiles).toContain('/path/file.bam.bai');
        });

        it('should detect missing FASTA file', async () => {
            let callCount = 0;
            (fs.access as any).mockImplementation((path: string) => {
                callCount++;
                if (callCount === 1) {
                    return Promise.resolve(); // POD5 exists
                }
                return Promise.reject(new Error('Not found')); // FASTA missing
            });

            const result = await FileResolver.validateSampleFiles(
                ['/path/file.pod5'],
                undefined,
                '/path/file.fasta'
            );

            expect(result.valid).toBe(false);
            expect(result.missingFiles).toContain('/path/file.fasta');
        });

        it('should handle multiple missing files', async () => {
            (fs.access as any).mockRejectedValue(new Error('Not found'));

            const result = await FileResolver.validateSampleFiles(
                ['/path/file1.pod5', '/path/file2.pod5'],
                '/path/file.bam',
                '/path/file.fasta'
            );

            expect(result.valid).toBe(false);
            expect(result.missingFiles).toHaveLength(5); // 2 POD5 + BAM + BAI + FASTA
        });

        it('should validate with only POD5 files', async () => {
            (fs.access as any).mockResolvedValue(undefined);

            const result = await FileResolver.validateSampleFiles(['/path/file.pod5']);

            expect(result.valid).toBe(true);
            expect(result.missingFiles).toHaveLength(0);
        });

        it('should validate multiple POD5 files', async () => {
            (fs.access as any).mockResolvedValue(undefined);

            const result = await FileResolver.validateSampleFiles([
                '/path/file1.pod5',
                '/path/file2.pod5',
                '/path/file3.pod5',
            ]);

            expect(result.valid).toBe(true);
            expect(result.missingFiles).toHaveLength(0);
        });
    });

    describe('getMissingFilesSummary', () => {
        it('should return success message when no files missing', () => {
            const summary = FileResolver.getMissingFilesSummary([]);

            expect(summary).toBe('All files found');
        });

        it('should list missing file names', () => {
            const summary = FileResolver.getMissingFilesSummary([
                '/path/to/file1.pod5',
                '/path/to/file2.bam',
            ]);

            expect(summary).toContain('Missing 2 file(s)');
            expect(summary).toContain('file1.pod5');
            expect(summary).toContain('file2.bam');
        });

        it('should handle single missing file', () => {
            const summary = FileResolver.getMissingFilesSummary(['/path/file.pod5']);

            expect(summary).toContain('Missing 1 file(s)');
            expect(summary).toContain('file.pod5');
        });

        it('should extract filenames from paths', () => {
            const summary = FileResolver.getMissingFilesSummary([
                '/very/long/path/to/data/file.pod5',
            ]);

            expect(summary).toContain('file.pod5');
            expect(summary).not.toContain('/very/long/path');
        });
    });
});
