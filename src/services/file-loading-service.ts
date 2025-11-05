/**
 * FileLoadingService - Centralized file loading operations
 *
 * Eliminates code duplication between File Panel and Samples Panel
 * by providing a single source of truth for file loading logic.
 *
 * Both panels use this service for:
 * - Loading POD5, BAM, and FASTA files
 * - Extracting file metadata (size, read counts, etc.)
 * - Consistent error handling
 * - Normalized result formats
 */

import { ExtensionState } from '../state/extension-state';
import {
    FileMetadata,
    POD5LoadResult,
    BAMLoadResult,
    FASTALoadResult,
    FileLoadResult,
    SampleLoadResult,
} from '../types/file-loading-types';

export class FileLoadingService {
    constructor(private state: ExtensionState) {}

    /**
     * Load a file and return normalized metadata
     * Shared entry point for both File Panel and Samples Panel
     *
     * @param filePath - Path to the file
     * @param fileType - Type of file to load ('pod5', 'bam', or 'fasta')
     * @returns FileLoadResult with success status and metadata
     */
    async loadFile(filePath: string, fileType: 'pod5' | 'bam' | 'fasta'): Promise<FileLoadResult> {
        switch (fileType) {
            case 'pod5':
                return this.loadPOD5(filePath);
            case 'bam':
                return this.loadBAM(filePath);
            case 'fasta':
                return this.loadFASTA(filePath);
        }
    }

    /**
     * Load a sample (POD5 + optional BAM + optional FASTA)
     * Used by Samples Panel to load complete sample bundles
     *
     * @param pod5Path - Required POD5 file path
     * @param bamPath - Optional BAM alignment file
     * @param fastaPath - Optional FASTA reference sequence
     * @returns SampleLoadResult with all file load results
     */
    async loadSample(
        pod5Path: string,
        bamPath?: string,
        fastaPath?: string
    ): Promise<SampleLoadResult> {
        // Load POD5 (required)
        const pod5Result = await this.loadPOD5(pod5Path);

        // Load BAM if provided
        let bamResult: BAMLoadResult | undefined;
        if (bamPath) {
            const result = await this.loadBAM(bamPath);
            if (result.success) {
                bamResult = result as BAMLoadResult;
            }
        }

        // Load FASTA if provided
        let fastaResult: FASTALoadResult | undefined;
        if (fastaPath) {
            const result = await this.loadFASTA(fastaPath);
            if (result.success) {
                fastaResult = result as FASTALoadResult;
            }
        }

        return {
            pod5Result: pod5Result as POD5LoadResult,
            bamResult,
            fastaResult,
        };
    }

    /**
     * Load a POD5 file with full metadata extraction
     * Returns normalized result usable by both panels
     *
     * @param filePath - Path to POD5 file
     * @returns POD5LoadResult with file info and read count
     */
    private async loadPOD5(filePath: string): Promise<POD5LoadResult> {
        try {
            // Verify API is available
            if (!this.state.squiggyAPI) {
                return {
                    success: false,
                    filePath,
                    fileType: 'pod5',
                    fileSize: 0,
                    fileSizeFormatted: '0 B',
                    readCount: 0,
                    error: 'Squiggy API not initialized',
                };
            }

            // Load via API
            const result = await this.state.squiggyAPI.loadPOD5(filePath);

            // Extract metadata
            const metadata = await this.extractFileMetadata(filePath);

            return {
                success: true,
                filePath,
                fileType: 'pod5',
                fileSize: metadata.fileSize,
                fileSizeFormatted: metadata.fileSizeFormatted,
                readCount: result.numReads,
                error: null,
            };
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            return {
                success: false,
                filePath,
                fileType: 'pod5',
                fileSize: 0,
                fileSizeFormatted: '0 B',
                readCount: 0,
                error: `Failed to load POD5: ${errorMessage}`,
            };
        }
    }

    /**
     * Load a BAM file with full metadata extraction
     * Returns aligned read count and reference information
     *
     * @param filePath - Path to BAM file
     * @returns BAMLoadResult with alignment metadata
     */
    private async loadBAM(filePath: string): Promise<BAMLoadResult> {
        try {
            // Verify API is available
            if (!this.state.squiggyAPI) {
                return {
                    success: false,
                    filePath,
                    fileType: 'bam',
                    fileSize: 0,
                    fileSizeFormatted: '0 B',
                    readCount: 0,
                    numReferences: 0,
                    hasModifications: false,
                    hasEventAlignment: false,
                    error: 'Squiggy API not initialized',
                };
            }

            // Load via API
            const result = await this.state.squiggyAPI.loadBAM(filePath);

            // Extract metadata
            const metadata = await this.extractFileMetadata(filePath);

            return {
                success: true,
                filePath,
                fileType: 'bam',
                fileSize: metadata.fileSize,
                fileSizeFormatted: metadata.fileSizeFormatted,
                readCount: result.numReads,
                numReferences: result.references?.length || 0,
                hasModifications: result.hasModifications || false,
                hasEventAlignment: result.hasEventAlignment || false,
                error: null,
            };
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            return {
                success: false,
                filePath,
                fileType: 'bam',
                fileSize: 0,
                fileSizeFormatted: '0 B',
                readCount: 0,
                numReferences: 0,
                hasModifications: false,
                hasEventAlignment: false,
                error: `Failed to load BAM: ${errorMessage}`,
            };
        }
    }

    /**
     * Load a FASTA file
     * Currently minimal validation - just confirms file is readable
     *
     * @param filePath - Path to FASTA file
     * @returns FASTALoadResult with file metadata
     */
    private async loadFASTA(filePath: string): Promise<FASTALoadResult> {
        try {
            // Verify API has FASTA loading
            if (!this.state.squiggyAPI?.loadFASTA) {
                return {
                    success: false,
                    filePath,
                    fileType: 'fasta',
                    fileSize: 0,
                    fileSizeFormatted: '0 B',
                    error: 'Squiggy API does not support FASTA loading',
                };
            }

            // Load via API
            await this.state.squiggyAPI.loadFASTA(filePath);

            // Extract metadata
            const metadata = await this.extractFileMetadata(filePath);

            return {
                success: true,
                filePath,
                fileType: 'fasta',
                fileSize: metadata.fileSize,
                fileSizeFormatted: metadata.fileSizeFormatted,
                error: null,
            };
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            return {
                success: false,
                filePath,
                fileType: 'fasta',
                fileSize: 0,
                fileSizeFormatted: '0 B',
                error: `Failed to load FASTA: ${errorMessage}`,
            };
        }
    }

    /**
     * Shared metadata extraction (file size, modification time, readability)
     * Centralizes file system operations used by all load methods
     *
     * @param filePath - Path to file
     * @returns FileMetadata with size and other file info
     */
    private async extractFileMetadata(filePath: string): Promise<FileMetadata> {
        const fs = await import('fs/promises');

        try {
            const stats = await fs.stat(filePath);

            return {
                fileSize: stats.size,
                fileSizeFormatted: this.formatFileSize(stats.size),
                lastModified: stats.mtime,
                isReadable: (stats.mode & 0o400) !== 0,
            };
        } catch (error) {
            // Return minimal metadata if stat fails
            return {
                fileSize: 0,
                fileSizeFormatted: '0 B',
                lastModified: new Date(),
                isReadable: false,
            };
        }
    }

    /**
     * Shared utility: format bytes to human-readable size
     * Used by all file loading methods for consistent formatting
     *
     * @param bytes - File size in bytes
     * @returns Human-readable size string (e.g., "2.5 MB")
     */
    private formatFileSize(bytes: number): string {
        const units = ['B', 'KB', 'MB', 'GB', 'TB'];
        let size = bytes;
        let unitIdx = 0;

        while (size >= 1024 && unitIdx < units.length - 1) {
            size /= 1024;
            unitIdx++;
        }

        // Format with 1 decimal place, or 0 decimals for bytes
        const precision = unitIdx === 0 ? 0 : 1;
        return `${size.toFixed(precision)} ${units[unitIdx]}`;
    }
}
