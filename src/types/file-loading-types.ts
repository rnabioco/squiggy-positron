/**
 * Type definitions for FileLoadingService
 *
 * Defines the interfaces for file loading operations, metadata extraction,
 * and result types used by File Panel and Samples Panel.
 */

/**
 * Metadata extracted from a file
 */
export interface FileMetadata {
    fileSize: number; // File size in bytes
    fileSizeFormatted: string; // Human-readable size (e.g., "2.5 MB")
    lastModified: Date; // File modification time
    isReadable: boolean; // Whether file is readable
}

/**
 * Base result interface for file loading operations
 */
export interface BaseFileLoadResult {
    success: boolean;
    filePath: string;
    fileType: 'pod5' | 'bam' | 'fasta';
    error: string | null;
}

/**
 * Result from loading a POD5 file
 */
export interface POD5LoadResult extends BaseFileLoadResult {
    fileType: 'pod5';
    fileSize: number;
    fileSizeFormatted: string;
    readCount: number;
}

/**
 * Result from loading a BAM file
 */
export interface BAMLoadResult extends BaseFileLoadResult {
    fileType: 'bam';
    fileSize: number;
    fileSizeFormatted: string;
    readCount: number;
    numReferences: number;
    hasModifications: boolean;
    hasEventAlignment: boolean;
}

/**
 * Result from loading a FASTA file
 */
export interface FASTALoadResult extends BaseFileLoadResult {
    fileType: 'fasta';
    fileSize: number;
    fileSizeFormatted: string;
}

/**
 * Union type for all file load results
 */
export type FileLoadResult = POD5LoadResult | BAMLoadResult | FASTALoadResult;

/**
 * Result of attempting to load multiple files (e.g., for a sample)
 */
export interface SampleLoadResult {
    pod5Result: POD5LoadResult;
    bamResult?: BAMLoadResult;
    fastaResult?: FASTALoadResult;
}
