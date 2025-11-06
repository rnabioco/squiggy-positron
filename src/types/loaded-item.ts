/**
 * LoadedItem - Unified representation of loaded files/samples
 *
 * This interface represents a single loaded item in the unified state.
 * It can be either a standalone POD5 file or a multi-sample bundle.
 *
 * Replaces:
 * - Legacy _currentPod5File, _currentBamFile, _currentFastaFile (single-file mode)
 * - Multi-sample _loadedSamples Map (multi-sample mode)
 *
 * With a single unified registry that both File Panel and Samples Panel use.
 */

export interface LoadedItem {
    /**
     * Unique identifier for this item
     *
     * Format: "type:identifier"
     * Examples:
     *   - "pod5:/path/to/file.pod5"
     *   - "sample:sampleName"
     *
     * This allows mixing files and samples in the same registry.
     */
    id: string;

    /**
     * Type of item: standalone file or sample bundle
     */
    type: 'pod5' | 'sample';

    // ========== File Paths ==========

    /**
     * Path to the POD5 file (required for both types)
     */
    pod5Path: string;

    /**
     * Path to the BAM alignment file (optional)
     */
    bamPath?: string;

    /**
     * Path to the FASTA reference sequence (optional)
     */
    fastaPath?: string;

    // ========== Metadata ==========

    /**
     * Number of reads in the POD5 file
     */
    readCount: number;

    /**
     * Whether BAM alignment is loaded
     */
    hasAlignments: boolean;

    /**
     * Whether FASTA reference is loaded
     */
    hasReference: boolean;

    /**
     * Whether BAM file has base modifications (MM/ML tags)
     */
    hasMods: boolean;

    /**
     * Whether BAM file has event alignment (mv tag)
     */
    hasEvents: boolean;

    // ========== File Info ==========

    /**
     * POD5 file size in bytes
     */
    fileSize: number;

    /**
     * Human-readable file size (e.g., "2.5 MB")
     */
    fileSizeFormatted: string;

    // ========== For Samples Only ==========

    /**
     * Human-readable sample name (e.g., "Sample1", "Yeast tRNA")
     * Defined only when type === 'sample'
     */
    sampleName?: string;
}
