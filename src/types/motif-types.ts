/**
 * TypeScript type definitions for motif search functionality
 */

/**
 * Represents a single motif match found in the reference sequence
 */
export interface MotifMatch {
    /** Chromosome/reference sequence name */
    chrom: string;
    /** 0-based genomic position of match start */
    position: number;
    /** Matched sequence (e.g., "GGACA" for DRACH match) */
    sequence: string;
    /** Strand where motif was found */
    strand: '+' | '-';
}

/**
 * Parameters for motif search
 */
export interface MotifSearchParams {
    /** FASTA file path */
    fastaFile: string;
    /** IUPAC motif pattern (e.g., "DRACH", "YGCY") */
    motif: string;
    /** Optional region filter ("chrom:start-end") */
    region?: string;
    /** Strand to search ('+', '-', or 'both') */
    strand?: '+' | '-' | 'both';
}

/**
 * Parameters for motif aggregate plot generation
 */
export interface MotifAggregatePlotParams {
    /** FASTA file path */
    fastaFile: string;
    /** IUPAC motif pattern */
    motif: string;
    /** Which motif match to plot (0-based index) */
    matchIndex: number;
    /** Window size around motif center (±bp) */
    window: number;
    /** Maximum number of reads to aggregate */
    maxReads: number;
    /** Normalization method */
    normalization: string;
    /** Theme */
    theme: string;
}

/**
 * State for motif search panel
 */
export interface MotifSearchState {
    /** Current motif pattern */
    pattern: string;
    /** Search results */
    matches: MotifMatch[];
    /** Index of selected match */
    selectedIndex: number | null;
    /** Window size for plotting */
    windowSize: number;
    /** Whether search is in progress */
    searching: boolean;
}
