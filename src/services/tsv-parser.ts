/**
 * TSV Parser - Parse sample manifest files
 *
 * Parses tab-separated value files containing sample metadata and file paths.
 * Expected format:
 *   sample_name\tpod5\tbam\tfasta
 *   sample_A\tdata/A.pod5\tdata/A.bam\tref.fa
 *
 * Required columns: sample_name, pod5
 * Optional columns: bam, fasta
 */

export interface TSVSampleSpec {
    /** User-facing sample name (must be unique) */
    sampleName: string;

    /** Path to POD5 file (required) */
    pod5Path: string;

    /** Path to BAM file (optional) */
    bamPath?: string;

    /** Path to FASTA reference file (optional) */
    fastaPath?: string;

    /** Line number in TSV (for error reporting) */
    lineNumber: number;
}

export interface TSVParseResult {
    /** Whether parsing succeeded */
    success: boolean;

    /** Parsed sample specifications */
    samples: TSVSampleSpec[];

    /** Critical errors that prevent parsing */
    errors: string[];

    /** Non-blocking warnings */
    warnings: string[];
}

export class TSVParser {
    /**
     * Parse TSV content into sample specifications
     *
     * @param content - Raw TSV file content or clipboard paste
     * @returns Parse result with samples or errors
     */
    static parse(content: string): TSVParseResult {
        const errors: string[] = [];
        const warnings: string[] = [];
        const samples: TSVSampleSpec[] = [];

        // Split into lines and filter empty lines
        const lines = content
            .split(/\r?\n/)
            .map((line) => line.trim())
            .filter((line) => line.length > 0 && !line.startsWith('#')); // Skip comments

        if (lines.length === 0) {
            errors.push('TSV file is empty');
            return { success: false, samples: [], errors, warnings };
        }

        // Auto-detect delimiter (tab or comma)
        const delimiter = this.detectDelimiter(lines[0]);

        // Parse header row
        const headerCells = lines[0].split(delimiter).map((cell) => cell.trim().toLowerCase());
        const columnMap = this.buildColumnMap(headerCells);

        // Validate required columns
        if (!columnMap.has('sample_name')) {
            errors.push('Missing required column: sample_name');
        }
        if (!columnMap.has('pod5')) {
            errors.push('Missing required column: pod5');
        }

        if (errors.length > 0) {
            return { success: false, samples: [], errors, warnings };
        }

        // Track duplicate sample names
        const seenNames = new Set<string>();

        // Parse data rows
        for (let i = 1; i < lines.length; i++) {
            const lineNumber = i + 1; // 1-indexed for user-friendly error messages
            const line = lines[i];
            const cells = line.split(delimiter).map((cell) => cell.trim());

            // Extract values using column map
            const sampleName = cells[columnMap.get('sample_name')!] || '';
            const pod5Path = cells[columnMap.get('pod5')!] || '';
            const bamPath = columnMap.has('bam') ? cells[columnMap.get('bam')!] : undefined;
            const fastaPath = columnMap.has('fasta')
                ? cells[columnMap.get('fasta')!]
                : undefined;

            // Validate required fields
            if (!sampleName) {
                errors.push(`Line ${lineNumber}: Missing sample_name`);
                continue;
            }

            if (!pod5Path || pod5Path === '-') {
                errors.push(`Line ${lineNumber}: Missing pod5 path for sample '${sampleName}'`);
                continue;
            }

            // Check for duplicates
            if (seenNames.has(sampleName)) {
                errors.push(`Line ${lineNumber}: Duplicate sample name '${sampleName}'`);
                continue;
            }
            seenNames.add(sampleName);

            // Handle optional fields (treat '-' as missing)
            const resolvedBamPath =
                bamPath && bamPath !== '-' && bamPath !== '' ? bamPath : undefined;
            const resolvedFastaPath =
                fastaPath && fastaPath !== '-' && fastaPath !== '' ? fastaPath : undefined;

            // Add warnings for missing optional files
            if (!resolvedBamPath) {
                warnings.push(
                    `Line ${lineNumber}: Sample '${sampleName}' has no BAM file (alignment features unavailable)`
                );
            }
            if (!resolvedFastaPath) {
                warnings.push(
                    `Line ${lineNumber}: Sample '${sampleName}' has no FASTA file (reference sequence unavailable)`
                );
            }

            // Create sample spec
            samples.push({
                sampleName,
                pod5Path,
                bamPath: resolvedBamPath,
                fastaPath: resolvedFastaPath,
                lineNumber,
            });
        }

        return {
            success: errors.length === 0,
            samples,
            errors,
            warnings,
        };
    }

    /**
     * Auto-detect delimiter (tab or comma)
     * Prefer tab, fall back to comma
     */
    private static detectDelimiter(headerLine: string): string {
        if (headerLine.includes('\t')) {
            return '\t';
        } else if (headerLine.includes(',')) {
            return ',';
        }
        return '\t'; // Default to tab
    }

    /**
     * Build column index map from header row
     * Maps column name → column index
     */
    private static buildColumnMap(headerCells: string[]): Map<string, number> {
        const map = new Map<string, number>();
        for (let i = 0; i < headerCells.length; i++) {
            map.set(headerCells[i], i);
        }
        return map;
    }
}
