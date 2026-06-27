/**
 * TSV Validator - Validate sample specifications from TSV
 *
 * Validates:
 * - File existence (POD5 required, BAM/FASTA optional)
 * - POD5/BAM read ID overlap (reuses existing validation logic)
 * - BAM/FASTA reference name overlap (reuses existing validation logic)
 * - Sample name conflicts with already-loaded samples
 */

import { TSVSampleSpec } from './tsv-parser';
import { TSVPathResolver, PathResolutionResult } from './tsv-path-resolver';

export interface ValidationResult {
    /** Sample name being validated */
    sampleName: string;

    /** Whether validation passed (no blocking errors) */
    valid: boolean;

    /** Blocking errors that prevent loading */
    errors: string[];

    /** Non-blocking warnings */
    warnings: string[];

    /** Resolved POD5 path (if found) */
    resolvedPod5?: string;

    /** Resolved BAM path (if found) */
    resolvedBam?: string;

    /** Resolved FASTA path (if found) */
    resolvedFasta?: string;

    /** Path resolution strategies used */
    resolutionStrategies?: {
        pod5?: string;
        bam?: string;
        fasta?: string;
    };
}

export class TSVValidator {
    /**
     * @param pathResolver - Path resolver for file lookups
     * @param existingLoadedSamples - Set of already-loaded sample names (to check conflicts)
     */
    constructor(
        private pathResolver: TSVPathResolver,
        private existingLoadedSamples: Set<string>
    ) {}

    /**
     * Validate a single sample specification
     *
     * TODO: Implement full validation logic
     * - Resolve all paths
     * - Check POD5 exists (BLOCK if missing)
     * - Check BAM exists (WARN if missing)
     * - Check FASTA exists (WARN if missing)
     * - Validate POD5/BAM overlap (reuse existing logic from file-commands.ts)
     * - Validate BAM/FASTA overlap (reuse existing logic)
     * - Check sample name conflicts
     *
     * @param spec - TSV sample specification to validate
     * @returns Validation result with errors/warnings and resolved paths
     */
    async validateSample(spec: TSVSampleSpec): Promise<ValidationResult> {
        const errors: string[] = [];
        const warnings: string[] = [];
        let resolvedPod5: string | undefined;
        let resolvedBam: string | undefined;
        let resolvedFasta: string | undefined;
        const resolutionStrategies: ValidationResult['resolutionStrategies'] = {};

        // Check for sample name conflicts
        if (this.existingLoadedSamples.has(spec.sampleName)) {
            errors.push(
                `Sample name '${spec.sampleName}' conflicts with already-loaded sample`
            );
        }

        // Validate POD5 (required)
        const pod5Result = await this.pathResolver.resolve(spec.pod5Path);
        if (pod5Result.resolvedPath) {
            resolvedPod5 = pod5Result.resolvedPath;
            resolutionStrategies.pod5 = pod5Result.strategy || 'unknown';
        } else {
            errors.push(
                `POD5 file not found: ${spec.pod5Path} (${pod5Result.error || 'unknown error'})`
            );
        }

        // Validate BAM (optional)
        if (spec.bamPath) {
            const bamResult = await this.pathResolver.resolve(spec.bamPath);
            if (bamResult.resolvedPath) {
                resolvedBam = bamResult.resolvedPath;
                resolutionStrategies.bam = bamResult.strategy || 'unknown';
            } else {
                warnings.push(
                    `BAM file not found: ${spec.bamPath} (alignment features will be unavailable)`
                );
            }
        }

        // Validate FASTA (optional)
        if (spec.fastaPath) {
            const fastaResult = await this.pathResolver.resolve(spec.fastaPath);
            if (fastaResult.resolvedPath) {
                resolvedFasta = fastaResult.resolvedPath;
                resolutionStrategies.fasta = fastaResult.strategy || 'unknown';
            } else {
                warnings.push(
                    `FASTA file not found: ${spec.fastaPath} (reference sequence will be unavailable)`
                );
            }
        }

        // TODO: Validate POD5/BAM overlap (requires loading files - expensive)
        // For now, defer to loading time
        // Future: Add optional quick validation using FileLoadingService

        return {
            sampleName: spec.sampleName,
            valid: errors.length === 0,
            errors,
            warnings,
            resolvedPod5,
            resolvedBam,
            resolvedFasta,
            resolutionStrategies,
        };
    }

    /**
     * Validate all samples in batch
     *
     * @param specs - Array of TSV sample specifications
     * @returns Array of validation results (parallel execution)
     */
    async validateBatch(specs: TSVSampleSpec[]): Promise<ValidationResult[]> {
        // Validate all samples in parallel for speed
        return Promise.all(specs.map((spec) => this.validateSample(spec)));
    }

    /**
     * Get summary of validation results
     *
     * @param results - Array of validation results
     * @returns Summary statistics
     */
    static getSummary(results: ValidationResult[]): {
        total: number;
        valid: number;
        invalid: number;
        withBam: number;
        withFasta: number;
        totalErrors: number;
        totalWarnings: number;
    } {
        return {
            total: results.length,
            valid: results.filter((r) => r.valid).length,
            invalid: results.filter((r) => !r.valid).length,
            withBam: results.filter((r) => r.resolvedBam).length,
            withFasta: results.filter((r) => r.resolvedFasta).length,
            totalErrors: results.reduce((sum, r) => sum + r.errors.length, 0),
            totalWarnings: results.reduce((sum, r) => sum + r.warnings.length, 0),
        };
    }
}
