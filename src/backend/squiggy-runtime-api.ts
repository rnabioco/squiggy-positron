/**
 * Squiggy-specific Runtime API
 *
 * High-level API for squiggy operations: loading POD5/BAM files,
 * generating plots, and reading squiggy-specific kernel state.
 *
 * Built on top of PositronRuntimeClient for low-level kernel communication.
 */

import { PositronRuntimeClient } from './positron-runtime-client';

/**
 * Result from loading a POD5 file
 */
export interface POD5LoadResult {
    numReads: number;
}

/**
 * Result from loading a BAM file
 */
export interface BAMLoadResult {
    numReads: number;
    hasModifications: boolean;
    modificationTypes: string[];
    hasProbabilities: boolean;
    hasEventAlignment: boolean;
}

/**
 * High-level API for squiggy operations in the Python kernel
 */
export class SquiggyRuntimeAPI {
    constructor(private readonly _client: PositronRuntimeClient) {}

    /**
     * Get access to the underlying Positron runtime client
     * For advanced use cases that need direct kernel access
     */
    get client(): PositronRuntimeClient {
        return this._client;
    }

    /**
     * Load a POD5 file
     *
     * Executes squiggy.load_pod5() in the kernel. The session object is stored
     * in _squiggy_session kernel variable accessible from console/notebooks.
     *
     * Does NOT preload read IDs - use getReadIds() to fetch them on-demand.
     */
    async loadPOD5(filePath: string): Promise<POD5LoadResult> {
        // Escape single quotes in path
        const escapedPath = filePath.replace(/'/g, "\\'");

        // Load file silently (no console output)
        // This populates the global _squiggy_session in Python
        await this._client.executeSilent(`
import squiggy
from squiggy.io import _squiggy_session
squiggy.load_pod5('${escapedPath}')
`);

        // Get read count by reading from session object (no print needed)
        const numReads = await this._client.getVariable('len(_squiggy_session.read_ids)');

        return { numReads: numReads as number };
    }

    /**
     * Get read IDs from loaded POD5 file
     *
     * @param offset Starting index (default 0)
     * @param limit Maximum number of read IDs to return (default all)
     */
    async getReadIds(offset: number = 0, limit?: number): Promise<string[]> {
        const sliceStr = limit ? `[${offset}:${offset + limit}]` : `[${offset}:]`;

        // Read from session object (no print needed)
        const readIds = await this._client.getVariable(`_squiggy_session.read_ids${sliceStr}`);

        return readIds as string[];
    }

    /**
     * Load a BAM file
     *
     * Does NOT preload reference mapping - use getReferences() and getReadsForReference()
     * to fetch data on-demand.
     */
    async loadBAM(filePath: string): Promise<BAMLoadResult> {
        const escapedPath = filePath.replace(/'/g, "\\'");

        // Load BAM silently (no console output)
        // This populates _squiggy_session.bam_info and .bam_path
        await this._client.executeSilent(`
import squiggy
from squiggy.io import _squiggy_session
squiggy.load_bam('${escapedPath}')
squiggy.get_read_to_reference_mapping()
`);

        // Read metadata directly from session object (no print needed)
        const numReads = await this._client.getVariable("_squiggy_session.bam_info['num_reads']");
        const hasModifications = await this._client.getVariable(
            "_squiggy_session.bam_info.get('has_modifications', False)"
        );
        const modificationTypes = await this._client.getVariable(
            "_squiggy_session.bam_info.get('modification_types', [])"
        );
        const hasProbabilities = await this._client.getVariable(
            "_squiggy_session.bam_info.get('has_probabilities', False)"
        );
        const hasEventAlignment = await this._client.getVariable(
            "_squiggy_session.bam_info.get('has_event_alignment', False)"
        );

        return {
            numReads: numReads as number,
            hasModifications: hasModifications as boolean,
            modificationTypes: (modificationTypes as unknown[]).map((x) => String(x)),
            hasProbabilities: hasProbabilities as boolean,
            hasEventAlignment: hasEventAlignment as boolean,
        };
    }

    /**
     * Get list of reference names from loaded BAM file
     */
    async getReferences(): Promise<string[]> {
        // Read keys directly from session object (no print needed)
        const references = await this._client.getVariable(
            'list(_squiggy_session.ref_mapping.keys()) if _squiggy_session.ref_mapping else []'
        );
        return references as string[];
    }

    /**
     * Get read IDs mapping to a specific reference
     * @deprecated Use getReadsForReferencePaginated instead for better performance
     */
    async getReadsForReference(referenceName: string): Promise<string[]> {
        const escapedRef = referenceName.replace(/'/g, "\\'");

        // Read directly from session object (no print needed)
        const readIds = await this._client.getVariable(
            `_squiggy_session.ref_mapping.get('${escapedRef}', []) if _squiggy_session.ref_mapping else []`
        );

        return readIds as string[];
    }

    /**
     * Get read IDs mapping to a specific reference with pagination support
     * Enables lazy loading for large reference groups
     */
    async getReadsForReferencePaginated(
        referenceName: string,
        offset: number = 0,
        limit: number | null = null
    ): Promise<{ readIds: string[]; totalCount: number }> {
        const escapedRef = referenceName.replace(/'/g, "\\'");

        // Get paginated reads using the new Python function
        const readIds = await this._client.getVariable(
            `squiggy.get_reads_for_reference_paginated('${escapedRef}', offset=${offset}, limit=${limit === null ? 'None' : limit})`
        );

        // Get total count for this reference
        const totalCount = await this._client.getVariable(
            `len(squiggy.io._squiggy_session.ref_mapping.get('${escapedRef}', []))`
        );

        return { readIds: readIds as string[], totalCount: totalCount as number };
    }

    /**
     * Generate a plot for read(s) and route to Plots pane
     *
     * Generates Bokeh plot which is automatically routed to Positron's Plots pane
     * via webbrowser.open() interception
     */
    async generatePlot(
        readIds: string[],
        mode: string = 'SINGLE',
        normalization: string = 'ZNORM',
        theme: string = 'LIGHT',
        showDwellTime: boolean = false,
        showBaseAnnotations: boolean = true,
        scaleDwellTime: boolean = false,
        minModProbability: number = 0.5,
        enabledModTypes: string[] = [],
        downsample: number = 5,
        showSignalPoints: boolean = false
    ): Promise<void> {
        const readIdsJson = JSON.stringify(readIds);
        const enabledModTypesJson = JSON.stringify(enabledModTypes);

        const code = `
import sys
import squiggy
import traceback

# Initialize error tracking
_squiggy_plot_error = None

try:
    # Generate plot - will be automatically routed to Plots pane via webbrowser.open()
    ${
        readIds.length === 1
            ? `squiggy.plot_read('${readIds[0]}', mode='${mode}', normalization='${normalization}', theme='${theme}', show_dwell_time=${showDwellTime ? 'True' : 'False'}, show_labels=${showBaseAnnotations ? 'True' : 'False'}, scale_dwell_time=${scaleDwellTime ? 'True' : 'False'}, min_mod_probability=${minModProbability}, enabled_mod_types=${enabledModTypesJson}, downsample=${downsample}, show_signal_points=${showSignalPoints ? 'True' : 'False'})`
            : `squiggy.plot_reads(${readIdsJson}, mode='${mode}', normalization='${normalization}', theme='${theme}', show_dwell_time=${showDwellTime ? 'True' : 'False'}, show_labels=${showBaseAnnotations ? 'True' : 'False'}, scale_dwell_time=${scaleDwellTime ? 'True' : 'False'}, min_mod_probability=${minModProbability}, enabled_mod_types=${enabledModTypesJson}, downsample=${downsample}, show_signal_points=${showSignalPoints ? 'True' : 'False'})`
    }
except Exception as e:
    _squiggy_plot_error = f"{type(e).__name__}: {str(e)}\\n{traceback.format_exc()}"
    print(f"ERROR generating plot: {_squiggy_plot_error}", file=sys.stderr)
`;

        try {
            // Execute silently - plot will appear in Plots pane automatically
            await this._client.executeSilent(code);

            // Check if there was an error during plot generation
            const plotError = await this.client
                .getVariable('_squiggy_plot_error')
                .catch(() => null);
            if (plotError !== null) {
                throw new Error(`Plot generation failed:\n${plotError}`);
            }

            // Clean up temporary variable
            await this._client.executeSilent(`
if '_squiggy_plot_error' in globals():
    del _squiggy_plot_error
`);
        } catch (error) {
            // Clean up on error
            await this.client
                .executeSilent(
                    `
if '_squiggy_plot_error' in globals():
    del _squiggy_plot_error
`
                )
                .catch(() => {});
            throw error;
        }
    }

    /**
     * Generate an aggregate plot for a reference sequence and route to Plots pane
     * @param referenceName - Name of reference sequence from BAM file
     * @param maxReads - Maximum number of reads to sample (default 100)
     * @param normalization - Normalization method (ZNORM, MAD, etc.)
     * @param theme - Color theme (LIGHT or DARK)
     */
    async generateAggregatePlot(
        referenceName: string,
        maxReads: number = 100,
        normalization: string = 'ZNORM',
        theme: string = 'LIGHT'
    ): Promise<void> {
        // Escape single quotes in reference name for Python string
        const escapedRefName = referenceName.replace(/'/g, "\\'");

        const code = `
import squiggy

# Generate aggregate plot - will be automatically routed to Plots pane
squiggy.plot_aggregate(
    reference_name='${escapedRefName}',
    max_reads=${maxReads},
    normalization='${normalization}',
    theme='${theme}'
)
`;

        try {
            // Execute silently - plot will appear in Plots pane automatically
            await this._client.executeSilent(code);
        } catch (error) {
            throw new Error(`Failed to generate aggregate plot: ${error}`);
        }
    }

    /**
     * Load and validate a FASTA file
     */
    async loadFASTA(fastaPath: string): Promise<void> {
        const escapedPath = fastaPath.replace(/'/g, "\\'");

        const code = `
import squiggy

# Load FASTA file using squiggy.load_fasta()
# This populates _squiggy_session.fasta_path and _squiggy_session.fasta_info
squiggy.load_fasta('${escapedPath}')
`;

        try {
            await this._client.executeSilent(code);
        } catch (error) {
            throw new Error(`Failed to load FASTA file: ${error}`);
        }
    }

    /**
     * Search for motif matches in FASTA file
     */
    async searchMotif(
        fastaFile: string,
        motif: string,
        region?: string,
        strand: string = 'both'
    ): Promise<any[]> {
        const escapedFastaPath = fastaFile.replace(/'/g, "\\'");
        const escapedMotif = motif.replace(/'/g, "\\'");
        const escapedRegion = region ? region.replace(/'/g, "\\'") : null;

        const searchCode = `
import squiggy

_squiggy_motif_matches = list(squiggy.search_motif(
    fasta_file='${escapedFastaPath}',
    motif='${escapedMotif}',
    region=${escapedRegion ? `'${escapedRegion}'` : 'None'},
    strand='${strand}'
))

_squiggy_motif_matches_json = [
    {'chrom': m.chrom, 'position': m.position,
     'sequence': m.sequence, 'strand': m.strand}
    for m in _squiggy_motif_matches
]
`;

        try {
            await this._client.executeSilent(searchCode);
            const matches = await this._client.getVariable('_squiggy_motif_matches_json');

            // Clean up temporary variables
            await this.client
                .executeSilent(
                    `
if '_squiggy_motif_matches' in globals():
    del _squiggy_motif_matches
if '_squiggy_motif_matches_json' in globals():
    del _squiggy_motif_matches_json
`
                )
                .catch(() => {});

            return (matches as any[]) || [];
        } catch (error) {
            throw new Error(`Failed to search motif: ${error}`);
        }
    }

    /**
     * Generate aggregate plot for all motif matches with asymmetric windows
     */
    async generateMotifAggregateAllPlot(
        fastaFile: string,
        motif: string,
        upstream: number = 10,
        downstream: number = 10,
        maxReadsPerMotif: number = 100,
        normalization: string = 'ZNORM',
        theme: string = 'LIGHT'
    ): Promise<void> {
        const escapedFastaPath = fastaFile.replace(/'/g, "\\'");
        const escapedMotif = motif.replace(/'/g, "\\'");

        const code = `
import squiggy

# Generate aggregate plot across all motif matches
squiggy.plot_motif_aggregate_all(
    fasta_file='${escapedFastaPath}',
    motif='${escapedMotif}',
    upstream=${upstream},
    downstream=${downstream},
    max_reads_per_motif=${maxReadsPerMotif},
    normalization='${normalization}',
    theme='${theme}'
)
`;

        try {
            await this._client.executeSilent(code);
        } catch (error) {
            throw new Error(`Failed to generate motif aggregate all plot: ${error}`);
        }
    }

    /**
     * Load a sample (POD5 + optional BAM/FASTA pair) with a custom name
     * Adds to the multi-sample session for later comparison
     *
     * @param sampleName - User-defined name for the sample
     * @param pod5Path - Path to POD5 file
     * @param bamPath - Optional path to BAM file
     * @param fastaPath - Optional path to FASTA file
     */
    async loadSample(
        sampleName: string,
        pod5Path: string,
        bamPath?: string,
        fastaPath?: string
    ): Promise<POD5LoadResult> {
        // Escape single quotes in paths
        const escapedSampleName = sampleName.replace(/'/g, "\\'");
        const escapedPod5Path = pod5Path.replace(/'/g, "\\'");
        const escapedBamPath = bamPath ? bamPath.replace(/'/g, "\\'") : null;
        const escapedFastaPath = fastaPath ? fastaPath.replace(/'/g, "\\'") : null;

        // Build Python code to load sample
        let code = `
import squiggy

# Load sample with custom name
squiggy.load_sample(
    '${escapedSampleName}',
    '${escapedPod5Path}'`;

        if (escapedBamPath) {
            code += `,\n    bam_path='${escapedBamPath}'`;
        }
        if (escapedFastaPath) {
            code += `,\n    fasta_path='${escapedFastaPath}'`;
        }

        code += `\n)`;

        try {
            // Load sample silently
            await this._client.executeSilent(code);

            // Get read count for this sample
            const numReads = await this._client.getVariable(
                `len(_squiggy_session.get_sample('${escapedSampleName}').read_ids)`
            );

            return { numReads: numReads as number };
        } catch (error) {
            throw new Error(`Failed to load sample '${sampleName}': ${error}`);
        }
    }

    /**
     * List all loaded samples
     *
     * @returns Array of sample names currently loaded in the session
     */
    async listSamples(): Promise<string[]> {
        try {
            const sampleNames = await this._client.getVariable('_squiggy_session.list_samples()');
            return (sampleNames as string[]) || [];
        } catch (error) {
            throw new Error(`Failed to list samples: ${error}`);
        }
    }

    /**
     * Get sample information
     */
    async getSampleInfo(sampleName: string): Promise<any> {
        const escapedName = sampleName.replace(/'/g, "\\'");

        try {
            const code = `
import json
_sample = _squiggy_session.get_sample('${escapedName}')
if _sample:
    _sample_info = {
        'name': _sample.name,
        'read_count': len(_sample.read_ids),
        'pod5_path': _sample.pod5_path,
        'has_bam': _sample.bam_path is not None,
        'has_fasta': _sample.fasta_path is not None
    }
else:
    _sample_info = None
_sample_info_json = json.dumps(_sample_info)
`;

            await this._client.executeSilent(code);
            const sampleInfo = await this._client.getVariable('_sample_info_json');

            // Clean up temporary variables
            await this.client
                .executeSilent(
                    `
if '_sample' in globals():
    del _sample
if '_sample_info' in globals():
    del _sample_info
if '_sample_info_json' in globals():
    del _sample_info_json
`
                )
                .catch(() => {});

            return sampleInfo ? JSON.parse(sampleInfo as string) : null;
        } catch (error) {
            throw new Error(`Failed to get sample info: ${error}`);
        }
    }

    /**
     * Remove a loaded sample from the session
     *
     * @param sampleName - Name of the sample to remove
     */
    async removeSample(sampleName: string): Promise<void> {
        const escapedName = sampleName.replace(/'/g, "\\'");

        try {
            const code = `
import squiggy
squiggy.remove_sample('${escapedName}')
`;
            await this._client.executeSilent(code);
        } catch (error) {
            throw new Error(`Failed to remove sample '${sampleName}': ${error}`);
        }
    }

    /**
     * Generate a delta comparison plot between two or more samples
     * Shows differences in aggregate statistics between samples (B - A)
     *
     * @param sampleNames - Array of sample names to compare (minimum 2 required)
     * @param normalization - Normalization method (ZNORM, MAD, MEDIAN, NONE)
     * @param theme - Color theme (LIGHT or DARK)
     */
    async generateDeltaPlot(
        sampleNames: string[],
        normalization: string = 'ZNORM',
        theme: string = 'LIGHT'
    ): Promise<void> {
        // Validate input
        if (!sampleNames || sampleNames.length < 2) {
            throw new Error('Delta comparison requires at least 2 samples');
        }

        // Convert sample names to JSON for safe Python serialization
        const sampleNamesJson = JSON.stringify(sampleNames);

        const code = `
import squiggy

# Generate delta comparison plot - will be automatically routed to Plots pane
squiggy.plot_delta_comparison(
    sample_names=${sampleNamesJson},
    normalization='${normalization}',
    theme='${theme}'
)
`;

        try {
            // Execute silently - plot will appear in Plots pane automatically
            await this._client.executeSilent(code);
        } catch (error) {
            throw new Error(`Failed to generate delta comparison plot: ${error}`);
        }
    }
}
