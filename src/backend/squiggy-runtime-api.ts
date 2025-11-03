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
    constructor(private readonly client: PositronRuntimeClient) {}

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
        await this.client.executeSilent(`
import squiggy
from squiggy.io import _squiggy_session
squiggy.load_pod5('${escapedPath}')
`);

        // Get read count by reading from session object (no print needed)
        const numReads = await this.client.getVariable('len(_squiggy_session.read_ids)');

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
        const readIds = await this.client.getVariable(`_squiggy_session.read_ids${sliceStr}`);

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
        await this.client.executeSilent(`
import squiggy
from squiggy.io import _squiggy_session
squiggy.load_bam('${escapedPath}')
squiggy.get_read_to_reference_mapping()
`);

        // Read metadata directly from session object (no print needed)
        const numReads = await this.client.getVariable("_squiggy_session.bam_info['num_reads']");
        const hasModifications = await this.client.getVariable(
            "_squiggy_session.bam_info.get('has_modifications', False)"
        );
        const modificationTypes = await this.client.getVariable(
            "_squiggy_session.bam_info.get('modification_types', [])"
        );
        const hasProbabilities = await this.client.getVariable(
            "_squiggy_session.bam_info.get('has_probabilities', False)"
        );
        const hasEventAlignment = await this.client.getVariable(
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
        const references = await this.client.getVariable(
            'list(_squiggy_session.ref_mapping.keys()) if _squiggy_session.ref_mapping else []'
        );
        return references as string[];
    }

    /**
     * Get read IDs mapping to a specific reference
     */
    async getReadsForReference(referenceName: string): Promise<string[]> {
        const escapedRef = referenceName.replace(/'/g, "\\'");

        // Read directly from session object (no print needed)
        const readIds = await this.client.getVariable(
            `_squiggy_session.ref_mapping.get('${escapedRef}', []) if _squiggy_session.ref_mapping else []`
        );

        return readIds as string[];
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
        downsample: number = 1,
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
            await this.client.executeSilent(code);

            // Check if there was an error during plot generation
            const plotError = await this.client
                .getVariable('_squiggy_plot_error')
                .catch(() => null);
            if (plotError !== null) {
                throw new Error(`Plot generation failed:\n${plotError}`);
            }

            // Clean up temporary variable
            await this.client.executeSilent(`
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
            await this.client.executeSilent(code);
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
            await this.client.executeSilent(code);
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
            await this.client.executeSilent(searchCode);
            const matches = await this.client.getVariable('_squiggy_motif_matches_json');

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
     * Generate motif-centered aggregate plot
     */
    async generateMotifAggregatePlot(
        fastaFile: string,
        motif: string,
        matchIndex: number,
        window: number = 50,
        maxReads: number = 100,
        normalization: string = 'ZNORM',
        theme: string = 'LIGHT'
    ): Promise<void> {
        const escapedFastaPath = fastaFile.replace(/'/g, "\\'");
        const escapedMotif = motif.replace(/'/g, "\\'");

        const code = `
import squiggy

# Generate motif aggregate plot - will be automatically routed to Plots pane
squiggy.plot_motif_aggregate(
    fasta_file='${escapedFastaPath}',
    motif='${escapedMotif}',
    match_index=${matchIndex},
    window=${window},
    max_reads=${maxReads},
    normalization='${normalization}',
    theme='${theme}'
)
`;

        try {
            await this.client.executeSilent(code);
        } catch (error) {
            throw new Error(`Failed to generate motif aggregate plot: ${error}`);
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
            await this.client.executeSilent(code);
        } catch (error) {
            throw new Error(`Failed to generate motif aggregate all plot: ${error}`);
        }
    }
}
