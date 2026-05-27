/**
 * Squiggy-specific Runtime API
 *
 * High-level API for squiggy operations: loading POD5/BAM files,
 * generating plots, and reading squiggy-specific kernel state.
 *
 * Built on top of PositronRuntimeClient for low-level kernel communication.
 */

import { RuntimeClient } from './runtime-client-interface';
import {
    POD5Error,
    BAMError,
    FASTAError,
    PlottingError,
    ValidationError,
} from '../utils/error-handler';
import { logger } from '../utils/logger';
import { MotifMatch } from '../types/motif-types';

/**
 * Result from loading a POD5 file via Python kernel
 * (Internal type - FileLoadingService constructs full POD5LoadResult)
 */
export interface POD5KernelResult {
    numReads: number;
}

/**
 * Result from loading a BAM file via Python kernel
 * (Internal type - FileLoadingService constructs full BAMLoadResult)
 */
export interface BAMKernelResult {
    numReads: number;
    hasModifications: boolean;
    modificationTypes: string[];
    hasProbabilities: boolean;
    hasEventAlignment: boolean;
    hasPrimers: boolean; // PT/pt tag present (primer/adapter trim regions)
    basecallModel?: string; // e.g., "rna004_130bps_sup@v5.1.0"
    isRna: boolean; // True if RNA basecalling model detected
}

/**
 * Result from loading a sample (POD5 + optional BAM/FASTA)
 */
export interface SampleLoadResult {
    name: string;
    numReads: number;
    hasBAM: boolean;
    hasFASTA: boolean;
    bamNumReads?: number;
    bamInfo?: {
        hasModifications: boolean;
        modificationTypes: string[];
        hasProbabilities: boolean;
        hasEventAlignment: boolean;
        hasPrimers?: boolean;
    };
}

/**
 * Metadata returned from Python load_sample() in a single round-trip.
 * Matches the _sample_meta dict built in loadSample().
 */
export interface SampleLoadMetadata {
    num_reads: number;
    has_bam: boolean;
    has_fasta: boolean;
    bam_info?: {
        num_reads: number;
        has_modifications: boolean;
        modification_types: string[];
        has_probabilities: boolean;
        has_event_alignment: boolean;
        has_primers: boolean;
        basecall_model?: string;
        is_rna: boolean;
    };
}

/**
 * High-level API for squiggy operations in the Python kernel
 *
 * Can work with either:
 * - PositronRuntimeClient (foreground session - for notebook API)
 * - SquiggyKernelManager (background session - for extension UI)
 */
export class SquiggyRuntimeAPI {
    constructor(private readonly _client: RuntimeClient) {}

    /**
     * Get access to the underlying runtime client
     * For advanced use cases that need direct kernel access
     */
    get client(): RuntimeClient {
        return this._client;
    }

    /**
     * Load a POD5 file
     *
     * Creates a Pod5File object (_sq_pod5) in the kernel.
     *
     * Does NOT preload read IDs - use getReadIds() to fetch them on-demand.
     * @throws POD5Error if loading fails
     */
    async loadPOD5(filePath: string): Promise<POD5KernelResult> {
        try {
            // Validate input
            if (!filePath || typeof filePath !== 'string') {
                throw new ValidationError('File path must be a non-empty string', 'filePath');
            }

            // Escape single quotes in path
            const escapedPath = filePath.replace(/'/g, "\\'");

            // Load file silently (no console output)
            // This creates a Pod5File object in the kernel
            await this._client.executeSilent(
                `
import squiggy
_sq_pod5 = squiggy.Pod5File('${escapedPath}')
`,
                true // Enable retry for transient failures
            );

            // Get read count by reading from OO API (no print needed)
            const numReads = await this._client.getVariable('len(_sq_pod5.read_ids)');

            return { numReads: numReads as number };
        } catch (error) {
            if (error instanceof ValidationError) {
                throw error;
            }
            const errorMessage =
                error instanceof Error
                    ? error.message
                    : 'Unknown error occurred while loading POD5 file';
            throw new POD5Error(`Failed to load POD5 file '${filePath}': ${errorMessage}`);
        }
    }

    /**
     * Get read IDs from loaded POD5 file
     *
     * @param offset Starting index (default 0)
     * @param limit Maximum number of read IDs to return (default all)
     */
    async getReadIds(offset: number = 0, limit?: number): Promise<string[]> {
        const sliceStr = limit ? `[${offset}:${offset + limit}]` : `[${offset}:]`;

        // Read from OO API (no print needed)
        const readIds = await this._client.getVariable(`_sq_pod5.read_ids${sliceStr}`);

        return readIds as string[];
    }

    /**
     * Load a BAM file
     *
     * Does NOT preload reference mapping - use getReferences() and
     * getReadsForReferencePaginated() to fetch data on-demand.
     * @throws BAMError if loading fails
     */
    async loadBAM(filePath: string): Promise<BAMKernelResult> {
        try {
            // Validate input
            if (!filePath || typeof filePath !== 'string') {
                throw new ValidationError('File path must be a non-empty string', 'filePath');
            }

            const escapedPath = filePath.replace(/'/g, "\\'");

            // Load BAM silently (no console output)
            // This creates a BamFile object in the kernel
            await this._client.executeSilent(
                `
import squiggy
_sq_bam = squiggy.BamFile('${escapedPath}')
`,
                true // Enable retry for transient failures
            );

            // Read metadata directly from OO API (no print needed)
            const numReads = await this._client.getVariable("_sq_bam.info['num_reads']");
            const hasModifications = await this._client.getVariable(
                "_sq_bam.info.get('has_modifications', False)"
            );
            const modificationTypes = await this._client.getVariable(
                "_sq_bam.info.get('modification_types', [])"
            );
            const hasProbabilities = await this._client.getVariable(
                "_sq_bam.info.get('has_probabilities', False)"
            );
            const hasEventAlignment = await this._client.getVariable(
                "_sq_bam.info.get('has_event_alignment', False)"
            );
            const basecallModel = await this._client.getVariable(
                "_sq_bam.info.get('basecall_model', None)"
            );
            const isRna = await this._client.getVariable("_sq_bam.info.get('is_rna', False)");
            const hasPrimers = await this._client.getVariable(
                "_sq_bam.info.get('has_primers', False)"
            );

            return {
                numReads: numReads as number,
                hasModifications: hasModifications as boolean,
                modificationTypes: (modificationTypes as unknown[]).map((x) => String(x)),
                hasProbabilities: hasProbabilities as boolean,
                hasEventAlignment: hasEventAlignment as boolean,
                hasPrimers: hasPrimers as boolean,
                basecallModel: basecallModel as string | undefined,
                isRna: isRna as boolean,
            };
        } catch (error) {
            if (error instanceof ValidationError) {
                throw error;
            }
            const errorMessage =
                error instanceof Error
                    ? error.message
                    : 'Unknown error occurred while loading BAM file';
            throw new BAMError(`Failed to load BAM file '${filePath}': ${errorMessage}`);
        }
    }

    /**
     * Get list of reference names from loaded BAM file
     */
    async getReferences(): Promise<string[]> {
        // Read keys directly from OO API (no print needed)
        const references = await this._client.getVariable(
            'list(_sq_bam.ref_mapping.keys()) if _sq_bam.ref_mapping else []'
        );
        return references as string[];
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

        // Get paginated reads using the OO API
        const readIds = await this._client.getVariable(
            `_sq_bam.get_reads_for_reference('${escapedRef}', offset=${offset}, limit=${limit === null ? 'None' : limit})`
        );

        // Get total count for this reference
        const totalCount = await this._client.getVariable(
            `len(_sq_bam.ref_mapping.get('${escapedRef}', []))`
        );

        return { readIds: readIds as string[], totalCount: totalCount as number };
    }

    /**
     * Generate a plot for read(s) and route to Plots pane
     *
     * Generates Bokeh plot which is automatically routed to Positron's Plots pane
     * via webbrowser.open() interception
     * @throws PlottingError if plot generation fails
     * @throws ValidationError if inputs are invalid
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
        showSignalPoints: boolean = false,
        sampleName?: string,
        coordinateSpace?: 'signal' | 'sequence',
        trimPrimers: boolean = true
    ): Promise<void> {
        try {
            // Validate inputs
            if (!readIds || readIds.length === 0) {
                throw new ValidationError('At least one read ID is required', 'readIds');
            }

            if (minModProbability < 0 || minModProbability > 1) {
                throw new ValidationError('Must be between 0 and 1', 'minModProbability');
            }

            const readIdsJson = JSON.stringify(readIds);
            const enabledModTypesJson = JSON.stringify(enabledModTypes);

            // Build pod5_file/bam_file parameters based on sample mode
            const pod5Param = sampleName
                ? `, pod5_file=_sq_samples['${sampleName}'].pod5`
                : ', pod5_file=_sq_pod5';
            const bamParam = sampleName
                ? `, bam_file=_sq_samples['${sampleName}'].bam`
                : ', bam_file=_sq_bam if "_sq_bam" in dir() else None';

            // Build coordinate space parameter if specified
            const coordinateSpaceParam = coordinateSpace
                ? `, coordinate_space='${coordinateSpace}'`
                : '';

            // Build the plot function call with proper multi-line formatting
            const plotCall =
                readIds.length === 1
                    ? `squiggy.plot_read(
    '${readIds[0]}',
    mode='${mode}',
    normalization='${normalization}',
    theme='${theme}',
    show_dwell_time=${showDwellTime ? 'True' : 'False'},
    show_labels=${showBaseAnnotations ? 'True' : 'False'},
    scale_dwell_time=${scaleDwellTime ? 'True' : 'False'},
    min_mod_probability=${minModProbability},
    enabled_mod_types=${enabledModTypesJson},
    downsample=${downsample},
    show_signal_points=${showSignalPoints ? 'True' : 'False'},
    trim_primers=${trimPrimers ? 'True' : 'False'}${pod5Param}${bamParam}${coordinateSpaceParam}
)`
                    : `squiggy.plot_reads(
    ${readIdsJson},
    mode='${mode}',
    normalization='${normalization}',
    theme='${theme}',
    show_dwell_time=${showDwellTime ? 'True' : 'False'},
    show_labels=${showBaseAnnotations ? 'True' : 'False'},
    scale_dwell_time=${scaleDwellTime ? 'True' : 'False'},
    min_mod_probability=${minModProbability},
    enabled_mod_types=${enabledModTypesJson},
    downsample=${downsample},
    show_signal_points=${showSignalPoints ? 'True' : 'False'},
    trim_primers=${trimPrimers ? 'True' : 'False'}${pod5Param}${bamParam}${coordinateSpaceParam}
)`;

            const code = `
import sys
import squiggy
import traceback

# Initialize error tracking
_squiggy_plot_error = None

try:
    # Generate plot and route to Plots pane
    _sq_fig = ${plotCall}
    from bokeh.io import show as _bokeh_show
    _bokeh_show(_sq_fig)
except Exception as e:
    _squiggy_plot_error = f"{type(e).__name__}: {str(e)}\\n{traceback.format_exc()}"
    print(f"ERROR generating plot: {_squiggy_plot_error}", file=sys.stderr)
`;

            // Execute silently - plot will appear in Plots pane automatically
            await this._client.executeSilent(code, true); // Enable retry

            // Check if there was an error during plot generation
            const plotError = await this.client
                .getVariable('_squiggy_plot_error')
                .catch(() => null);
            if (plotError !== null) {
                throw new PlottingError(`${plotError}`);
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

            if (error instanceof ValidationError || error instanceof PlottingError) {
                throw error;
            }

            const errorMessage = error instanceof Error ? error.message : 'Unknown plotting error';
            throw new PlottingError(`Plot generation failed: ${errorMessage}`);
        }
    }

    /**
     * Generate a multi-sample plot with per-read color assignment
     *
     * Supports plotting reads from multiple samples with sample-based coloring.
     * Each read is colored according to its sample, with alpha = 1/N for better
     * visualization of overlapping reads.
     *
     * @param readIds - List of read IDs to plot
     * @param readSampleMap - Dict mapping read_id → sample_name
     * @param readColors - Dict mapping read_id → color hex string
     * @param mode - Plot mode (OVERLAY or STACKED)
     * @param normalization - Normalization method (ZNORM, MAD, etc.)
     * @param theme - Color theme (LIGHT or DARK)
     * @param showDwellTime - Show dwell time coloring (EVENTALIGN only)
     * @param showBaseAnnotations - Show base annotations (EVENTALIGN only)
     * @param scaleDwellTime - Scale x-axis by dwell time (EVENTALIGN only)
     * @param minModProbability - Minimum modification probability threshold
     * @param enabledModTypes - List of modification types to display
     * @param downsample - Downsampling factor
     * @param showSignalPoints - Show individual signal points
     * @param coordinateSpace - Coordinate system ('signal' or 'sequence')
     */
    async generateMultiSamplePlot(
        readIds: string[],
        readSampleMap: Record<string, string>,
        readColors: Record<string, string>,
        mode: string = 'OVERLAY',
        normalization: string = 'ZNORM',
        theme: string = 'LIGHT',
        showDwellTime: boolean = false,
        showBaseAnnotations: boolean = true,
        scaleDwellTime: boolean = false,
        minModProbability: number = 0.5,
        enabledModTypes: string[] = [],
        downsample: number = 5,
        showSignalPoints: boolean = false,
        coordinateSpace: 'signal' | 'sequence' = 'signal'
    ): Promise<void> {
        const readIdsJson = JSON.stringify(readIds);
        const readColorsJson = JSON.stringify(readColors);
        const enabledModTypesJson = JSON.stringify(enabledModTypes);

        // For multi-sample plots, use the first sample's pod5 as the pod5_file
        // (all reads must be from loaded samples)
        const firstSampleName = Object.values(readSampleMap)[0];
        const pod5FileExpr = firstSampleName
            ? `_sq_samples['${firstSampleName}'].pod5`
            : '_sq_pod5';
        const bamFileExpr = firstSampleName
            ? `_sq_samples['${firstSampleName}'].bam`
            : '_sq_bam if "_sq_bam" in dir() else None';

        const code = `
import sys
import squiggy
import traceback

# Initialize error tracking
_squiggy_plot_error = None

try:
    # Generate multi-sample plot
    _sq_fig = squiggy.plot_reads(
        ${readIdsJson},
        mode='${mode}',
        normalization='${normalization}',
        theme='${theme}',
        show_dwell_time=${showDwellTime ? 'True' : 'False'},
        show_labels=${showBaseAnnotations ? 'True' : 'False'},
        scale_dwell_time=${scaleDwellTime ? 'True' : 'False'},
        min_mod_probability=${minModProbability},
        enabled_mod_types=${enabledModTypesJson},
        downsample=${downsample},
        show_signal_points=${showSignalPoints ? 'True' : 'False'},
        read_colors=${readColorsJson},
        coordinate_space='${coordinateSpace}',
        pod5_file=${pod5FileExpr},
        bam_file=${bamFileExpr}
    )
    from bokeh.io import show as _bokeh_show
    _bokeh_show(_sq_fig)
except Exception as e:
    _squiggy_plot_error = f"{type(e).__name__}: {str(e)}\\n{traceback.format_exc()}"
    print(f"ERROR generating multi-sample plot: {_squiggy_plot_error}", file=sys.stderr)
`;

        try {
            // Execute silently - plot will appear in Plots pane automatically
            await this._client.executeSilent(code);

            // Check if there was an error during plot generation
            const plotError = await this.client
                .getVariable('_squiggy_plot_error')
                .catch(() => null);
            if (plotError !== null) {
                throw new Error(`Multi-sample plot generation failed:\n${plotError}`);
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
     * @param showModifications - Show modifications heatmap panel (default true)
     * @param showPileup - Show base pileup panel (default true)
     * @param showDwellTime - Show dwell time track panel (default true)
     * @param showSignal - Show signal track panel (default true)
     * @param showQuality - Show quality track panel (default true)
     * @throws PlottingError if plot generation fails
     * @throws ValidationError if inputs are invalid
     */
    async generateAggregatePlot(
        referenceName: string,
        maxReads: number = 100,
        normalization: string = 'ZNORM',
        theme: string = 'LIGHT',
        showModifications: boolean = true,
        modificationThreshold: number = 0.5,
        enabledModTypes: string[] = [],
        showPileup: boolean = true,
        showDwellTime: boolean = true,
        showSignal: boolean = true,
        showQuality: boolean = true,
        showCoverage: boolean = true,
        clipXAxisToAlignment: boolean = true,
        transformCoordinates: boolean = true,
        sampleName?: string,
        minModFrequency: number = 0.2,
        minModifiedReads: number = 5,
        rnaMode: boolean = false,
        trimPrimers: boolean = true,
        primer5p?: string,
        adapter3p?: string
    ): Promise<void> {
        try {
            // Validate inputs
            if (!referenceName || typeof referenceName !== 'string') {
                throw new ValidationError(
                    'Reference name must be a non-empty string',
                    'referenceName'
                );
            }

            if (maxReads <= 0) {
                throw new ValidationError('Must be greater than 0', 'maxReads');
            }

            if (modificationThreshold < 0 || modificationThreshold > 1) {
                throw new ValidationError('Must be between 0 and 1', 'modificationThreshold');
            }

            if (minModFrequency < 0 || minModFrequency > 1) {
                throw new ValidationError('Must be between 0 and 1', 'minModFrequency');
            }

            if (minModifiedReads < 1) {
                throw new ValidationError('Must be at least 1', 'minModifiedReads');
            }

            // Escape single quotes in reference name and sample name for Python strings
            const escapedRefName = referenceName.replace(/'/g, "\\'");
            const escapedSampleName = sampleName ? sampleName.replace(/'/g, "\\'") : '';

            // Build modification filter dict if modifications are enabled
            const modFilterDict =
                enabledModTypes.length > 0
                    ? `{${enabledModTypes.map((mt) => `'${mt}': ${modificationThreshold}`).join(', ')}}`
                    : 'None';

            // Build pod5_file/bam_file/fasta_file parameters based on sample mode
            const pod5Param = sampleName
                ? `pod5_file=_sq_samples['${escapedSampleName}'].pod5`
                : 'pod5_file=_sq_pod5';
            const bamParam = sampleName
                ? `bam_file=_sq_samples['${escapedSampleName}'].bam`
                : 'bam_file=_sq_bam if "_sq_bam" in dir() else None';
            const fastaParam = sampleName
                ? `fasta_file=_sq_samples['${escapedSampleName}'].fasta`
                : 'fasta_file=_sq_fasta if "_sq_fasta" in dir() else None';

            // Build optional primer sequence parameters
            const primerParams =
                primer5p || adapter3p
                    ? `${primer5p ? `,\n    primer_5p='${primer5p.replace(/'/g, "\\'")}'` : ''}${adapter3p ? `,\n    adapter_3p='${adapter3p.replace(/'/g, "\\'")}'` : ''}`
                    : '';

            // Use plot_pileup() when signal and dwell time are disabled (no mv tag required)
            // This allows aggregate-style plots for BAM files without move tables
            const usePileupOnly = !showSignal && !showDwellTime;

            const code = usePileupOnly
                ? `
import squiggy

# Generate pileup-only plot (no mv tag required)
_sq_fig = squiggy.plot_pileup(
    reference_name='${escapedRefName}',
    max_reads=${maxReads},
    theme='${theme}',
    show_modifications=${showModifications ? 'True' : 'False'},
    mod_filter=${modFilterDict},
    min_mod_frequency=${minModFrequency},
    min_modified_reads=${minModifiedReads},
    show_pileup=${showPileup ? 'True' : 'False'},
    show_quality=${showQuality ? 'True' : 'False'},
    show_coverage=${showCoverage ? 'True' : 'False'},
    clip_x_to_alignment=${clipXAxisToAlignment ? 'True' : 'False'},
    transform_coordinates=${transformCoordinates ? 'True' : 'False'},
    rna_mode=${rnaMode ? 'True' : 'False'},
    ${bamParam},
    ${fastaParam}
)
from bokeh.io import show as _bokeh_show
_bokeh_show(_sq_fig)
`
                : `
import squiggy

# Generate aggregate plot
_sq_fig = squiggy.plot_aggregate(
    reference_name='${escapedRefName}',
    max_reads=${maxReads},
    normalization='${normalization}',
    theme='${theme}',
    show_modifications=${showModifications ? 'True' : 'False'},
    mod_filter=${modFilterDict},
    min_mod_frequency=${minModFrequency},
    min_modified_reads=${minModifiedReads},
    show_pileup=${showPileup ? 'True' : 'False'},
    show_dwell_time=${showDwellTime ? 'True' : 'False'},
    show_signal=${showSignal ? 'True' : 'False'},
    show_quality=${showQuality ? 'True' : 'False'},
    show_coverage=${showCoverage ? 'True' : 'False'},
    clip_x_to_alignment=${clipXAxisToAlignment ? 'True' : 'False'},
    transform_coordinates=${transformCoordinates ? 'True' : 'False'},
    rna_mode=${rnaMode ? 'True' : 'False'},
    trim_primers=${trimPrimers ? 'True' : 'False'}${primerParams},
    ${pod5Param},
    ${bamParam},
    ${fastaParam}
)
from bokeh.io import show as _bokeh_show
_bokeh_show(_sq_fig)
`;

            // Execute silently - plot will appear in Plots pane automatically
            await this._client.executeSilent(code, true); // Enable retry
        } catch (error) {
            if (error instanceof ValidationError || error instanceof PlottingError) {
                throw error;
            }
            const errorMessage =
                error instanceof Error ? error.message : 'Unknown aggregate plot error';
            throw new PlottingError(`Failed to generate aggregate plot: ${errorMessage}`);
        }
    }

    /**
     * Load and validate a FASTA file
     * @throws FASTAError if loading fails
     */
    async loadFASTA(fastaPath: string): Promise<void> {
        try {
            // Validate input
            if (!fastaPath || typeof fastaPath !== 'string') {
                throw new ValidationError('File path must be a non-empty string', 'fastaPath');
            }

            const escapedPath = fastaPath.replace(/'/g, "\\'");

            const code = `
import squiggy

# Load FASTA file using OO API
_sq_fasta = squiggy.FastaFile('${escapedPath}')
`;

            await this._client.executeSilent(code, true); // Enable retry
        } catch (error) {
            if (error instanceof ValidationError) {
                throw error;
            }
            const errorMessage =
                error instanceof Error
                    ? error.message
                    : 'Unknown error occurred while loading FASTA file';
            throw new FASTAError(`Failed to load FASTA file '${fastaPath}': ${errorMessage}`);
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
    ): Promise<MotifMatch[]> {
        const escapedFastaPath = fastaFile.replace(/'/g, "\\'");
        const escapedMotif = motif.replace(/'/g, "\\'");
        const escapedRegion = region ? region.replace(/'/g, "\\'") : null;

        const searchCode = `
import squiggy

_squiggy_motif_matches = list(squiggy.search_motif(
    fasta_file=squiggy.FastaFile('${escapedFastaPath}'),
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

            return (matches as MotifMatch[]) || [];
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            throw new Error(`Failed to search motif: ${errorMessage}`);
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
_sq_fig = squiggy.plot_motif_aggregate_all(
    motif='${escapedMotif}',
    pod5_file=_sq_pod5,
    bam_file=_sq_bam if '_sq_bam' in dir() else None,
    fasta_file=squiggy.FastaFile('${escapedFastaPath}'),
    upstream=${upstream},
    downstream=${downstream},
    max_reads_per_motif=${maxReadsPerMotif},
    normalization='${normalization}',
    theme='${theme}'
)
from bokeh.io import show as _bokeh_show
_bokeh_show(_sq_fig)
`;

        try {
            await this._client.executeSilent(code);
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            throw new Error(`Failed to generate motif aggregate all plot: ${errorMessage}`);
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
    ): Promise<SampleLoadMetadata> {
        // Escape single quotes in paths
        const escapedSampleName = sampleName.replace(/'/g, "\\'");
        const escapedPod5Path = pod5Path.replace(/'/g, "\\'");
        const escapedBamPath = bamPath ? bamPath.replace(/'/g, "\\'") : null;
        const escapedFastaPath = fastaPath ? fastaPath.replace(/'/g, "\\'") : null;

        // Build Python code to load sample AND return all metadata in one call
        let code = `
import squiggy
import json as _json

# Load sample with OO API
_sq_samples['${escapedSampleName}'] = squiggy.Sample(
    '${escapedSampleName}',
    '${escapedPod5Path}'`;

        if (escapedBamPath) {
            code += `,\n    bam_path='${escapedBamPath}'`;
        }
        if (escapedFastaPath) {
            code += `,\n    fasta_path='${escapedFastaPath}'`;
        }

        code += `\n)

# Return all metadata in one shot to avoid multiple round-trips
_s = _sq_samples['${escapedSampleName}']
_sample_meta = {
    'num_reads': len(_s.read_ids),
    'has_bam': _s.bam is not None,
    'has_fasta': _s.fasta is not None,
}
if _s.bam and _s.bam.info:
    _sample_meta['bam_info'] = {
        'num_reads': _s.bam.info.get('num_reads', 0),
        'has_modifications': _s.bam.info.get('has_modifications', False),
        'modification_types': _s.bam.info.get('modification_types', []),
        'has_probabilities': _s.bam.info.get('has_probabilities', False),
        'has_event_alignment': _s.bam.info.get('has_event_alignment', False),
        'has_primers': _s.bam.info.get('has_primers', False),
        'basecall_model': _s.bam.info.get('basecall_model'),
        'is_rna': _s.bam.info.get('is_rna', False),
    }
del _s
`;

        try {
            logger.debug(
                `[loadSample] Starting to load sample '${sampleName}' with POD5: ${pod5Path}${bamPath ? ` BAM: ${bamPath}` : ''}`
            );
            const startTime = Date.now();

            // Load sample and populate metadata variable in one execution
            await this._client.executeSilent(code);
            const loadTime = Date.now();
            logger.debug(`[loadSample] Python load completed in ${loadTime - startTime}ms`);

            // Single round-trip to get all metadata
            const meta = (await this._client.getVariable('_sample_meta')) as SampleLoadMetadata;
            await this._client.executeSilent('del _sample_meta');
            const totalTime = Date.now();
            logger.debug(
                `[loadSample] Got metadata in ${totalTime - loadTime}ms (total: ${totalTime - startTime}ms)`
            );

            return meta;
        } catch (error) {
            logger.error(`[loadSample] Error loading sample '${sampleName}'`, error);
            const errorMessage = error instanceof Error ? error.message : String(error);
            throw new Error(`Failed to load sample '${sampleName}': ${errorMessage}`);
        }
    }

    /**
     * List all loaded samples
     *
     * @returns Array of sample names currently loaded in the session
     */
    async listSamples(): Promise<string[]> {
        try {
            const sampleNames = await this._client.getVariable('list(_sq_samples.keys())');
            return (sampleNames as string[]) || [];
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            throw new Error(`Failed to list samples: ${errorMessage}`);
        }
    }

    /**
     * Get sample information
     */
    async getSampleInfo(sampleName: string): Promise<any> {
        const escapedName = sampleName.replace(/'/g, "\\'");

        try {
            const code = `
_sample = _sq_samples.get('${escapedName}')
if _sample:
    _sample_info = {
        'name': _sample.name,
        'read_count': len(_sample.read_ids),
        'pod5_path': _sample.pod5.path if _sample.pod5 else None,
        'has_bam': _sample.bam is not None,
        'has_fasta': _sample.fasta is not None
    }
    # Add reference information if BAM is loaded
    if _sample.bam and _sample.bam.info and 'references' in _sample.bam.info:
        _sample_info['references'] = _sample.bam.info['references']
else:
    _sample_info = None
`;

            await this._client.executeSilent(code);
            const sampleInfo = await this._client.getVariable('_sample_info');

            // Clean up temporary variables
            await this.client
                .executeSilent(
                    `
if '_sample' in globals():
    del _sample
if '_sample_info' in globals():
    del _sample_info
`
                )
                .catch(() => {});

            // getVariable already handles JSON parsing, so sampleInfo is a JavaScript object
            return sampleInfo;
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            throw new Error(`Failed to get sample info: ${errorMessage}`);
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
_sq_samples['${escapedName}'].close()
del _sq_samples['${escapedName}']
`;
            await this._client.executeSilent(code);
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            throw new Error(`Failed to remove sample '${sampleName}': ${errorMessage}`);
        }
    }

    /**
     * Get read IDs and references for a sample in a single query (optimized batch)
     *
     * This combines two expensive getVariable() calls into one to reduce kernel round-trips.
     * Each getVariable() call requires 3 sequential executeSilent() calls (json.dumps, getSessionVariables, cleanup),
     * so batching two queries saves significant overhead.
     *
     * @param sampleName - Name of the sample
     * @returns Object with readIds and references
     */
    async getReadIdsAndReferencesForSample(
        sampleName: string
    ): Promise<{ readIds: string[]; references: string[] }> {
        const escapedName = sampleName.replace(/'/g, "\\'");

        try {
            logger.debug(
                `[getReadIdsAndReferencesForSample] Fetching data for sample '${sampleName}'...`
            );
            const startTime = Date.now();

            // Execute setup code first to create variables
            // Uses ref_counts (reference → read count) which is built once during sample load
            const setupCode = `
_sample = _sq_samples.get('${escapedName}')
if _sample:
    _read_ids = _sample.read_ids
    if _sample.bam and _sample.bam.info and 'ref_counts' in _sample.bam.info:
        _refs = list(_sample.bam.info['ref_counts'].keys())
    else:
        _refs = []
else:
    _read_ids = []
    _refs = []
_result = {'read_ids': _read_ids, 'references': _refs}
`;
            await this._client.executeSilent(setupCode);
            const result = await this._client.getVariable('_result');
            const elapsed = Date.now() - startTime;

            const data = (result as { read_ids: string[]; references: string[] }) || {
                read_ids: [],
                references: [],
            };
            logger.debug(
                `[getReadIdsAndReferencesForSample] Got ${data.read_ids?.length || 0} reads and ${data.references?.length || 0} references in ${elapsed}ms`
            );

            // Clean up temporary variables
            await this._client
                .executeSilent(
                    `
if '_sample' in globals():
    del _sample
if '_read_ids' in globals():
    del _read_ids
if '_refs' in globals():
    del _refs
if '_result' in globals():
    del _result
`
                )
                .catch(() => {});

            return {
                readIds: data.read_ids || [],
                references: data.references || [],
            };
        } catch (error) {
            logger.warning(
                `Failed to get read IDs and references for sample '${sampleName}'`,
                error
            );
            return { readIds: [], references: [] };
        }
    }

    /**
     * Get read IDs for a specific sample from the multi-sample registry
     *
     * @param sampleName - Name of the sample
     * @returns Array of read IDs in that sample
     *
     * NOTE: Prefer getReadIdsAndReferencesForSample() when you also need references,
     * as it batches both queries into a single call.
     */
    async getReadIdsForSample(
        sampleName: string,
        offset: number = 0,
        limit?: number
    ): Promise<string[]> {
        const escapedName = sampleName.replace(/'/g, "\\'");

        try {
            logger.debug(
                `[getReadIdsForSample] Fetching read IDs for sample '${sampleName}' (offset: ${offset}, limit: ${limit || 'all'})...`
            );
            const startTime = Date.now();

            // Build slice string for read IDs
            const sliceStr = limit ? `[${offset}:${offset + limit}]` : `[${offset}:]`;

            // Two-step approach: execute code to create temp variable, then read it
            // This avoids the syntax error from passing statements to getVariable()
            const tempVar = '_temp_read_ids_' + Math.random().toString(36).substr(2, 9);

            await this._client.executeSilent(`
_sample = _sq_samples.get('${escapedName}')
${tempVar} = _sample.read_ids${sliceStr} if _sample else []
`);

            // Now read the temp variable as a single expression
            const readIds = await this._client.getVariable(tempVar);

            // Clean up temp variables
            await this._client
                .executeSilent(
                    `
if '${tempVar}' in globals():
    del ${tempVar}
if '_sample' in globals():
    del _sample
`
                )
                .catch(() => {});

            const elapsed = Date.now() - startTime;
            const readIdArray = (readIds as string[]) || [];
            logger.debug(
                `[getReadIdsForSample] Got ${readIdArray.length} read IDs in ${elapsed}ms`
            );
            return readIdArray;
        } catch (error) {
            logger.warning(`Failed to get read IDs for sample '${sampleName}'`, error);
            return [];
        }
    }

    /**
     * Get reference names for a specific sample from the multi-sample registry
     *
     * @param sampleName - Name of the sample
     * @returns Array of reference names from that sample's BAM
     *
     * NOTE: Prefer getReadIdsAndReferencesForSample() when you also need read IDs,
     * as it batches both queries into a single call.
     */
    async getReferencesForSample(sampleName: string): Promise<string[]> {
        const escapedName = sampleName.replace(/'/g, "\\'");

        try {
            // Extract reference names from the sample's BAM without serializing the Sample object
            // Note: We use executeSilent + temp variable instead of getVariable
            // because getVariable wraps code in json.dumps() which doesn't work with multi-line code
            const tempVar = '_squiggy_temp_refs_' + Math.random().toString(36).substr(2, 9);

            await this._client.executeSilent(`
_sample = _sq_samples.get('${escapedName}')
if _sample and _sample.bam and _sample.bam.ref_mapping:
    ${tempVar} = list(_sample.bam.ref_mapping.keys())
else:
    ${tempVar} = []
`);

            // Read the temp variable
            const references = await this._client.getVariable(tempVar);

            // Clean up temp variables
            await this._client
                .executeSilent(
                    `
if '${tempVar}' in globals():
    del ${tempVar}
if '_sample' in globals():
    del _sample
`
                )
                .catch(() => {});

            return (references as string[]) || [];
        } catch (error) {
            logger.warning(`Failed to get references for sample '${sampleName}'`, error);
            return [];
        }
    }

    /**
     * Get read counts for all references in a sample (optimized batch)
     *
     * This fetches read counts for all references in a single query,
     * avoiding N separate getVariable() calls when you have multiple references.
     *
     * Uses pre-computed ref_counts (built during sample load), so this is instant.
     *
     * @param sampleName - Name of the sample
     * @returns Map of reference name to read count
     */
    async getReadsCountForAllReferencesSample(
        sampleName: string
    ): Promise<{ [referenceName: string]: number }> {
        const escapedName = sampleName.replace(/'/g, "\\'");

        try {
            // Execute setup code first to create variables
            const setupCode = `
_sample = _sq_samples.get('${escapedName}')
if _sample and _sample.bam and _sample.bam.info and 'ref_counts' in _sample.bam.info:
    _counts = _sample.bam.info['ref_counts']
else:
    _counts = {}
`;
            await this._client.executeSilent(setupCode);
            const counts = await this._client.getVariable('_counts');

            // Clean up temporary variables
            await this._client
                .executeSilent(
                    `
if '_sample' in globals():
    del _sample
if '_counts' in globals():
    del _counts
`
                )
                .catch(() => {});

            return (counts as { [referenceName: string]: number }) || {};
        } catch (error) {
            logger.warning(`Failed to get reference read counts for sample '${sampleName}'`, error);
            return {};
        }
    }

    /**
     * Get read IDs for a specific reference within a specific sample
     *
     * @param sampleName - Name of the sample
     * @param referenceName - Name of the reference
     * @returns Array of read IDs aligned to that reference
     *
     * NOTE: Prefer getReadsCountForAllReferencesSample() when you need counts for all references,
     * as it batches all queries into a single call.
     */
    async getReadsForReferenceSample(sampleName: string, referenceName: string): Promise<string[]> {
        const escapedName = sampleName.replace(/'/g, "\\'");
        const escapedRef = referenceName.replace(/'/g, "\\'");

        try {
            // Execute setup code first to create variables
            const setupCode = `
_sample = _sq_samples.get('${escapedName}')
if _sample and _sample.bam and _sample.bam.ref_mapping:
    _reads = _sample.bam.ref_mapping.get('${escapedRef}', [])
else:
    _reads = []
`;
            await this._client.executeSilent(setupCode);
            const readIds = await this._client.getVariable('_reads');

            // Clean up temporary variables
            await this._client
                .executeSilent(
                    `
if '_sample' in globals():
    del _sample
if '_reads' in globals():
    del _reads
`
                )
                .catch(() => {});

            return (readIds as string[]) || [];
        } catch (error) {
            logger.warning(
                `Failed to get reads for reference '${referenceName}' in sample '${sampleName}'`,
                error
            );
            return [];
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
    async generateSignalOverlayComparison(
        sampleNames: string[],
        normalization: string = 'ZNORM',
        theme: string = 'LIGHT',
        maxReads?: number | null
    ): Promise<void> {
        // Validate input
        if (!sampleNames || sampleNames.length < 2) {
            throw new Error('Signal overlay comparison requires at least 2 samples');
        }

        // Build maxReads parameter if provided
        const maxReadsParam =
            maxReads !== undefined && maxReads !== null ? `, max_reads=${maxReads}` : '';

        // Build list of Sample objects from _sq_samples dict
        const sampleListExpr = sampleNames
            .map((name) => `_sq_samples['${name.replace(/'/g, "\\'")}']`)
            .join(', ');

        const code = `
import squiggy

# Generate signal overlay comparison plot
_sq_fig = squiggy.plot_signal_overlay_comparison(
    [${sampleListExpr}],
    normalization='${normalization}',
    theme='${theme}'${maxReadsParam}
)
from bokeh.io import show as _bokeh_show
_bokeh_show(_sq_fig)
`;

        try {
            // Execute silently - plot will appear in Plots pane automatically
            await this.client.executeSilent(code);
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);

            // Provide helpful error messages for common issues
            if (errorMessage.includes('not found')) {
                throw new Error(
                    `Sample not found in Python session. ` +
                        `This can happen if samples were loaded through the UI but the Python backend needs to be re-synchronized. ` +
                        `Try loading the samples again using "Load Sample Data" in the File Explorer.`
                );
            }

            throw new Error(`Failed to generate signal overlay comparison plot: ${errorMessage}`);
        }
    }

    async generateDeltaPlot(
        sampleNames: string[],
        referenceName: string,
        normalization: string = 'ZNORM',
        theme: string = 'LIGHT',
        maxReads?: number | null
    ): Promise<void> {
        // Validate input
        if (!sampleNames || sampleNames.length < 2) {
            throw new Error('Delta comparison requires at least 2 samples');
        }

        // Escape single quotes in reference name for Python strings
        const escapedRefName = referenceName.replace(/'/g, "\\'");

        // Build maxReads parameter if provided
        const maxReadsParam =
            maxReads !== undefined && maxReads !== null ? `, max_reads=${maxReads}` : '';

        // Build list of Sample objects from _sq_samples dict
        const sampleListExpr = sampleNames
            .map((name) => `_sq_samples['${name.replace(/'/g, "\\'")}']`)
            .join(', ');

        const code = `
import squiggy

# Generate delta comparison plot
_sq_fig = squiggy.plot_delta_comparison(
    [${sampleListExpr}],
    reference_name='${escapedRefName}',
    normalization='${normalization}',
    theme='${theme}'${maxReadsParam}
)
from bokeh.io import show as _bokeh_show
_bokeh_show(_sq_fig)
`;

        try {
            // Execute silently - plot will appear in Plots pane automatically
            await this._client.executeSilent(code);
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            throw new Error(`Failed to generate delta comparison plot: ${errorMessage}`);
        }
    }

    /**
     * Generate aggregate comparison plot for multiple samples
     *
     * Creates a visualization comparing aggregate statistics (signal, dwell time,
     * quality) from 2+ samples overlaid on the same axes.
     *
     * @param sampleNames - Array of sample names to compare (minimum 2)
     * @param referenceName - Name of reference sequence from BAM files
     * @param metrics - Array of metrics to display: 'signal', 'dwell_time', 'quality'
     * @param maxReads - Maximum reads per sample (default: auto-calculated minimum)
     * @param normalization - Normalization method (ZNORM, MAD, MEDIAN, NONE)
     * @param theme - Color theme (LIGHT or DARK)
     * @param sampleColors - Optional object mapping sample names to hex colors
     */
    async generateAggregateComparison(
        sampleNames: string[],
        referenceName: string,
        metrics: string[] = ['signal', 'dwell_time', 'quality'],
        maxReads?: number | null,
        normalization: string = 'ZNORM',
        theme: string = 'LIGHT',
        sampleColors?: Record<string, string>,
        trimPrimers: boolean = true
    ): Promise<void> {
        // Validate input
        if (!sampleNames || sampleNames.length < 2) {
            throw new Error('Aggregate comparison requires at least 2 samples');
        }

        // Validate reference name
        if (!referenceName) {
            throw new Error('Reference name is required for aggregate comparison');
        }

        // Convert arrays and objects to JSON for safe Python serialization
        const metricsJson = JSON.stringify(metrics);

        // Build maxReads parameter if provided
        const maxReadsParam =
            maxReads !== undefined && maxReads !== null ? `, max_reads=${maxReads}` : '';

        // Build sample_colors parameter if provided
        const sampleColorsParam = sampleColors
            ? `, sample_colors=${JSON.stringify(sampleColors)}`
            : '';

        // Escape single quotes in reference name
        const escapedRefName = referenceName.replace(/'/g, "\\'");

        // Build list of Sample objects from _sq_samples dict
        const sampleListExpr = sampleNames
            .map((name) => `_sq_samples['${name.replace(/'/g, "\\'")}']`)
            .join(', ');

        const code = `
import squiggy

# Generate aggregate comparison plot
_sq_fig = squiggy.plot_aggregate_comparison(
    [${sampleListExpr}],
    reference_name='${escapedRefName}',
    metrics=${metricsJson},
    normalization='${normalization}',
    theme='${theme}',
    trim_primers=${trimPrimers ? 'True' : 'False'}${maxReadsParam}${sampleColorsParam}
)
from bokeh.io import show as _bokeh_show
_bokeh_show(_sq_fig)
`;

        try {
            // Execute silently - plot will appear in Plots pane automatically
            await this._client.executeSilent(code);
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);

            // Provide helpful error messages for common issues
            if (errorMessage.includes('not found')) {
                throw new Error(
                    `Sample not found in Python session. ` +
                        `This can happen if samples were loaded through the UI but the Python backend needs to be re-synchronized. ` +
                        `Try loading the samples again using "Load Sample Data" in the File Explorer.`
                );
            }

            throw new Error(`Failed to generate aggregate comparison plot: ${errorMessage}`);
        }
    }
}
