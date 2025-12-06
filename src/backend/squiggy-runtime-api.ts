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
     * Executes squiggy.load_pod5() in the kernel. The session object is stored
     * in squiggy_kernel kernel variable accessible from console/notebooks.
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
            // This populates the global squiggy_kernel in Python
            await this._client.executeSilent(
                `
import squiggy
from squiggy.io import squiggy_kernel
squiggy.load_pod5('${escapedPath}')
`,
                true // Enable retry for transient failures
            );

            // Get read count by reading from session object (no print needed)
            const numReads = await this._client.getVariable('len(squiggy_kernel._read_ids)');

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

        // Read from session object (no print needed)
        const readIds = await this._client.getVariable(`squiggy_kernel._read_ids${sliceStr}`);

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
            // This populates squiggy_kernel._bam_info and .bam_path
            await this._client.executeSilent(
                `
import squiggy
from squiggy.io import squiggy_kernel
squiggy.load_bam('${escapedPath}')
squiggy.get_read_to_reference_mapping()
`,
                true // Enable retry for transient failures
            );

            // Read metadata directly from session object (no print needed)
            const numReads = await this._client.getVariable(
                "squiggy_kernel._bam_info['num_reads']"
            );
            const hasModifications = await this._client.getVariable(
                "squiggy_kernel._bam_info.get('has_modifications', False)"
            );
            const modificationTypes = await this._client.getVariable(
                "squiggy_kernel._bam_info.get('modification_types', [])"
            );
            const hasProbabilities = await this._client.getVariable(
                "squiggy_kernel._bam_info.get('has_probabilities', False)"
            );
            const hasEventAlignment = await this._client.getVariable(
                "squiggy_kernel._bam_info.get('has_event_alignment', False)"
            );
            const basecallModel = await this._client.getVariable(
                "squiggy_kernel._bam_info.get('basecall_model', None)"
            );
            const isRna = await this._client.getVariable(
                "squiggy_kernel._bam_info.get('is_rna', False)"
            );

            return {
                numReads: numReads as number,
                hasModifications: hasModifications as boolean,
                modificationTypes: (modificationTypes as unknown[]).map((x) => String(x)),
                hasProbabilities: hasProbabilities as boolean,
                hasEventAlignment: hasEventAlignment as boolean,
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
        // Read keys directly from session object (no print needed)
        const references = await this._client.getVariable(
            'list(squiggy_kernel._ref_mapping.keys()) if squiggy_kernel._ref_mapping else []'
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

        // Get paginated reads using the new Python function
        const readIds = await this._client.getVariable(
            `squiggy.get_reads_for_reference_paginated('${escapedRef}', offset=${offset}, limit=${limit === null ? 'None' : limit})`
        );

        // Get total count for this reference
        const totalCount = await this._client.getVariable(
            `len(squiggy.io.squiggy_kernel._ref_mapping.get('${escapedRef}', []))`
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
        trimAdapters: boolean = false
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

            // Build sample name parameter if in multi-sample mode
            const sampleNameParam = sampleName ? `, sample_name='${sampleName}'` : '';

            // Build coordinate space parameter if specified
            const coordinateSpaceParam = coordinateSpace
                ? `, coordinate_space='${coordinateSpace}'`
                : '';

            // Build trim adapters parameter
            const trimAdaptersParam = `,
    trim_adapters=${trimAdapters ? 'True' : 'False'}`;

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
    show_signal_points=${showSignalPoints ? 'True' : 'False'}${sampleNameParam}${coordinateSpaceParam}${trimAdaptersParam}
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
    show_signal_points=${showSignalPoints ? 'True' : 'False'}${sampleNameParam}${coordinateSpaceParam}${trimAdaptersParam}
)`;

            const code = `
import sys
import squiggy
import traceback

# Initialize error tracking
_squiggy_plot_error = None

try:
    # Generate plot - will be automatically routed to Plots pane via webbrowser.open()
    ${plotCall}
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
        const readSampleMapJson = JSON.stringify(readSampleMap);
        const readColorsJson = JSON.stringify(readColors);
        const enabledModTypesJson = JSON.stringify(enabledModTypes);

        const code = `
import sys
import squiggy
import traceback

# Initialize error tracking
_squiggy_plot_error = None

try:
    # Generate multi-sample plot
    squiggy.plot_reads(
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
        read_sample_map=${readSampleMapJson},
        read_colors=${readColorsJson},
        coordinate_space='${coordinateSpace}'
    )
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
        trimAdapters: boolean = false
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

            // Build sample name parameter if in multi-sample mode
            const sampleNameParam = sampleName ? `, sample_name='${escapedSampleName}'` : '';

            // Use plot_pileup() when signal and dwell time are disabled (no mv tag required)
            // This allows aggregate-style plots for BAM files without move tables
            const usePileupOnly = !showSignal && !showDwellTime;

            const code = usePileupOnly
                ? `
import squiggy

# Generate pileup-only plot (no mv tag required) - will be automatically routed to Plots pane
squiggy.plot_pileup(
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
    rna_mode=${rnaMode ? 'True' : 'False'}${sampleNameParam}
)
`
                : `
import squiggy

# Generate aggregate plot - will be automatically routed to Plots pane
squiggy.plot_aggregate(
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
    trim_adapters=${trimAdapters ? 'True' : 'False'}${sampleNameParam}
)
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

# Load FASTA file using squiggy.load_fasta()
# This populates squiggy_kernel._fasta_path and squiggy_kernel._fasta_info
squiggy.load_fasta('${escapedPath}')
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
    ): Promise<POD5KernelResult> {
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
            logger.debug(
                `[loadSample] Starting to load sample '${sampleName}' with POD5: ${pod5Path}${bamPath ? ` BAM: ${bamPath}` : ''}`
            );
            const startTime = Date.now();

            // Load sample silently
            logger.debug(`[loadSample] Executing Python code to load sample...`);
            await this._client.executeSilent(code);
            const executeSilentTime = Date.now();
            logger.debug(
                `[loadSample] executeSilent completed in ${executeSilentTime - startTime}ms`
            );

            // Get read count for this sample
            logger.debug(`[loadSample] Querying read count for sample '${sampleName}'...`);
            const numReads = await this._client.getVariable(
                `len(squiggy.squiggy_kernel.get_sample('${escapedSampleName}')._read_ids)`
            );
            const queryTime = Date.now();
            logger.debug(
                `[loadSample] Got ${numReads} reads in ${queryTime - executeSilentTime}ms (total: ${queryTime - startTime}ms)`
            );

            return { numReads: numReads as number };
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
            const sampleNames = await this._client.getVariable(
                'squiggy.squiggy_kernel.list_samples()'
            );
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
from squiggy import squiggy_kernel
_sample = squiggy_kernel.get_sample('${escapedName}')
if _sample:
    _sample_info = {
        'name': _sample.name,
        'read_count': len(_sample._read_ids),
        'pod5_path': _sample.pod5_path,
        'has_bam': _sample.bam_path is not None,
        'has_fasta': _sample.fasta_path is not None
    }
    # Add reference information if BAM is loaded
    if _sample._bam_info and 'references' in _sample._bam_info:
        _sample_info['references'] = _sample._bam_info['references']
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
import squiggy
squiggy.remove_sample('${escapedName}')
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
from squiggy import squiggy_kernel
_sample = squiggy_kernel.get_sample('${escapedName}')
if _sample:
    _read_ids = _sample._read_ids
    if _sample._bam_info and 'ref_counts' in _sample._bam_info:
        _refs = list(_sample._bam_info['ref_counts'].keys())
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
from squiggy import squiggy_kernel
_sample = squiggy_kernel.get_sample('${escapedName}')
${tempVar} = _sample._read_ids${sliceStr} if _sample else []
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
from squiggy import squiggy_kernel
_sample = squiggy_kernel.get_sample('${escapedName}')
if _sample and _sample._bam_info and 'ref_mapping' in _sample._bam_info:
    ${tempVar} = list(_sample._bam_info['ref_mapping'].keys())
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
from squiggy import squiggy_kernel
_sample = squiggy_kernel.get_sample('${escapedName}')
if _sample and _sample._bam_info and 'ref_counts' in _sample._bam_info:
    _counts = _sample._bam_info['ref_counts']
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
from squiggy import squiggy_kernel
_sample = squiggy_kernel.get_sample('${escapedName}')
if _sample and _sample._bam_info and 'ref_mapping' in _sample._bam_info:
    _reads = _sample._bam_info['ref_mapping'].get('${escapedRef}', [])
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

        // Convert sample names to JSON for safe Python serialization
        const sampleNamesJson = JSON.stringify(sampleNames);

        // Build maxReads parameter if provided
        const maxReadsParam =
            maxReads !== undefined && maxReads !== null ? `, max_reads=${maxReads}` : '';

        const code = `
import squiggy

# Generate signal overlay comparison plot - will be automatically routed to Plots pane
squiggy.plot_signal_overlay_comparison(
    sample_names=${sampleNamesJson},
    normalization='${normalization}',
    theme='${theme}'${maxReadsParam}
)
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

        // Convert sample names to JSON for safe Python serialization
        const sampleNamesJson = JSON.stringify(sampleNames);

        // Escape single quotes in reference name for Python strings
        const escapedRefName = referenceName.replace(/'/g, "\\'");

        // Build maxReads parameter if provided
        const maxReadsParam =
            maxReads !== undefined && maxReads !== null ? `, max_reads=${maxReads}` : '';

        const code = `
import squiggy

# Generate delta comparison plot - will be automatically routed to Plots pane
squiggy.plot_delta_comparison(
    sample_names=${sampleNamesJson},
    reference_name='${escapedRefName}',
    normalization='${normalization}',
    theme='${theme}'${maxReadsParam}
)
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
        sampleColors?: Record<string, string>
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
        const sampleNamesJson = JSON.stringify(sampleNames);
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

        const code = `
import squiggy

# Generate aggregate comparison plot - will be automatically routed to Plots pane
squiggy.plot_aggregate_comparison(
    sample_names=${sampleNamesJson},
    reference_name='${escapedRefName}',
    metrics=${metricsJson},
    normalization='${normalization}',
    theme='${theme}'${maxReadsParam}${sampleColorsParam}
)
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
