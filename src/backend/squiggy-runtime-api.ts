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
        showSignalPoints: boolean = false,
        sampleName?: string
    ): Promise<void> {
        const readIdsJson = JSON.stringify(readIds);
        const enabledModTypesJson = JSON.stringify(enabledModTypes);

        // Build sample name parameter if in multi-sample mode
        const sampleNameParam = sampleName ? `, sample_name='${sampleName}'` : '';

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
    show_signal_points=${showSignalPoints ? 'True' : 'False'}${sampleNameParam}
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
    show_signal_points=${showSignalPoints ? 'True' : 'False'}${sampleNameParam}
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
     * @param showModifications - Show modifications heatmap panel (default true)
     * @param showPileup - Show base pileup panel (default true)
     * @param showDwellTime - Show dwell time track panel (default true)
     * @param showSignal - Show signal track panel (default true)
     * @param showQuality - Show quality track panel (default true)
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
        clipXAxisToAlignment: boolean = true,
        sampleName?: string
    ): Promise<void> {
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

        const code = `
import squiggy

# Generate aggregate plot - will be automatically routed to Plots pane
squiggy.plot_aggregate(
    reference_name='${escapedRefName}',
    max_reads=${maxReads},
    normalization='${normalization}',
    theme='${theme}',
    show_modifications=${showModifications ? 'True' : 'False'},
    mod_filter=${modFilterDict},
    show_pileup=${showPileup ? 'True' : 'False'},
    show_dwell_time=${showDwellTime ? 'True' : 'False'},
    show_signal=${showSignal ? 'True' : 'False'},
    show_quality=${showQuality ? 'True' : 'False'},
    clip_x_to_alignment=${clipXAxisToAlignment ? 'True' : 'False'}${sampleNameParam}
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
            console.log(
                `[loadSample] Starting to load sample '${sampleName}' with POD5: ${pod5Path}${bamPath ? ` BAM: ${bamPath}` : ''}`
            );
            const startTime = Date.now();

            // Load sample silently
            console.log(`[loadSample] Executing Python code to load sample...`);
            await this._client.executeSilent(code);
            const executeSilentTime = Date.now();
            console.log(
                `[loadSample] executeSilent completed in ${executeSilentTime - startTime}ms`
            );

            // Get read count for this sample
            console.log(`[loadSample] Querying read count for sample '${sampleName}'...`);
            const numReads = await this._client.getVariable(
                `len(_squiggy_session.get_sample('${escapedSampleName}').read_ids)`
            );
            const queryTime = Date.now();
            console.log(
                `[loadSample] Got ${numReads} reads in ${queryTime - executeSilentTime}ms (total: ${queryTime - startTime}ms)`
            );

            return { numReads: numReads as number };
        } catch (error) {
            console.error(`[loadSample] Error loading sample '${sampleName}':`, error);
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
            const sampleInfoJson = await this._client.getVariable('_sample_info_json');

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

            // Handle the case where sampleInfoJson is a string representation of JSON
            if (!sampleInfoJson) {
                return null;
            }

            try {
                // If it's a string like "null", parse it correctly
                const jsonString = String(sampleInfoJson);
                const parsed = JSON.parse(jsonString);
                return parsed;
            } catch (parseError) {
                // If parsing fails, sample likely doesn't exist in registry
                console.warn(`Could not parse sample info for '${escapedName}':`, parseError);
                return null;
            }
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
            console.log(
                `[getReadIdsAndReferencesForSample] Fetching data for sample '${sampleName}'...`
            );
            const startTime = Date.now();

            // Execute setup code first to create variables
            // Uses ref_counts (reference → read count) which is built once during sample load
            const setupCode = `
_sample = _squiggy_session.get_sample('${escapedName}')
if _sample:
    _read_ids = _sample.read_ids
    if _sample.bam_info and 'ref_counts' in _sample.bam_info:
        _refs = list(_sample.bam_info['ref_counts'].keys())
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
            console.log(
                `[getReadIdsAndReferencesForSample] Got ${data.read_ids?.length || 0} reads and ${data.references?.length || 0} references in ${elapsed}ms`
            );

            return {
                readIds: data.read_ids || [],
                references: data.references || [],
            };
        } catch (error) {
            console.warn(
                `Failed to get read IDs and references for sample '${sampleName}':`,
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
    async getReadIdsForSample(sampleName: string): Promise<string[]> {
        const escapedName = sampleName.replace(/'/g, "\\'");

        try {
            console.log(`[getReadIdsForSample] Fetching read IDs for sample '${sampleName}'...`);
            const startTime = Date.now();

            // Extract read IDs safely without trying to serialize the Sample object
            const code = `
_sample = _squiggy_session.get_sample('${escapedName}')
_read_ids = _sample.read_ids if _sample else []
_read_ids
`;
            const readIds = await this._client.getVariable(code);
            const elapsed = Date.now() - startTime;
            const readIdArray = (readIds as string[]) || [];
            console.log(`[getReadIdsForSample] Got ${readIdArray.length} read IDs in ${elapsed}ms`);
            return readIdArray;
        } catch (error) {
            console.warn(`Failed to get read IDs for sample '${sampleName}':`, error);
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
_sample = _squiggy_session.get_sample('${escapedName}')
if _sample and _sample.bam_info and 'ref_mapping' in _sample.bam_info:
    ${tempVar} = list(_sample.bam_info['ref_mapping'].keys())
else:
    ${tempVar} = []
`);

            // Read the temp variable
            const references = await this._client.getVariable(tempVar);

            // Clean up
            await this._client.executeSilent(`del ${tempVar}`);

            return (references as string[]) || [];
        } catch (error) {
            console.warn(`Failed to get references for sample '${sampleName}':`, error);
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
_sample = _squiggy_session.get_sample('${escapedName}')
if _sample and _sample.bam_info and 'ref_counts' in _sample.bam_info:
    _counts = _sample.bam_info['ref_counts']
else:
    _counts = {}
`;
            await this._client.executeSilent(setupCode);
            const counts = await this._client.getVariable('_counts');
            return (counts as { [referenceName: string]: number }) || {};
        } catch (error) {
            console.warn(`Failed to get reference read counts for sample '${sampleName}':`, error);
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
_sample = _squiggy_session.get_sample('${escapedName}')
if _sample and _sample.bam_info and 'ref_mapping' in _sample.bam_info:
    _reads = _sample.bam_info['ref_mapping'].get('${escapedRef}', [])
else:
    _reads = []
`;
            await this._client.executeSilent(setupCode);
            const readIds = await this._client.getVariable('_reads');
            return (readIds as string[]) || [];
        } catch (error) {
            console.warn(
                `Failed to get reads for reference '${referenceName}' in sample '${sampleName}':`,
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

        // Build maxReads parameter if provided
        const maxReadsParam =
            maxReads !== undefined && maxReads !== null ? `, max_reads=${maxReads}` : '';

        const code = `
import squiggy

# Generate delta comparison plot - will be automatically routed to Plots pane
squiggy.plot_delta_comparison(
    sample_names=${sampleNamesJson},
    normalization='${normalization}',
    theme='${theme}'${maxReadsParam}
)
`;

        try {
            // Execute silently - plot will appear in Plots pane automatically
            await this._client.executeSilent(code);
        } catch (error) {
            throw new Error(`Failed to generate delta comparison plot: ${error}`);
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

            throw new Error(`Failed to generate aggregate comparison plot: ${error}`);
        }
    }
}
