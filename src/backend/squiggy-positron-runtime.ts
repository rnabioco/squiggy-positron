/**
 * Positron Runtime Backend
 *
 * Uses Positron's runtime API to execute Python code in the active Jupyter kernel.
 * Based on patterns from positron/extensions/positron-python source code.
 */

import * as positron from 'positron';

export class PositronRuntime {
    /**
     * Check if Positron runtime is available
     */
    isAvailable(): boolean {
        try {
            // Check if positron global is defined
            return typeof positron !== 'undefined' && typeof positron.runtime !== 'undefined';
        } catch {
            return false;
        }
    }

    /**
     * Ensure the Python kernel is ready to execute code
     * Waits up to 10 seconds for kernel to be ready after restart
     */
    private async ensureKernelReady(): Promise<void> {
        const session = await positron.runtime.getForegroundSession();

        if (!session) {
            throw new Error('No Python kernel is running. Please start a Python console first.');
        }

        // Check if session has onDidChangeRuntimeState event (may not exist in all Positron versions)
        const hasStateEvent = typeof (session as any).onDidChangeRuntimeState === 'function';

        if (hasStateEvent) {
            // Use event-based approach if available
            return this.ensureKernelReadyViaEvents(session);
        } else {
            // Fallback to polling approach
            return this.ensureKernelReadyViaPolling();
        }
    }

    /**
     * Ensure kernel ready using event-based approach (Positron API with onDidChangeRuntimeState)
     */
    private async ensureKernelReadyViaEvents(session: any): Promise<void> {
        return new Promise<void>((resolve, reject) => {
            const timeout = setTimeout(() => {
                reject(new Error('Timeout waiting for Python kernel to be ready'));
            }, 10000); // 10 second timeout

            // Function to check if current state is ready
            const checkState = (state: string) => {
                console.log(`Squiggy: Kernel state is ${state}`);

                // Ready states - can execute code
                if (state === 'ready' || state === 'idle' || state === 'busy') {
                    clearTimeout(timeout);
                    resolve();
                    return true;
                }

                // Failed states - cannot execute code
                if (state === 'offline' || state === 'exited') {
                    clearTimeout(timeout);
                    reject(new Error(`Python kernel is ${state}. Please start a Python console.`));
                    return true;
                }

                // Transitioning states - wait
                // uninitialized, initializing, starting, restarting
                return false;
            };

            // Listen for state changes
            const disposable = session.onDidChangeRuntimeState((state: string) => {
                if (checkState(state)) {
                    disposable.dispose();
                }
            });

            // Also check current state immediately (in case already ready)
            Promise.resolve(
                positron.runtime.executeCode(
                    'python',
                    '1+1',
                    false,
                    true,
                    positron.RuntimeCodeExecutionMode.Silent
                )
            )
                .then(() => {
                    // Kernel responded - it's ready
                    clearTimeout(timeout);
                    disposable.dispose();
                    resolve();
                })
                .catch(() => {
                    // Kernel didn't respond - wait for state change
                    // The onDidChangeRuntimeState listener will handle it
                });
        });
    }

    /**
     * Ensure kernel ready using polling approach (fallback for older Positron versions)
     */
    private async ensureKernelReadyViaPolling(): Promise<void> {
        const startTime = Date.now();
        const maxWaitMs = 10000; // 10 seconds
        const retryDelayMs = 500; // 500ms between retries

        while (Date.now() - startTime < maxWaitMs) {
            try {
                // Try to execute a simple test command
                await positron.runtime.executeCode(
                    'python',
                    '1+1',
                    false,
                    true,
                    positron.RuntimeCodeExecutionMode.Silent
                );
                // Success - kernel is ready
                console.log('Squiggy: Kernel is ready (polling check)');
                return;
            } catch (_error) {
                // Kernel not ready yet, wait and retry
                await new Promise((resolve) => setTimeout(resolve, retryDelayMs));
            }
        }

        // Timeout reached
        throw new Error('Timeout waiting for Python kernel to be ready');
    }

    /**
     * Execute Python code in the active kernel
     *
     * @param code Python code to execute
     * @param focus Whether to focus the console
     * @param allowIncomplete Whether to allow incomplete statements
     * @param mode Execution mode (silent by default to hide imports)
     * @param observer Optional observer for capturing output
     * @returns Promise that resolves with the result object containing MIME type mappings
     */
    async executeCode(
        code: string,
        focus: boolean = false,
        allowIncomplete: boolean = true,
        mode: positron.RuntimeCodeExecutionMode = positron.RuntimeCodeExecutionMode.Silent,
        observer?: positron.RuntimeCodeExecutionObserver
    ): Promise<Record<string, unknown>> {
        if (!this.isAvailable()) {
            throw new Error('Positron runtime not available');
        }

        // Ensure kernel is ready before executing code
        await this.ensureKernelReady();

        try {
            return await positron.runtime.executeCode(
                'python',
                code,
                focus,
                allowIncomplete,
                mode,
                undefined, // errorBehavior
                observer
            );
        } catch (error) {
            throw new Error(`Failed to execute Python code: ${error}`);
        }
    }

    /**
     * Execute code silently without console output
     *
     * @param code Python code to execute
     * @returns Promise that resolves when execution completes
     */
    async executeSilent(code: string): Promise<void> {
        await this.executeCode(
            code,
            false, // focus=false
            true,
            positron.RuntimeCodeExecutionMode.Silent
        );
    }

    /**
     * Execute code and capture printed output via observer
     *
     * NOTE: This will show output in console. Use only when absolutely necessary.
     * Prefer getVariable() for reading data from Python memory.
     *
     * @param code Python code to execute
     * @returns Promise that resolves with captured output
     */
    private async executeWithOutput(code: string): Promise<string> {
        return new Promise((resolve, reject) => {
            let output = '';

            this.executeCode(
                code,
                false,
                true,
                positron.RuntimeCodeExecutionMode.Silent, // Try Silent mode
                {
                    onOutput: (message: string) => {
                        output += message;
                    },
                    onFinished: () => {
                        resolve(output);
                    },
                }
            ).catch(reject);
        });
    }

    /**
     * Get a Python variable value directly from the kernel
     *
     * @param varName Python variable name (can include indexing like 'var[0:10]')
     * @returns Promise that resolves with the variable value
     */
    async getVariable(varName: string): Promise<unknown> {
        const session = await positron.runtime.getForegroundSession();
        if (!session || session.runtimeMetadata.languageId !== 'python') {
            throw new Error('No active Python session');
        }

        // Convert the Python value to JSON in Python, then read that
        const tempVar = '_squiggy_temp_' + Math.random().toString(36).substr(2, 9);

        try {
            await this.executeSilent(`
import json
${tempVar} = json.dumps(${varName})
`);

            const [[variable]] = await positron.runtime.getSessionVariables(
                session.metadata.sessionId,
                [[tempVar]]
            );

            // Clean up temp variable
            await this.executeSilent(`
if '${tempVar}' in globals():
    del ${tempVar}
`);

            if (!variable) {
                throw new Error(`Variable ${varName} not found`);
            }

            // display_value contains the JSON string (as a Python string repr)
            // We need to parse it: Python repr -> actual string -> JSON parse
            // e.g., "'[1,2,3]'" -> "[1,2,3]" -> [1,2,3]
            const jsonString = variable.display_value;

            // Remove outer quotes if present (Python string repr)
            const cleaned = jsonString.replace(/^['"]|['"]$/g, '');

            return JSON.parse(cleaned);
        } catch (error) {
            // Clean up temp variable on error
            await this.executeSilent(
                `
if '${tempVar}' in globals():
    del ${tempVar}
`
            ).catch(() => {}); // Ignore cleanup errors
            throw new Error(`Failed to get variable ${varName}: ${error}`);
        }
    }

    /**
     * Load a POD5 file
     *
     * Executes squiggy.load_pod5() in the kernel. The reader and read_ids
     * are stored in kernel variables accessible from console/notebooks.
     *
     * Does NOT preload read IDs - use getReadIds() to fetch them on-demand.
     */
    async loadPOD5(filePath: string): Promise<{ numReads: number }> {
        // Escape single quotes in path
        const escapedPath = filePath.replace(/'/g, "\\'");

        // Load file silently (no console output)
        // This populates _squiggy_session.reader and _squiggy_session.read_ids
        await this.executeSilent(`
import squiggy
squiggy.load_pod5('${escapedPath}')
`);

        // Get read count from session (no global variables created)
        const numReads = await this.getVariable('len(squiggy.io._squiggy_session.read_ids)');

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

        // Read from session instead of global variable
        const readIds = await this.getVariable(`squiggy.io._squiggy_session.read_ids${sliceStr}`);

        return readIds as string[];
    }

    /**
     * Load a BAM file
     *
     * Does NOT preload reference mapping - use getReferences() and getReadsForReference()
     * to fetch data on-demand.
     */
    async loadBAM(filePath: string): Promise<{
        numReads: number;
        hasModifications: boolean;
        modificationTypes: string[];
        hasProbabilities: boolean;
        hasEventAlignment: boolean;
    }> {
        const escapedPath = filePath.replace(/'/g, "\\'");

        // Load BAM silently (no console output)
        // This populates _squiggy_session.bam_info and _squiggy_session.ref_mapping
        await this.executeSilent(`
import squiggy
squiggy.load_bam('${escapedPath}')
`);

        // Read metadata from session (no global variables created)
        const numReads = await this.getVariable(
            "squiggy.io._squiggy_session.bam_info['num_reads']"
        );
        const hasModifications = await this.getVariable(
            "squiggy.io._squiggy_session.bam_info.get('has_modifications', False)"
        );
        const modificationTypes = await this.getVariable(
            "squiggy.io._squiggy_session.bam_info.get('modification_types', [])"
        );
        const hasProbabilities = await this.getVariable(
            "squiggy.io._squiggy_session.bam_info.get('has_probabilities', False)"
        );
        const hasEventAlignment = await this.getVariable(
            "squiggy.io._squiggy_session.bam_info.get('has_event_alignment', False)"
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
        // Read from session instead of global variable
        const references = await this.getVariable(
            'list(squiggy.io._squiggy_session.ref_mapping.keys())'
        );
        return references as string[];
    }

    /**
     * Get read IDs mapping to a specific reference
     */
    async getReadsForReference(referenceName: string): Promise<string[]> {
        const escapedRef = referenceName.replace(/'/g, "\\'");

        // Read from session instead of global variable
        const readIds = await this.getVariable(
            `squiggy.io._squiggy_session.ref_mapping.get('${escapedRef}', [])`
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
            await this.executeSilent(code);

            // Check if there was an error during plot generation
            const plotError = await this.getVariable('_squiggy_plot_error').catch(() => null);
            if (plotError !== null) {
                throw new Error(`Plot generation failed:\n${plotError}`);
            }

            // Clean up temporary variable
            await this.executeSilent(`
if '_squiggy_plot_error' in globals():
    del _squiggy_plot_error
`);
        } catch (error) {
            // Clean up on error
            await this.executeSilent(
                `
if '_squiggy_plot_error' in globals():
    del _squiggy_plot_error
`
            ).catch(() => {});
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
            await this.executeSilent(code);
        } catch (error) {
            throw new Error(`Failed to generate aggregate plot: ${error}`);
        }
    }

    /**
     * Check if squiggy package is installed in the kernel
     */
    async isSquiggyInstalled(): Promise<boolean> {
        const code = `
try:
    import squiggy
    # Verify package has expected functions
    _squiggy_installed = (hasattr(squiggy, 'load_pod5') and
                          hasattr(squiggy, 'load_bam') and
                          hasattr(squiggy, 'plot_read'))
except ImportError:
    _squiggy_installed = False
`;

        try {
            await this.executeSilent(code);
            const result = await this.getVariable('_squiggy_installed');
            await this.executeSilent('del _squiggy_installed').catch(() => {});
            return result === true;
        } catch {
            return false; // ImportError or other exception means not installed
        }
    }

    /**
     * Get squiggy package version
     */
    async getSquiggyVersion(): Promise<string | null> {
        try {
            // Import squiggy first, then get version
            await this.executeSilent('import squiggy');
            const version = await this.getVariable('squiggy.__version__');
            return version as string | null;
        } catch {
            // squiggy not installed or no __version__ attribute
            return null;
        }
    }

    /**
     * Detect Python environment type
     *
     * @returns Information about the Python environment
     */
    async detectEnvironmentType(): Promise<{
        isVirtualEnv: boolean;
        isConda: boolean;
        isExternallyManaged: boolean;
        pythonPath: string;
        prefix: string;
        basePrefix: string;
    }> {
        const code = `
import sys
import os
import json

result = {
    'is_venv': hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix),
    'is_conda': 'CONDA_PREFIX' in os.environ or 'CONDA_DEFAULT_ENV' in os.environ,
    'is_externally_managed': os.path.exists(os.path.join(sys.prefix, 'EXTERNALLY-MANAGED')),
    'python_path': sys.executable,
    'prefix': sys.prefix,
    'base_prefix': getattr(sys, 'base_prefix', sys.prefix)
}
_squiggy_env_info = json.dumps(result)
`;

        try {
            await this.executeSilent(code);
            const envInfo = await this.getVariable('_squiggy_env_info');
            await this.executeSilent('del _squiggy_env_info');

            // getVariable() already parses JSON, so envInfo is already a JS object
            // Python returns snake_case keys, convert to camelCase for TypeScript
            const result = envInfo as any;
            return {
                isVirtualEnv: result.is_venv,
                isConda: result.is_conda,
                isExternallyManaged: result.is_externally_managed,
                pythonPath: result.python_path,
                prefix: result.prefix,
                basePrefix: result.base_prefix,
            };
        } catch (error) {
            throw new Error(`Failed to detect environment type: ${error}`);
        }
    }

    /**
     * Install squiggy package to the kernel from extension directory
     *
     * @param extensionPath Path to the extension directory containing pyproject.toml
     */
    async installSquiggy(extensionPath: string): Promise<void> {
        // Detect environment first
        const envInfo = await this.detectEnvironmentType();

        // Refuse installation on externally-managed system Python
        if (envInfo.isExternallyManaged && !envInfo.isVirtualEnv && !envInfo.isConda) {
            throw new Error(
                'EXTERNALLY_MANAGED_ENVIRONMENT: Cannot install squiggy in externally-managed Python environment.\n\n' +
                    `Your Python installation (${envInfo.pythonPath}) is managed by your system package manager ` +
                    '(e.g., Homebrew, apt, dnf) and cannot be modified directly.\n\n' +
                    'Please create a virtual environment first:\n' +
                    '1. Run: python3 -m venv .venv\n' +
                    '2. Select the new environment in Positron (Interpreter selector)\n' +
                    '3. Restart the Python console\n' +
                    '4. Try opening the POD5 file again\n\n' +
                    `Or install squiggy manually: pip install -e ${extensionPath}`
            );
        }

        // Use JSON serialization for cross-platform path safety (Windows backslashes, etc.)
        const pathJson = JSON.stringify(extensionPath);

        const code = `
import subprocess
import sys

# Check if pip is available
pip_check = subprocess.run(
    [sys.executable, '-m', 'pip', '--version'],
    capture_output=True,
    text=True
)
if pip_check.returncode != 0:
    raise Exception('pip is not available in this Python environment. Please install pip first.')

# Path is already JSON-stringified by TypeScript, interpolate directly
extension_path = ${pathJson}

# Install with timeout (5 minutes)
result = subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '-e', extension_path],
    capture_output=True,
    text=True,
    timeout=300
)
if result.returncode != 0:
    raise Exception(f'Installation failed: {result.stderr}')
print('SUCCESS')
`;

        try {
            const output = await this.executeWithOutput(code);
            if (!output.includes('SUCCESS')) {
                throw new Error(`Installation failed: ${output}`);
            }
        } catch (error) {
            throw new Error(`Failed to install squiggy: ${error}`);
        }
    }

    /**
     * Load and validate a FASTA file
     */
    async loadFASTA(fastaPath: string): Promise<void> {
        await this.ensureKernelReady();

        const code = `
import squiggy

# Load FASTA file using squiggy.load_fasta()
# This populates _squiggy_session.fasta_path and _squiggy_session.fasta_info
squiggy.load_fasta(${JSON.stringify(fastaPath)})
`;

        await this.executeSilent(code);
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
        await this.ensureKernelReady();

        // Execute search and store results in kernel variable
        const searchCode = `
import squiggy

# Search for motif matches
_squiggy_motif_matches = list(squiggy.search_motif(
    fasta_file=${JSON.stringify(fastaFile)},
    motif=${JSON.stringify(motif)},
    region=${region ? JSON.stringify(region) : 'None'},
    strand=${JSON.stringify(strand)}
))

# Convert matches to dicts for JSON serialization
_squiggy_motif_matches_json = [
    {
        'chrom': m.chrom,
        'position': m.position,
        'sequence': m.sequence,
        'strand': m.strand
    }
    for m in _squiggy_motif_matches
]
`;

        await this.executeSilent(searchCode);

        // Retrieve matches using getVariable
        const matches = await this.getVariable('_squiggy_motif_matches_json');

        // Clean up temporary variables
        await this.executeSilent(
            `
if '_squiggy_motif_matches' in globals():
    del _squiggy_motif_matches
if '_squiggy_motif_matches_json' in globals():
    del _squiggy_motif_matches_json
`
        ).catch(() => {});

        return (matches as any[]) || [];
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
        await this.ensureKernelReady();

        const code = `
import squiggy

# Generate motif aggregate plot
html = squiggy.plot_motif_aggregate(
    fasta_file=${JSON.stringify(fastaFile)},
    motif=${JSON.stringify(motif)},
    match_index=${matchIndex},
    window=${window},
    max_reads=${maxReads},
    normalization=${JSON.stringify(normalization)},
    theme=${JSON.stringify(theme)}
)

# Route to Positron Plots pane
squiggy.io._route_to_plots_pane(html)
`;

        await this.executeCode(code, false, true, positron.RuntimeCodeExecutionMode.Silent);
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
        await this.ensureKernelReady();

        const code = `
import squiggy

# Generate aggregate plot across all motif matches
html = squiggy.plot_motif_aggregate_all(
    fasta_file=${JSON.stringify(fastaFile)},
    motif=${JSON.stringify(motif)},
    upstream=${upstream},
    downstream=${downstream},
    max_reads_per_motif=${maxReadsPerMotif},
    normalization=${JSON.stringify(normalization)},
    theme=${JSON.stringify(theme)}
)

# Route to Positron Plots pane
squiggy.io._route_to_plots_pane(html)
`;

        await this.executeCode(code, false, true, positron.RuntimeCodeExecutionMode.Silent);
    }
}
