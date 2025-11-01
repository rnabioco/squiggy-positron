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
            throw new Error(
                'No Python kernel is running. Please start a Python console first.'
            );
        }

        // Check if session has runtimeMetadata to determine state
        // Note: We need to check the actual current state
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
            const disposable = session.onDidChangeRuntimeState((state: any) => {
                if (checkState(state)) {
                    disposable.dispose();
                }
            });

            // Also check current state immediately (in case already ready)
            // We'll try to execute a simple test to see if kernel responds
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
    ): Promise<Record<string, any>> {
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
    async getVariable(varName: string): Promise<any> {
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
            await this.executeSilent(`
if '${tempVar}' in globals():
    del ${tempVar}
`).catch(() => {}); // Ignore cleanup errors
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
        await this.executeSilent(`
import squiggy
_squiggy_reader, _squiggy_read_ids = squiggy.load_pod5('${escapedPath}')
`);

        // Get read count by reading variable directly (no print needed)
        const numReads = await this.getVariable('len(_squiggy_read_ids)');

        return { numReads };
    }

    /**
     * Get read IDs from loaded POD5 file
     *
     * @param offset Starting index (default 0)
     * @param limit Maximum number of read IDs to return (default all)
     */
    async getReadIds(offset: number = 0, limit?: number): Promise<string[]> {
        const sliceStr = limit ? `[${offset}:${offset + limit}]` : `[${offset}:]`;

        // Read variable slice directly (no print needed)
        const readIds = await this.getVariable(`_squiggy_read_ids${sliceStr}`);

        return readIds;
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
    }> {
        const escapedPath = filePath.replace(/'/g, "\\'");

        // Load BAM silently (no console output)
        await this.executeSilent(`
import squiggy
_squiggy_bam_info = squiggy.load_bam('${escapedPath}')
_squiggy_ref_mapping = squiggy.get_read_to_reference_mapping()
`);

        // Read metadata directly from variables (no print needed)
        const numReads = await this.getVariable("_squiggy_bam_info['num_reads']");
        const hasModifications = await this.getVariable(
            "_squiggy_bam_info.get('has_modifications', False)"
        );
        const modificationTypes = await this.getVariable(
            "_squiggy_bam_info.get('modification_types', [])"
        );
        const hasProbabilities = await this.getVariable(
            "_squiggy_bam_info.get('has_probabilities', False)"
        );

        return {
            numReads,
            hasModifications,
            modificationTypes: modificationTypes.map((x: any) => String(x)),
            hasProbabilities,
        };
    }

    /**
     * Get list of reference names from loaded BAM file
     */
    async getReferences(): Promise<string[]> {
        // Read keys directly from variable (no print needed)
        const references = await this.getVariable('list(_squiggy_ref_mapping.keys())');
        return references;
    }

    /**
     * Get read IDs mapping to a specific reference
     */
    async getReadsForReference(referenceName: string): Promise<string[]> {
        const escapedRef = referenceName.replace(/'/g, "\\'");

        // Read directly from variable (no print needed)
        const readIds = await this.getVariable(`_squiggy_ref_mapping.get('${escapedRef}', [])`);

        return readIds;
    }

    /**
     * Generate a plot for read(s)
     *
     * Creates HTML in kernel and writes to temp file, returns file path
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
    ): Promise<string> {
        const readIdsJson = JSON.stringify(readIds);

        const enabledModTypesJson = JSON.stringify(enabledModTypes);

        const code = `
import sys
import squiggy
import tempfile
import json
import traceback

# Initialize error tracking
_squiggy_plot_error = None
_squiggy_plot_path = None

try:
    # Note: We don't reload the squiggy module here because it would reset global state
    # (loaded POD5/BAM files). If you're developing and need to reload, restart the kernel.

    # Generate plot HTML
    ${
        readIds.length === 1
            ? `html = squiggy.plot_read('${readIds[0]}', mode='${mode}', normalization='${normalization}', theme='${theme}', show_dwell_time=${showDwellTime ? 'True' : 'False'}, show_labels=${showBaseAnnotations ? 'True' : 'False'}, scale_dwell_time=${scaleDwellTime ? 'True' : 'False'}, min_mod_probability=${minModProbability}, enabled_mod_types=${enabledModTypesJson}, downsample=${downsample}, show_signal_points=${showSignalPoints ? 'True' : 'False'})`
            : `html = squiggy.plot_reads(${readIdsJson}, mode='${mode}', normalization='${normalization}', theme='${theme}', show_dwell_time=${showDwellTime ? 'True' : 'False'}, show_labels=${showBaseAnnotations ? 'True' : 'False'}, scale_dwell_time=${scaleDwellTime ? 'True' : 'False'}, min_mod_probability=${minModProbability}, enabled_mod_types=${enabledModTypesJson}, downsample=${downsample}, show_signal_points=${showSignalPoints ? 'True' : 'False'})`
    }

    # Write to temp file
    _squiggy_temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False)
    _squiggy_temp_file.write(html)
    _squiggy_temp_file.close()

    # Store path in kernel variable (NO print!)
    _squiggy_plot_path = _squiggy_temp_file.name
except Exception as e:
    _squiggy_plot_error = f"{type(e).__name__}: {str(e)}\\n{traceback.format_exc()}"
    # Also print to console for debugging
    print(f"ERROR generating plot: {_squiggy_plot_error}", file=sys.stderr)
`;

        try {
            // Execute silently (won't throw even if Python code has errors)
            await this.executeSilent(code);

            // Check if there was an error during plot generation
            // Note: Python None becomes null in JavaScript after JSON parsing
            const plotError = await this.getVariable('_squiggy_plot_error').catch(() => null);
            if (plotError !== null) {
                throw new Error(`Plot generation failed:\n${plotError}`);
            }

            // Read the plot file path
            // Note: Python None becomes null in JavaScript after JSON parsing
            const filePath = await this.getVariable('_squiggy_plot_path').catch(() => null);
            if (filePath === null) {
                // If no error was reported but also no path, something went wrong silently
                throw new Error('Plot generation failed - no file path returned. The plot generation code may have been interrupted or failed silently.');
            }

            // Clean up temporary variables (ignore errors if they don't exist)
            await this.executeSilent(`
if '_squiggy_plot_path' in globals():
    del _squiggy_plot_path
if '_squiggy_plot_error' in globals():
    del _squiggy_plot_error
if '_squiggy_temp_file' in globals():
    del _squiggy_temp_file
`);

            return filePath;
        } catch (error) {
            // Clean up on error (ignore if variables don't exist)
            await this.executeSilent(`
if '_squiggy_plot_path' in globals():
    del _squiggy_plot_path
if '_squiggy_plot_error' in globals():
    del _squiggy_plot_error
if '_squiggy_temp_file' in globals():
    del _squiggy_temp_file
`).catch(() => {}); // Ignore cleanup errors
            throw error; // Re-throw the original error with full message
        }
    }

    /**
     * Generate an aggregate plot for a reference sequence
     * @param referenceName - Name of reference sequence from BAM file
     * @param maxReads - Maximum number of reads to sample (default 100)
     * @param normalization - Normalization method (ZNORM, MAD, etc.)
     * @param theme - Color theme (LIGHT or DARK)
     * @returns Path to temp file containing Bokeh HTML
     */
    async generateAggregatePlot(
        referenceName: string,
        maxReads: number = 100,
        normalization: string = 'ZNORM',
        theme: string = 'LIGHT'
    ): Promise<string> {
        // Escape single quotes in reference name for Python string
        const escapedRefName = referenceName.replace(/'/g, "\\'");

        const code = `
import squiggy
import tempfile

# Generate aggregate plot HTML
html = squiggy.plot_aggregate(
    reference_name='${escapedRefName}',
    max_reads=${maxReads},
    normalization='${normalization}',
    theme='${theme}'
)

# Write to temp file
_squiggy_temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False)
_squiggy_temp_file.write(html)
_squiggy_temp_file.close()

# Store path in kernel variable (NO print!)
_squiggy_plot_path = _squiggy_temp_file.name
`;

        try {
            // Execute silently
            await this.executeSilent(code);

            // Read variable directly from kernel memory
            const filePath = await this.getVariable('_squiggy_plot_path');

            // Clean up temporary variables
            await this.executeSilent('del _squiggy_plot_path, _squiggy_temp_file');

            return filePath;
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
    print('True')
except ImportError:
    print('False')
`;

        try {
            const output = await this.executeWithOutput(code);
            return output.trim() === 'True';
        } catch {
            return false;
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
            return version;
        } catch {
            // squiggy not installed or no __version__ attribute
            return null;
        }
    }

    /**
     * Install squiggy package to the kernel
     */
    async installSquiggy(): Promise<void> {
        const code = `
import subprocess
import sys
result = subprocess.run(
    [sys.executable, '-m', 'pip', 'install', 'squiggy'],
    capture_output=True,
    text=True
)
if result.returncode != 0:
    raise Exception(f'Installation failed: {result.stderr}')
print('SUCCESS')
`;

        try {
            const output = await this.executeWithOutput(code);
            if (output.trim() !== 'SUCCESS') {
                throw new Error(`Installation failed: ${output}`);
            }
        } catch (error) {
            throw new Error(`Failed to install squiggy: ${error}`);
        }
    }
}
