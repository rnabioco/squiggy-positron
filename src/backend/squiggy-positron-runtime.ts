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
     * Execute Python code in the active kernel
     *
     * @param code Python code to execute
     * @param focus Whether to focus the console
     * @param allowIncomplete Whether to allow incomplete statements
     * @param mode Execution mode (silent by default to hide imports)
     * @param observer Optional observer for capturing output
     */
    async executeCode(
        code: string,
        focus: boolean = false,
        allowIncomplete: boolean = true,
        mode: positron.RuntimeCodeExecutionMode = positron.RuntimeCodeExecutionMode.Silent,
        observer?: positron.RuntimeCodeExecutionObserver
    ): Promise<void> {
        if (!this.isAvailable()) {
            throw new Error('Positron runtime not available');
        }

        try {
            await positron.runtime.executeCode(
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
     * Execute code and capture output
     *
     * @param code Python code to execute
     * @returns Promise that resolves with captured output
     */
    async executeWithOutput(code: string): Promise<string> {
        return new Promise((resolve, reject) => {
            let output = '';
            let errorOutput = '';

            this.executeCode(
                code,
                false,
                true,
                positron.RuntimeCodeExecutionMode.Silent,
                {
                    onOutput: (message: string) => {
                        output += message;
                    },
                    onError: (message: string) => {
                        errorOutput += message;
                    },
                    onFinished: () => {
                        if (errorOutput) {
                            reject(new Error(errorOutput));
                        } else {
                            resolve(output);
                        }
                    },
                }
            ).catch(reject);
        });
    }

    /**
     * Load a POD5 file
     *
     * Executes squiggy.load_pod5() in the kernel. The reader and read_ids
     * are stored in kernel variables accessible from console/notebooks.
     */
    async loadPOD5(filePath: string): Promise<{ numReads: number; readIds?: string[] }> {
        // Escape single quotes in path
        const escapedPath = filePath.replace(/'/g, "\\'");

        // Execute load command - this creates variables in the kernel
        const code = `
import squiggy
import json
_squiggy_reader, _squiggy_read_ids = squiggy.load_pod5('${escapedPath}')
print(json.dumps({
    'num_reads': len(_squiggy_read_ids),
    'preview_ids': _squiggy_read_ids[:100]
}))
`;

        try {
            const output = await this.executeWithOutput(code);
            const data = JSON.parse(output.trim());
            return {
                numReads: data.num_reads,
                readIds: data.preview_ids,
            };
        } catch (error) {
            throw new Error(`Failed to load POD5 file: ${error}`);
        }
    }

    /**
     * Load a BAM file
     */
    async loadBAM(filePath: string): Promise<{
        numReads: number;
        referenceToReads: Record<string, string[]>;
        hasModifications: boolean;
        modificationTypes: string[];
        hasProbabilities: boolean;
    }> {
        const escapedPath = filePath.replace(/'/g, "\\'");

        const code = `
import squiggy
import json
_squiggy_bam_info = squiggy.load_bam('${escapedPath}')
_squiggy_ref_mapping = squiggy.get_read_to_reference_mapping()
print(json.dumps({
    'num_reads': _squiggy_bam_info['num_reads'],
    'reference_to_reads': _squiggy_ref_mapping,
    'has_modifications': _squiggy_bam_info.get('has_modifications', False),
    'modification_types': _squiggy_bam_info.get('modification_types', []),
    'has_probabilities': _squiggy_bam_info.get('has_probabilities', False)
}))
`;

        try {
            const output = await this.executeWithOutput(code);
            const data = JSON.parse(output.trim());
            return {
                numReads: data.num_reads,
                referenceToReads: data.reference_to_reads || {},
                hasModifications: data.has_modifications || false,
                modificationTypes: data.modification_types || [],
                hasProbabilities: data.has_probabilities || false,
            };
        } catch (error) {
            throw new Error(`Failed to load BAM file: ${error}`);
        }
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
        enabledModTypes: string[] = []
    ): Promise<string> {
        const readIdsJson = JSON.stringify(readIds);

        const enabledModTypesJson = JSON.stringify(enabledModTypes);

        const code = `
import squiggy
import tempfile
import json

# Generate plot HTML
${
    readIds.length === 1
        ? `html = squiggy.plot_read('${readIds[0]}', mode='${mode}', normalization='${normalization}', theme='${theme}', show_dwell_time=${showDwellTime ? 'True' : 'False'}, show_labels=${showBaseAnnotations ? 'True' : 'False'}, scale_dwell_time=${scaleDwellTime ? 'True' : 'False'}, min_mod_probability=${minModProbability}, enabled_mod_types=${enabledModTypesJson})`
        : `html = squiggy.plot_reads(${readIdsJson}, mode='${mode}', normalization='${normalization}', theme='${theme}', show_dwell_time=${showDwellTime ? 'True' : 'False'}, show_labels=${showBaseAnnotations ? 'True' : 'False'}, scale_dwell_time=${scaleDwellTime ? 'True' : 'False'}, min_mod_probability=${minModProbability}, enabled_mod_types=${enabledModTypesJson})`
}

# Write to temp file
temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False)
temp_file.write(html)
temp_file.close()

print(temp_file.name)
`;

        try {
            const output = await this.executeWithOutput(code);
            return output.trim();
        } catch (error) {
            throw new Error(`Failed to generate plot: ${error}`);
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
        const code = `
try:
    import squiggy
    print(squiggy.__version__)
except (ImportError, AttributeError):
    print('unavailable')
`;

        try {
            const output = await this.executeWithOutput(code);
            const version = output.trim();
            return version === 'unavailable' ? null : version;
        } catch {
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
