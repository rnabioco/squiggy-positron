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
     * @param observer Optional observer for capturing output
     */
    async executeCode(
        code: string,
        focus: boolean = false,
        allowIncomplete: boolean = true,
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
                undefined, // mode
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

            this.executeCode(code, false, true, {
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
                }
            }).catch(reject);
        });
    }

    /**
     * Load a POD5 file
     *
     * Executes squiggy.load_pod5() in the kernel. The reader and read_ids
     * are stored in kernel variables accessible from console/notebooks.
     */
    async loadPOD5(filePath: string): Promise<{ numReads: number, readIds?: string[] }> {
        // Escape single quotes in path
        const escapedPath = filePath.replace(/'/g, "\\'");

        // Execute load command - this creates variables in the kernel
        const code = `
import squiggy
import json
_squiggy_reader, _squiggy_read_ids = squiggy.load_pod5('${escapedPath}')
# Print structured output for extension to parse
print('SQUIGGY_LOADED:' + json.dumps({
    'num_reads': len(_squiggy_read_ids),
    'preview_ids': _squiggy_read_ids[:100]  # First 100 for tree view
}))
`;

        try {
            const output = await this.executeWithOutput(code);

            // Parse output for structured data
            const match = output.match(/SQUIGGY_LOADED:(\{.*\})/);
            if (match) {
                const data = JSON.parse(match[1]);
                return {
                    numReads: data.num_reads,
                    readIds: data.preview_ids
                };
            } else {
                // Fallback: just show success message
                return { numReads: 0 };
            }
        } catch (error) {
            throw new Error(`Failed to load POD5 file: ${error}`);
        }
    }

    /**
     * Load a BAM file
     */
    async loadBAM(filePath: string): Promise<{ numReads: number }> {
        const escapedPath = filePath.replace(/'/g, "\\'");

        const code = `
import squiggy
import json
_squiggy_bam_info = squiggy.load_bam('${escapedPath}')
print('SQUIGGY_BAM_LOADED:' + json.dumps({
    'num_reads': _squiggy_bam_info['num_reads']
}))
`;

        try {
            const output = await this.executeWithOutput(code);

            const match = output.match(/SQUIGGY_BAM_LOADED:(\{.*\})/);
            if (match) {
                const data = JSON.parse(match[1]);
                return { numReads: data.num_reads };
            } else {
                return { numReads: 0 };
            }
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
        theme: string = 'LIGHT'
    ): Promise<string> {
        const readIdsJson = JSON.stringify(readIds);

        const code = `
import squiggy
import tempfile
import json

# Generate plot HTML
${readIds.length === 1
    ? `html = squiggy.plot_read('${readIds[0]}', mode='${mode}', normalization='${normalization}', theme='${theme}')`
    : `html = squiggy.plot_reads(${readIdsJson}, mode='${mode}', normalization='${normalization}', theme='${theme}')`
}

# Write to temp file
import os
temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False)
temp_file.write(html)
temp_file.close()

# Print file path for extension
print('SQUIGGY_PLOT_FILE:' + temp_file.name)
`;

        try {
            const output = await this.executeWithOutput(code);

            // Parse output for file path
            const match = output.match(/SQUIGGY_PLOT_FILE:(.*)/);
            if (match) {
                const filePath = match[1].trim();
                return filePath;
            } else {
                throw new Error('Failed to get plot file path from output');
            }
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
    print('SQUIGGY_INSTALLED:True')
except ImportError:
    print('SQUIGGY_INSTALLED:False')
`;

        try {
            const output = await this.executeWithOutput(code);
            return output.includes('SQUIGGY_INSTALLED:True');
        } catch {
            return false;
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
if result.returncode == 0:
    print('SQUIGGY_INSTALL:SUCCESS')
else:
    print('SQUIGGY_INSTALL:FAILED')
    print(result.stderr)
`;

        try {
            const output = await this.executeWithOutput(code);
            if (!output.includes('SQUIGGY_INSTALL:SUCCESS')) {
                throw new Error(`Installation failed: ${output}`);
            }
        } catch (error) {
            throw new Error(`Failed to install squiggy: ${error}`);
        }
    }
}
