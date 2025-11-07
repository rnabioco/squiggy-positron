/**
 * Python Package Management
 *
 * Handles detection and installation of the squiggy Python package
 * in the active Python environment.
 */

import * as vscode from 'vscode';
import { PositronRuntimeClient } from './positron-runtime-client';
import { ExternallyManagedEnvironmentError } from '../utils/error-handler';

/**
 * Information about the Python environment
 */
export interface EnvironmentInfo {
    isVirtualEnv: boolean;
    isConda: boolean;
    isExternallyManaged: boolean;
    pythonPath: string;
    prefix: string;
    basePrefix: string;
}

/**
 * Manages squiggy package installation and detection
 */
export class PackageManager {
    constructor(private readonly client: PositronRuntimeClient) {}

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
            await this.client.executeSilent(code);
            const result = await this.client.getVariable('_squiggy_installed');
            await this.client.executeSilent('del _squiggy_installed').catch(() => {});
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
            await this.client.executeSilent('import squiggy');
            const version = await this.client.getVariable('squiggy.__version__');
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
    async detectEnvironmentType(): Promise<EnvironmentInfo> {
        const code = `
import sys
import os

_squiggy_env_info = {
    'is_venv': hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix),
    'is_conda': 'CONDA_PREFIX' in os.environ or 'CONDA_DEFAULT_ENV' in os.environ,
    'is_externally_managed': os.path.exists(os.path.join(sys.prefix, 'EXTERNALLY-MANAGED')),
    'python_path': sys.executable,
    'prefix': sys.prefix,
    'base_prefix': getattr(sys, 'base_prefix', sys.prefix)
}
`;

        try {
            await this.client.executeSilent(code);
            // getVariable() will handle JSON encoding/decoding
            const envInfo = await this.client.getVariable('_squiggy_env_info');
            await this.client.executeSilent('del _squiggy_env_info');

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
            throw new ExternallyManagedEnvironmentError(envInfo.pythonPath);
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
            const output = await this.client.executeWithOutput(code);
            if (!output.includes('SUCCESS')) {
                throw new Error(`Installation failed: ${output}`);
            }
        } catch (error) {
            throw new Error(`Failed to install squiggy: ${error}`);
        }
    }

    /**
     * Show manual installation guide with copy-able commands
     */
    async showManualInstallationGuide(extensionPath: string): Promise<void> {
        const items = [
            {
                label: '1️⃣ Install uv',
                detail: 'curl -LsSf https://astral.sh/uv/install.sh | sh',
                description: '⭐ RECOMMENDED: Install uv package manager',
            },
            {
                label: '2️⃣ Create Virtual Environment with uv',
                detail: 'uv venv',
                description: '⭐ Create .venv in your project directory',
            },
            {
                label: '3️⃣ Activate venv',
                detail: 'source .venv/bin/activate',
                description: '⭐ Activate the virtual environment',
            },
            {
                label: '4️⃣ Install Squiggy with uv',
                detail: `uv pip install -e "${extensionPath}"`,
                description: '⭐ Install squiggy in editable mode',
            },
            {
                label: '5️⃣ Select Environment in Positron',
                detail: 'Use the Interpreter selector (bottom-right) to choose .venv',
                description: '⭐ Switch to your new virtual environment',
            },
            {
                label: '━━━━━━━━━━━━━━━━━━━━━━━━',
                detail: '',
                description: 'Alternative: Traditional pip (requires Python 3.8+, pip 21.3+)',
            },
            {
                label: '🐍 Alt: Create venv with pip',
                detail: 'python3 -m venv .venv',
                description: 'Only if you cannot use uv',
            },
            {
                label: '🐍 Alt: Install with pip',
                detail: `pip install -e "${extensionPath}"`,
                description: 'Only if you cannot use uv (may fail with old pip)',
            },
        ];

        const selected = await vscode.window.showQuickPick(items, {
            placeHolder: 'Click to copy command to clipboard',
            title: '⭐ Recommended: uv workflow (fast, modern, reliable)',
        });

        if (selected && selected.detail && selected.detail.length > 0) {
            await vscode.env.clipboard.writeText(selected.detail);
            vscode.window.showInformationMessage(`Copied to clipboard: ${selected.detail}`);
        }
    }

    /**
     * Prompt user to install squiggy package
     * @returns Action selected by user ('install', 'manual', or 'cancel')
     */
    async promptInstallSquiggy(): Promise<'install' | 'manual' | 'cancel'> {
        const choice = await vscode.window.showInformationMessage(
            'Squiggy requires the Python package "squiggy" to be installed in your active Python environment.',
            'Install Automatically',
            'Manual Instructions',
            'Cancel'
        );

        if (choice === 'Install Automatically') {
            return 'install';
        } else if (choice === 'Manual Instructions') {
            return 'manual';
        } else {
            return 'cancel';
        }
    }

    /**
     * Install squiggy package via pip with progress notification
     *
     * @param extensionPath Path to extension directory
     * @returns true if installation succeeded, false otherwise
     */
    async installSquiggyWithProgress(extensionPath: string): Promise<boolean> {
        try {
            return await vscode.window.withProgress(
                {
                    location: vscode.ProgressLocation.Notification,
                    title: 'Installing squiggy Python package...',
                    cancellable: false,
                },
                async () => {
                    try {
                        await this.installSquiggy(extensionPath);

                        vscode.window.showInformationMessage(
                            'Successfully installed squiggy Python package!'
                        );
                        return true;
                    } catch (error) {
                        const errorMessage = error instanceof Error ? error.message : String(error);

                        // Detect PEP 668 externally-managed environment errors
                        if (
                            errorMessage.includes('EXTERNALLY_MANAGED_ENVIRONMENT') ||
                            errorMessage.includes('externally-managed-environment') ||
                            errorMessage.includes('EXTERNALLY-MANAGED')
                        ) {
                            // Show detailed error with option to see manual instructions
                            const choice = await vscode.window.showErrorMessage(
                                'Cannot install squiggy: Python environment is externally managed by your ' +
                                    'system package manager. Please create a virtual environment first.',
                                'Show Instructions',
                                'Dismiss'
                            );

                            if (choice === 'Show Instructions') {
                                await this.showManualInstallationGuide(extensionPath);
                            }
                        } else if (
                            errorMessage.includes('setup.py" or "setup.cfg" not found') ||
                            errorMessage.includes('editable mode currently requires a setuptools-based build')
                        ) {
                            // Detect old pip version that doesn't support pyproject.toml editable installs
                            const choice = await vscode.window.showErrorMessage(
                                'Cannot install squiggy: Your pip version is too old to install packages ' +
                                    'with pyproject.toml in editable mode. Please upgrade pip or create a ' +
                                    'virtual environment with a newer Python version.',
                                'Show Instructions',
                                'Dismiss'
                            );

                            if (choice === 'Show Instructions') {
                                await this.showManualInstallationGuide(extensionPath);
                            }
                        } else {
                            // Generic installation error
                            vscode.window.showErrorMessage(
                                `Failed to install squiggy package: ${errorMessage}`
                            );
                        }
                        return false;
                    }
                }
            );
        } catch {
            return false;
        }
    }
}
