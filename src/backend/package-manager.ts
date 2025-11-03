/**
 * Python Package Management
 *
 * Handles detection and installation of the squiggy Python package
 * in the active Python environment.
 */

import * as vscode from 'vscode';
import * as path from 'path';
import { PositronRuntimeClient } from './positron-runtime-client';

/**
 * Manages squiggy package installation and detection using uv
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
     * Create a virtual environment using uv and install squiggy into it
     *
     * @param extensionPath Path to the extension directory
     * @returns Promise that resolves with venv path on success
     */
    private async createVenvWithUv(extensionPath: string): Promise<string> {
        // Determine venv location
        const workspaceFolders = vscode.workspace.workspaceFolders;
        let venvPath: string;

        if (workspaceFolders && workspaceFolders.length > 0) {
            // Use workspace folder if available
            const workspaceRoot = workspaceFolders[0].uri.fsPath;
            venvPath = path.join(workspaceRoot, '.venv');
        } else {
            // No workspace - create in user's home directory
            const homeDir = process.env.HOME || process.env.USERPROFILE || '~';
            venvPath = path.join(homeDir, '.squiggy', 'venv');
        }

        // Create venv and install in Python
        const pathJson = JSON.stringify(extensionPath);
        const venvPathJson = JSON.stringify(venvPath);

        const code = `
import subprocess
import sys

try:
    _squiggy_extension_path = ${pathJson}
    _squiggy_venv_path = ${venvPathJson}

    # Use uv to create venv and install in one command
    print("Creating virtual environment with uv...")
    print(f"Installing squiggy from {_squiggy_extension_path}...")

    _squiggy_result = subprocess.run(
        ['uv', 'venv', _squiggy_venv_path],
        capture_output=True,
        text=True,
        timeout=120
    )

    # Print output
    if _squiggy_result.stdout:
        print(_squiggy_result.stdout)
    if _squiggy_result.stderr:
        print(_squiggy_result.stderr, file=sys.stderr)

    if _squiggy_result.returncode != 0:
        _squiggy_venv_result = {
            'success': False,
            'error': 'venv_creation_failed',
            'message': 'Failed to create virtual environment with uv'
        }
    else:
        # Install squiggy using uv pip install
        # Need to use the Python executable path, not the venv directory
        print("Installing squiggy...")
        import os
        if os.name == 'nt':  # Windows
            _squiggy_python_path = os.path.join(_squiggy_venv_path, 'Scripts', 'python.exe')
        else:  # macOS/Linux
            _squiggy_python_path = os.path.join(_squiggy_venv_path, 'bin', 'python')

        _squiggy_install_result = subprocess.run(
            ['uv', 'pip', 'install', '--python', _squiggy_python_path, '-e', _squiggy_extension_path],
            capture_output=True,
            text=True,
            timeout=300
        )

        # Print output
        if _squiggy_install_result.stdout:
            print(_squiggy_install_result.stdout)
        if _squiggy_install_result.stderr:
            print(_squiggy_install_result.stderr, file=sys.stderr)

        if _squiggy_install_result.returncode == 0:
            _squiggy_venv_result = {
                'success': True,
                'venv_path': _squiggy_venv_path
            }
        else:
            _squiggy_venv_result = {
                'success': False,
                'error': 'install_failed',
                'message': 'Failed to install squiggy with uv'
            }
finally:
    # Clean up temporary variables (but NOT _squiggy_venv_result - will be deleted after reading)
    for _squiggy_var in ['_squiggy_extension_path', '_squiggy_venv_path', '_squiggy_result', '_squiggy_install_result', '_squiggy_python_path']:
        try:
            del globals()[_squiggy_var]
        except:
            pass
`;

        try {
            await this.client.executeSilent(code);
            const result = (await this.client.getVariable('_squiggy_venv_result')) as any;
            await this.client.executeSilent('del _squiggy_venv_result');

            if (!result.success) {
                throw new Error(result.message);
            }

            return venvPath;
        } catch (error) {
            throw new Error(`Failed to create venv and install squiggy: ${error}`);
        }
    }

    /**
     * Check if uv is installed
     */
    async isUvInstalled(): Promise<boolean> {
        const code = `
import subprocess
try:
    _squiggy_uv_check = subprocess.run(['uv', '--version'], capture_output=True)
    _squiggy_uv_available = _squiggy_uv_check.returncode == 0
except FileNotFoundError:
    _squiggy_uv_available = False
finally:
    try:
        del _squiggy_uv_check
    except:
        pass
`;

        try {
            await this.client.executeSilent(code);
            const result = await this.client.getVariable('_squiggy_uv_available');
            await this.client.executeSilent('del _squiggy_uv_available').catch(() => {});
            return result === true;
        } catch {
            return false;
        }
    }

    /**
     * Install squiggy package to the kernel from extension directory
     *
     * @param extensionPath Path to the extension directory containing pyproject.toml
     */
    async installSquiggy(extensionPath: string): Promise<void> {
        // Check if uv is installed first
        const uvInstalled = await this.isUvInstalled();

        if (!uvInstalled) {
            throw new Error(
                `UV_NOT_INSTALLED:uv is not installed. ` +
                    `Squiggy requires uv to manage Python environments and dependencies.`
            );
        }

        // Use uv to create venv and install
        const venvPath = await this.createVenvWithUv(extensionPath);

        // Throw special error with venv path to inform user
        throw new Error(
            `VENV_CREATED:${venvPath}:Created virtual environment and installed squiggy. ` +
                `Please select the Python interpreter at '${venvPath}' using the interpreter ` +
                `selector (bottom-right corner), then restart the Python console.`
        );
    }

    /**
     * Show manual installation guide with copy-able commands
     */
    async showManualInstallationGuide(extensionPath: string): Promise<void> {
        const items = [
            {
                label: '1️⃣ Install uv',
                detail: 'curl -LsSf https://astral.sh/uv/install.sh | sh',
                description: 'Fast Python package installer (macOS/Linux)',
            },
            {
                label: '1️⃣ Install uv (Windows)',
                detail: 'powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"',
                description: 'Fast Python package installer',
            },
            {
                label: '2️⃣ Create Virtual Environment',
                detail: 'uv venv .venv',
                description: 'Creates .venv in current directory',
            },
            {
                label: '3️⃣ Install Squiggy Package',
                detail: `uv pip install --python .venv -e "${extensionPath}"`,
                description: 'Install into the venv',
            },
            {
                label: '4️⃣ Select Environment in Positron',
                detail: 'Use the Interpreter selector in bottom-right corner',
                description: 'Choose .venv/bin/python',
            },
        ];

        const selected = await vscode.window.showQuickPick(items, {
            placeHolder: 'Select a command to copy to clipboard',
            title: 'Manual Installation Steps',
        });

        if (selected && selected.detail) {
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

                        // Handle automatic venv creation success
                        if (errorMessage.startsWith('VENV_CREATED:')) {
                            const parts = errorMessage.split(':');
                            const venvPath = parts[1];

                            // Determine Python executable path
                            const isWindows = process.platform === 'win32';
                            const pythonPath = isWindows
                                ? path.join(venvPath, 'Scripts', 'python.exe')
                                : path.join(venvPath, 'bin', 'python');

                            // Automatically set the Python interpreter
                            try {
                                await vscode.commands.executeCommand(
                                    'python.setInterpreter',
                                    { path: pythonPath }
                                );

                                // Show success message
                                const choice = await vscode.window.showInformationMessage(
                                    `✓ Virtual environment created and configured!\n\n` +
                                        `Location: ${venvPath}\n\n` +
                                        `The Python interpreter has been automatically selected.\n` +
                                        `Please restart the Python console to use squiggy.`,
                                    'Restart Python Console',
                                    'OK'
                                );

                                if (choice === 'Restart Python Console') {
                                    // Restart the Python runtime
                                    await vscode.commands.executeCommand('workbench.action.positronConsole.restartRuntime');
                                }
                            } catch (error) {
                                // Fallback to manual selection if auto-select fails
                                const choice = await vscode.window.showInformationMessage(
                                    `✓ Virtual environment created successfully!\n\n` +
                                        `Location: ${venvPath}\n\n` +
                                        `Please select the Python interpreter:\n` +
                                        `${pythonPath}\n\n` +
                                        `Then restart the Python console.`,
                                    'Open Interpreter Selector',
                                    'OK'
                                );

                                if (choice === 'Open Interpreter Selector') {
                                    vscode.commands.executeCommand('python.setInterpreter');
                                }
                            }

                            // Return false because squiggy is NOT installed in current kernel yet
                            return false;
                        }

                        // Handle UV not installed error
                        if (errorMessage.startsWith('UV_NOT_INSTALLED:')) {
                            const choice = await vscode.window.showErrorMessage(
                                'Squiggy requires uv to manage Python environments.\n\n' +
                                    'uv is a fast, modern Python package installer and environment manager.\n\n' +
                                    'Please install uv first, then try again.',
                                'Install uv',
                                'Manual Instructions',
                                'Dismiss'
                            );

                            if (choice === 'Install uv') {
                                vscode.env.openExternal(vscode.Uri.parse('https://docs.astral.sh/uv/getting-started/installation/'));
                            } else if (choice === 'Manual Instructions') {
                                await this.showManualInstallationGuide(extensionPath);
                            }
                            return false;
                        }

                        // Handle other errors
                        vscode.window.showErrorMessage(
                            `Failed to install squiggy package: ${errorMessage}`
                        );
                        return false;
                    }
                }
            );
        } catch {
            return false;
        }
    }
}
