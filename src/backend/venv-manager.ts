/**
 * Virtual Environment Manager
 *
 * Manages the Squiggy extension's isolated Python virtual environment.
 * Creates a venv at ~/.venvs/squiggy (or $SQUIGGY_VENV) and installs
 * the bundled squiggy-positron package using uv.
 */

import * as vscode from 'vscode';
import * as path from 'path';
import * as os from 'os';
import * as fs from 'fs';
import { exec } from 'child_process';
import { promisify } from 'util';
import { logger } from '../utils/logger';

const execAsync = promisify(exec);

/**
 * Result of a venv setup operation
 */
export interface VenvSetupResult {
    success: boolean;
    pythonPath: string | null;
    error?: string;
    errorType?: 'UV_NOT_INSTALLED' | 'PYTHON_NOT_FOUND' | 'VENV_CREATE_FAILED' | 'INSTALL_FAILED';
}

/**
 * Manages the Squiggy extension's isolated Python virtual environment
 */
export class VenvManager {
    private readonly venvPath: string;

    constructor() {
        // Default: ~/.venvs/squiggy, overridable via SQUIGGY_VENV env var
        // Using ~/.venvs/ because Positron automatically discovers venvs in this location
        // (see positron-python globalVirtualEnvironmentLocator.ts)
        this.venvPath = process.env.SQUIGGY_VENV || path.join(os.homedir(), '.venvs', 'squiggy');
    }

    /**
     * Get the venv path
     */
    getVenvPath(): string {
        return this.venvPath;
    }

    /**
     * Get path to venv's Python executable
     */
    getVenvPython(): string {
        if (process.platform === 'win32') {
            return path.join(this.venvPath, 'Scripts', 'python.exe');
        }
        return path.join(this.venvPath, 'bin', 'python');
    }

    /**
     * Check if uv is installed
     */
    async isUvInstalled(): Promise<boolean> {
        try {
            await execAsync('uv --version');
            return true;
        } catch {
            return false;
        }
    }

    /**
     * Check if venv exists and has squiggy installed
     */
    async isVenvValid(): Promise<boolean> {
        const pythonPath = this.getVenvPython();

        // Check if venv python exists
        if (!fs.existsSync(pythonPath)) {
            return false;
        }

        // Check if squiggy is importable
        try {
            await execAsync(`"${pythonPath}" -c "import squiggy"`);
            return true;
        } catch {
            return false;
        }
    }

    /**
     * Create venv using uv
     */
    async createVenv(): Promise<void> {
        // Ensure parent directory exists
        const parentDir = path.dirname(this.venvPath);
        if (!fs.existsSync(parentDir)) {
            fs.mkdirSync(parentDir, { recursive: true });
        }

        // Create venv with uv (requires Python 3.12+)
        logger.info(`Creating venv at ${this.venvPath}`);
        const { stderr } = await execAsync(`uv venv "${this.venvPath}" --python 3.12`);
        if (stderr && !stderr.includes('Using CPython')) {
            logger.warning(`uv venv stderr: ${stderr}`);
        }
    }

    /**
     * Install bundled package using uv
     */
    async installPackage(extensionPath: string): Promise<void> {
        const pythonPath = this.getVenvPython();
        const packagePath = path.join(extensionPath, 'squiggy');

        // Verify package exists
        if (!fs.existsSync(packagePath)) {
            throw new Error(`Bundled package not found at ${packagePath}`);
        }

        logger.info(`Installing squiggy package from ${packagePath}`);

        // Install with uv pip (editable mode for bundled package)
        const { stderr } = await execAsync(
            `uv pip install --python "${pythonPath}" -e "${extensionPath}"`
        );
        if (stderr && !stderr.includes('Installed')) {
            logger.warning(`uv pip install stderr: ${stderr}`);
        }
    }

    /**
     * Get installed package version
     */
    async getInstalledVersion(): Promise<string | null> {
        try {
            const pythonPath = this.getVenvPython();
            const { stdout } = await execAsync(
                `"${pythonPath}" -c "import squiggy; print(squiggy.__version__)"`
            );
            return stdout.trim();
        } catch {
            return null;
        }
    }

    /**
     * Get bundled package version from pyproject.toml
     */
    getBundledVersion(extensionPath: string): string | null {
        try {
            const pyprojectPath = path.join(extensionPath, 'pyproject.toml');
            if (!fs.existsSync(pyprojectPath)) {
                return null;
            }

            const content = fs.readFileSync(pyprojectPath, 'utf8');
            const match = content.match(/^version\s*=\s*["']([^"']+)["']/m);
            return match ? match[1] : null;
        } catch {
            return null;
        }
    }

    /**
     * Upgrade package if version mismatch
     * @returns true if upgrade was performed
     */
    async upgradeIfNeeded(extensionPath: string): Promise<boolean> {
        const installedVersion = await this.getInstalledVersion();
        const bundledVersion = this.getBundledVersion(extensionPath);

        if (!installedVersion || !bundledVersion) {
            return false;
        }

        if (installedVersion !== bundledVersion) {
            logger.info(`Upgrading squiggy from ${installedVersion} to ${bundledVersion}`);
            await this.installPackage(extensionPath);
            return true;
        }

        return false;
    }

    /**
     * Set venv as Positron's Python interpreter
     *
     * NOTE: This is a no-op. Positron automatically discovers venvs in
     * ~/.venvs/ and sets the discovered venv as the preferred runtime.
     * The dedicated kernel uses getPreferredRuntime() which returns the
     * squiggy venv (Python 3.12) after Positron discovers it.
     *
     * We don't programmatically select the interpreter because that can
     * trigger UI dialogs that hang.
     */
    async setAsInterpreter(): Promise<void> {
        const pythonPath = this.getVenvPython();
        logger.info(`Squiggy venv ready at ${pythonPath}`);
        // No-op: Positron discovers venvs in ~/.venvs/ automatically
        // and sets them as the preferred runtime
    }

    /**
     * Delete venv for reset
     */
    async deleteVenv(): Promise<void> {
        if (fs.existsSync(this.venvPath)) {
            logger.info(`Deleting venv at ${this.venvPath}`);
            fs.rmSync(this.venvPath, { recursive: true, force: true });
        }
    }

    /**
     * Full setup: create venv if needed, install package, set interpreter
     */
    async ensureVenv(extensionPath: string): Promise<VenvSetupResult> {
        // Check if uv is installed
        if (!(await this.isUvInstalled())) {
            return {
                success: false,
                pythonPath: null,
                error: 'uv is not installed',
                errorType: 'UV_NOT_INSTALLED',
            };
        }

        try {
            if (await this.isVenvValid()) {
                // Venv exists and is valid - check for upgrade
                const upgraded = await this.upgradeIfNeeded(extensionPath);
                if (upgraded) {
                    logger.info('Squiggy package upgraded');
                }
            } else {
                // Need to create/recreate venv
                await this.createVenv();
                await this.installPackage(extensionPath);
            }

            // Set as interpreter
            await this.setAsInterpreter();

            return {
                success: true,
                pythonPath: this.getVenvPython(),
            };
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            logger.error(`Venv setup failed: ${errorMessage}`);

            // Determine error type
            let errorType: VenvSetupResult['errorType'] = 'VENV_CREATE_FAILED';
            if (errorMessage.includes('No interpreter found')) {
                errorType = 'PYTHON_NOT_FOUND';
            } else if (errorMessage.includes('pip install')) {
                errorType = 'INSTALL_FAILED';
            }

            return {
                success: false,
                pythonPath: null,
                error: errorMessage,
                errorType,
            };
        }
    }
}

/**
 * Show error notification with appropriate actions
 */
export async function showVenvSetupError(result: VenvSetupResult): Promise<void> {
    switch (result.errorType) {
        case 'UV_NOT_INSTALLED': {
            const choice = await vscode.window.showErrorMessage(
                'Squiggy requires uv to manage Python environments.\n\n' +
                    'Install with: curl -LsSf https://astral.sh/uv/install.sh | sh',
                'Copy Install Command',
                'View Documentation'
            );
            if (choice === 'Copy Install Command') {
                await vscode.env.clipboard.writeText(
                    'curl -LsSf https://astral.sh/uv/install.sh | sh'
                );
                vscode.window.showInformationMessage('Install command copied to clipboard');
            } else if (choice === 'View Documentation') {
                vscode.env.openExternal(vscode.Uri.parse('https://docs.astral.sh/uv/'));
            }
            break;
        }

        case 'PYTHON_NOT_FOUND': {
            const choice = await vscode.window.showErrorMessage(
                'Squiggy requires Python 3.12 or later.\n\n' +
                    'uv could not find a compatible Python installation.',
                'Download Python',
                'Retry'
            );
            if (choice === 'Download Python') {
                vscode.env.openExternal(vscode.Uri.parse('https://www.python.org/downloads/'));
            } else if (choice === 'Retry') {
                vscode.commands.executeCommand('squiggy.resetVenv');
            }
            break;
        }

        case 'VENV_CREATE_FAILED':
        case 'INSTALL_FAILED': {
            const choice = await vscode.window.showErrorMessage(
                `Failed to set up Squiggy environment: ${result.error}`,
                'View Logs',
                'Reset & Retry'
            );
            if (choice === 'View Logs') {
                vscode.commands.executeCommand('squiggy.showLogs');
            } else if (choice === 'Reset & Retry') {
                vscode.commands.executeCommand('squiggy.resetVenv');
            }
            break;
        }

        default: {
            vscode.window.showErrorMessage(`Squiggy setup failed: ${result.error}`);
        }
    }
}
