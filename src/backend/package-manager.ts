/**
 * Python Package Management
 *
 * Detects the squiggy-positron Python package and provides
 * installation guidance. Does NOT auto-install.
 */

import * as vscode from 'vscode';
import { PositronRuntimeClient } from './positron-runtime-client';

/**
 * Manages squiggy package detection and provides installation guidance
 */
export class PackageManager {
    private static readonly REQUIRED_VERSION = '0.1.7'; // Minimum required version
    private static readonly PACKAGE_NAME = 'squiggy-positron';

    constructor(private readonly client: PositronRuntimeClient) {}

    /**
     * Check if squiggy package is installed in the kernel
     */
    async isSquiggyInstalled(): Promise<boolean> {
        try {
            // Use a self-contained lambda expression to avoid variable persistence issues
            const checkExpression = `(lambda: (
                __import__('squiggy') and
                hasattr(__import__('squiggy'), 'load_pod5') and
                hasattr(__import__('squiggy'), 'load_bam') and
                hasattr(__import__('squiggy'), 'plot_read')
            ))()`;

            const result = await this.client.getVariable(checkExpression);
            return result === true;
        } catch {
            // Import error or other exception means not installed
            return false;
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
     * Check if installed version is compatible with extension requirements
     */
    async isVersionCompatible(): Promise<{ compatible: boolean; version: string | null }> {
        const version = await this.getSquiggyVersion();
        if (!version) {
            return { compatible: false, version: null };
        }

        // Simple version comparison (assuming semver-like format)
        const required = PackageManager.REQUIRED_VERSION.split('.').map(Number);
        const installed = version.split('.').map(Number);

        // Check major.minor.patch - installed must be >= required
        for (let i = 0; i < 3; i++) {
            const req = required[i] || 0;
            const inst = installed[i] || 0;
            if (inst > req) {
                return { compatible: true, version };
            }
            if (inst < req) {
                return { compatible: false, version };
            }
        }

        return { compatible: true, version };
    }

    /**
     * Show installation instructions with copy-able command
     */
    async showInstallationInstructions(): Promise<void> {
        const installCommand = `uv pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ ${PackageManager.PACKAGE_NAME}`;

        const choice = await vscode.window.showErrorMessage(
            `Squiggy requires the ${PackageManager.PACKAGE_NAME} Python package.\n\n` +
                `Please install it in your active Python environment:\n\n` +
                `${installCommand}\n\n` +
                `Note: Installing from TestPyPI temporarily until next PyPI release.`,
            'Copy Install Command',
            'Open Documentation',
            'Dismiss'
        );

        if (choice === 'Copy Install Command') {
            await vscode.env.clipboard.writeText(installCommand);
            vscode.window.showInformationMessage(
                `Copied to clipboard: ${installCommand}\n\nPaste this in your terminal to install.`
            );
        } else if (choice === 'Open Documentation') {
            vscode.env.openExternal(
                vscode.Uri.parse('https://github.com/rnabioco/squiggy-positron#installation')
            );
        }
    }

    /**
     * Show version incompatibility warning
     */
    async showVersionWarning(installedVersion: string): Promise<void> {
        const upgradeCommand = `uv pip install --upgrade --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ ${PackageManager.PACKAGE_NAME}`;

        const choice = await vscode.window.showWarningMessage(
            `Squiggy extension requires ${PackageManager.PACKAGE_NAME} >= ${PackageManager.REQUIRED_VERSION}\n\n` +
                `Installed version: ${installedVersion}\n\n` +
                `Please upgrade:\n${upgradeCommand}\n\n` +
                `Note: Installing from TestPyPI temporarily until next PyPI release.`,
            'Copy Upgrade Command',
            'Dismiss'
        );

        if (choice === 'Copy Upgrade Command') {
            await vscode.env.clipboard.writeText(upgradeCommand);
            vscode.window.showInformationMessage(
                `Copied to clipboard: ${upgradeCommand}\n\nPaste this in your terminal to upgrade.`
            );
        }
    }

    /**
     * Verify package installation and show appropriate guidance
     * @returns true if package is installed and compatible, false otherwise
     */
    async verifyPackage(): Promise<boolean> {
        const isInstalled = await this.isSquiggyInstalled();

        if (!isInstalled) {
            await this.showInstallationInstructions();
            return false;
        }

        const { compatible, version } = await this.isVersionCompatible();

        if (!compatible && version) {
            await this.showVersionWarning(version);
            return false;
        }

        return true;
    }
}
