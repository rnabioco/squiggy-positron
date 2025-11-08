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
            // Always clean up, even if getVariable failed
            await this.client
                .executeSilent("if '_squiggy_installed' in globals(): del _squiggy_installed")
                .catch(() => {});
            return result === true;
        } catch {
            // Clean up on error path too
            await this.client
                .executeSilent("if '_squiggy_installed' in globals(): del _squiggy_installed")
                .catch(() => {});
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
