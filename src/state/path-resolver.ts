/**
 * Path Resolver
 *
 * Handles resolution of extension-relative paths and path normalization
 * for session state restoration.
 */

import * as vscode from 'vscode';
import * as path from 'path';
import { logger } from '../utils/logger';

export class PathResolver {
    /**
     * Resolve Python package paths
     *
     * Converts paths like "<package:squiggy>/data/file.pod5"
     * to actual file system paths by querying Python for package location.
     *
     * @param filePath Path to resolve (may contain <package:name> placeholder)
     * @param positronClient Positron client to query Python
     * @returns Absolute file path or original if not a package path
     */
    static async resolvePythonPackagePath(filePath: string, positronClient?: any): Promise<string> {
        // Check if this is a package path
        const packageMatch = filePath.match(/^<package:([^>]+)>\/(.+)$/);
        if (!packageMatch) {
            return filePath; // Not a package path, return as-is
        }

        const [, packageName, relativePath] = packageMatch;

        // Query Python for package location
        if (positronClient) {
            try {
                // Execute Python code to get package path
                // Note: We use executeSilent + a temp variable approach instead of getVariable
                // because getVariable wraps code in json.dumps() which doesn't work with imports
                const tempVar = '_squiggy_temp_path_' + Math.random().toString(36).substr(2, 9);

                await positronClient.executeSilent(`
import importlib.util
import os
spec = importlib.util.find_spec('${packageName}')
if spec and spec.origin:
    package_dir = os.path.dirname(spec.origin)
    ${tempVar} = os.path.join(package_dir, '${relativePath}')
else:
    ${tempVar} = None
`);

                // Now read the variable
                const result = await positronClient.getVariable(tempVar);

                // Clean up temp variables
                await positronClient
                    .executeSilent(
                        `
if '${tempVar}' in globals():
    del ${tempVar}
if 'spec' in globals():
    del spec
if 'package_dir' in globals():
    del package_dir
`
                    )
                    .catch(() => {});

                if (result && result !== 'None' && result !== 'null') {
                    return result;
                }
            } catch (error) {
                logger.error(`Failed to resolve package path for ${packageName}:`, error);
            }
        }

        // Fallback: return original path (will fail with helpful error)
        return filePath;
    }
    /**
     * Resolve paths relative to a session file's directory
     *
     * Used for pipeline-generated sessions (e.g., squiggy-session.json from aa-tRNA)
     * where file paths are stored relative to the session file's location.
     *
     * @param relativePath Path from session file (may be relative or absolute)
     * @param sessionFileDir Directory containing the session file
     * @returns Absolute file path
     */
    static resolveSessionRelativePath(relativePath: string, sessionFileDir: string): string {
        // If already absolute, return as-is
        if (path.isAbsolute(relativePath)) {
            return relativePath;
        }

        // Resolve relative to session file directory
        return path.join(sessionFileDir, relativePath);
    }

    /**
     * Resolve extension-relative paths to absolute paths
     *
     * Converts paths like "${extensionPath}/out/test-data/file.pod5"
     * to absolute paths using the extension URI.
     *
     * @param filePath Path to resolve (may contain ${extensionPath} placeholder)
     * @param extensionUri Extension URI for resolving relative paths
     * @returns Absolute file path
     */
    static resolveExtensionPath(filePath: string, extensionUri: vscode.Uri): string {
        // Check if path contains extension placeholder
        if (filePath.includes('${extensionPath}')) {
            // Replace placeholder with extension URI path
            const relativePath = filePath.replace('${extensionPath}', '');
            return vscode.Uri.joinPath(extensionUri, relativePath).fsPath;
        }

        // If path is already absolute, return as-is
        if (path.isAbsolute(filePath)) {
            return filePath;
        }

        // If relative path without placeholder, resolve relative to extension
        return vscode.Uri.joinPath(extensionUri, filePath).fsPath;
    }

    /**
     * Convert absolute path to workspace-relative if possible
     *
     * This makes sessions more portable when shared within the same workspace.
     *
     * @param absolutePath Absolute file path
     * @param workspaceUri Workspace root URI
     * @returns Workspace-relative path or original absolute path
     */
    static makeWorkspaceRelative(absolutePath: string, workspaceUri?: vscode.Uri): string {
        if (!workspaceUri) {
            return absolutePath;
        }

        const workspaceRoot = workspaceUri.fsPath;
        if (absolutePath.startsWith(workspaceRoot)) {
            // Return path relative to workspace
            return path.relative(workspaceRoot, absolutePath);
        }

        // Not within workspace, return absolute
        return absolutePath;
    }

    /**
     * Resolve workspace-relative path to absolute
     *
     * @param relativePath Workspace-relative path
     * @param workspaceUri Workspace root URI
     * @returns Absolute file path
     */
    static resolveWorkspacePath(relativePath: string, workspaceUri: vscode.Uri): string {
        // If already absolute, return as-is
        if (path.isAbsolute(relativePath)) {
            return relativePath;
        }

        return path.join(workspaceUri.fsPath, relativePath);
    }

    /**
     * Normalize path separators for cross-platform compatibility
     *
     * @param filePath Path to normalize
     * @returns Normalized path
     */
    static normalizePath(filePath: string): string {
        return path.normalize(filePath);
    }

    /**
     * Check if path is extension-relative (for demo sessions)
     *
     * @param filePath Path to check
     * @returns True if path is within extension directory
     */
    static isExtensionRelative(filePath: string, extensionUri: vscode.Uri): boolean {
        const extensionRoot = extensionUri.fsPath;
        return filePath.startsWith(extensionRoot) || filePath.includes('${extensionPath}');
    }

    /**
     * Get display name for path (for UI)
     *
     * Returns just the filename for extension-relative paths,
     * or the full path for user files.
     *
     * @param filePath Path to get display name for
     * @param extensionUri Extension URI
     * @returns Display-friendly path
     */
    static getDisplayPath(filePath: string, extensionUri: vscode.Uri): string {
        if (this.isExtensionRelative(filePath, extensionUri)) {
            return `(Demo) ${path.basename(filePath)}`;
        }

        // For user files, show full path
        return filePath;
    }
}
