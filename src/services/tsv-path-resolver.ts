/**
 * TSV Path Resolver - Resolve relative paths from TSV files
 *
 * Handles multiple path resolution strategies for TSV-specified file paths:
 * - Absolute paths: Use as-is
 * - TSV-relative: Relative to TSV file location
 * - Workspace-relative: Relative to workspace root
 * - Auto: Try multiple strategies
 */

import * as path from 'path';
import { promises as fs } from 'fs';

export enum PathResolutionStrategy {
    /** Relative to TSV file directory */
    TsvRelative = 'tsv-relative',

    /** Relative to workspace root directory */
    WorkspaceRelative = 'workspace',

    /** Use path as-is (absolute paths) */
    Absolute = 'absolute',

    /** Try multiple strategies (absolute → tsv-relative → workspace-relative) */
    Auto = 'auto',
}

export interface PathResolutionResult {
    /** Successfully resolved path, or null if not found */
    resolvedPath: string | null;

    /** Strategy that succeeded */
    strategy: PathResolutionStrategy | null;

    /** Error message if resolution failed */
    error?: string;
}

export class TSVPathResolver {
    /**
     * @param tsvFilePath - Path to the TSV file (null if pasted from clipboard)
     * @param workspaceRoot - Workspace root directory
     */
    constructor(
        private tsvFilePath: string | null,
        private workspaceRoot: string
    ) {}

    /**
     * Resolve a file path using the specified strategy
     *
     * @param rawPath - Path from TSV file (may be relative or absolute)
     * @param strategy - Resolution strategy to use
     * @returns Resolved absolute path or null if file not found
     */
    async resolve(
        rawPath: string,
        strategy: PathResolutionStrategy = PathResolutionStrategy.Auto
    ): Promise<PathResolutionResult> {
        // TODO: Implement path resolution logic
        // For now, return a stub implementation

        // Handle absolute paths
        if (path.isAbsolute(rawPath)) {
            const exists = await this.fileExists(rawPath);
            if (exists) {
                return {
                    resolvedPath: rawPath,
                    strategy: PathResolutionStrategy.Absolute,
                };
            } else {
                return {
                    resolvedPath: null,
                    strategy: null,
                    error: `File not found: ${rawPath}`,
                };
            }
        }

        // Auto strategy: try multiple approaches
        if (strategy === PathResolutionStrategy.Auto) {
            // Try TSV-relative first (if TSV path available)
            if (this.tsvFilePath) {
                const tsvRelative = await this.resolveTsvRelative(rawPath);
                if (tsvRelative.resolvedPath) {
                    return tsvRelative;
                }
            }

            // Try workspace-relative
            const workspaceRelative = await this.resolveWorkspaceRelative(rawPath);
            if (workspaceRelative.resolvedPath) {
                return workspaceRelative;
            }

            return {
                resolvedPath: null,
                strategy: null,
                error: `Could not resolve path: ${rawPath} (tried TSV-relative and workspace-relative)`,
            };
        }

        // Single strategy
        switch (strategy) {
            case PathResolutionStrategy.TsvRelative:
                return this.resolveTsvRelative(rawPath);
            case PathResolutionStrategy.WorkspaceRelative:
                return this.resolveWorkspaceRelative(rawPath);
            default:
                return {
                    resolvedPath: null,
                    strategy: null,
                    error: `Unknown strategy: ${strategy}`,
                };
        }
    }

    /**
     * Resolve path relative to TSV file directory
     */
    private async resolveTsvRelative(rawPath: string): Promise<PathResolutionResult> {
        if (!this.tsvFilePath) {
            return {
                resolvedPath: null,
                strategy: null,
                error: 'Cannot use TSV-relative resolution without TSV file path',
            };
        }

        const tsvDir = path.dirname(this.tsvFilePath);
        const resolvedPath = path.resolve(tsvDir, rawPath);

        const exists = await this.fileExists(resolvedPath);
        if (exists) {
            return {
                resolvedPath,
                strategy: PathResolutionStrategy.TsvRelative,
            };
        } else {
            return {
                resolvedPath: null,
                strategy: null,
                error: `File not found (TSV-relative): ${resolvedPath}`,
            };
        }
    }

    /**
     * Resolve path relative to workspace root
     */
    private async resolveWorkspaceRelative(rawPath: string): Promise<PathResolutionResult> {
        const resolvedPath = path.resolve(this.workspaceRoot, rawPath);

        const exists = await this.fileExists(resolvedPath);
        if (exists) {
            return {
                resolvedPath,
                strategy: PathResolutionStrategy.WorkspaceRelative,
            };
        } else {
            return {
                resolvedPath: null,
                strategy: null,
                error: `File not found (workspace-relative): ${resolvedPath}`,
            };
        }
    }

    /**
     * Check if file exists
     */
    private async fileExists(filePath: string): Promise<boolean> {
        try {
            await fs.access(filePath);
            return true;
        } catch {
            return false;
        }
    }
}
