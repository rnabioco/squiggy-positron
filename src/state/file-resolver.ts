/**
 * File Resolver
 *
 * Handles resolution of missing file paths during session restoration.
 * Provides file picker dialogs for user files and error handling for demo files.
 */

import * as vscode from 'vscode';
import * as fs from 'fs/promises';
import { PathResolver } from './path-resolver';

export type FileType = 'POD5' | 'BAM' | 'FASTA';

export interface FileResolutionResult {
    resolved: boolean;
    newPath?: string;
    error?: string;
}

export class FileResolver {
    /**
     * Resolve a file path, prompting user if file is missing
     *
     * For demo session files: Show error without prompting
     * For user files: Prompt to locate file
     *
     * @param filePath Path to resolve
     * @param fileType Type of file (for dialog filters)
     * @param isDemo Whether this is from a demo session
     * @param extensionUri Extension URI for demo file checks
     * @returns Resolution result with new path or error
     */
    static async resolveFilePath(
        filePath: string,
        fileType: FileType,
        isDemo: boolean,
        extensionUri: vscode.Uri
    ): Promise<FileResolutionResult> {
        // Check if file exists
        const exists = await this.fileExists(filePath);

        if (exists) {
            return { resolved: true, newPath: filePath };
        }

        // File doesn't exist
        // For demo files, show error instead of prompting
        if (isDemo || PathResolver.isExtensionRelative(filePath, extensionUri)) {
            return {
                resolved: false,
                error: `Demo file not found: ${filePath}. Extension may need to be reinstalled.`,
            };
        }

        // For user files, prompt to locate
        return await this.promptForFile(filePath, fileType);
    }

    /**
     * Check if file exists
     */
    private static async fileExists(filePath: string): Promise<boolean> {
        try {
            await fs.access(filePath);
            return true;
        } catch {
            return false;
        }
    }

    /**
     * Prompt user to locate a missing file
     */
    private static async promptForFile(
        originalPath: string,
        fileType: FileType
    ): Promise<FileResolutionResult> {
        const fileName = originalPath.split('/').pop() || 'file';

        const response = await vscode.window.showWarningMessage(
            `File not found: ${fileName}. Would you like to locate it?`,
            { modal: true },
            'Locate File',
            'Skip'
        );

        if (response !== 'Locate File') {
            return {
                resolved: false,
                error: `File not found and user chose to skip: ${originalPath}`,
            };
        }

        // Show file picker with appropriate filter
        const filters = this.getFileFilters(fileType);
        const uris = await vscode.window.showOpenDialog({
            canSelectMany: false,
            filters,
            title: `Locate ${fileType} file: ${fileName}`,
        });

        if (!uris || uris.length === 0) {
            return {
                resolved: false,
                error: `User cancelled file selection for: ${originalPath}`,
            };
        }

        return {
            resolved: true,
            newPath: uris[0].fsPath,
        };
    }

    /**
     * Get file filters for open dialog based on file type
     */
    private static getFileFilters(fileType: FileType): { [name: string]: string[] } {
        switch (fileType) {
            case 'POD5':
                return {
                    'POD5 Files': ['pod5'],
                    'All Files': ['*'],
                };
            case 'BAM':
                return {
                    'BAM Files': ['bam'],
                    'All Files': ['*'],
                };
            case 'FASTA':
                return {
                    'FASTA Files': ['fasta', 'fa', 'fna'],
                    'All Files': ['*'],
                };
        }
    }

    /**
     * Resolve multiple POD5 files for a sample
     *
     * @param pod5Paths Array of POD5 paths
     * @param isDemo Whether this is from demo session
     * @param extensionUri Extension URI
     * @returns Array of resolution results
     */
    static async resolvePOD5Files(
        pod5Paths: string[],
        isDemo: boolean,
        extensionUri: vscode.Uri
    ): Promise<FileResolutionResult[]> {
        const results: FileResolutionResult[] = [];

        for (const pod5Path of pod5Paths) {
            const result = await this.resolveFilePath(pod5Path, 'POD5', isDemo, extensionUri);
            results.push(result);
        }

        return results;
    }

    /**
     * Check if all files for a sample are accessible
     *
     * @param pod5Paths POD5 file paths
     * @param bamPath Optional BAM path
     * @param fastaPath Optional FASTA path
     * @returns True if all files exist
     */
    static async validateSampleFiles(
        pod5Paths: string[],
        bamPath?: string,
        fastaPath?: string
    ): Promise<{ valid: boolean; missingFiles: string[] }> {
        const missingFiles: string[] = [];

        // Check POD5 files
        for (const pod5Path of pod5Paths) {
            const exists = await this.fileExists(pod5Path);
            if (!exists) {
                missingFiles.push(pod5Path);
            }
        }

        // Check BAM file
        if (bamPath) {
            const exists = await this.fileExists(bamPath);
            if (!exists) {
                missingFiles.push(bamPath);
            }

            // Also check for BAM index
            const baiPath = `${bamPath}.bai`;
            const baiExists = await this.fileExists(baiPath);
            if (!baiExists) {
                missingFiles.push(baiPath);
            }
        }

        // Check FASTA file
        if (fastaPath) {
            const exists = await this.fileExists(fastaPath);
            if (!exists) {
                missingFiles.push(fastaPath);
            }
        }

        return {
            valid: missingFiles.length === 0,
            missingFiles,
        };
    }

    /**
     * Get a summary of missing files for display
     */
    static getMissingFilesSummary(missingFiles: string[]): string {
        if (missingFiles.length === 0) {
            return 'All files found';
        }

        const fileNames = missingFiles.map((f) => f.split('/').pop()).join(', ');
        return `Missing ${missingFiles.length} file(s): ${fileNames}`;
    }
}
