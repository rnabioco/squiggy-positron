/**
 * Session State Manager
 *
 * Handles session persistence to workspace state and file export/import.
 * Provides validation and demo session generation.
 */

import * as vscode from 'vscode';
import * as fs from 'fs/promises';
import * as crypto from 'crypto';
import { SessionState, ValidationResult } from '../types/squiggy-session-types';
import { logger } from '../utils/logger';

const SESSION_STATE_KEY = 'squiggy.sessionState';
const SESSION_VERSION = '1.0.0';

export class SessionStateManager {
    /**
     * Save session to workspace state with metadata
     */
    static async saveSession(state: SessionState, context: vscode.ExtensionContext): Promise<void> {
        // Get extension version from package.json
        const packageJson = context.extension?.packageJSON;
        const extensionVersion = packageJson?.version || 'unknown';

        // Get Positron/VSCode version
        const positronVersion = vscode.version;

        // Calculate file checksums
        const fileChecksums = await this.calculateFileChecksums(state);

        // Add metadata
        const sessionWithMetadata: SessionState = {
            ...state,
            version: state.version || SESSION_VERSION,
            timestamp: new Date().toISOString(),
            extensionVersion,
            positronVersion,
            fileChecksums,
        };

        await context.workspaceState.update(SESSION_STATE_KEY, sessionWithMetadata);
    }

    /**
     * Calculate MD5 checksums for all files in session
     */
    private static async calculateFileChecksums(
        state: SessionState
    ): Promise<SessionState['fileChecksums']> {
        const checksums: NonNullable<SessionState['fileChecksums']> = {};

        // Process all samples
        for (const [_sampleName, sample] of Object.entries(state.samples)) {
            // POD5 files
            for (const pod5Path of sample.pod5Paths) {
                checksums[pod5Path] = await this.getFileInfo(pod5Path);
            }

            // BAM file
            if (sample.bamPath) {
                checksums[sample.bamPath] = await this.getFileInfo(sample.bamPath);
            }

            // FASTA file
            if (sample.fastaPath) {
                checksums[sample.fastaPath] = await this.getFileInfo(sample.fastaPath);
            }
        }

        return checksums;
    }

    /**
     * Get file info (MD5, size, last modified)
     */
    private static async getFileInfo(
        filePath: string
    ): Promise<{ md5?: string; size?: number; lastModified?: string }> {
        try {
            const stats = await fs.stat(filePath);
            const content = await fs.readFile(filePath);
            const hash = crypto.createHash('md5');
            hash.update(content);

            return {
                md5: hash.digest('hex'),
                size: stats.size,
                lastModified: stats.mtime.toISOString(),
            };
        } catch (_error) {
            // If file doesn't exist or can't be read, return empty
            return {};
        }
    }

    /**
     * Load session from workspace state
     */
    static async loadSession(context: vscode.ExtensionContext): Promise<SessionState | null> {
        const session = context.workspaceState.get<SessionState>(SESSION_STATE_KEY);

        if (!session) {
            return null;
        }

        // Validate schema
        const validation = this.validateSession(session);
        if (!validation.valid) {
            logger.warning(`Session state validation failed: ${validation.errors.join(', ')}`);
            return null;
        }

        return session;
    }

    /**
     * Export session to JSON file with full metadata
     */
    static async exportSession(
        state: SessionState,
        filePath: string,
        context?: vscode.ExtensionContext
    ): Promise<void> {
        // Add metadata
        let sessionWithMetadata: SessionState = {
            ...state,
            version: state.version || SESSION_VERSION,
            timestamp: new Date().toISOString(),
        };

        // Add extension version if context available
        if (context) {
            const packageJson = context.extension?.packageJSON;
            sessionWithMetadata.extensionVersion = packageJson?.version || 'unknown';
            sessionWithMetadata.positronVersion = vscode.version;
        }

        // Calculate file checksums if not already present
        if (!sessionWithMetadata.fileChecksums) {
            sessionWithMetadata.fileChecksums = await this.calculateFileChecksums(state);
        }

        const json = JSON.stringify(sessionWithMetadata, null, 2);
        await fs.writeFile(filePath, json, 'utf-8');
    }

    /**
     * Import session from JSON file
     */
    static async importSession(filePath: string): Promise<SessionState> {
        const json = await fs.readFile(filePath, 'utf-8');
        const session = JSON.parse(json) as SessionState;

        // Validate schema
        const validation = this.validateSession(session);
        if (!validation.valid) {
            throw new Error(`Invalid session file: ${validation.errors.join(', ')}`);
        }

        return session;
    }

    /**
     * Clear saved session from workspace state
     */
    static async clearSession(context: vscode.ExtensionContext): Promise<void> {
        await context.workspaceState.update(SESSION_STATE_KEY, undefined);
    }

    /**
     * Get demo session using packaged test data from Python package
     *
     * Note: Demo data is bundled with the squiggy Python package,
     * not the extension. The actual paths will be resolved at runtime
     * by querying the Python package location.
     */
    static getDemoSession(_extensionUri: vscode.Uri): SessionState {
        // Use placeholder paths - these will be resolved by Python at runtime
        // The format is: <package:squiggy>/data/filename
        // This gets resolved by the extension when loading the session
        const pod5Path = '<package:squiggy>/data/yeast_trna_reads.pod5';
        const bamPath = '<package:squiggy>/data/yeast_trna_mappings.bam';
        const fastaPath = '<package:squiggy>/data/yeast_trna.fa';

        return {
            version: SESSION_VERSION,
            timestamp: new Date().toISOString(),
            sessionName: 'Demo: Yeast tRNA Reads',
            isDemo: true,

            // Single demo sample
            samples: {
                Yeast_tRNA: {
                    pod5Paths: [pod5Path],
                    bamPath: bamPath,
                    fastaPath: fastaPath,
                },
            },

            // Default plot options showcasing key features
            plotOptions: {
                mode: 'EVENTALIGN',
                normalization: 'ZNORM',
                showDwellTime: false,
                showBaseAnnotations: true,
                scaleDwellTime: false,
                downsample: 5,
                showSignalPoints: false,
            },

            // No modification filters for demo (yeast data doesn't have mods)
            modificationFilters: undefined,

            // UI state
            ui: {
                expandedSamples: ['Yeast_tRNA'],
                selectedSamplesForComparison: [],
                selectedReadExplorerSample: 'Yeast_tRNA', // Auto-select the demo sample
            },
        };
    }

    /**
     * Validate session state schema
     */
    static validateSession(session: any): ValidationResult {
        const errors: string[] = [];

        // Check required fields
        if (!session.version) {
            errors.push('Missing version field');
        }

        if (!session.timestamp) {
            errors.push('Missing timestamp field');
        }

        if (!session.samples || typeof session.samples !== 'object') {
            errors.push('Missing or invalid samples field');
        }

        if (!session.plotOptions || typeof session.plotOptions !== 'object') {
            errors.push('Missing or invalid plotOptions field');
        }

        // Validate samples structure
        if (session.samples) {
            for (const [sampleName, sampleData] of Object.entries(session.samples)) {
                const sample = sampleData as any;

                if (!Array.isArray(sample.pod5Paths) || sample.pod5Paths.length === 0) {
                    errors.push(`Sample ${sampleName}: pod5Paths must be non-empty array`);
                }

                // Validate paths are strings
                if (
                    sample.pod5Paths &&
                    !sample.pod5Paths.every((p: any) => typeof p === 'string')
                ) {
                    errors.push(`Sample ${sampleName}: pod5Paths must contain only strings`);
                }

                if (sample.bamPath && typeof sample.bamPath !== 'string') {
                    errors.push(`Sample ${sampleName}: bamPath must be string`);
                }

                if (sample.fastaPath && typeof sample.fastaPath !== 'string') {
                    errors.push(`Sample ${sampleName}: fastaPath must be string`);
                }
            }
        }

        // Validate plot options structure
        if (session.plotOptions) {
            const required = [
                'mode',
                'normalization',
                'showDwellTime',
                'showBaseAnnotations',
                'scaleDwellTime',
                'downsample',
                'showSignalPoints',
            ];

            for (const field of required) {
                if (!(field in session.plotOptions)) {
                    errors.push(`plotOptions: Missing ${field}`);
                }
            }
        }

        // Validate modification filters if present
        if (session.modificationFilters) {
            if (typeof session.modificationFilters.minProbability !== 'number') {
                errors.push('modificationFilters: minProbability must be number');
            }

            if (!Array.isArray(session.modificationFilters.enabledModTypes)) {
                errors.push('modificationFilters: enabledModTypes must be array');
            }
        }

        // Validate UI state if present
        if (session.ui) {
            if (session.ui.expandedSamples && !Array.isArray(session.ui.expandedSamples)) {
                errors.push('ui: expandedSamples must be array');
            }

            if (
                session.ui.selectedSamplesForComparison &&
                !Array.isArray(session.ui.selectedSamplesForComparison)
            ) {
                errors.push('ui: selectedSamplesForComparison must be array');
            }
        }

        return {
            valid: errors.length === 0,
            errors,
        };
    }

    /**
     * Check if a session has unsaved changes compared to workspace state
     */
    static async hasUnsavedChanges(
        currentState: SessionState,
        context: vscode.ExtensionContext
    ): Promise<boolean> {
        const savedState = await this.loadSession(context);

        if (!savedState) {
            return true; // No saved state means changes exist
        }

        // Compare JSON representations (excluding timestamp)
        const current = { ...currentState, timestamp: '' };
        const saved = { ...savedState, timestamp: '' };

        return JSON.stringify(current) !== JSON.stringify(saved);
    }

    /**
     * Migrate session from older schema versions
     */
    static migrateSession(session: SessionState): SessionState {
        // Currently only one version, but prepare for future migrations
        if (session.version === SESSION_VERSION) {
            return session;
        }

        // Add migration logic for older versions here
        // For now, just update version
        return {
            ...session,
            version: SESSION_VERSION,
        };
    }
}
