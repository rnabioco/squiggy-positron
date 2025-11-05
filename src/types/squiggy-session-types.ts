/**
 * Session State Types
 *
 * TypeScript interfaces for session persistence and restoration
 */

/**
 * Plot options configuration
 */
export interface PlotOptionsState {
    mode: string;
    normalization: string;
    showDwellTime: boolean;
    showBaseAnnotations: boolean;
    scaleDwellTime: boolean;
    downsample: number;
    showSignalPoints: boolean;
}

/**
 * Modification filter configuration
 */
export interface ModificationFiltersState {
    minProbability: number;
    enabledModTypes: string[];
}

/**
 * Sample information (aligns with multi-sample architecture from #79)
 */
export interface SampleSessionState {
    pod5Paths: string[]; // Multiple POD5s per sample
    bamPath?: string;
    fastaPath?: string;
}

/**
 * UI state for panels
 */
export interface UIState {
    expandedSamples?: string[];
    selectedSamplesForComparison?: string[];
}

/**
 * Complete session state
 */
export interface SessionState {
    // Metadata
    version: string; // Schema version (e.g., "1.0.0")
    timestamp: string; // ISO 8601 timestamp of when session was saved
    sessionName?: string; // Optional user-provided name
    isDemo?: boolean; // Flag for built-in demo session

    // Extension info
    extensionVersion?: string; // Squiggy extension version (from package.json)
    positronVersion?: string; // Positron/VSCode version

    // File checksums (for validation)
    fileChecksums?: {
        [filePath: string]: {
            md5?: string;
            size?: number;
            lastModified?: string; // ISO 8601 timestamp
        };
    };

    // Sample-centric structure (aligns with #79)
    samples: {
        [sampleName: string]: SampleSessionState;
    };

    // Plot configuration
    plotOptions: PlotOptionsState;

    // Modification filters (if applicable)
    modificationFilters?: ModificationFiltersState;

    // UI state
    ui?: UIState;
}

/**
 * Validation result for session state
 */
export interface ValidationResult {
    valid: boolean;
    errors: string[];
}
