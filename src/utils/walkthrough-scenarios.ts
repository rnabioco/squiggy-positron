/**
 * Predefined UX Walkthrough Scenarios
 *
 * Common user flows for automated testing
 */

import { WalkthroughScenario } from './walkthrough-runner';
import { ExtensionState } from '../state/extension-state';

/**
 * Basic workflow: Load data, view reads, plot
 */
export const basicWorkflow: WalkthroughScenario = {
    name: 'Basic Workflow',
    description: 'Load test data, explore reads, and generate plots',
    steps: [
        {
            name: 'Load test data',
            command: 'squiggy.loadTestData',
            description: 'Load sample POD5 and BAM files',
            delay: 0,
        },
        {
            name: 'Wait for data loading',
            delay: 2000,
            validate: async (state: ExtensionState) => {
                return state.getLoadedItems().length > 0;
            },
        },
        {
            name: 'Refresh read list',
            command: 'squiggy.refreshReads',
            description: 'Update read explorer with loaded data',
            delay: 500,
        },
        {
            name: 'Wait for reads to populate',
            delay: 1500,
        },
        {
            name: 'Show extension logs',
            command: 'squiggy.showLogs',
            description: 'Display extension output channel',
            delay: 500,
        },
    ],
};

/**
 * Session management workflow
 */
export const sessionWorkflow: WalkthroughScenario = {
    name: 'Session Management',
    description: 'Test session save, restore, and export functionality',
    steps: [
        {
            name: 'Load demo session',
            command: 'squiggy.loadDemoSession',
            description: 'Load pre-configured demo session',
            delay: 0,
        },
        {
            name: 'Wait for session load',
            delay: 2000,
            validate: async (state: ExtensionState) => {
                return state.getAllSampleNames().length > 0;
            },
        },
        {
            name: 'Save session',
            command: 'squiggy.saveSession',
            description: 'Save current session state',
            delay: 1000,
        },
        {
            name: 'Clear state',
            command: 'squiggy.clearState',
            description: 'Clear all loaded data',
            delay: 1000,
        },
        {
            name: 'Verify cleared state',
            delay: 500,
            validate: async (state: ExtensionState) => {
                return state.getLoadedItems().length === 0;
            },
        },
        {
            name: 'Restore session',
            command: 'squiggy.restoreSession',
            description: 'Restore previously saved session',
            delay: 1000,
        },
        {
            name: 'Wait for restore',
            delay: 2000,
            validate: async (state: ExtensionState) => {
                return state.getAllSampleNames().length > 0;
            },
        },
    ],
};

/**
 * Multi-sample comparison workflow
 */
export const comparisonWorkflow: WalkthroughScenario = {
    name: 'Multi-Sample Comparison',
    description: 'Load multiple samples and generate comparison plots',
    steps: [
        {
            name: 'Load test multi-read dataset',
            command: 'squiggy.loadTestMultiReadDataset',
            description: 'Load multiple sample dataset for comparison',
            delay: 0,
        },
        {
            name: 'Wait for samples to load',
            delay: 3000,
            validate: async (state: ExtensionState) => {
                return state.getAllSampleNames().length >= 2;
            },
        },
        {
            name: 'Refresh reads',
            command: 'squiggy.refreshReads',
            description: 'Update read explorer',
            delay: 500,
        },
        {
            name: 'Wait for UI update',
            delay: 1000,
        },
    ],
};

/**
 * Plot generation workflow
 */
export const plottingWorkflow: WalkthroughScenario = {
    name: 'Plot Generation',
    description: 'Test various plot types and options',
    steps: [
        {
            name: 'Load test data',
            command: 'squiggy.loadTestData',
            description: 'Load sample data',
            delay: 0,
        },
        {
            name: 'Wait for data loading',
            delay: 2000,
        },
        {
            name: 'Refresh reads',
            command: 'squiggy.refreshReads',
            delay: 500,
        },
        {
            name: 'Wait for reads',
            delay: 1500,
        },
        // Note: Actual plotting requires read selection in the UI
        // This is a limitation of command-based automation
        {
            name: 'Show logs',
            command: 'squiggy.showLogs',
            delay: 500,
        },
    ],
};

/**
 * File operations workflow
 */
export const fileOperations: WalkthroughScenario = {
    name: 'File Operations',
    description: 'Test opening and closing files',
    steps: [
        {
            name: 'Load test data',
            command: 'squiggy.loadTestData',
            description: 'Load test POD5 and BAM',
            delay: 0,
        },
        {
            name: 'Wait for load',
            delay: 2000,
        },
        {
            name: 'Close BAM file',
            command: 'squiggy.closeBAM',
            description: 'Close BAM file',
            delay: 1000,
        },
        {
            name: 'Close POD5 file',
            command: 'squiggy.closePOD5',
            description: 'Close POD5 file',
            delay: 1000,
        },
        {
            name: 'Verify closed',
            delay: 500,
            validate: async (state: ExtensionState) => {
                return state.getLoadedItems().length === 0;
            },
        },
    ],
};

/**
 * State management workflow
 */
export const stateManagement: WalkthroughScenario = {
    name: 'State Management',
    description: 'Test state clearing and session management',
    steps: [
        {
            name: 'Load demo session',
            command: 'squiggy.loadDemoSession',
            delay: 0,
        },
        {
            name: 'Wait for load',
            delay: 2000,
        },
        {
            name: 'Clear all state',
            command: 'squiggy.clearState',
            delay: 1000,
        },
        {
            name: 'Verify cleared',
            delay: 500,
            validate: async (state: ExtensionState) => {
                return state.getLoadedItems().length === 0;
            },
        },
        {
            name: 'Clear saved session',
            command: 'squiggy.clearSession',
            delay: 500,
        },
    ],
};

/**
 * All predefined scenarios
 */
export const allScenarios: WalkthroughScenario[] = [
    basicWorkflow,
    sessionWorkflow,
    comparisonWorkflow,
    plottingWorkflow,
    fileOperations,
    stateManagement,
];

/**
 * Get scenario by name
 */
export function getScenario(name: string): WalkthroughScenario | undefined {
    return allScenarios.find(
        (s) => s.name.toLowerCase() === name.toLowerCase() || s.name.toLowerCase().includes(name.toLowerCase())
    );
}

/**
 * List all scenario names
 */
export function listScenarios(): string[] {
    return allScenarios.map((s) => s.name);
}
