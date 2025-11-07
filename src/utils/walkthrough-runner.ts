/**
 * UX Walkthrough Runner
 *
 * Executes automated UI walkthroughs with logging and state capture
 */

import * as vscode from 'vscode';
import { WalkthroughLogger } from './walkthrough-logger';
import { ExtensionState } from '../state/extension-state';

export interface WalkthroughStep {
    name: string;
    command?: string;
    args?: any[];
    delay?: number; // Delay before this step (ms)
    validate?: (state: ExtensionState) => Promise<boolean>;
    description?: string;
}

export interface WalkthroughScenario {
    name: string;
    description: string;
    steps: WalkthroughStep[];
}

export class WalkthroughRunner {
    private logger: WalkthroughLogger;
    private state: ExtensionState;
    private outputChannel: vscode.OutputChannel;

    constructor(state: ExtensionState, outputChannel: vscode.OutputChannel) {
        this.logger = new WalkthroughLogger();
        this.state = state;
        this.outputChannel = outputChannel;
    }

    /**
     * Execute a walkthrough scenario
     */
    async run(scenario: WalkthroughScenario): Promise<void> {
        this.outputChannel.show();
        this.logger.start(scenario.name);

        this.outputChannel.appendLine('='.repeat(80));
        this.outputChannel.appendLine(`Starting Walkthrough: ${scenario.name}`);
        this.outputChannel.appendLine(`Description: ${scenario.description}`);
        this.outputChannel.appendLine('='.repeat(80));
        this.outputChannel.appendLine('');

        try {
            for (let i = 0; i < scenario.steps.length; i++) {
                const step = scenario.steps[i];

                // Delay before step
                if (step.delay && step.delay > 0) {
                    await this.sleep(step.delay);
                }

                this.outputChannel.appendLine(`[${i + 1}/${scenario.steps.length}] ${step.name}`);
                if (step.description) {
                    this.outputChannel.appendLine(`    ${step.description}`);
                }

                // Execute command if specified
                if (step.command) {
                    await this.logger.logCommand(step.command, step.args || [], async () => {
                        const result = await vscode.commands.executeCommand(
                            step.command!,
                            ...(step.args || [])
                        );
                        this.outputChannel.appendLine(`    ✅ Command executed: ${step.command}`);
                        return result;
                    });
                }

                // Validate state if specified
                if (step.validate) {
                    const passed = await step.validate(this.state);
                    this.logger.logCheck(step.name, passed);
                    if (!passed) {
                        this.outputChannel.appendLine(`    ⚠️  Validation failed for step: ${step.name}`);
                    } else {
                        this.outputChannel.appendLine(`    ✓ Validation passed`);
                    }
                }

                this.outputChannel.appendLine('');
            }

            this.logger.end();
            this.outputChannel.appendLine('');
            this.outputChannel.appendLine(this.logger.generateReport());

            const summary = this.logger.getSummary();
            if (summary.errorCount === 0) {
                vscode.window.showInformationMessage(
                    `Walkthrough completed: ${summary.successCount} steps successful`
                );
            } else {
                vscode.window.showWarningMessage(
                    `Walkthrough completed with ${summary.errorCount} errors`
                );
            }
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            this.outputChannel.appendLine(`❌ Walkthrough failed: ${errorMessage}`);
            this.logger.end();
            vscode.window.showErrorMessage(`Walkthrough failed: ${errorMessage}`);
            throw error;
        }
    }

    /**
     * Capture current extension state
     */
    captureState(description: string): void {
        const stateSnapshot = {
            loadedItems: this.state.getLoadedItems().length,
            samples: this.state.getAllSampleNames(),
            comparisonItems: this.state.getComparisonItems(),
            currentPlotReadIds: this.state.currentPlotReadIds,
            hasPod5: this.state.getLoadedItems().some((i) => i.type === 'pod5' || i.type === 'sample'),
            hasBam: this.state.getLoadedItems().some((i) => i.bamPath !== undefined),
        };

        this.logger.logState(description, stateSnapshot);
        this.outputChannel.appendLine(`    📸 State captured: ${description}`);
        this.outputChannel.appendLine(`       ${JSON.stringify(stateSnapshot, null, 2).split('\n').join('\n       ')}`);
    }

    /**
     * Get the logger instance for manual logging
     */
    getLogger(): WalkthroughLogger {
        return this.logger;
    }

    /**
     * Helper to sleep for specified duration
     */
    private sleep(ms: number): Promise<void> {
        return new Promise((resolve) => setTimeout(resolve, ms));
    }
}
