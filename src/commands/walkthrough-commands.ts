/**
 * Walkthrough Commands
 *
 * Commands for running automated UX walkthroughs
 */

import * as vscode from 'vscode';
import { ExtensionState } from '../state/extension-state';
import { WalkthroughRunner } from '../utils/walkthrough-runner';
import {
    allScenarios,
    getScenario,
    listScenarios,
} from '../utils/walkthrough-scenarios';

/**
 * Register walkthrough commands
 */
export function registerWalkthroughCommands(
    context: vscode.ExtensionContext,
    state: ExtensionState
): void {
    // Create shared output channel for walkthroughs
    const walkthroughChannel = vscode.window.createOutputChannel('Squiggy Walkthrough');

    // Run a specific walkthrough scenario
    context.subscriptions.push(
        vscode.commands.registerCommand('squiggy.runWalkthrough', async (scenarioName?: string) => {
            try {
                // If no scenario name provided, show quick pick
                if (!scenarioName) {
                    const scenarios = listScenarios();
                    const selected = await vscode.window.showQuickPick(scenarios, {
                        placeHolder: 'Select a walkthrough scenario to run',
                    });

                    if (!selected) {
                        return; // User cancelled
                    }

                    scenarioName = selected;
                }

                // Get scenario
                const scenario = getScenario(scenarioName);
                if (!scenario) {
                    vscode.window.showErrorMessage(`Scenario not found: ${scenarioName}`);
                    return;
                }

                // Show confirmation
                const proceed = await vscode.window.showInformationMessage(
                    `Run walkthrough: ${scenario.name}?\n${scenario.description}`,
                    'Run',
                    'Cancel'
                );

                if (proceed !== 'Run') {
                    return;
                }

                // Run walkthrough
                const runner = new WalkthroughRunner(state, walkthroughChannel);
                await runner.run(scenario);
            } catch (error) {
                const errorMessage = error instanceof Error ? error.message : String(error);
                vscode.window.showErrorMessage(`Walkthrough error: ${errorMessage}`);
            }
        })
    );

    // Run all walkthrough scenarios
    context.subscriptions.push(
        vscode.commands.registerCommand('squiggy.runAllWalkthroughs', async () => {
            try {
                const proceed = await vscode.window.showWarningMessage(
                    `This will run ${allScenarios.length} walkthrough scenarios. This may take several minutes and will clear your current state. Continue?`,
                    'Run All',
                    'Cancel'
                );

                if (proceed !== 'Run All') {
                    return;
                }

                walkthroughChannel.show();
                walkthroughChannel.clear();

                let successCount = 0;
                let failureCount = 0;

                for (let i = 0; i < allScenarios.length; i++) {
                    const scenario = allScenarios[i];

                    walkthroughChannel.appendLine('');
                    walkthroughChannel.appendLine('='.repeat(80));
                    walkthroughChannel.appendLine(
                        `Running scenario ${i + 1}/${allScenarios.length}: ${scenario.name}`
                    );
                    walkthroughChannel.appendLine('='.repeat(80));
                    walkthroughChannel.appendLine('');

                    try {
                        const runner = new WalkthroughRunner(state, walkthroughChannel);
                        await runner.run(scenario);
                        successCount++;

                        // Clean up between scenarios
                        await vscode.commands.executeCommand('squiggy.clearState');
                        await new Promise((resolve) => setTimeout(resolve, 1000));
                    } catch (error) {
                        failureCount++;
                        walkthroughChannel.appendLine(`❌ Scenario failed: ${error}`);
                    }
                }

                walkthroughChannel.appendLine('');
                walkthroughChannel.appendLine('='.repeat(80));
                walkthroughChannel.appendLine('SUMMARY');
                walkthroughChannel.appendLine('='.repeat(80));
                walkthroughChannel.appendLine(`Total scenarios: ${allScenarios.length}`);
                walkthroughChannel.appendLine(`Successful: ${successCount}`);
                walkthroughChannel.appendLine(`Failed: ${failureCount}`);
                walkthroughChannel.appendLine('='.repeat(80));

                if (failureCount === 0) {
                    vscode.window.showInformationMessage(
                        `All ${allScenarios.length} walkthroughs completed successfully!`
                    );
                } else {
                    vscode.window.showWarningMessage(
                        `Walkthroughs completed: ${successCount} succeeded, ${failureCount} failed`
                    );
                }
            } catch (error) {
                const errorMessage = error instanceof Error ? error.message : String(error);
                vscode.window.showErrorMessage(`Walkthrough batch error: ${errorMessage}`);
            }
        })
    );

    // List available walkthroughs
    context.subscriptions.push(
        vscode.commands.registerCommand('squiggy.listWalkthroughs', async () => {
            const scenarios = allScenarios.map((s) => `• ${s.name}: ${s.description}`).join('\n');

            walkthroughChannel.clear();
            walkthroughChannel.show();
            walkthroughChannel.appendLine('Available Walkthrough Scenarios:');
            walkthroughChannel.appendLine('='.repeat(80));
            walkthroughChannel.appendLine(scenarios);
            walkthroughChannel.appendLine('='.repeat(80));
            walkthroughChannel.appendLine('');
            walkthroughChannel.appendLine('Run a walkthrough with:');
            walkthroughChannel.appendLine('  Command Palette → "Squiggy: Run UX Walkthrough"');
        })
    );
}
