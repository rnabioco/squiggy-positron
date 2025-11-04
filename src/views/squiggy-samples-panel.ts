/**
 * Sample Comparison Panel Webview View
 *
 * Manages loaded samples for multi-sample comparison with selection,
 * comparison workflow, and sample management (unload).
 */

import * as vscode from 'vscode';
import { BaseWebviewProvider } from './base-webview-provider';
import {
    SamplesIncomingMessage,
    UpdateSamplesMessage,
    SampleItem,
    ClearSamplesMessage,
} from '../types/messages';
import { ExtensionState } from '../state/extension-state';
// SampleInfo and formatFileSize unused - reserved for future features

export class SamplesPanelProvider extends BaseWebviewProvider {
    public static readonly viewType = 'squiggyComparisonSamples';

    private _state: ExtensionState;
    private _samples: SampleItem[] = [];
    private _selectedSamples: Set<string> = new Set();

    // Event emitter for comparison requests
    private _onDidRequestComparison = new vscode.EventEmitter<string[]>();
    public readonly onDidRequestComparison = this._onDidRequestComparison.event;

    // Event emitter for sample unload requests
    private _onDidRequestUnload = new vscode.EventEmitter<string>();
    public readonly onDidRequestUnload = this._onDidRequestUnload.event;

    constructor(extensionUri: vscode.Uri, state: ExtensionState) {
        super(extensionUri);
        this._state = state;
    }

    protected getTitle(): string {
        return 'Sample Comparison Manager';
    }

    protected async handleMessage(message: SamplesIncomingMessage): Promise<void> {
        switch (message.type) {
            case 'ready':
                // Webview is ready, send initial state
                this.updateView();
                break;

            case 'selectSample':
                if (message.selected) {
                    this._selectedSamples.add(message.sampleName);
                } else {
                    this._selectedSamples.delete(message.sampleName);
                }
                break;

            case 'startComparison':
                if (message.sampleNames.length >= 2) {
                    this._onDidRequestComparison.fire(message.sampleNames);
                } else {
                    vscode.window.showWarningMessage(
                        'Please select at least 2 samples for comparison'
                    );
                }
                break;

            case 'unloadSample': {
                // Ask for confirmation
                const confirm = await vscode.window.showWarningMessage(
                    `Unload sample "${message.sampleName}"?`,
                    { modal: true },
                    'Yes',
                    'Cancel'
                );

                if (confirm === 'Yes') {
                    this._onDidRequestUnload.fire(message.sampleName);
                    this._selectedSamples.delete(message.sampleName);
                }
                break;
            }
        }
    }

    protected updateView(): void {
        console.log('[SamplesPanelProvider] updateView called');
        console.log('[SamplesPanelProvider] _view exists:', !!this._view);

        if (!this._view) {
            console.log('[SamplesPanelProvider] No view to update');
            return;
        }

        // Rebuild samples list from extension state
        const sampleNames = this._state.getAllSampleNames();
        console.log('[SamplesPanelProvider] Sample names from state:', sampleNames);

        this._samples = Array.from(sampleNames)
            .map((name) => {
                const sampleInfo = this._state.getSample(name);
                console.log(`[SamplesPanelProvider] Sample '${name}' info:`, sampleInfo);
                if (!sampleInfo) {
                    return null;
                }

                const sampleItem: SampleItem = {
                    name: sampleInfo.name,
                    pod5Path: sampleInfo.pod5Path,
                    bamPath: sampleInfo.bamPath,
                    fastaPath: sampleInfo.fastaPath,
                    readCount: sampleInfo.readCount,
                    hasBam: sampleInfo.hasBam,
                    hasFasta: sampleInfo.hasFasta,
                };
                return sampleItem;
            })
            .filter((item): item is SampleItem => item !== null);

        console.log('[SamplesPanelProvider] Built samples array:', this._samples);

        if (this._samples.length === 0) {
            console.log('[SamplesPanelProvider] Sending clearSamples message');
            const message: ClearSamplesMessage = {
                type: 'clearSamples',
            };
            this.postMessage(message);
        } else {
            console.log(
                '[SamplesPanelProvider] Sending updateSamples message with',
                this._samples.length,
                'samples'
            );
            const message: UpdateSamplesMessage = {
                type: 'updateSamples',
                samples: this._samples,
            };
            this.postMessage(message);
        }
    }

    /**
     * Update samples display when samples are added
     */
    public async refresh(): Promise<void> {
        // If view doesn't exist yet, try to show it
        if (!this._view) {
            console.log('[SamplesPanelProvider] View not yet created, showing panel...');
            try {
                await vscode.commands.executeCommand('squiggyComparisonSamples.focus');
                // Wait a bit for view to be created
                await new Promise((resolve) => setTimeout(resolve, 500));
            } catch (error) {
                console.error('[SamplesPanelProvider] Error showing panel:', error);
            }
        }

        this.updateView();
    }

    /**
     * Get currently selected sample names
     */
    public getSelectedSamples(): string[] {
        return Array.from(this._selectedSamples);
    }

    /**
     * Clear selection
     */
    public clearSelection(): void {
        this._selectedSamples.clear();
    }
}
