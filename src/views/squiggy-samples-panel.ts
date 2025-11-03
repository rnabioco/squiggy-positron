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
import { ExtensionState, SampleInfo } from '../state/extension-state';
import { formatFileSize } from '../utils/format-utils';

export class SamplesPanelProvider extends BaseWebviewProvider {
    public static readonly viewType = 'squiggySamplesPanel';

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
                    vscode.window.showWarningMessage('Please select at least 2 samples for comparison');
                }
                break;

            case 'unloadSample':
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

    protected updateView(): void {
        if (!this._view) {
            console.log('SamplesPanelProvider: No view to update');
            return;
        }

        // Rebuild samples list from extension state
        this._samples = Array.from(this._state.getAllSampleNames())
            .map((name) => {
                const sampleInfo = this._state.getSample(name);
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

        if (this._samples.length === 0) {
            const message: ClearSamplesMessage = {
                type: 'clearSamples',
            };
            this.postMessage(message);
        } else {
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
    public refresh(): void {
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
