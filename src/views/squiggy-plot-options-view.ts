/**
 * Plot Options Webview View
 *
 * Provides controls for plot configuration in the sidebar
 */

import * as vscode from 'vscode';
import { BaseWebviewProvider } from './base-webview-provider';
import {
    PlotOptionsIncomingMessage,
    UpdatePlotOptionsMessage,
    UpdateBamStatusMessage,
} from '../types/messages';

export class PlotOptionsViewProvider extends BaseWebviewProvider {
    public static readonly viewType = 'squiggyPlotOptions';

    private _plotMode: string = 'SINGLE';
    private _normalization: string = 'ZNORM';
    private _showDwellTime: boolean = false;
    private _showBaseAnnotations: boolean = true;
    private _scaleDwellTime: boolean = false;
    private _downsample: number = 5;
    private _showSignalPoints: boolean = false;
    private _hasBamFile: boolean = false;

    // Event emitter for when options change that should trigger refresh
    private _onDidChangeOptions = new vscode.EventEmitter<void>();
    public readonly onDidChangeOptions = this._onDidChangeOptions.event;

    protected getTitle(): string {
        return 'Squiggy Plot Options';
    }

    protected async handleMessage(message: PlotOptionsIncomingMessage): Promise<void> {
        if (message.type === 'ready') {
            this.updateView();
            return;
        }

        if (message.type === 'optionsChanged') {
            // Update internal state
            this._plotMode = message.options.mode;
            this._normalization = message.options.normalization;
            this._showDwellTime = message.options.showDwellTime;
            this._showBaseAnnotations = message.options.showBaseAnnotations;
            this._scaleDwellTime = message.options.scaleDwellTime;
            this._downsample = message.options.downsample;
            this._showSignalPoints = message.options.showSignalPoints;

            // Fire change event
            this._onDidChangeOptions.fire();
        }
    }

    protected updateView(): void {
        // Don't check isVisible - if we have a view and received 'ready',
        // the webview is ready to receive messages
        if (!this._view) {
            return;
        }

        // Send all current option values to the webview
        const updateMessage: UpdatePlotOptionsMessage = {
            type: 'updatePlotOptions',
            options: {
                mode: this._plotMode as any,
                normalization: this._normalization as any,
                showDwellTime: this._showDwellTime,
                showBaseAnnotations: this._showBaseAnnotations,
                scaleDwellTime: this._scaleDwellTime,
                downsample: this._downsample,
                showSignalPoints: this._showSignalPoints,
            },
        };
        this.postMessage(updateMessage);
    }

    /**
     * Get current plot options
     */
    public getOptions() {
        return {
            mode: this._plotMode,
            normalization: this._normalization,
            showDwellTime: this._showDwellTime,
            showBaseAnnotations: this._showBaseAnnotations,
            scaleDwellTime: this._scaleDwellTime,
            downsample: this._downsample,
            showSignalPoints: this._showSignalPoints,
        };
    }

    /**
     * Update BAM file status and available plot modes
     */
    public updateBamStatus(hasBam: boolean) {
        this._hasBamFile = hasBam;

        // If BAM loaded, default to EVENTALIGN
        if (hasBam && this._plotMode === 'SINGLE') {
            this._plotMode = 'EVENTALIGN';
            this._updateConfig('defaultPlotMode', 'EVENTALIGN');
        }
        // If BAM unloaded and currently in EVENTALIGN mode, switch to SINGLE
        else if (!hasBam && this._plotMode === 'EVENTALIGN') {
            this._plotMode = 'SINGLE';
            this._updateConfig('defaultPlotMode', 'SINGLE');
        }

        // Update webview
        const message: UpdateBamStatusMessage = {
            type: 'updateBamStatus',
            hasBam: this._hasBamFile,
        };
        this.postMessage(message);
    }

    /**
     * Update workspace configuration
     */
    private _updateConfig(key: string, value: any): void {
        const config = vscode.workspace.getConfiguration('squiggy');
        config.update(key, value, vscode.ConfigurationTarget.Workspace);
    }
}
