/**
 * Motif Search Panel Webview View
 *
 * Provides UI for searching motifs in FASTA files and generating aggregate plots
 */

import * as vscode from 'vscode';
import { BaseWebviewProvider } from './base-webview-provider';
import { MotifMatch } from '../types/motif-types';
import { IncomingWebviewMessage } from '../types/messages';

export class MotifSearchPanelProvider extends BaseWebviewProvider {
    public static readonly viewType = 'squiggyMotifSearch';

    private _matches: MotifMatch[] = [];
    private _currentMotif: string = 'DRACH';
    private _searching: boolean = false;

    constructor(
        extensionUri: vscode.Uri,
        private readonly state: any // ExtensionState
    ) {
        super(extensionUri);
    }

    protected getTitle(): string {
        return 'Squiggy Motif Search';
    }

    protected async handleMessage(message: IncomingWebviewMessage): Promise<void> {
        // Handle common messages
        if (message.type === 'ready') {
            this.updateView();
            return;
        }

        // Handle motif-specific messages (use type assertion for custom messages)
        const motifMessage = message as any;
        if (motifMessage.type === 'searchMotif' && motifMessage.motif) {
            await this.searchMotif(motifMessage.motif);
        } else if (motifMessage.type === 'plotMotif') {
            if (
                motifMessage.matchIndex !== undefined &&
                motifMessage.motif &&
                motifMessage.window !== undefined
            ) {
                await this.plotMotif(
                    motifMessage.motif,
                    motifMessage.matchIndex,
                    motifMessage.window
                );
            }
        }
    }

    protected updateView(): void {
        if (!this._view) {
            return;
        }

        const message = {
            type: 'updateMatches',
            matches: this._matches,
            searching: this._searching,
        };
        this.postMessage(message as any);
    }

    /**
     * Search for motif matches
     */
    private async searchMotif(motif: string): Promise<void> {
        if (!this.state.currentFastaFile) {
            vscode.window.showErrorMessage('No FASTA file loaded. Use "Open FASTA File" first.');
            return;
        }

        this._currentMotif = motif;
        this._searching = true;
        this.updateView();

        try {
            if (this.state.usePositron && this.state.squiggyAPI) {
                const matches = await this.state.squiggyAPI.searchMotif(
                    this.state.currentFastaFile,
                    motif
                );
                this._matches = matches;
            } else {
                vscode.window.showErrorMessage(
                    'Motif search requires Positron runtime. Please use Positron IDE.'
                );
                this._matches = [];
            }
        } catch (error) {
            vscode.window.showErrorMessage(
                `Failed to search motif: ${error instanceof Error ? error.message : String(error)}`
            );
            this._matches = [];
        } finally {
            this._searching = false;
            this.updateView();
        }
    }

    /**
     * Generate motif aggregate plot
     */
    private async plotMotif(motif: string, matchIndex: number, window: number): Promise<void> {
        if (!this.state.currentFastaFile) {
            vscode.window.showErrorMessage('No FASTA file loaded.');
            return;
        }

        try {
            await vscode.commands.executeCommand('squiggy.plotMotifAggregate', {
                fastaFile: this.state.currentFastaFile,
                motif: motif,
                matchIndex: matchIndex,
                window: window,
            });
        } catch (error) {
            vscode.window.showErrorMessage(
                `Failed to plot motif: ${error instanceof Error ? error.message : String(error)}`
            );
        }
    }

    /**
     * Public API to set matches (if needed from external commands)
     */
    public setMatches(matches: MotifMatch[]): void {
        this._matches = matches;
        this.updateView();
    }

    /**
     * Clear matches
     */
    public clear(): void {
        this._matches = [];
        this._currentMotif = 'DRACH';
        this._searching = false;
        this.updateView();
    }
}
