/**
 * Reads View Pane - WebviewViewProvider for React-based reads panel
 *
 * Provides a multi-column table view of reads from POD5 files,
 * optionally grouped by reference (when BAM loaded).
 *
 * Follows Positron Variables panel pattern with ViewPane container.
 */

import * as vscode from 'vscode';
import { BaseWebviewProvider } from './base-webview-provider';
import { ReadItem, ReadListItem, ReferenceGroupItem } from '../types/squiggy-reads-types';
import { ReadsViewIncomingMessage } from '../types/messages';
import { ExtensionState } from '../state/extension-state';

export class ReadsViewPane extends BaseWebviewProvider {
    public static readonly viewType = 'squiggyReadList';

    private _hasReferences: boolean = false;
    private _readItems: ReadListItem[] = [];
    private _referenceToReads?: Map<string, ReadItem[]>;
    private _state: ExtensionState;

    constructor(extensionUri: vscode.Uri, state: ExtensionState) {
        super(extensionUri);
        this._state = state;
    }

    protected getTitle(): string {
        return 'Squiggy Reads Explorer';
    }

    protected async handleMessage(message: ReadsViewIncomingMessage): Promise<void> {
        switch (message.type) {
            case 'plotRead':
                vscode.commands.executeCommand('squiggy.plotRead', message.readId);
                break;
            case 'plotAggregate':
                vscode.commands.executeCommand('squiggy.plotAggregate', message.referenceName);
                break;
            case 'loadMore':
                // POD5 pagination: request more reads
                vscode.commands.executeCommand('squiggy.internal.loadMoreReads');
                break;
            case 'expandReference':
                // BAM lazy loading: fetch reads for specific reference
                vscode.commands.executeCommand(
                    'squiggy.internal.expandReference',
                    message.referenceName,
                    message.offset,
                    message.limit
                );
                break;
            case 'selectSample':
                // User selected a different sample in the dropdown
                this._state.selectedReadExplorerSample = message.sampleName;
                // Reload reads for selected sample
                vscode.commands.executeCommand(
                    'squiggy.internal.loadReadsForSample',
                    message.sampleName
                );
                break;
            case 'ready':
                // Webview is ready, send initial state
                this.updateView();
                break;
        }
    }

    protected updateView(): void {
        // Don't check isVisible - if we have a view and received 'ready',
        // the webview is ready to receive messages
        if (!this._view) {
            console.log('[ReadsViewPane] updateView called but _view is null, skipping');
            return;
        }

        // Send available samples first
        const availableSamples = this.getAvailableSamples();
        const selectedSample = this.getSelectedSample();
        console.log(
            '[ReadsViewPane] Sending setAvailableSamples:',
            availableSamples,
            'selected:',
            selectedSample
        );
        this.postMessage({
            type: 'setAvailableSamples',
            samples: availableSamples,
            selectedSample,
        });

        // Always send update message (even if empty, to clear the view)
        if (this._hasReferences && this._referenceToReads) {
            this.postMessage({
                type: 'updateReads',
                reads: this._readItems,
                groupedByReference: true,
                referenceToReads: Array.from(this._referenceToReads.entries()),
            });
        } else {
            // Send update even if empty - this clears the webview
            this.postMessage({
                type: 'updateReads',
                reads: this._readItems,
                groupedByReference: false,
            });
        }
    }

    /**
     * Set reads without reference grouping (flat list)
     */
    public setReads(readIds: string[]): void {
        this._hasReferences = false;
        this._readItems = readIds.map((readId) => ({
            type: 'read' as const,
            readId,
            indentLevel: 0,
        }));

        this.updateView();
    }

    /**
     * Set reads grouped by reference (BAM loaded)
     */
    public setReadsGrouped(referenceToReads: Map<string, ReadItem[]>): void {
        this._hasReferences = true;
        this._referenceToReads = referenceToReads;

        // Flatten into list with reference headers (initially all collapsed)
        const items: ReadListItem[] = [];
        for (const [referenceName, reads] of referenceToReads.entries()) {
            // Add reference group header
            items.push({
                type: 'reference',
                referenceName,
                readCount: reads.length,
                isExpanded: false,
                indentLevel: 0,
            } as ReferenceGroupItem);

            // Reads will be shown when expanded (handled in webview)
        }

        this._readItems = items;
        this.updateView();
    }

    /**
     * Set reference headers only (for lazy loading mode)
     * Reads will be fetched when user expands each reference
     */
    public setReferencesOnly(references: { referenceName: string; readCount: number }[]): void {
        this._hasReferences = true;
        this._referenceToReads = new Map(); // Empty initially

        // Build _readItems with reference headers so state persists across visibility changes
        const items: ReadListItem[] = [];
        for (const ref of references) {
            items.push({
                type: 'reference',
                referenceName: ref.referenceName,
                readCount: ref.readCount,
                isExpanded: false,
                indentLevel: 0,
            } as ReferenceGroupItem);
        }
        this._readItems = items;

        // Send specific message for initial load, but also update persistent state
        this.postMessage({
            type: 'setReferencesOnly',
            references,
        });
    }

    /**
     * Append reads to the flat list (for POD5 pagination)
     */
    public appendReads(newReadIds: string[]): void {
        const newItems = newReadIds.map((readId) => ({
            type: 'read' as const,
            readId,
            indentLevel: 0 as 0 | 1,
        }));

        this._readItems = [...this._readItems, ...newItems];

        this.postMessage({
            type: 'appendReads',
            reads: newItems,
        });
    }

    /**
     * Set reads for a specific reference (lazy loading for BAM mode)
     */
    public setReadsForReference(
        referenceName: string,
        readIds: string[],
        offset: number,
        totalCount: number
    ): void {
        const reads = readIds.map((readId) => ({
            type: 'read' as const,
            readId,
            referenceName,
            indentLevel: 1 as 0 | 1,
        }));

        this.postMessage({
            type: 'setReadsForReference',
            referenceName,
            reads,
            offset,
            totalCount,
        });
    }

    /**
     * Send loading state to webview
     */
    public setLoading(isLoading: boolean, message?: string): void {
        this.postMessage({
            type: 'setLoading',
            isLoading,
            message,
        });
    }

    /**
     * Get list of available samples
     */
    public getAvailableSamples(): string[] {
        return this._state.getAllSampleNames();
    }

    /**
     * Get currently selected sample
     */
    public getSelectedSample(): string | null {
        return this._state.selectedReadExplorerSample;
    }

    /**
     * Refresh the view
     */
    public refresh(): void {
        this.updateView();
    }
}
