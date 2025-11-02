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

export class ReadsViewPane extends BaseWebviewProvider {
    public static readonly viewType = 'squiggyReadList';

    private _hasReferences: boolean = false;
    private _readItems: ReadListItem[] = [];
    private _referenceToReads?: Map<string, ReadItem[]>;

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
            return;
        }

        // Re-send data based on current mode
        if (this._hasReferences && this._referenceToReads) {
            this.postMessage({
                type: 'updateReads',
                reads: this._readItems,
                groupedByReference: true,
                referenceToReads: Array.from(this._referenceToReads.entries()),
            });
        } else if (this._readItems.length > 0) {
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
     * Refresh the view
     */
    public refresh(): void {
        this.updateView();
    }
}
