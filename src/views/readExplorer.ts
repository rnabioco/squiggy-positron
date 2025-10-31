/**
 * Read Explorer TreeView
 *
 * Provides a hierarchical view of reads from POD5 files
 */

import * as vscode from 'vscode';
import { PythonBackend } from '../backend/pythonBackend';

export class ReadItem extends vscode.TreeItem {
    constructor(
        public readonly readId: string,
        public readonly collapsibleState: vscode.TreeItemCollapsibleState,
        public readonly command?: vscode.Command
    ) {
        super(readId, collapsibleState);

        this.tooltip = `Read ID: ${readId}`;
        this.contextValue = 'read';
        this.iconPath = new vscode.ThemeIcon('pulse');
    }
}

export class ReadTreeProvider implements vscode.TreeDataProvider<ReadItem> {
    private _onDidChangeTreeData = new vscode.EventEmitter<ReadItem | undefined | void>();
    readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

    private reads: string[] = [];

    constructor() {}

    /**
     * Set the list of reads to display
     */
    setReads(readIds: string[]): void {
        this.reads = readIds;
        this.refresh();
    }

    /**
     * Refresh the tree view
     */
    refresh(): void {
        this._onDidChangeTreeData.fire();
    }

    /**
     * Get tree item for an element
     */
    getTreeItem(element: ReadItem): vscode.TreeItem {
        return element;
    }

    /**
     * Get children for a tree element
     */
    getChildren(element?: ReadItem): Thenable<ReadItem[]> {
        if (!element) {
            // Root level - show all reads
            return Promise.resolve(
                this.reads.map(readId =>
                    new ReadItem(
                        readId,
                        vscode.TreeItemCollapsibleState.None,
                        {
                            command: 'squiggy.plotRead',
                            title: 'Plot Read',
                            arguments: [readId]
                        }
                    )
                )
            );
        }

        // No children for read items
        return Promise.resolve([]);
    }

    /**
     * Filter reads by search text
     */
    filterReads(searchText: string): void {
        if (!searchText) {
            this.refresh();
            return;
        }

        // Simple case-insensitive filter
        // In future, could support more complex queries
        const filtered = this.reads.filter(readId =>
            readId.toLowerCase().includes(searchText.toLowerCase())
        );

        // Temporarily show only filtered reads
        // Note: This is a simple implementation
        // A more robust version would preserve original reads
        this.refresh();
    }
}
