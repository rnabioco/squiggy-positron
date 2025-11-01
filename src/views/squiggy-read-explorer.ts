/**
 * Read Explorer TreeView
 *
 * Provides a hierarchical view of reads from POD5 files grouped by reference
 */

import * as vscode from 'vscode';
// import { PythonBackend } from '../backend/squiggy-python-backend';

type TreeItemType = 'reference' | 'read';

export class ReadItem extends vscode.TreeItem {
    constructor(
        public readonly label: string,
        public readonly itemType: TreeItemType,
        public readonly readId: string,
        public readonly collapsibleState: vscode.TreeItemCollapsibleState,
        public readonly command?: vscode.Command
    ) {
        super(label, collapsibleState);

        if (itemType === 'reference') {
            this.tooltip = `Reference: ${label}`;
            this.contextValue = 'reference';
            this.iconPath = new vscode.ThemeIcon('symbol-namespace');
        } else {
            this.tooltip = `Read ID: ${readId}`;
            this.contextValue = 'read';
            this.iconPath = new vscode.ThemeIcon('pulse');
        }
    }
}

export class ReadTreeProvider implements vscode.TreeDataProvider<ReadItem> {
    private _onDidChangeTreeData = new vscode.EventEmitter<ReadItem | undefined | void>();
    readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

    // Map of reference name to list of read IDs
    private referenceToReads: Map<string, string[]> = new Map();
    private originalReferenceToReads: Map<string, string[]> = new Map(); // Store original data
    private hasReferences: boolean = false;
    private currentFilter: string = '';

    constructor() {}

    /**
     * Set reads without reference grouping (flat list)
     */
    setReads(readIds: string[]): void {
        this.hasReferences = false;
        this.originalReferenceToReads.clear();
        this.originalReferenceToReads.set('_ungrouped', readIds);
        this.currentFilter = '';
        this.applyFilter();
    }

    /**
     * Set reads grouped by reference
     */
    setReadsGrouped(referenceToReads: Map<string, string[]>): void {
        this.hasReferences = true;
        this.originalReferenceToReads = new Map(referenceToReads);
        this.currentFilter = '';
        this.applyFilter();
    }

    /**
     * Refresh the tree view
     */
    refresh(): void {
        this._onDidChangeTreeData.fire();
    }

    /**
     * Apply current filter to the data
     */
    private applyFilter(): void {
        if (!this.currentFilter) {
            this.referenceToReads = new Map(this.originalReferenceToReads);
            this.refresh();
            return;
        }

        const searchLower = this.currentFilter.toLowerCase();
        const filtered = new Map<string, string[]>();

        for (const [refName, readIds] of this.originalReferenceToReads.entries()) {
            // Check if reference name matches
            const refMatches = refName.toLowerCase().includes(searchLower);

            // Check which read IDs match
            const matchingReads = readIds.filter((readId) =>
                readId.toLowerCase().includes(searchLower)
            );

            // Include reference if either:
            // 1. Reference name matches (show all its reads)
            // 2. Some read IDs match (show only matching reads)
            if (refMatches) {
                filtered.set(refName, readIds); // Show all reads
            } else if (matchingReads.length > 0) {
                filtered.set(refName, matchingReads); // Show only matching reads
            }
        }

        this.referenceToReads = filtered;
        this.refresh();
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
            // Root level
            if (this.hasReferences) {
                // Show references as collapsible groups
                const references: ReadItem[] = [];
                for (const [refName, readIds] of this.referenceToReads.entries()) {
                    references.push(
                        new ReadItem(
                            `${refName} (${readIds.length})`,
                            'reference',
                            refName,
                            vscode.TreeItemCollapsibleState.Collapsed
                        )
                    );
                }
                return Promise.resolve(references);
            } else {
                // Flat list of reads (no BAM loaded)
                const readIds = this.referenceToReads.get('_ungrouped') || [];
                return Promise.resolve(
                    readIds.map(
                        (readId) =>
                            new ReadItem(
                                readId,
                                'read',
                                readId,
                                vscode.TreeItemCollapsibleState.None,
                                {
                                    command: 'squiggy.plotRead',
                                    title: 'Plot Read',
                                    arguments: [readId],
                                }
                            )
                    )
                );
            }
        }

        // Children of reference - show reads
        if (element.itemType === 'reference') {
            const readIds = this.referenceToReads.get(element.readId) || [];
            return Promise.resolve(
                readIds.map(
                    (readId) =>
                        new ReadItem(readId, 'read', readId, vscode.TreeItemCollapsibleState.None, {
                            command: 'squiggy.plotRead',
                            title: 'Plot Read',
                            arguments: [readId],
                        })
                )
            );
        }

        // No children for read items
        return Promise.resolve([]);
    }

    /**
     * Filter reads by search text (searches both reference names and read IDs)
     */
    filterReads(searchText: string): void {
        this.currentFilter = searchText;
        this.applyFilter();
    }
}
