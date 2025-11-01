/**
 * Base Modifications Panel Webview View
 *
 * Displays base modification information from BAM files with MM/ML tags
 */

import * as vscode from 'vscode';
import { BaseWebviewProvider } from './base-webview-provider';
import {
    ModificationsIncomingMessage,
    UpdateModInfoMessage,
    ClearModsMessage,
} from '../types/messages';

export class ModificationsPanelProvider extends BaseWebviewProvider {
    public static readonly viewType = 'squiggyModificationsPanel';

    private _hasModifications: boolean = false;
    private _modificationTypes: string[] = [];
    private _hasProbabilities: boolean = false;
    private _minProbability: number = 0.5;
    private _enabledModTypes: Set<string> = new Set();

    // Event emitter for when filter options change
    private _onDidChangeFilters = new vscode.EventEmitter<void>();
    public readonly onDidChangeFilters = this._onDidChangeFilters.event;

    protected getTitle(): string {
        return 'Squiggy Modifications';
    }

    protected async handleMessage(message: ModificationsIncomingMessage): Promise<void> {
        if (message.type === 'ready') {
            this.updateView();
            return;
        }

        if (message.type === 'filtersChanged') {
            this._minProbability = message.minProbability;
            this._enabledModTypes = new Set(message.enabledModTypes);
            this._onDidChangeFilters.fire();
        }
    }

    protected updateView(): void {
        if (!this.isVisible) {
            return;
        }

        if (this._hasModifications) {
            const message: UpdateModInfoMessage = {
                type: 'updateModInfo',
                hasModifications: this._hasModifications,
                modificationTypes: this._modificationTypes,
                hasProbabilities: this._hasProbabilities,
            };
            this.postMessage(message);
        } else {
            const message: ClearModsMessage = {
                type: 'clearMods',
            };
            this.postMessage(message);
        }
    }

    /**
     * Update modification info display
     */
    public setModificationInfo(
        hasModifications: boolean,
        modificationTypes: string[],
        hasProbabilities: boolean
    ) {
        this._hasModifications = hasModifications;
        this._modificationTypes = modificationTypes;
        this._hasProbabilities = hasProbabilities;

        // Initialize all modification types as enabled by default
        this._enabledModTypes = new Set(modificationTypes);

        this.updateView();
    }

    /**
     * Get current filter settings
     */
    public getFilters() {
        return {
            minProbability: this._minProbability,
            enabledModTypes: Array.from(this._enabledModTypes),
        };
    }

    /**
     * Clear modification info (when no BAM loaded or BAM has no modifications)
     */
    public clear() {
        this._hasModifications = false;
        this._modificationTypes = [];
        this._hasProbabilities = false;
        this.updateView();
    }
}
