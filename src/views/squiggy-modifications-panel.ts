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
import { logger } from '../utils/logger';

export class ModificationsPanelProvider extends BaseWebviewProvider {
    public static readonly viewType = 'squiggyModificationsPanel';

    private _hasModifications: boolean = false;
    private _modificationTypes: string[] = [];
    private _hasProbabilities: boolean = false;
    private _minProbability: number = 0.5;
    private _minFrequency: number = 0.2;
    private _minModifiedReads: number = 5;
    private _enabledModTypes: Set<string> = new Set();

    // Event emitter for when filter options change
    private _onDidChangeFilters = new vscode.EventEmitter<void>();
    public readonly onDidChangeFilters = this._onDidChangeFilters.event;

    protected getTitle(): string {
        return 'Squiggy Modifications';
    }

    protected async handleMessage(message: ModificationsIncomingMessage): Promise<void> {
        logger.debug('[ModificationsPanel] Received message:', message);

        if (message.type === 'ready') {
            logger.debug(
                '[ModificationsPanel] Webview ready, hasModifications:',
                this._hasModifications
            );
            this.updateView();
            return;
        }

        if (message.type === 'filtersChanged') {
            this._minProbability = message.minProbability;
            this._minFrequency = message.minFrequency;
            this._minModifiedReads = message.minModifiedReads;
            this._enabledModTypes = new Set(message.enabledModTypes);
            this._onDidChangeFilters.fire();
        }
    }

    protected updateView(): void {
        // Don't check isVisible here - if we have a view and received 'ready',
        // the webview is ready to receive messages even if not technically "visible" yet
        if (!this._view) {
            logger.debug(
                '[ModificationsPanel] updateView: No view available, data will be sent when webview is ready'
            );
            return;
        }

        if (this._hasModifications) {
            const message: UpdateModInfoMessage = {
                type: 'updateModInfo',
                hasModifications: this._hasModifications,
                modificationTypes: this._modificationTypes,
                hasProbabilities: this._hasProbabilities,
            };
            logger.debug('[ModificationsPanel] Sending updateModInfo message:', message);
            this.postMessage(message);
        } else {
            const message: ClearModsMessage = {
                type: 'clearMods',
            };
            logger.debug('[ModificationsPanel] Sending clearMods message');
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
        logger.debug(
            '[ModificationsPanel] setModificationInfo called:',
            hasModifications,
            modificationTypes
        );
        this._hasModifications = hasModifications;
        this._modificationTypes = modificationTypes;
        this._hasProbabilities = hasProbabilities;

        // Initialize all modification types as enabled by default
        this._enabledModTypes = new Set(modificationTypes);

        logger.debug('[ModificationsPanel] Calling updateView, _view exists:', !!this._view);
        this.updateView();
    }

    /**
     * Get current filter settings
     */
    public getFilters() {
        return {
            minProbability: this._minProbability,
            minFrequency: this._minFrequency,
            minModifiedReads: this._minModifiedReads,
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
