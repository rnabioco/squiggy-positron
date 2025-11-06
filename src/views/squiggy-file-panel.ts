/**
 * File Panel Webview View - React-based
 *
 * Provides file management with sortable table layout
 * Subscribes to unified extension state for cross-panel synchronization
 */

import * as vscode from 'vscode';
import * as path from 'path';
import { BaseWebviewProvider } from './base-webview-provider';
import { FilePanelIncomingMessage, UpdateFilesMessage, FileItem } from '../types/messages';
import { formatFileSize } from '../utils/format-utils';
import { ExtensionState } from '../state/extension-state';
import { LoadedItem } from '../types/loaded-item';

export class FilePanelProvider extends BaseWebviewProvider {
    public static readonly viewType = 'squiggyFilePanel';

    private _files: FileItem[] = [];
    private _disposables: vscode.Disposable[] = [];

    /**
     * Constructor with optional ExtensionState for unified state subscription
     * @param extensionUri - Extension URI
     * @param state - Optional ExtensionState for subscribing to unified state changes
     */
    constructor(
        extensionUri: vscode.Uri,
        private state?: ExtensionState
    ) {
        super(extensionUri);

        // Subscribe to unified state changes if state is provided
        if (this.state) {
            const disposable = this.state.onLoadedItemsChanged((items: LoadedItem[]) => {
                this._handleLoadedItemsChanged(items);
            });
            this._disposables.push(disposable);
        }
    }

    /**
     * Handle unified state changes - convert LoadedItem[] to FileItem[] for display
     * Expands each LoadedItem into separate rows for POD5, BAM, and FASTA files
     * @private
     */
    private _handleLoadedItemsChanged(items: LoadedItem[]): void {
        // Convert LoadedItem[] to FileItem[] for the UI
        // Each LoadedItem can expand to multiple rows (POD5 + BAM + FASTA)
        this._files = [];

        for (const item of items) {
            // Add POD5 row
            this._files.push({
                path: item.pod5Path,
                filename: path.basename(item.pod5Path),
                type: 'POD5',
                size: item.fileSize,
                sizeFormatted: item.fileSizeFormatted,
                numReads: item.readCount,
                hasMods: item.hasMods,
                hasEvents: item.hasEvents,
            });

            // Add BAM row if present
            if (item.bamPath) {
                this._files.push({
                    path: item.bamPath,
                    filename: path.basename(item.bamPath),
                    type: 'BAM',
                    size: 0, // BAM size not tracked in LoadedItem yet
                    sizeFormatted: 'Unknown',
                    numReads: item.readCount, // BAM alignment count matches POD5
                    numRefs: item.hasAlignments ? 1 : 0,
                    hasMods: item.hasMods,
                    hasEvents: item.hasEvents,
                });
            }

            // Add FASTA row if present
            if (item.fastaPath) {
                this._files.push({
                    path: item.fastaPath,
                    filename: path.basename(item.fastaPath),
                    type: 'FASTA',
                    size: 0, // FASTA size not tracked in LoadedItem yet
                    sizeFormatted: 'Unknown',
                    // FASTA is reference sequence, not reads
                    hasMods: false,
                    hasEvents: false,
                });
            }
        }

        console.log(
            'FilePanelProvider: Unified state changed, now showing',
            this._files.length,
            'file rows from',
            items.length,
            'items'
        );
        this.updateView();
    }

    protected getTitle(): string {
        return 'Squiggy File Explorer';
    }

    protected async handleMessage(message: FilePanelIncomingMessage): Promise<void> {
        console.log('[FilePanelProvider] Received message:', message.type);
        switch (message.type) {
            case 'openFile':
                if (message.fileType === 'POD5') {
                    vscode.commands.executeCommand('squiggy.openPOD5');
                } else if (message.fileType === 'BAM') {
                    vscode.commands.executeCommand('squiggy.openBAM');
                } else if (message.fileType === 'FASTA') {
                    vscode.commands.executeCommand('squiggy.openFASTA');
                }
                break;
            case 'closeFile':
                if (message.fileType === 'POD5') {
                    vscode.commands.executeCommand('squiggy.closePOD5');
                } else if (message.fileType === 'BAM') {
                    vscode.commands.executeCommand('squiggy.closeBAM');
                } else if (message.fileType === 'FASTA') {
                    vscode.commands.executeCommand('squiggy.closeFASTA');
                }
                break;
            case 'addFiles':
                // New workflow: add POD5/BAM files (will be auto-matched and appear in Sample Manager)
                console.log('[FilePanelProvider] Executing squiggy.loadSamplesFromUI command');
                vscode.commands.executeCommand('squiggy.loadSamplesFromUI');
                break;
            case 'addReference':
                // New workflow: add FASTA reference file
                vscode.commands.executeCommand('squiggy.setSessionFasta');
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
            console.log('FilePanelProvider: No view to update');
            return;
        }

        console.log('FilePanelProvider: Sending updateFiles with', this._files.length, 'files');
        const message: UpdateFilesMessage = {
            type: 'updateFiles',
            files: this._files,
        };
        this.postMessage(message);
    }

    /**
     * Set POD5 file info
     */
    public setPOD5(fileInfo: { path: string; numReads: number; size: number }) {
        console.log('FilePanelProvider.setPOD5 called with:', fileInfo);
        // Remove existing POD5 file
        this._files = this._files.filter((f) => f.type !== 'POD5');

        // Add new POD5 file
        this._files.push({
            path: fileInfo.path,
            filename: path.basename(fileInfo.path),
            type: 'POD5',
            size: fileInfo.size,
            sizeFormatted: formatFileSize(fileInfo.size),
            numReads: fileInfo.numReads,
        });

        console.log('FilePanelProvider._files after setPOD5:', this._files);
        this.updateView();
    }

    /**
     * Set BAM file info
     */
    public setBAM(fileInfo: {
        path: string;
        numReads: number;
        numRefs: number;
        size: number;
        hasMods: boolean;
        hasEvents: boolean;
    }) {
        // Remove existing BAM file
        this._files = this._files.filter((f) => f.type !== 'BAM');

        // Add new BAM file
        this._files.push({
            path: fileInfo.path,
            filename: path.basename(fileInfo.path),
            type: 'BAM',
            size: fileInfo.size,
            sizeFormatted: formatFileSize(fileInfo.size),
            numReads: fileInfo.numReads,
            numRefs: fileInfo.numRefs,
            hasMods: fileInfo.hasMods,
            hasEvents: fileInfo.hasEvents,
        });

        this.updateView();
    }

    /**
     * Clear POD5 file
     */
    public clearPOD5() {
        this._files = this._files.filter((f) => f.type !== 'POD5');
        this.updateView();
    }

    /**
     * Clear BAM file
     */
    public clearBAM() {
        this._files = this._files.filter((f) => f.type !== 'BAM');
        this.updateView();
    }

    /**
     * Set FASTA file info
     */
    public setFASTA(fileInfo: { path: string; size: number }) {
        // Remove existing FASTA file
        this._files = this._files.filter((f) => f.type !== 'FASTA');

        // Add new FASTA file
        this._files.push({
            path: fileInfo.path,
            filename: path.basename(fileInfo.path),
            type: 'FASTA',
            size: fileInfo.size,
            sizeFormatted: formatFileSize(fileInfo.size),
        });

        this.updateView();
    }

    /**
     * Clear FASTA file
     */
    public clearFASTA() {
        this._files = this._files.filter((f) => f.type !== 'FASTA');
        this.updateView();
    }

    /**
     * Check if any POD5 file is loaded
     */
    public hasPOD5(): boolean {
        return this._files.some((f) => f.type === 'POD5');
    }

    /**
     * Check if any BAM file is loaded
     */
    public hasBAM(): boolean {
        return this._files.some((f) => f.type === 'BAM');
    }

    /**
     * Check if any files are loaded
     */
    public hasAnyFiles(): boolean {
        return this._files.length > 0;
    }

    /**
     * Dispose method to clean up subscriptions
     * Called when the provider is no longer needed
     */
    public dispose(): void {
        // Clean up event subscriptions
        for (const disposable of this._disposables) {
            disposable.dispose();
        }
        this._disposables = [];
    }
}
