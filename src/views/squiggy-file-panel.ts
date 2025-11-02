/**
 * File Panel Webview View - React-based
 *
 * Provides file management with sortable table layout
 */

import * as vscode from 'vscode';
import * as path from 'path';
import { BaseWebviewProvider } from './base-webview-provider';
import { FilePanelIncomingMessage, UpdateFilesMessage, FileItem } from '../types/messages';
import { formatFileSize } from '../utils/format-utils';

export class FilePanelProvider extends BaseWebviewProvider {
    public static readonly viewType = 'squiggyFilePanel';

    private _files: FileItem[] = [];

    protected getTitle(): string {
        return 'Squiggy File Explorer';
    }

    protected async handleMessage(message: FilePanelIncomingMessage): Promise<void> {
        switch (message.type) {
            case 'openFile':
                if (message.fileType === 'POD5') {
                    vscode.commands.executeCommand('squiggy.openPOD5');
                } else if (message.fileType === 'BAM') {
                    vscode.commands.executeCommand('squiggy.openBAM');
                }
                break;
            case 'closeFile':
                if (message.fileType === 'POD5') {
                    vscode.commands.executeCommand('squiggy.closePOD5');
                } else if (message.fileType === 'BAM') {
                    vscode.commands.executeCommand('squiggy.closeBAM');
                }
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
}
