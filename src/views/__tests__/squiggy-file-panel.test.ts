/**
 * Tests for File Panel Provider
 *
 * Tests the FilePanelProvider webview implementation.
 * Target: >80% coverage of squiggy-file-panel.ts
 */

import { describe, it, expect, beforeEach, jest, afterEach } from '@jest/globals';
import * as vscode from 'vscode';
import { FilePanelProvider } from '../squiggy-file-panel';
import { ExtensionState } from '../../state/extension-state';
import { LoadedItem } from '../../types/loaded-item';

// Mock formatFileSize utility
jest.mock('../../utils/format-utils', () => ({
    formatFileSize: jest.fn((size: number) => `${(size / 1024).toFixed(1)} KB`),
}));

describe('FilePanelProvider', () => {
    let provider: FilePanelProvider;
    let mockWebviewView: any;
    let mockState: any;

    beforeEach(() => {
        const extensionUri = vscode.Uri.file('/mock/extension');

        mockState = {
            onLoadedItemsChanged: jest.fn().mockReturnValue({ dispose: jest.fn() }),
        } as any;

        // Mock webview view
        mockWebviewView = {
            webview: {
                options: {},
                html: '',
                postMessage: jest.fn(),
                asWebviewUri: (uri: vscode.Uri) => uri,
                onDidReceiveMessage: jest.fn(),
            },
            visible: true,
            onDidChangeVisibility: jest.fn(),
            onDidDispose: jest.fn(),
        };

        jest.clearAllMocks();

        // Create provider after clearing mocks
        provider = new FilePanelProvider(extensionUri, mockState);
    });

    afterEach(() => {
        jest.clearAllMocks();
    });

    describe('Provider Properties', () => {
        it('should have correct viewType', () => {
            expect(FilePanelProvider.viewType).toBe('squiggyFilePanel');
        });

        it('should return correct title', () => {
            const title = (provider as any).getTitle();
            expect(title).toBe('Squiggy File Explorer');
        });
    });

    describe('Constructor', () => {
        it('should subscribe to state changes when state is provided', () => {
            expect(mockState.onLoadedItemsChanged).toHaveBeenCalled();
        });

        it('should not subscribe when state is not provided', () => {
            const mockState2 = {
                onLoadedItemsChanged: jest.fn(),
            };
            const provider2 = new FilePanelProvider(vscode.Uri.file('/mock'));

            expect(mockState2.onLoadedItemsChanged).not.toHaveBeenCalled();
        });
    });

    describe('resolveWebviewView', () => {
        it('should set up webview when resolved', () => {
            provider.resolveWebviewView(mockWebviewView, {} as any, {} as any);

            expect(mockWebviewView.webview.options).toBeDefined();
            expect(mockWebviewView.webview.html).toBeTruthy();
        });
    });

    describe('Message Handling', () => {
        beforeEach(() => {
            provider.resolveWebviewView(mockWebviewView, {} as any, {} as any);
            (vscode.commands.executeCommand as any).mockResolvedValue(undefined);
        });

        it('should handle openFile message for POD5', async () => {
            const messageHandler = mockWebviewView.webview.onDidReceiveMessage.mock.calls[0][0];

            await messageHandler({ type: 'openFile', fileType: 'POD5' });

            expect(vscode.commands.executeCommand).toHaveBeenCalledWith('squiggy.openPOD5');
        });

        it('should handle openFile message for BAM', async () => {
            const messageHandler = mockWebviewView.webview.onDidReceiveMessage.mock.calls[0][0];

            await messageHandler({ type: 'openFile', fileType: 'BAM' });

            expect(vscode.commands.executeCommand).toHaveBeenCalledWith('squiggy.openBAM');
        });

        it('should handle openFile message for FASTA', async () => {
            const messageHandler = mockWebviewView.webview.onDidReceiveMessage.mock.calls[0][0];

            await messageHandler({ type: 'openFile', fileType: 'FASTA' });

            expect(vscode.commands.executeCommand).toHaveBeenCalledWith('squiggy.openFASTA');
        });

        it('should handle closeFile message for POD5', async () => {
            const messageHandler = mockWebviewView.webview.onDidReceiveMessage.mock.calls[0][0];

            await messageHandler({ type: 'closeFile', fileType: 'POD5' });

            expect(vscode.commands.executeCommand).toHaveBeenCalledWith('squiggy.closePOD5');
        });

        it('should handle closeFile message for BAM', async () => {
            const messageHandler = mockWebviewView.webview.onDidReceiveMessage.mock.calls[0][0];

            await messageHandler({ type: 'closeFile', fileType: 'BAM' });

            expect(vscode.commands.executeCommand).toHaveBeenCalledWith('squiggy.closeBAM');
        });

        it('should handle closeFile message for FASTA', async () => {
            const messageHandler = mockWebviewView.webview.onDidReceiveMessage.mock.calls[0][0];

            await messageHandler({ type: 'closeFile', fileType: 'FASTA' });

            expect(vscode.commands.executeCommand).toHaveBeenCalledWith('squiggy.closeFASTA');
        });

        it('should handle addFiles message', async () => {
            const messageHandler = mockWebviewView.webview.onDidReceiveMessage.mock.calls[0][0];

            await messageHandler({ type: 'addFiles' });

            expect(vscode.commands.executeCommand).toHaveBeenCalledWith(
                'squiggy.loadSamplesFromUI'
            );
        });

        it('should handle addReference message', async () => {
            const messageHandler = mockWebviewView.webview.onDidReceiveMessage.mock.calls[0][0];

            await messageHandler({ type: 'addReference' });

            expect(vscode.commands.executeCommand).toHaveBeenCalledWith('squiggy.setSessionFasta');
        });

        it('should handle ready message and update view', async () => {
            const messageHandler = mockWebviewView.webview.onDidReceiveMessage.mock.calls[0][0];

            await messageHandler({ type: 'ready' });

            expect(mockWebviewView.webview.postMessage).toHaveBeenCalledWith({
                type: 'updateFiles',
                files: [],
            });
        });
    });

    describe('setPOD5', () => {
        beforeEach(() => {
            provider.resolveWebviewView(mockWebviewView, {} as any, {} as any);
            jest.clearAllMocks();
        });

        it('should add POD5 file and update view', () => {
            provider.setPOD5({
                path: '/path/to/file.pod5',
                numReads: 1000,
                size: 2048000,
            });

            expect(mockWebviewView.webview.postMessage).toHaveBeenCalledWith({
                type: 'updateFiles',
                files: [
                    {
                        path: '/path/to/file.pod5',
                        filename: 'file.pod5',
                        type: 'POD5',
                        size: 2048000,
                        sizeFormatted: '2000.0 KB',
                        numReads: 1000,
                    },
                ],
            });
        });

        it('should replace existing POD5 file', () => {
            provider.setPOD5({
                path: '/path/to/file1.pod5',
                numReads: 1000,
                size: 2048000,
            });
            jest.clearAllMocks();

            provider.setPOD5({
                path: '/path/to/file2.pod5',
                numReads: 2000,
                size: 4096000,
            });

            expect(mockWebviewView.webview.postMessage).toHaveBeenCalledWith({
                type: 'updateFiles',
                files: [
                    {
                        path: '/path/to/file2.pod5',
                        filename: 'file2.pod5',
                        type: 'POD5',
                        size: 4096000,
                        sizeFormatted: '4000.0 KB',
                        numReads: 2000,
                    },
                ],
            });
        });
    });

    describe('setBAM', () => {
        beforeEach(() => {
            provider.resolveWebviewView(mockWebviewView, {} as any, {} as any);
            jest.clearAllMocks();
        });

        it('should add BAM file and update view', () => {
            provider.setBAM({
                path: '/path/to/file.bam',
                numReads: 1000,
                numRefs: 3,
                size: 5120000,
                hasMods: true,
                hasEvents: false,
            });

            expect(mockWebviewView.webview.postMessage).toHaveBeenCalledWith({
                type: 'updateFiles',
                files: [
                    {
                        path: '/path/to/file.bam',
                        filename: 'file.bam',
                        type: 'BAM',
                        size: 5120000,
                        sizeFormatted: '5000.0 KB',
                        numReads: 1000,
                        numRefs: 3,
                        hasMods: true,
                        hasEvents: false,
                    },
                ],
            });
        });

        it('should replace existing BAM file', () => {
            provider.setBAM({
                path: '/path/to/file1.bam',
                numReads: 1000,
                numRefs: 3,
                size: 5120000,
                hasMods: false,
                hasEvents: true,
            });
            jest.clearAllMocks();

            provider.setBAM({
                path: '/path/to/file2.bam',
                numReads: 2000,
                numRefs: 5,
                size: 10240000,
                hasMods: true,
                hasEvents: false,
            });

            expect(mockWebviewView.webview.postMessage).toHaveBeenCalledWith({
                type: 'updateFiles',
                files: [
                    {
                        path: '/path/to/file2.bam',
                        filename: 'file2.bam',
                        type: 'BAM',
                        size: 10240000,
                        sizeFormatted: '10000.0 KB',
                        numReads: 2000,
                        numRefs: 5,
                        hasMods: true,
                        hasEvents: false,
                    },
                ],
            });
        });
    });

    describe('setFASTA', () => {
        beforeEach(() => {
            provider.resolveWebviewView(mockWebviewView, {} as any, {} as any);
            jest.clearAllMocks();
        });

        it('should add FASTA file and update view', () => {
            provider.setFASTA({
                path: '/path/to/reference.fasta',
                size: 1024000,
            });

            expect(mockWebviewView.webview.postMessage).toHaveBeenCalledWith({
                type: 'updateFiles',
                files: [
                    {
                        path: '/path/to/reference.fasta',
                        filename: 'reference.fasta',
                        type: 'FASTA',
                        size: 1024000,
                        sizeFormatted: '1000.0 KB',
                    },
                ],
            });
        });

        it('should replace existing FASTA file', () => {
            provider.setFASTA({
                path: '/path/to/ref1.fasta',
                size: 1024000,
            });
            jest.clearAllMocks();

            provider.setFASTA({
                path: '/path/to/ref2.fasta',
                size: 2048000,
            });

            expect(mockWebviewView.webview.postMessage).toHaveBeenCalledWith({
                type: 'updateFiles',
                files: [
                    {
                        path: '/path/to/ref2.fasta',
                        filename: 'ref2.fasta',
                        type: 'FASTA',
                        size: 2048000,
                        sizeFormatted: '2000.0 KB',
                    },
                ],
            });
        });
    });

    describe('Clear Methods', () => {
        beforeEach(() => {
            provider.resolveWebviewView(mockWebviewView, {} as any, {} as any);
            provider.setPOD5({ path: '/path/to/file.pod5', numReads: 1000, size: 2048000 });
            provider.setBAM({
                path: '/path/to/file.bam',
                numReads: 1000,
                numRefs: 3,
                size: 5120000,
                hasMods: true,
                hasEvents: false,
            });
            provider.setFASTA({ path: '/path/to/ref.fasta', size: 1024000 });
            jest.clearAllMocks();
        });

        it('should clear POD5 file', () => {
            provider.clearPOD5();

            expect(mockWebviewView.webview.postMessage).toHaveBeenCalledWith({
                type: 'updateFiles',
                files: expect.arrayContaining([
                    expect.objectContaining({ type: 'BAM' }),
                    expect.objectContaining({ type: 'FASTA' }),
                ]),
            });
            expect(
                mockWebviewView.webview.postMessage.mock.calls[0][0].files.some(
                    (f: any) => f.type === 'POD5'
                )
            ).toBe(false);
        });

        it('should clear BAM file', () => {
            provider.clearBAM();

            expect(mockWebviewView.webview.postMessage).toHaveBeenCalledWith({
                type: 'updateFiles',
                files: expect.arrayContaining([
                    expect.objectContaining({ type: 'POD5' }),
                    expect.objectContaining({ type: 'FASTA' }),
                ]),
            });
            expect(
                mockWebviewView.webview.postMessage.mock.calls[0][0].files.some(
                    (f: any) => f.type === 'BAM'
                )
            ).toBe(false);
        });

        it('should clear FASTA file', () => {
            provider.clearFASTA();

            expect(mockWebviewView.webview.postMessage).toHaveBeenCalledWith({
                type: 'updateFiles',
                files: expect.arrayContaining([
                    expect.objectContaining({ type: 'POD5' }),
                    expect.objectContaining({ type: 'BAM' }),
                ]),
            });
            expect(
                mockWebviewView.webview.postMessage.mock.calls[0][0].files.some(
                    (f: any) => f.type === 'FASTA'
                )
            ).toBe(false);
        });
    });

    describe('Check Methods', () => {
        beforeEach(() => {
            provider.resolveWebviewView(mockWebviewView, {} as any, {} as any);
        });

        it('should return true when POD5 is loaded', () => {
            provider.setPOD5({ path: '/path/to/file.pod5', numReads: 1000, size: 2048000 });

            expect(provider.hasPOD5()).toBe(true);
        });

        it('should return false when POD5 is not loaded', () => {
            expect(provider.hasPOD5()).toBe(false);
        });

        it('should return true when BAM is loaded', () => {
            provider.setBAM({
                path: '/path/to/file.bam',
                numReads: 1000,
                numRefs: 3,
                size: 5120000,
                hasMods: true,
                hasEvents: false,
            });

            expect(provider.hasBAM()).toBe(true);
        });

        it('should return false when BAM is not loaded', () => {
            expect(provider.hasBAM()).toBe(false);
        });

        it('should return true when any files are loaded', () => {
            provider.setPOD5({ path: '/path/to/file.pod5', numReads: 1000, size: 2048000 });

            expect(provider.hasAnyFiles()).toBe(true);
        });

        it('should return false when no files are loaded', () => {
            expect(provider.hasAnyFiles()).toBe(false);
        });
    });

    describe('Unified State Integration', () => {
        let stateCallback: (items: LoadedItem[]) => void;

        beforeEach(() => {
            // Save the callback registered during provider construction
            stateCallback = mockState.onLoadedItemsChanged.mock.calls[0][0];
        });

        it('should convert LoadedItem to FileItems', () => {
            provider.resolveWebviewView(mockWebviewView, {} as any, {} as any);
            jest.clearAllMocks();

            const items: LoadedItem[] = [
                {
                    id: 'pod5:/path/to/sample1.pod5',
                    type: 'pod5',
                    pod5Path: '/path/to/sample1.pod5',
                    bamPath: '/path/to/sample1.bam',
                    fastaPath: undefined,
                    fileSize: 2048000,
                    fileSizeFormatted: '2.0 MB',
                    readCount: 1000,
                    hasMods: true,
                    hasEvents: false,
                    hasAlignments: true,
                    hasReference: false,
                },
            ];

            stateCallback(items);

            // Should create 2 rows: POD5 + BAM
            expect(mockWebviewView.webview.postMessage).toHaveBeenCalledWith({
                type: 'updateFiles',
                files: [
                    {
                        path: '/path/to/sample1.pod5',
                        filename: 'sample1.pod5',
                        type: 'POD5',
                        size: 2048000,
                        sizeFormatted: '2.0 MB',
                        numReads: 1000,
                        hasMods: true,
                        hasEvents: false,
                    },
                    {
                        path: '/path/to/sample1.bam',
                        filename: 'sample1.bam',
                        type: 'BAM',
                        size: 0,
                        sizeFormatted: 'Unknown',
                        numReads: 1000,
                        numRefs: 1,
                        hasMods: true,
                        hasEvents: false,
                    },
                ],
            });
        });

        it('should handle LoadedItem with FASTA', () => {
            provider.resolveWebviewView(mockWebviewView, {} as any, {} as any);
            jest.clearAllMocks();

            const items: LoadedItem[] = [
                {
                    id: 'pod5:/path/to/sample1.pod5',
                    type: 'pod5',
                    pod5Path: '/path/to/sample1.pod5',
                    bamPath: '/path/to/sample1.bam',
                    fastaPath: '/path/to/reference.fasta',
                    fileSize: 2048000,
                    fileSizeFormatted: '2.0 MB',
                    readCount: 1000,
                    hasMods: true,
                    hasEvents: true,
                    hasAlignments: true,
                    hasReference: true,
                },
            ];

            stateCallback(items);

            // Should create 3 rows: POD5 + BAM + FASTA
            expect(mockWebviewView.webview.postMessage).toHaveBeenCalledWith({
                type: 'updateFiles',
                files: [
                    expect.objectContaining({ type: 'POD5' }),
                    expect.objectContaining({ type: 'BAM' }),
                    {
                        path: '/path/to/reference.fasta',
                        filename: 'reference.fasta',
                        type: 'FASTA',
                        size: 0,
                        sizeFormatted: 'Unknown',
                        hasMods: false,
                        hasEvents: false,
                    },
                ],
            });
        });

        it('should handle LoadedItem without BAM', () => {
            provider.resolveWebviewView(mockWebviewView, {} as any, {} as any);
            jest.clearAllMocks();

            const items: LoadedItem[] = [
                {
                    id: 'pod5:/path/to/sample1.pod5',
                    type: 'pod5',
                    pod5Path: '/path/to/sample1.pod5',
                    bamPath: undefined,
                    fastaPath: undefined,
                    fileSize: 2048000,
                    fileSizeFormatted: '2.0 MB',
                    readCount: 1000,
                    hasMods: false,
                    hasEvents: false,
                    hasAlignments: false,
                    hasReference: false,
                },
            ];

            stateCallback(items);

            // Should create 1 row: POD5 only
            expect(mockWebviewView.webview.postMessage).toHaveBeenCalledWith({
                type: 'updateFiles',
                files: [
                    {
                        path: '/path/to/sample1.pod5',
                        filename: 'sample1.pod5',
                        type: 'POD5',
                        size: 2048000,
                        sizeFormatted: '2.0 MB',
                        numReads: 1000,
                        hasMods: false,
                        hasEvents: false,
                    },
                ],
            });
        });

        it('should handle multiple LoadedItems', () => {
            provider.resolveWebviewView(mockWebviewView, {} as any, {} as any);
            jest.clearAllMocks();

            const items: LoadedItem[] = [
                {
                    id: 'pod5:/path/to/sample1.pod5',
                    type: 'pod5',
                    pod5Path: '/path/to/sample1.pod5',
                    bamPath: '/path/to/sample1.bam',
                    fastaPath: undefined,
                    fileSize: 2048000,
                    fileSizeFormatted: '2.0 MB',
                    readCount: 1000,
                    hasMods: true,
                    hasEvents: false,
                    hasAlignments: true,
                    hasReference: false,
                },
                {
                    id: 'pod5:/path/to/sample2.pod5',
                    type: 'pod5',
                    pod5Path: '/path/to/sample2.pod5',
                    bamPath: undefined,
                    fastaPath: undefined,
                    fileSize: 1024000,
                    fileSizeFormatted: '1.0 MB',
                    readCount: 500,
                    hasMods: false,
                    hasEvents: false,
                    hasAlignments: false,
                    hasReference: false,
                },
            ];

            stateCallback(items);

            // Should create 3 rows: sample1 (POD5+BAM) + sample2 (POD5)
            const files = mockWebviewView.webview.postMessage.mock.calls[0][0].files;
            expect(files).toHaveLength(3);
            expect(files[0].type).toBe('POD5');
            expect(files[0].filename).toBe('sample1.pod5');
            expect(files[1].type).toBe('BAM');
            expect(files[1].filename).toBe('sample1.bam');
            expect(files[2].type).toBe('POD5');
            expect(files[2].filename).toBe('sample2.pod5');
        });
    });

    describe('updateView', () => {
        beforeEach(() => {
            provider.resolveWebviewView(mockWebviewView, {} as any, {} as any);
        });

        it('should not post message if view not available', () => {
            const newProvider = new FilePanelProvider(vscode.Uri.file('/mock'));

            (newProvider as any).updateView();

            expect(mockWebviewView.webview.postMessage).not.toHaveBeenCalled();
        });
    });

    describe('dispose', () => {
        it('should dispose all subscriptions', () => {
            const mockDisposable = { dispose: jest.fn() };
            mockState.onLoadedItemsChanged.mockReturnValue(mockDisposable);

            const provider2 = new FilePanelProvider(vscode.Uri.file('/mock'), mockState);

            provider2.dispose();

            expect(mockDisposable.dispose).toHaveBeenCalled();
        });

        it('should handle dispose when no subscriptions exist', () => {
            const provider2 = new FilePanelProvider(vscode.Uri.file('/mock'));

            expect(() => provider2.dispose()).not.toThrow();
        });
    });
});
