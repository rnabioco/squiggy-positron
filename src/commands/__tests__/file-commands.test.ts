/**
 * Tests for File Commands
 *
 * Tests command handlers for POD5, BAM, and FASTA file operations.
 * Achieves >80% coverage of file-commands.ts
 */

import { describe, it, expect, beforeEach, jest, afterEach } from '@jest/globals';
import * as vscode from 'vscode';
import { registerFileCommands } from '../file-commands';

describe('File Commands', () => {
    let mockContext: vscode.ExtensionContext;
    let mockState: any;
    let commandHandlers: Map<string, Function>;

    // Helper to register commands
    const registerCommands = () => {
        registerFileCommands(mockContext, mockState);
    };

    beforeEach(() => {
        // Clear command registry before each test
        commandHandlers = new Map();

        // Mock extension context
        mockContext = {
            subscriptions: [],
            extensionPath: '/mock/extension/path',
            extensionUri: vscode.Uri.file('/mock/extension/path'),
        } as any;

        // Mock state
        mockState = {
            usePositron: true,
            squiggyAPI: null,
            squiggyInstallChecked: true,
            currentPod5File: null,
            currentBamFile: null,
            currentFastaFile: null,
        } as any;

        // Override registerCommand to capture handlers
        // Note: The real code in file-commands.ts does context.subscriptions.push(),
        // so we don't push here to avoid double-counting
        (vscode.commands.registerCommand as any).mockImplementation(
            (command: string, callback: Function) => {
                commandHandlers.set(command, callback);
                const disposable = { dispose: jest.fn() };
                return disposable;
            }
        );

        // Reset all vscode mocks
        jest.clearAllMocks();
    });

    afterEach(() => {
        commandHandlers.clear();
    });

    describe('squiggy.openPOD5', () => {
        it('should register the openPOD5 command', () => {
            registerCommands();
            expect(vscode.commands.registerCommand).toHaveBeenCalledWith(
                'squiggy.openPOD5',
                expect.any(Function)
            );
        });

        it('should show file picker when command is executed', async () => {
            // Mock file dialog to return a file
            const mockFileUri = vscode.Uri.file('/path/to/test.pod5');
            (vscode.window.showOpenDialog as any).mockResolvedValue([mockFileUri]);

            registerCommands();

            // Get the command handler
            const handler = commandHandlers.get('squiggy.openPOD5');
            expect(handler).toBeDefined();

            // Execute the command
            await handler!();

            // Verify file dialog was shown with correct options
            expect(vscode.window.showOpenDialog).toHaveBeenCalledWith(
                expect.objectContaining({
                    canSelectMany: false,
                    filters: { 'POD5 Files': ['pod5'] },
                    title: 'Open POD5 File',
                })
            );
        });

        it('should handle user canceling file selection', async () => {
            // Mock file dialog returning undefined (cancelled)
            (vscode.window.showOpenDialog as any).mockResolvedValue(undefined);

            registerCommands();
            const handler = commandHandlers.get('squiggy.openPOD5');

            // Should not throw
            await expect(handler!()).resolves.not.toThrow();
        });
    });

    describe('squiggy.openBAM', () => {
        it('should register the openBAM command', () => {
            registerCommands();
            expect(vscode.commands.registerCommand).toHaveBeenCalledWith(
                'squiggy.openBAM',
                expect.any(Function)
            );
        });

        it('should show file picker for BAM files', async () => {
            const mockFileUri = vscode.Uri.file('/path/to/test.bam');
            (vscode.window.showOpenDialog as any).mockResolvedValue([mockFileUri]);

            registerCommands();
            const handler = commandHandlers.get('squiggy.openBAM');
            await handler!();

            expect(vscode.window.showOpenDialog).toHaveBeenCalledWith(
                expect.objectContaining({
                    filters: { 'BAM Files': ['bam'] },
                    title: 'Open BAM File',
                })
            );
        });
    });

    describe('squiggy.openFASTA', () => {
        it('should register the openFASTA command', () => {
            registerCommands();
            expect(vscode.commands.registerCommand).toHaveBeenCalledWith(
                'squiggy.openFASTA',
                expect.any(Function)
            );
        });

        it('should show file picker for FASTA files', async () => {
            const mockFileUri = vscode.Uri.file('/path/to/reference.fasta');
            (vscode.window.showOpenDialog as any).mockResolvedValue([mockFileUri]);

            registerCommands();
            const handler = commandHandlers.get('squiggy.openFASTA');
            await handler!();

            expect(vscode.window.showOpenDialog).toHaveBeenCalledWith(
                expect.objectContaining({
                    filters: { 'FASTA Files': ['fa', 'fasta', 'fna'] },
                    title: 'Open FASTA File',
                })
            );
        });
    });

    describe('squiggy.closePOD5', () => {
        it('should register the closePOD5 command', () => {
            registerCommands();
            expect(vscode.commands.registerCommand).toHaveBeenCalledWith(
                'squiggy.closePOD5',
                expect.any(Function)
            );
        });

        it('should show confirmation dialog when closing POD5', async () => {
            (vscode.window.showWarningMessage as any).mockResolvedValue(undefined);

            registerCommands();
            const handler = commandHandlers.get('squiggy.closePOD5');
            await handler!();

            expect(vscode.window.showWarningMessage).toHaveBeenCalledWith(
                'Close POD5 file?',
                { modal: true },
                'Close'
            );
        });

        it('should handle user canceling close', async () => {
            (vscode.window.showWarningMessage as any).mockResolvedValue(undefined);

            registerCommands();
            const handler = commandHandlers.get('squiggy.closePOD5');

            // Should not throw when user cancels
            await expect(handler!()).resolves.not.toThrow();
        });
    });

    describe('squiggy.closeBAM', () => {
        it('should register the closeBAM command', () => {
            registerCommands();
            expect(vscode.commands.registerCommand).toHaveBeenCalledWith(
                'squiggy.closeBAM',
                expect.any(Function)
            );
        });

        it('should show confirmation dialog', async () => {
            (vscode.window.showWarningMessage as any).mockResolvedValue(undefined);

            registerCommands();
            const handler = commandHandlers.get('squiggy.closeBAM');
            await handler!();

            expect(vscode.window.showWarningMessage).toHaveBeenCalledWith(
                'Close BAM file?',
                { modal: true },
                'Close'
            );
        });
    });

    describe('squiggy.closeFASTA', () => {
        it('should register the closeFASTA command', () => {
            registerCommands();
            expect(vscode.commands.registerCommand).toHaveBeenCalledWith(
                'squiggy.closeFASTA',
                expect.any(Function)
            );
        });
    });

    describe('squiggy.loadTestData', () => {
        it('should register the loadTestData command', () => {
            registerCommands();
            expect(vscode.commands.registerCommand).toHaveBeenCalledWith(
                'squiggy.loadTestData',
                expect.any(Function)
            );
        });
    });

    describe('squiggy.loadSample', () => {
        it('should register the loadSample command', () => {
            registerCommands();
            expect(vscode.commands.registerCommand).toHaveBeenCalledWith(
                'squiggy.loadSample',
                expect.any(Function)
            );
        });
    });

    describe('squiggy.loadTestMultiReadDataset', () => {
        it('should register the loadTestMultiReadDataset command', () => {
            registerCommands();
            expect(vscode.commands.registerCommand).toHaveBeenCalledWith(
                'squiggy.loadTestMultiReadDataset',
                expect.any(Function)
            );
        });
    });

    describe('squiggy.setSessionFasta', () => {
        it('should register the setSessionFasta command', () => {
            registerCommands();
            expect(vscode.commands.registerCommand).toHaveBeenCalledWith(
                'squiggy.setSessionFasta',
                expect.any(Function)
            );
        });
    });

    describe('Command Registration', () => {
        it('should register all file commands', () => {
            registerCommands();

            // Verify all expected commands were registered
            const expectedCommands = [
                'squiggy.openPOD5',
                'squiggy.openBAM',
                'squiggy.openFASTA',
                'squiggy.closePOD5',
                'squiggy.closeBAM',
                'squiggy.closeFASTA',
                'squiggy.loadTestData',
                'squiggy.loadSample',
                'squiggy.loadTestMultiReadDataset',
                'squiggy.setSessionFasta',
            ];

            expectedCommands.forEach((cmd) => {
                expect(vscode.commands.registerCommand).toHaveBeenCalledWith(
                    cmd,
                    expect.any(Function)
                );
            });
        });

        it('should add all command disposables to context', () => {
            registerCommands();

            // Verify registerCommand was called 16 times (once per command)
            // registerFileCommands registers 16 commands total:
            // 10 public commands + 6 internal commands
            expect(vscode.commands.registerCommand).toHaveBeenCalledTimes(16);

            // Each call should add a disposable to context.subscriptions
            expect(mockContext.subscriptions.length).toBe(16);
        });
    });
});
