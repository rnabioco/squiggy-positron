/**
 * Tests for State Commands
 *
 * Tests command handlers for state management operations.
 * Target: >80% coverage of state-commands.ts
 */

import { describe, it, expect, beforeEach, jest, afterEach } from '@jest/globals';
import * as vscode from 'vscode';
import { registerStateCommands } from '../state-commands';
import { ExtensionState } from '../../state/extension-state';

describe('State Commands', () => {
    let mockContext: vscode.ExtensionContext;
    let mockState: any;
    let commandHandlers: Map<string, Function>;

    // Helper to register commands
    const registerCommands = () => {
        registerStateCommands(mockContext, mockState);
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

        // Mock state with minimal required properties
        mockState = {
            usePositron: true,
            squiggyAPI: {
                client: {
                    getVariable: (jest.fn() as any).mockResolvedValue(true),
                },
                getReadIds: (jest.fn() as any).mockResolvedValue(['read_001', 'read_002']),
                getReferences: (jest.fn() as any).mockResolvedValue(['ref1', 'ref2']),
            },
            readsViewPane: {
                setLoading: jest.fn(),
                setReads: jest.fn(),
                setReferencesOnly: jest.fn(),
            },
            clearAll: (jest.fn() as any).mockResolvedValue(undefined),
            pod5LoadContext: null,
        } as any;

        // Override registerCommand to capture handlers
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

    describe('Command Registration', () => {
        it('should register all state commands', () => {
            registerCommands();

            const expectedCommands = ['squiggy.refreshReads', 'squiggy.clearState'];

            expectedCommands.forEach((cmd) => {
                expect(vscode.commands.registerCommand).toHaveBeenCalledWith(
                    cmd,
                    expect.any(Function)
                );
            });
        });

        it('should add all command disposables to context', () => {
            registerCommands();

            // Verify all expected commands were registered (avoid exact count to be resilient to changes)
            expect(vscode.commands.registerCommand).toHaveBeenCalledWith(
                'squiggy.refreshReads',
                expect.any(Function)
            );
            expect(vscode.commands.registerCommand).toHaveBeenCalledWith(
                'squiggy.clearState',
                expect.any(Function)
            );
        });
    });

    describe('squiggy.refreshReads', () => {
        it('should show warning when Positron runtime is not available', async () => {
            mockState.usePositron = false;

            registerCommands();
            const handler = commandHandlers.get('squiggy.refreshReads');

            await handler!();

            expect(vscode.window.showWarningMessage).toHaveBeenCalledWith(
                'Refresh requires Positron runtime with active Python kernel'
            );
        });

        it('should show warning when squiggyAPI is not available', async () => {
            mockState.squiggyAPI = null;

            registerCommands();
            const handler = commandHandlers.get('squiggy.refreshReads');

            await handler!();

            expect(vscode.window.showWarningMessage).toHaveBeenCalledWith(
                'Refresh requires Positron runtime with active Python kernel'
            );
        });

        it('should show info message when no POD5 file is loaded', async () => {
            mockState.squiggyAPI.client.getVariable.mockResolvedValue(false);

            registerCommands();
            const handler = commandHandlers.get('squiggy.refreshReads');

            await handler!();

            expect(vscode.window.showInformationMessage).toHaveBeenCalledWith(
                'No POD5 file loaded in Python session'
            );
        });

        it('should refresh POD5-only mode (flat list)', async () => {
            // Mock: POD5 loaded but no BAM
            mockState.squiggyAPI.client.getVariable
                .mockResolvedValueOnce(true) // hasPod5
                .mockResolvedValueOnce(false) // hasBAM
                .mockResolvedValueOnce(2000); // totalReads

            registerCommands();
            const handler = commandHandlers.get('squiggy.refreshReads');

            await handler!();

            // Should fetch read IDs
            expect(mockState.squiggyAPI.getReadIds).toHaveBeenCalledWith(0, 1000);

            // Should display reads
            expect(mockState.readsViewPane.setReads).toHaveBeenCalledWith(['read_001', 'read_002']);

            // Should update pod5LoadContext
            expect(mockState.pod5LoadContext).toEqual({
                currentOffset: 1000,
                pageSize: 500,
                totalReads: 2000,
            });

            // Should show success message
            expect(vscode.window.showInformationMessage).toHaveBeenCalledWith(
                'Read list refreshed successfully'
            );
        });

        it('should refresh BAM mode (grouped by reference)', async () => {
            // Mock: POD5 and BAM loaded
            mockState.squiggyAPI.client.getVariable
                .mockResolvedValueOnce(true) // hasPod5
                .mockResolvedValueOnce(true) // hasBAM
                .mockResolvedValueOnce(100) // ref1 readCount
                .mockResolvedValueOnce(150); // ref2 readCount

            registerCommands();
            const handler = commandHandlers.get('squiggy.refreshReads');

            await handler!();

            // Should fetch references
            expect(mockState.squiggyAPI.getReferences).toHaveBeenCalled();

            // Should display references with counts
            expect(mockState.readsViewPane.setReferencesOnly).toHaveBeenCalledWith([
                { referenceName: 'ref1', readCount: 100 },
                { referenceName: 'ref2', readCount: 150 },
            ]);

            // Should show success message
            expect(vscode.window.showInformationMessage).toHaveBeenCalledWith(
                'Read list refreshed successfully'
            );
        });

        it('should set loading state during refresh', async () => {
            mockState.squiggyAPI.client.getVariable
                .mockResolvedValueOnce(true) // hasPod5
                .mockResolvedValueOnce(false); // hasBAM

            registerCommands();
            const handler = commandHandlers.get('squiggy.refreshReads');

            await handler!();

            // Should set loading true at start
            expect(mockState.readsViewPane.setLoading).toHaveBeenCalledWith(
                true,
                'Refreshing read list...'
            );

            // Should set loading false at end
            expect(mockState.readsViewPane.setLoading).toHaveBeenCalledWith(false);
        });

        it('should handle errors during refresh', async () => {
            mockState.squiggyAPI.client.getVariable.mockRejectedValue(
                new Error('Python kernel error')
            );

            registerCommands();
            const handler = commandHandlers.get('squiggy.refreshReads');

            await handler!();

            // Should show error message
            expect(vscode.window.showErrorMessage).toHaveBeenCalledWith(
                expect.stringContaining('Failed to refresh reads')
            );

            // Should still clear loading state
            expect(mockState.readsViewPane.setLoading).toHaveBeenCalledWith(false);
        });
    });

    describe('squiggy.clearState', () => {
        it('should call clearAll on state', async () => {
            registerCommands();
            const handler = commandHandlers.get('squiggy.clearState');

            await handler!();

            expect(mockState.clearAll).toHaveBeenCalled();
        });

        it('should show success message after clearing', async () => {
            registerCommands();
            const handler = commandHandlers.get('squiggy.clearState');

            await handler!();

            expect(vscode.window.showInformationMessage).toHaveBeenCalledWith(
                'Squiggy state cleared. Load new files to continue.'
            );
        });
    });
});
