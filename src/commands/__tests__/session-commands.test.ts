/**
 * Tests for Session Commands
 *
 * Tests command handlers for session management operations.
 * Target: >80% coverage of session-commands.ts
 */

import { describe, it, expect, beforeEach, jest, afterEach } from '@jest/globals';
import * as vscode from 'vscode';
import {
    registerSessionCommands,
    saveSessionCommand,
    restoreSessionCommand,
    exportSessionCommand,
    importSessionCommand,
    clearSessionCommand,
    loadDemoSessionCommand,
} from '../session-commands';

// Mock SessionStateManager
jest.mock('../../state/session-state-manager', () => ({
    SessionStateManager: {
        saveSession: (jest.fn() as any).mockResolvedValue(undefined),
        loadSession: (jest.fn() as any).mockResolvedValue(null),
        hasUnsavedChanges: (jest.fn() as any).mockResolvedValue(false),
        exportSession: (jest.fn() as any).mockResolvedValue(undefined),
        importSession: (jest.fn() as any).mockResolvedValue(null),
        clearSession: (jest.fn() as any).mockResolvedValue(undefined),
        getDemoSession: jest.fn().mockReturnValue(null),
    },
}));

import { SessionStateManager } from '../../state/session-state-manager';

describe('Session Commands', () => {
    let mockContext: vscode.ExtensionContext;
    let mockState: any;
    let commandHandlers: Map<string, Function>;

    beforeEach(() => {
        commandHandlers = new Map();

        mockContext = {
            subscriptions: [],
            extensionPath: '/mock/extension/path',
            extensionUri: vscode.Uri.file('/mock/extension/path'),
        } as any;

        mockState = {
            toSessionState: jest.fn().mockReturnValue({
                samples: { sample1: {} },
                sessionName: 'Test Session',
            }),
            fromSessionState: (jest.fn() as any).mockResolvedValue(undefined),
            loadDemoSession: (jest.fn() as any).mockResolvedValue(undefined),
        } as any;

        (vscode.commands.registerCommand as any).mockImplementation(
            (command: string, callback: Function) => {
                commandHandlers.set(command, callback);
                const disposable = { dispose: jest.fn() };
                return disposable;
            }
        );

        // Mock workspace
        (vscode.workspace as any).name = 'TestWorkspace';

        jest.clearAllMocks();
    });

    afterEach(() => {
        commandHandlers.clear();
    });

    describe('Command Registration', () => {
        it('should register all session commands', () => {
            registerSessionCommands(mockContext, mockState);

            const expectedCommands = [
                'squiggy.saveSession',
                'squiggy.restoreSession',
                'squiggy.exportSession',
                'squiggy.importSession',
                'squiggy.clearSession',
                'squiggy.loadDemoSession',
            ];

            expectedCommands.forEach((cmd) => {
                expect(vscode.commands.registerCommand).toHaveBeenCalledWith(
                    cmd,
                    expect.any(Function)
                );
            });
        });
    });

    describe('saveSessionCommand', () => {
        it('should show warning when no data is loaded', async () => {
            mockState.toSessionState.mockReturnValue({ samples: {} });

            await saveSessionCommand(mockState, mockContext);

            expect(vscode.window.showWarningMessage).toHaveBeenCalledWith(
                'No data loaded to save. Load a POD5 file first.'
            );
        });

        it('should save session without name when user skips input', async () => {
            (vscode.window.showInputBox as any).mockResolvedValue('');

            await saveSessionCommand(mockState, mockContext);

            expect(SessionStateManager.saveSession).toHaveBeenCalled();
            expect(vscode.window.showInformationMessage).toHaveBeenCalledWith(
                expect.stringContaining('saved to TestWorkspace workspace state')
            );
        });

        it('should save session with name when user provides it', async () => {
            (vscode.window.showInputBox as any).mockResolvedValue('My Session');

            await saveSessionCommand(mockState, mockContext);

            expect(SessionStateManager.saveSession).toHaveBeenCalledWith(
                expect.objectContaining({ sessionName: 'My Session' }),
                mockContext
            );
        });

        it('should not save when user cancels', async () => {
            (vscode.window.showInputBox as any).mockResolvedValue(undefined);

            await saveSessionCommand(mockState, mockContext);

            expect(SessionStateManager.saveSession).not.toHaveBeenCalled();
        });

        it('should handle errors during save', async () => {
            (vscode.window.showInputBox as any).mockResolvedValue('Test');
            (SessionStateManager.saveSession as any).mockRejectedValue(new Error('Save failed'));

            await saveSessionCommand(mockState, mockContext);

            expect(vscode.window.showErrorMessage).toHaveBeenCalledWith(
                expect.stringContaining('Failed to save session')
            );
        });
    });

    describe('restoreSessionCommand', () => {
        it('should show warning when no saved session exists', async () => {
            (SessionStateManager.loadSession as any).mockResolvedValue(null);

            await restoreSessionCommand(mockState, mockContext);

            expect(vscode.window.showWarningMessage).toHaveBeenCalledWith(
                'No saved session found for this workspace.'
            );
        });

        it('should restore session when found', async () => {
            const savedSession = { samples: { sample1: {} }, sessionName: 'Saved' };
            (SessionStateManager.loadSession as any).mockResolvedValue(savedSession);

            await restoreSessionCommand(mockState, mockContext);

            expect(mockState.fromSessionState).toHaveBeenCalledWith(savedSession, mockContext);
        });

        it('should prompt for confirmation when unsaved changes exist', async () => {
            const savedSession = { samples: { sample1: {} } };
            (SessionStateManager.loadSession as any).mockResolvedValue(savedSession);
            (SessionStateManager.hasUnsavedChanges as any).mockResolvedValue(true);
            (vscode.window.showWarningMessage as any).mockResolvedValue('Restore');

            await restoreSessionCommand(mockState, mockContext);

            expect(vscode.window.showWarningMessage).toHaveBeenCalledWith(
                expect.stringContaining('unsaved changes'),
                expect.any(Object),
                'Restore'
            );
        });

        it('should cancel restore if user chooses cancel', async () => {
            const savedSession = { samples: { sample1: {} } };
            (SessionStateManager.loadSession as any).mockResolvedValue(savedSession);
            (SessionStateManager.hasUnsavedChanges as any).mockResolvedValue(true);
            // Modal cancel button returns undefined
            (vscode.window.showWarningMessage as any).mockResolvedValue(undefined);

            await restoreSessionCommand(mockState, mockContext);

            expect(mockState.fromSessionState).not.toHaveBeenCalled();
        });

        it('should handle errors during restore', async () => {
            (SessionStateManager.loadSession as any).mockRejectedValue(new Error('Load failed'));

            await restoreSessionCommand(mockState, mockContext);

            expect(vscode.window.showErrorMessage).toHaveBeenCalledWith(
                expect.stringContaining('Failed to restore session')
            );
        });
    });

    describe('exportSessionCommand', () => {
        it('should show warning when no data is loaded', async () => {
            mockState.toSessionState.mockReturnValue({ samples: {} });

            await exportSessionCommand(mockState, mockContext);

            expect(vscode.window.showWarningMessage).toHaveBeenCalledWith(
                'No data loaded to export. Load a POD5 file first.'
            );
        });

        it('should export session when file is selected', async () => {
            const fileUri = vscode.Uri.file('/path/to/session.json');
            (vscode.window.showSaveDialog as any).mockResolvedValue(fileUri);

            await exportSessionCommand(mockState, mockContext);

            expect(SessionStateManager.exportSession).toHaveBeenCalled();
        });

        it('should not export when user cancels', async () => {
            (vscode.window.showSaveDialog as any).mockResolvedValue(undefined);

            await exportSessionCommand(mockState, mockContext);

            expect(SessionStateManager.exportSession).not.toHaveBeenCalled();
        });
    });

    describe('importSessionCommand', () => {
        it('should import session when file is selected and user confirms', async () => {
            const fileUri = vscode.Uri.file('/path/to/session.json');
            const importedSession = { samples: { sample1: {} } };
            (vscode.window.showOpenDialog as any).mockResolvedValue([fileUri]);
            (vscode.window.showWarningMessage as any).mockResolvedValue('Import');
            (SessionStateManager.importSession as any).mockResolvedValue(importedSession);

            await importSessionCommand(mockState, mockContext);

            expect(mockState.fromSessionState).toHaveBeenCalledWith(importedSession, mockContext);
        });

        it('should import directly when no existing data', async () => {
            const fileUri = vscode.Uri.file('/path/to/session.json');
            const importedSession = { samples: { sample1: {} } };
            mockState.toSessionState.mockReturnValue({ samples: {} });
            (vscode.window.showOpenDialog as any).mockResolvedValue([fileUri]);
            (SessionStateManager.importSession as any).mockResolvedValue(importedSession);

            await importSessionCommand(mockState, mockContext);

            expect(mockState.fromSessionState).toHaveBeenCalledWith(importedSession, mockContext);
        });

        it('should not import when user cancels confirmation', async () => {
            const fileUri = vscode.Uri.file('/path/to/session.json');
            (vscode.window.showOpenDialog as any).mockResolvedValue([fileUri]);
            (vscode.window.showWarningMessage as any).mockResolvedValue('Cancel');

            await importSessionCommand(mockState, mockContext);

            expect(SessionStateManager.importSession).not.toHaveBeenCalled();
        });

        it('should show error when import fails', async () => {
            const fileUri = vscode.Uri.file('/path/to/session.json');
            mockState.toSessionState.mockReturnValue({ samples: {} }); // No existing data
            (vscode.window.showOpenDialog as any).mockResolvedValue([fileUri]);
            (SessionStateManager.importSession as any).mockRejectedValue(
                new Error('Invalid session file')
            );

            await importSessionCommand(mockState, mockContext);

            expect(vscode.window.showErrorMessage).toHaveBeenCalledWith(
                expect.stringContaining('Failed to import session')
            );
        });

        it('should not import when user cancels file dialog', async () => {
            (vscode.window.showOpenDialog as any).mockResolvedValue(undefined);

            await importSessionCommand(mockState, mockContext);

            expect(SessionStateManager.importSession).not.toHaveBeenCalled();
        });
    });

    describe('clearSessionCommand', () => {
        it('should clear session from workspace state after confirmation', async () => {
            (vscode.window.showWarningMessage as any).mockResolvedValue('Clear');

            await clearSessionCommand(mockContext);

            expect(SessionStateManager.clearSession).toHaveBeenCalledWith(mockContext);
            expect(vscode.window.showInformationMessage).toHaveBeenCalledWith(
                expect.stringContaining('Saved session cleared')
            );
        });

        it('should not clear when user cancels', async () => {
            (vscode.window.showWarningMessage as any).mockResolvedValue('Cancel');

            await clearSessionCommand(mockContext);

            expect(SessionStateManager.clearSession).not.toHaveBeenCalled();
        });

        it('should handle errors during clear', async () => {
            (vscode.window.showWarningMessage as any).mockResolvedValue('Clear');
            (SessionStateManager.clearSession as any).mockRejectedValue(new Error('Clear failed'));

            await clearSessionCommand(mockContext);

            expect(vscode.window.showErrorMessage).toHaveBeenCalledWith(
                expect.stringContaining('Failed to clear session')
            );
        });
    });

    describe('loadDemoSessionCommand', () => {
        it('should load demo session directly when no current data', async () => {
            mockState.toSessionState.mockReturnValue({ samples: {} });

            await loadDemoSessionCommand(mockState, mockContext);

            expect(mockState.loadDemoSession).toHaveBeenCalledWith(mockContext);
        });

        it('should prompt for confirmation when current data exists', async () => {
            (vscode.window.showWarningMessage as any).mockResolvedValue('Load Demo');

            await loadDemoSessionCommand(mockState, mockContext);

            expect(vscode.window.showWarningMessage).toHaveBeenCalledWith(
                'Loading demo session will replace current data. Continue?',
                { modal: true },
                'Load Demo'
            );
            expect(mockState.loadDemoSession).toHaveBeenCalledWith(mockContext);
        });

        it('should not load when user cancels', async () => {
            // Modal cancel button returns undefined
            (vscode.window.showWarningMessage as any).mockResolvedValue(undefined);

            await loadDemoSessionCommand(mockState, mockContext);

            expect(mockState.loadDemoSession).not.toHaveBeenCalled();
        });

        it('should handle errors during demo load', async () => {
            mockState.toSessionState.mockReturnValue({ samples: {} });
            mockState.loadDemoSession.mockRejectedValue(new Error('Demo load failed'));

            await loadDemoSessionCommand(mockState, mockContext);

            expect(vscode.window.showErrorMessage).toHaveBeenCalledWith(
                expect.stringContaining('Failed to load demo session')
            );
        });
    });
});
