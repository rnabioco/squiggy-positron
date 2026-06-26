/**
 * Tests for SquiggyKernelManager lifecycle (Issue #186)
 *
 * Focus: the concurrency guard on start() and the dispose() teardown contract.
 */

import { describe, it, expect, jest, beforeEach } from '@jest/globals';
import * as positron from 'positron';
import { SquiggyKernelManager, SquiggyKernelState } from '../squiggy-kernel-manager';

jest.mock('positron');
jest.mock('vscode');

describe('SquiggyKernelManager', () => {
    const makeRuntime = () => ({
        runtimeId: 'rt-squiggy',
        runtimeName: 'Squiggy',
        runtimeVersion: '3.12',
        runtimePath: '/home/u/.venvs/squiggy/bin/python',
        languageId: 'python',
    });

    // Fake session whose state-change listeners reach Ready asynchronously
    // (mirrors the real kernel, which never fires synchronously on registration).
    const makeSession = (sessionId = 'sess-1') => ({
        metadata: { sessionId, sessionName: 'Squiggy Dedicated Kernel', sessionMode: 'console' },
        onDidChangeRuntimeState: jest.fn((cb: (s: string) => void) => {
            setTimeout(() => cb(positron.RuntimeState.Ready), 0);
            return { dispose: jest.fn() };
        }),
        onDidEndSession: jest.fn(() => ({ dispose: jest.fn() })),
    });

    beforeEach(() => {
        jest.clearAllMocks();
        (positron.runtime.getRegisteredRuntimes as any).mockResolvedValue([makeRuntime()]);
        (positron.runtime.getPreferredRuntime as any).mockResolvedValue(makeRuntime());
        (positron.runtime.getActiveSessions as any).mockResolvedValue([]);
        (positron.runtime.deleteSession as any).mockResolvedValue(undefined);
        // Reject the readiness probe ('1+1') so waitForReady resolves via the
        // state-change event (after setState(Ready)); resolve everything else
        // (setupPythonPath / verifySquiggyAvailable).
        (positron.runtime.executeCode as any).mockImplementation(
            async (_lang: string, code: string) => {
                if (code === '1+1') {
                    throw new Error('not ready');
                }
                return {};
            }
        );
    });

    it('coalesces concurrent start() calls into a single session', async () => {
        (positron.runtime.startLanguageRuntime as any).mockResolvedValue(makeSession());

        const mgr = new SquiggyKernelManager('/ext/path'); // no context => no reconnect
        await Promise.all([mgr.start(), mgr.start(), mgr.start()]);

        expect(positron.runtime.startLanguageRuntime).toHaveBeenCalledTimes(1);
        expect(mgr.getState()).toBe(SquiggyKernelState.Ready);
        expect(mgr.getSessionId()).toBe('sess-1');
    });

    it('does not start a second session once already running', async () => {
        (positron.runtime.startLanguageRuntime as any).mockResolvedValue(makeSession());

        const mgr = new SquiggyKernelManager('/ext/path');
        await mgr.start();
        await mgr.start(); // already running -> early return

        expect(positron.runtime.startLanguageRuntime).toHaveBeenCalledTimes(1);
    });

    it('throws when start() is called after dispose()', async () => {
        const mgr = new SquiggyKernelManager('/ext/path');
        mgr.dispose();

        await expect(mgr.start()).rejects.toThrow(/disposed/i);
        expect(positron.runtime.startLanguageRuntime).not.toHaveBeenCalled();
    });

    it('dispose() is idempotent and deletes the session once', async () => {
        (positron.runtime.startLanguageRuntime as any).mockResolvedValue(makeSession());

        const mgr = new SquiggyKernelManager('/ext/path');
        await mgr.start();

        mgr.dispose();
        mgr.dispose(); // second call is a no-op

        expect(positron.runtime.deleteSession).toHaveBeenCalledTimes(1);
        expect(mgr.getSessionId()).toBeUndefined();
    });
});
