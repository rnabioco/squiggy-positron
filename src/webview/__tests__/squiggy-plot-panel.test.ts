/**
 * Tests for Squiggle Plot Panel
 *
 * Tests the SquigglePlotPanel webview implementation.
 * Target: >80% coverage of squiggy-plot-panel.ts
 */

import { describe, it, expect, beforeEach, jest, afterEach } from '@jest/globals';
import * as vscode from 'vscode';
import { SquigglePlotPanel } from '../squiggy-plot-panel';
import { promises as fs } from 'fs';

// Mock fs.promises
jest.mock('fs', () => ({
    promises: {
        writeFile: jest.fn(),
    },
}));

describe('SquigglePlotPanel', () => {
    let mockPanel: any;
    let mockWebview: any;
    const extensionUri = vscode.Uri.file('/mock/extension');

    beforeEach(() => {
        // Mock webview
        mockWebview = {
            html: '',
            options: {},
            postMessage: jest.fn(),
            onDidReceiveMessage: jest.fn(),
            asWebviewUri: (uri: vscode.Uri) => uri,
        };

        // Mock webview panel
        mockPanel = {
            webview: mockWebview,
            title: '',
            reveal: jest.fn(),
            dispose: jest.fn(),
            onDidDispose: jest.fn((callback: () => void) => {
                // Store callback for later invocation if needed
                mockPanel._onDidDisposeCallback = callback;
                return { dispose: jest.fn() };
            }),
            viewColumn: vscode.ViewColumn.One,
        };

        // Mock window.createWebviewPanel
        (vscode.window.createWebviewPanel as any) = jest.fn(() => mockPanel);

        // Clear any existing panel
        (SquigglePlotPanel as any).currentPanel = undefined;

        jest.clearAllMocks();
    });

    afterEach(() => {
        jest.clearAllMocks();
        (SquigglePlotPanel as any).currentPanel = undefined;
    });

    describe('createOrShow', () => {
        it('should create a new panel when none exists', () => {
            const panel = SquigglePlotPanel.createOrShow(extensionUri);

            expect(vscode.window.createWebviewPanel).toHaveBeenCalledWith(
                'squigglePlot',
                'Squiggle Plot',
                vscode.ViewColumn.One,
                {
                    enableScripts: true,
                    retainContextWhenHidden: true,
                    localResourceRoots: [extensionUri],
                }
            );
            expect(panel).toBeDefined();
            expect((SquigglePlotPanel as any).currentPanel).toBe(panel);
        });

        it('should reveal existing panel instead of creating new one', () => {
            const panel1 = SquigglePlotPanel.createOrShow(extensionUri);
            jest.clearAllMocks();

            const panel2 = SquigglePlotPanel.createOrShow(extensionUri);

            expect(panel1).toBe(panel2);
            expect(vscode.window.createWebviewPanel).not.toHaveBeenCalled();
            expect(mockPanel.reveal).toHaveBeenCalled();
        });

        it('should use active text editor column if available', () => {
            (vscode.window as any).activeTextEditor = {
                viewColumn: vscode.ViewColumn.Two,
            };

            SquigglePlotPanel.createOrShow(extensionUri);

            expect(vscode.window.createWebviewPanel).toHaveBeenCalledWith(
                'squigglePlot',
                'Squiggle Plot',
                vscode.ViewColumn.Two,
                expect.any(Object)
            );

            // Clean up
            delete (vscode.window as any).activeTextEditor;
        });

        it('should register onDidDispose listener', () => {
            SquigglePlotPanel.createOrShow(extensionUri);

            expect(mockPanel.onDidDispose).toHaveBeenCalled();
        });

        it('should register message handler', () => {
            SquigglePlotPanel.createOrShow(extensionUri);

            expect(mockWebview.onDidReceiveMessage).toHaveBeenCalled();
        });
    });

    describe('Message Handling', () => {
        it('should handle alert messages', () => {
            SquigglePlotPanel.createOrShow(extensionUri);

            const messageHandler = mockWebview.onDidReceiveMessage.mock.calls[0][0];
            messageHandler({ command: 'alert', text: 'Test alert' });

            expect(vscode.window.showInformationMessage).toHaveBeenCalledWith('Test alert');
        });

        it('should ignore unknown message commands', () => {
            SquigglePlotPanel.createOrShow(extensionUri);

            const messageHandler = mockWebview.onDidReceiveMessage.mock.calls[0][0];
            messageHandler({ command: 'unknown', data: 'test' });

            // Should not throw
            expect(vscode.window.showInformationMessage).not.toHaveBeenCalled();
        });
    });

    describe('setPlot', () => {
        let panel: SquigglePlotPanel;

        beforeEach(() => {
            panel = SquigglePlotPanel.createOrShow(extensionUri);
            jest.clearAllMocks();
            jest.useFakeTimers();
        });

        afterEach(() => {
            jest.useRealTimers();
        });

        it('should set loading content immediately', () => {
            panel.setPlot('<div>Bokeh plot HTML</div>', ['read_001']);

            expect(mockWebview.html).toContain('Loading plot...');
            expect(mockWebview.html).toContain('spinner');
        });

        it('should set plot content after delay', () => {
            panel.setPlot('<div>Bokeh plot HTML</div>', ['read_001']);

            // Fast-forward timers
            jest.advanceTimersByTime(100);

            expect(mockWebview.html).toContain('<div>Bokeh plot HTML</div>');
            expect(mockWebview.html).toContain('plot-container');
        });

        it('should update panel title for single read', () => {
            panel.setPlot('<div>Bokeh plot HTML</div>', ['read_001']);

            expect(mockPanel.title).toBe('Squiggle Plot: read_001');
        });

        it('should update panel title for multiple reads', () => {
            panel.setPlot('<div>Bokeh plot HTML</div>', ['read_001', 'read_002', 'read_003']);

            expect(mockPanel.title).toBe('Squiggle Plot: 3 reads');
        });

        it('should include Content Security Policy in plot HTML', () => {
            panel.setPlot('<div>Bokeh plot HTML</div>', ['read_001']);
            jest.advanceTimersByTime(100);

            expect(mockWebview.html).toContain('Content-Security-Policy');
            expect(mockWebview.html).toContain('https://cdn.bokeh.org');
            expect(mockWebview.html).toContain('https://cdn.pydata.org');
        });

        it('should allow unsafe-inline and unsafe-eval for Bokeh', () => {
            panel.setPlot('<div>Bokeh plot HTML</div>', ['read_001']);
            jest.advanceTimersByTime(100);

            expect(mockWebview.html).toContain("script-src 'unsafe-inline' 'unsafe-eval'");
            expect(mockWebview.html).toContain("style-src 'unsafe-inline'");
        });
    });

    describe('exportPlot', () => {
        let panel: SquigglePlotPanel;

        beforeEach(() => {
            panel = SquigglePlotPanel.createOrShow(extensionUri);
            panel.setPlot('<div>Test Bokeh HTML</div>', ['read_001']);
            jest.clearAllMocks();
        });

        it('should export HTML successfully', async () => {
            (fs.writeFile as any).mockResolvedValue(undefined);

            await panel.exportPlot('/path/to/output.html');

            expect(fs.writeFile).toHaveBeenCalledWith(
                '/path/to/output.html',
                '<div>Test Bokeh HTML</div>',
                'utf-8'
            );
            // Success now shown via status bar, not pop-up
            expect(vscode.window.showInformationMessage).not.toHaveBeenCalled();
        });

        it('should show error for PNG export', async () => {
            await panel.exportPlot('/path/to/output.png');

            expect(fs.writeFile).not.toHaveBeenCalled();
            expect(vscode.window.showErrorMessage).toHaveBeenCalledWith(
                'PNG/SVG export not yet implemented'
            );
        });

        it('should show error for SVG export', async () => {
            await panel.exportPlot('/path/to/output.svg');

            expect(fs.writeFile).not.toHaveBeenCalled();
            expect(vscode.window.showErrorMessage).toHaveBeenCalledWith(
                'PNG/SVG export not yet implemented'
            );
        });

        it('should handle file write errors', async () => {
            (fs.writeFile as any).mockRejectedValue(new Error('Write failed'));

            await expect(panel.exportPlot('/path/to/output.html')).rejects.toThrow('Write failed');
        });
    });

    describe('dispose', () => {
        it('should clear currentPanel reference', () => {
            const panel = SquigglePlotPanel.createOrShow(extensionUri);

            panel.dispose();

            expect((SquigglePlotPanel as any).currentPanel).toBeUndefined();
        });

        it('should dispose the underlying panel', () => {
            const panel = SquigglePlotPanel.createOrShow(extensionUri);

            panel.dispose();

            expect(mockPanel.dispose).toHaveBeenCalled();
        });

        it('should dispose all registered disposables', () => {
            const panel = SquigglePlotPanel.createOrShow(extensionUri);

            // The panel registers at least 2 disposables (onDidDispose and onDidReceiveMessage)
            const disposeCallsBefore = mockPanel.onDidDispose.mock.results.length;
            expect(disposeCallsBefore).toBeGreaterThan(0);

            panel.dispose();

            // Disposables should have been cleared
            // We can't directly verify they were disposed, but we can check the dispose was called
            expect(mockPanel.dispose).toHaveBeenCalled();
        });

        it('should trigger dispose when panel is closed', () => {
            const panel = SquigglePlotPanel.createOrShow(extensionUri);

            // Simulate panel being closed by user
            mockPanel._onDidDisposeCallback();

            expect((SquigglePlotPanel as any).currentPanel).toBeUndefined();
        });
    });

    describe('HTML Content Generation', () => {
        it('should generate loading content with spinner', () => {
            const panel = SquigglePlotPanel.createOrShow(extensionUri);
            panel.setPlot('<div>Test</div>', ['read_001']);

            const loadingHtml = mockWebview.html;

            expect(loadingHtml).toContain('Loading plot...');
            expect(loadingHtml).toContain('spinner');
            expect(loadingHtml).toContain('animation: spin');
        });

        it('should generate webview content with plot container', () => {
            const panel = SquigglePlotPanel.createOrShow(extensionUri);
            jest.useFakeTimers();

            panel.setPlot('<div class="bokeh-plot">Test Plot</div>', ['read_001']);
            jest.advanceTimersByTime(100);

            const plotHtml = mockWebview.html;

            expect(plotHtml).toContain('<div id="plot-container">');
            expect(plotHtml).toContain('<div class="bokeh-plot">Test Plot</div>');

            jest.useRealTimers();
        });

        it('should include DOCTYPE and proper HTML structure', () => {
            const panel = SquigglePlotPanel.createOrShow(extensionUri);
            jest.useFakeTimers();

            panel.setPlot('<div>Test</div>', ['read_001']);
            jest.advanceTimersByTime(100);

            const html = mockWebview.html;

            expect(html).toContain('<!DOCTYPE html>');
            expect(html).toContain('<html lang="en">');
            expect(html).toContain('<head>');
            expect(html).toContain('<body>');

            jest.useRealTimers();
        });

        it('should set viewport meta tag', () => {
            const panel = SquigglePlotPanel.createOrShow(extensionUri);
            jest.useFakeTimers();

            panel.setPlot('<div>Test</div>', ['read_001']);
            jest.advanceTimersByTime(100);

            expect(mockWebview.html).toContain(
                '<meta name="viewport" content="width=device-width, initial-scale=1.0">'
            );

            jest.useRealTimers();
        });
    });
});
