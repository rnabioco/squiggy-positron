/**
 * Tests for webview utility functions
 */

import { describe, it, expect, beforeEach } from '@jest/globals';
import * as vscode from 'vscode';
import { getWebviewOptions, getReactWebviewHtml } from '../webview-utils';

describe('webview-utils', () => {
    let mockExtensionUri: vscode.Uri;

    beforeEach(() => {
        mockExtensionUri = vscode.Uri.file('/mock/extension/path');
    });

    describe('getWebviewOptions', () => {
        it('should return webview options with scripts enabled', () => {
            const options = getWebviewOptions(mockExtensionUri);

            expect(options.enableScripts).toBe(true);
        });

        it('should include localResourceRoots', () => {
            const options = getWebviewOptions(mockExtensionUri);

            expect(options.localResourceRoots).toBeDefined();
            expect(Array.isArray(options.localResourceRoots)).toBe(true);
        });

        it('should include enableScripts and localResourceRoots', () => {
            const options = getWebviewOptions(mockExtensionUri);

            // Main assertions - these are the most important properties
            expect(options.enableScripts).toBe(true);
            expect(options.localResourceRoots).toBeDefined();
        });
    });

    describe('getReactWebviewHtml', () => {
        let mockWebview: vscode.Webview;

        beforeEach(() => {
            mockWebview = {
                asWebviewUri: (uri: vscode.Uri) => {
                    return vscode.Uri.parse(`webview://resource${uri.fsPath}`);
                },
                cspSource: 'mock-csp-source',
            } as any;
        });

        it('should generate HTML with correct title', () => {
            const html = getReactWebviewHtml(mockWebview, mockExtensionUri, 'Test Title');

            expect(html).toContain('<title>Test Title</title>');
        });

        it('should include DOCTYPE and html structure', () => {
            const html = getReactWebviewHtml(mockWebview, mockExtensionUri, 'Test');

            expect(html).toContain('<!DOCTYPE html>');
            expect(html).toContain('<html');
            expect(html).toContain('</html>');
            expect(html).toContain('<head>');
            expect(html).toContain('<body>');
        });

        it('should include root div for React mounting', () => {
            const html = getReactWebviewHtml(mockWebview, mockExtensionUri, 'Test');

            expect(html).toContain('<div id="root"></div>');
        });

        it('should include webview script reference', () => {
            const html = getReactWebviewHtml(mockWebview, mockExtensionUri, 'Test');

            expect(html).toContain('<script');
            expect(html).toContain('webview.js');
        });

        it('should include Content Security Policy', () => {
            const html = getReactWebviewHtml(mockWebview, mockExtensionUri, 'Test');

            expect(html).toContain('Content-Security-Policy');
            expect(html).toContain('default-src');
        });

        it('should include charset meta tag', () => {
            const html = getReactWebviewHtml(mockWebview, mockExtensionUri, 'Test');

            expect(html).toContain('<meta charset="UTF-8"');
        });

        it('should include viewport meta tag', () => {
            const html = getReactWebviewHtml(mockWebview, mockExtensionUri, 'Test');

            expect(html).toContain('<meta name="viewport"');
        });
    });
});
