/**
 * Tests for PathResolver
 *
 * Tests path resolution logic for extension-relative, workspace-relative,
 * and Python package paths.
 * Target: >80% coverage of path-resolver.ts
 */

import { describe, it, expect, jest, beforeEach } from '@jest/globals';
import * as vscode from 'vscode';
import * as path from 'path';
import { PathResolver } from '../path-resolver';

describe('PathResolver', () => {
    let mockExtensionUri: vscode.Uri;
    let mockWorkspaceUri: vscode.Uri;

    beforeEach(() => {
        mockExtensionUri = vscode.Uri.file('/mock/extension');
        mockWorkspaceUri = vscode.Uri.file('/mock/workspace');
    });

    describe('resolvePythonPackagePath', () => {
        it('should return path as-is if not a package path', async () => {
            const result = await PathResolver.resolvePythonPackagePath('/regular/path.pod5');

            expect(result).toBe('/regular/path.pod5');
        });

        it('should return path as-is for paths without package syntax', async () => {
            const result = await PathResolver.resolvePythonPackagePath('data/file.pod5');

            expect(result).toBe('data/file.pod5');
        });

        it('should resolve package path with positron client', async () => {
            const mockClient = {
                executeSilent: jest.fn(async () => {}),
                getVariable: jest.fn(async () => '/python/lib/squiggy/data/file.pod5'),
            };

            const result = await PathResolver.resolvePythonPackagePath(
                '<package:squiggy>/data/file.pod5',
                mockClient
            );

            expect(result).toBe('/python/lib/squiggy/data/file.pod5');
            expect(mockClient.executeSilent).toHaveBeenCalled();
            expect(mockClient.getVariable).toHaveBeenCalled();
        });

        it('should clean up temporary variable after resolution', async () => {
            const mockClient = {
                executeSilent: jest.fn(async () => {}),
                getVariable: jest.fn(async () => '/python/lib/squiggy/data/file.pod5'),
            };

            await PathResolver.resolvePythonPackagePath(
                '<package:squiggy>/data/file.pod5',
                mockClient
            );

            // Check that cleanup was called (should be 2 executeSilent calls: one for setup, one for cleanup)
            expect(mockClient.executeSilent).toHaveBeenCalledTimes(2);
            const cleanupCall = (mockClient.executeSilent as any).mock.calls[1][0];
            expect(cleanupCall).toContain('del _squiggy_temp_path_');
        });

        it('should return original path if positron client is not provided', async () => {
            const result = await PathResolver.resolvePythonPackagePath(
                '<package:squiggy>/data/file.pod5',
                undefined
            );

            expect(result).toBe('<package:squiggy>/data/file.pod5');
        });

        it('should return original path if resolution returns None', async () => {
            const mockClient = {
                executeSilent: jest.fn(async () => {}),
                getVariable: jest.fn(async () => 'None'),
            };

            const result = await PathResolver.resolvePythonPackagePath(
                '<package:squiggy>/data/file.pod5',
                mockClient
            );

            expect(result).toBe('<package:squiggy>/data/file.pod5');
        });

        it('should return original path if resolution returns null', async () => {
            const mockClient = {
                executeSilent: jest.fn(async () => {}),
                getVariable: jest.fn(async () => 'null'),
            };

            const result = await PathResolver.resolvePythonPackagePath(
                '<package:squiggy>/data/file.pod5',
                mockClient
            );

            expect(result).toBe('<package:squiggy>/data/file.pod5');
        });

        it('should handle errors during package resolution', async () => {
            const mockClient = {
                executeSilent: jest.fn(async () => {
                    throw new Error('Python execution failed');
                }),
                getVariable: jest.fn(),
            };

            const result = await PathResolver.resolvePythonPackagePath(
                '<package:squiggy>/data/file.pod5',
                mockClient
            );

            // Should return original path as fallback
            expect(result).toBe('<package:squiggy>/data/file.pod5');
        });
    });

    describe('resolveExtensionPath', () => {
        it('should resolve path with ${extensionPath} placeholder', () => {
            const result = PathResolver.resolveExtensionPath(
                '${extensionPath}/data/file.pod5',
                mockExtensionUri
            );

            expect(result).toContain('data');
            expect(result).toContain('file.pod5');
        });

        it('should return absolute path as-is', () => {
            const absolutePath = '/absolute/path/file.pod5';

            const result = PathResolver.resolveExtensionPath(absolutePath, mockExtensionUri);

            expect(result).toBe(absolutePath);
        });

        it('should resolve relative path relative to extension', () => {
            const result = PathResolver.resolveExtensionPath('data/file.pod5', mockExtensionUri);

            expect(result).toContain('data');
            expect(result).toContain('file.pod5');
        });
    });

    describe('makeWorkspaceRelative', () => {
        it('should return absolute path if no workspace URI', () => {
            const absolutePath = '/some/absolute/path.pod5';

            const result = PathResolver.makeWorkspaceRelative(absolutePath, undefined);

            expect(result).toBe(absolutePath);
        });

        it('should make path workspace-relative if within workspace', () => {
            const absolutePath = path.join(mockWorkspaceUri.fsPath, 'subfolder', 'file.pod5');

            const result = PathResolver.makeWorkspaceRelative(absolutePath, mockWorkspaceUri);

            expect(result).toBe(path.join('subfolder', 'file.pod5'));
        });

        it('should return absolute path if not within workspace', () => {
            const absolutePath = '/other/location/file.pod5';

            const result = PathResolver.makeWorkspaceRelative(absolutePath, mockWorkspaceUri);

            expect(result).toBe(absolutePath);
        });
    });

    describe('resolveWorkspacePath', () => {
        it('should return absolute path as-is', () => {
            const absolutePath = '/absolute/path/file.pod5';

            const result = PathResolver.resolveWorkspacePath(absolutePath, mockWorkspaceUri);

            expect(result).toBe(absolutePath);
        });

        it('should resolve relative path to workspace', () => {
            const result = PathResolver.resolveWorkspacePath('data/file.pod5', mockWorkspaceUri);

            expect(result).toBe(path.join(mockWorkspaceUri.fsPath, 'data', 'file.pod5'));
        });
    });

    describe('normalizePath', () => {
        it('should normalize path separators', () => {
            const result = PathResolver.normalizePath('/path/with/../dots/./file.pod5');

            // Path should be normalized (exact result depends on platform)
            expect(result).toBeTruthy();
            expect(result).toContain('file.pod5');
        });

        it('should handle already normalized paths', () => {
            const normalPath = '/path/to/file.pod5';

            const result = PathResolver.normalizePath(normalPath);

            expect(result).toBeTruthy();
        });
    });

    describe('isExtensionRelative', () => {
        it('should return true for paths starting with extension root', () => {
            const filePath = path.join(mockExtensionUri.fsPath, 'data', 'file.pod5');

            const result = PathResolver.isExtensionRelative(filePath, mockExtensionUri);

            expect(result).toBe(true);
        });

        it('should return true for paths with ${extensionPath} placeholder', () => {
            const result = PathResolver.isExtensionRelative(
                '${extensionPath}/data/file.pod5',
                mockExtensionUri
            );

            expect(result).toBe(true);
        });

        it('should return false for non-extension paths', () => {
            const result = PathResolver.isExtensionRelative(
                '/other/path/file.pod5',
                mockExtensionUri
            );

            expect(result).toBe(false);
        });
    });

    describe('getDisplayPath', () => {
        it('should prefix demo for extension-relative paths', () => {
            const filePath = path.join(mockExtensionUri.fsPath, 'data', 'file.pod5');

            const result = PathResolver.getDisplayPath(filePath, mockExtensionUri);

            expect(result).toContain('(Demo)');
            expect(result).toContain('file.pod5');
        });

        it('should show full path for user files', () => {
            const filePath = '/user/data/file.pod5';

            const result = PathResolver.getDisplayPath(filePath, mockExtensionUri);

            expect(result).toBe(filePath);
        });

        it('should handle paths with ${extensionPath} placeholder', () => {
            const result = PathResolver.getDisplayPath(
                '${extensionPath}/data/file.pod5',
                mockExtensionUri
            );

            expect(result).toContain('(Demo)');
            expect(result).toContain('file.pod5');
        });
    });
});
