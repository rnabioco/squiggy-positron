/**
 * Tests for SessionStateManager
 *
 * Tests session persistence, validation, import/export, and demo session generation.
 * Target: >75% coverage of session-state-manager.ts
 */

import { describe, it, expect, jest, beforeEach } from '@jest/globals';
import * as vscode from 'vscode';
import * as fs from 'fs/promises';
import * as crypto from 'crypto';
import { SessionStateManager } from '../session-state-manager';
import { SessionState } from '../../types/squiggy-session-types';
import { logger } from '../../utils/logger';

// Mock fs.promises
jest.mock('fs/promises');

// Mock logger
jest.mock('../../utils/logger');

// Mock crypto
jest.mock('crypto', () => ({
    createHash: jest.fn(() => ({
        update: jest.fn(),
        digest: jest.fn(() => 'mock-md5-hash'),
    })),
}));

describe('SessionStateManager', () => {
    let mockContext: any;
    let mockExtensionUri: vscode.Uri;
    let validSession: SessionState;

    beforeEach(() => {
        mockExtensionUri = vscode.Uri.file('/mock/extension');

        mockContext = {
            workspaceState: {
                get: jest.fn(),
                update: jest.fn(),
            },
            extension: {
                packageJSON: {
                    version: '1.2.3',
                },
            },
        };

        validSession = {
            version: '1.0.0',
            timestamp: new Date().toISOString(),
            sessionName: 'Test Session',
            isDemo: false,
            samples: {
                Sample1: {
                    pod5Paths: ['/path/file.pod5'],
                },
            },
            plotOptions: {
                mode: 'SINGLE',
                normalization: 'ZNORM',
                showDwellTime: false,
                showBaseAnnotations: true,
                scaleDwellTime: false,
                downsample: 5,
                showSignalPoints: false,
            },
        };

        jest.clearAllMocks();
    });

    describe('saveSession', () => {
        it('should save session with metadata to workspace state', async () => {
            (fs.stat as any).mockResolvedValue({
                size: 1024,
                mtime: new Date('2024-01-01'),
            });
            (fs.readFile as any).mockResolvedValue(Buffer.from('file content'));

            await SessionStateManager.saveSession(validSession, mockContext);

            expect(mockContext.workspaceState.update).toHaveBeenCalled();
            const savedSession = mockContext.workspaceState.update.mock.calls[0][1];
            expect(savedSession.version).toBe('1.0.0');
            expect(savedSession.extensionVersion).toBe('1.2.3');
            expect(savedSession.positronVersion).toBe(vscode.version);
            expect(savedSession.fileChecksums).toBeDefined();
        });

        it('should calculate file checksums for POD5 files', async () => {
            (fs.stat as any).mockResolvedValue({
                size: 1024,
                mtime: new Date('2024-01-01'),
            });
            (fs.readFile as any).mockResolvedValue(Buffer.from('file content'));

            await SessionStateManager.saveSession(validSession, mockContext);

            const savedSession = mockContext.workspaceState.update.mock.calls[0][1];
            expect(savedSession.fileChecksums['/path/file.pod5']).toBeDefined();
            expect(savedSession.fileChecksums['/path/file.pod5'].md5).toBe('mock-md5-hash');
        });

        it('should calculate checksums for BAM files', async () => {
            const sessionWithBam: SessionState = {
                ...validSession,
                samples: {
                    Sample1: {
                        pod5Paths: ['/path/file.pod5'],
                        bamPath: '/path/file.bam',
                    },
                },
            };

            (fs.stat as any).mockResolvedValue({
                size: 2048,
                mtime: new Date('2024-01-01'),
            });
            (fs.readFile as any).mockResolvedValue(Buffer.from('content'));

            await SessionStateManager.saveSession(sessionWithBam, mockContext);

            const savedSession = mockContext.workspaceState.update.mock.calls[0][1];
            expect(savedSession.fileChecksums['/path/file.bam']).toBeDefined();
        });

        it('should calculate checksums for FASTA files', async () => {
            const sessionWithFasta: SessionState = {
                ...validSession,
                samples: {
                    Sample1: {
                        pod5Paths: ['/path/file.pod5'],
                        fastaPath: '/path/file.fasta',
                    },
                },
            };

            (fs.stat as any).mockResolvedValue({
                size: 512,
                mtime: new Date('2024-01-01'),
            });
            (fs.readFile as any).mockResolvedValue(Buffer.from('fasta'));

            await SessionStateManager.saveSession(sessionWithFasta, mockContext);

            const savedSession = mockContext.workspaceState.update.mock.calls[0][1];
            expect(savedSession.fileChecksums['/path/file.fasta']).toBeDefined();
        });

        it('should handle file read errors gracefully', async () => {
            (fs.stat as any).mockRejectedValue(new Error('File not found'));
            (fs.readFile as any).mockRejectedValue(new Error('Cannot read'));

            await SessionStateManager.saveSession(validSession, mockContext);

            const savedSession = mockContext.workspaceState.update.mock.calls[0][1];
            expect(savedSession.fileChecksums['/path/file.pod5']).toEqual({});
        });

        it('should use default version if not provided', async () => {
            const sessionNoVersion = { ...validSession } as any;
            delete sessionNoVersion.version;

            (fs.stat as any).mockResolvedValue({
                size: 1024,
                mtime: new Date(),
            });
            (fs.readFile as any).mockResolvedValue(Buffer.from('content'));

            await SessionStateManager.saveSession(sessionNoVersion, mockContext);

            const savedSession = mockContext.workspaceState.update.mock.calls[0][1];
            expect(savedSession.version).toBe('1.0.0');
        });
    });

    describe('loadSession', () => {
        it('should load and return valid session', async () => {
            mockContext.workspaceState.get.mockReturnValue(validSession);

            const result = await SessionStateManager.loadSession(mockContext);

            expect(result).toEqual(validSession);
        });

        it('should return null if no session saved', async () => {
            mockContext.workspaceState.get.mockReturnValue(undefined);

            const result = await SessionStateManager.loadSession(mockContext);

            expect(result).toBeNull();
        });

        it('should return null for invalid session', async () => {
            const invalidSession = { version: '1.0.0' }; // Missing required fields
            mockContext.workspaceState.get.mockReturnValue(invalidSession);

            const result = await SessionStateManager.loadSession(mockContext);

            expect(result).toBeNull();
            expect(logger.warning).toHaveBeenCalledWith(
                expect.stringContaining('Session state validation failed')
            );
        });
    });

    describe('exportSession', () => {
        it('should export session to JSON file', async () => {
            (fs.stat as any).mockResolvedValue({
                size: 1024,
                mtime: new Date('2024-01-01'),
            });
            (fs.readFile as any).mockResolvedValue(Buffer.from('content'));
            (fs.writeFile as any).mockResolvedValue(undefined);

            await SessionStateManager.exportSession(
                validSession,
                '/export/session.json',
                mockContext
            );

            expect(fs.writeFile).toHaveBeenCalled();
            const writtenJson = (fs.writeFile as any).mock.calls[0][1];
            const parsed = JSON.parse(writtenJson);
            expect(parsed.version).toBe('1.0.0');
            expect(parsed.extensionVersion).toBe('1.2.3');
        });

        it('should export without context', async () => {
            (fs.stat as any).mockResolvedValue({
                size: 1024,
                mtime: new Date('2024-01-01'),
            });
            (fs.readFile as any).mockResolvedValue(Buffer.from('content'));
            (fs.writeFile as any).mockResolvedValue(undefined);

            await SessionStateManager.exportSession(validSession, '/export/session.json');

            expect(fs.writeFile).toHaveBeenCalled();
            const writtenJson = (fs.writeFile as any).mock.calls[0][1];
            const parsed = JSON.parse(writtenJson);
            expect(parsed.extensionVersion).toBeUndefined();
        });

        it('should use existing checksums if present', async () => {
            const sessionWithChecksums = {
                ...validSession,
                fileChecksums: { '/path/file.pod5': { md5: 'existing-hash' } },
            };

            (fs.writeFile as any).mockResolvedValue(undefined);

            await SessionStateManager.exportSession(sessionWithChecksums, '/export/session.json');

            const writtenJson = (fs.writeFile as any).mock.calls[0][1];
            const parsed = JSON.parse(writtenJson);
            expect(parsed.fileChecksums['/path/file.pod5'].md5).toBe('existing-hash');
        });
    });

    describe('importSession', () => {
        it('should import and validate session from file', async () => {
            (fs.readFile as any).mockResolvedValue(JSON.stringify(validSession));

            const result = await SessionStateManager.importSession('/import/session.json');

            expect(result).toEqual(validSession);
        });

        it('should throw error for invalid session file', async () => {
            const invalidSession = { version: '1.0.0' }; // Missing required fields
            (fs.readFile as any).mockResolvedValue(JSON.stringify(invalidSession));

            await expect(SessionStateManager.importSession('/import/session.json')).rejects.toThrow(
                'Invalid session file'
            );
        });

        it('should handle JSON parse errors', async () => {
            (fs.readFile as any).mockResolvedValue('invalid json{');

            await expect(
                SessionStateManager.importSession('/import/session.json')
            ).rejects.toThrow();
        });
    });

    describe('clearSession', () => {
        it('should clear session from workspace state', async () => {
            await SessionStateManager.clearSession(mockContext);

            expect(mockContext.workspaceState.update).toHaveBeenCalledWith(
                'squiggy.sessionState',
                undefined
            );
        });
    });

    describe('getDemoSession', () => {
        it('should return demo session with packaged data paths', () => {
            const demoSession = SessionStateManager.getDemoSession(mockExtensionUri);

            expect(demoSession.isDemo).toBe(true);
            expect(demoSession.sessionName).toContain('Demo');
            expect(demoSession.samples.Yeast_tRNA).toBeDefined();
            expect(demoSession.samples.Yeast_tRNA.pod5Paths[0]).toContain('<package:squiggy>');
        });

        it('should use EVENTALIGN mode for demo', () => {
            const demoSession = SessionStateManager.getDemoSession(mockExtensionUri);

            expect(demoSession.plotOptions.mode).toBe('EVENTALIGN');
            expect(demoSession.plotOptions.showBaseAnnotations).toBe(true);
        });

        it('should have expanded UI state for demo sample', () => {
            const demoSession = SessionStateManager.getDemoSession(mockExtensionUri);

            expect(demoSession.ui?.expandedSamples).toContain('Yeast_tRNA');
        });
    });

    describe('validateSession', () => {
        it('should validate valid session', () => {
            const result = SessionStateManager.validateSession(validSession);

            expect(result.valid).toBe(true);
            expect(result.errors).toHaveLength(0);
        });

        it('should detect missing version', () => {
            const invalid = { ...validSession } as any;
            delete invalid.version;

            const result = SessionStateManager.validateSession(invalid);

            expect(result.valid).toBe(false);
            expect(result.errors).toContain('Missing version field');
        });

        it('should detect missing timestamp', () => {
            const invalid = { ...validSession } as any;
            delete invalid.timestamp;

            const result = SessionStateManager.validateSession(invalid);

            expect(result.valid).toBe(false);
            expect(result.errors).toContain('Missing timestamp field');
        });

        it('should detect missing samples', () => {
            const invalid = { ...validSession } as any;
            delete invalid.samples;

            const result = SessionStateManager.validateSession(invalid);

            expect(result.valid).toBe(false);
            expect(result.errors).toContain('Missing or invalid samples field');
        });

        it('should detect missing plotOptions', () => {
            const invalid = { ...validSession } as any;
            delete invalid.plotOptions;

            const result = SessionStateManager.validateSession(invalid);

            expect(result.valid).toBe(false);
            expect(result.errors).toContain('Missing or invalid plotOptions field');
        });

        it('should validate sample pod5Paths is non-empty array', () => {
            const invalid = {
                ...validSession,
                samples: {
                    Sample1: { pod5Paths: [] },
                },
            };

            const result = SessionStateManager.validateSession(invalid);

            expect(result.valid).toBe(false);
            expect(result.errors[0]).toContain('pod5Paths must be non-empty array');
        });

        it('should validate pod5Paths contains only strings', () => {
            const invalid = {
                ...validSession,
                samples: {
                    Sample1: { pod5Paths: [123, 'string'] as any },
                },
            };

            const result = SessionStateManager.validateSession(invalid);

            expect(result.valid).toBe(false);
            expect(result.errors[0]).toContain('pod5Paths must contain only strings');
        });

        it('should validate bamPath is string', () => {
            const invalid = {
                ...validSession,
                samples: {
                    Sample1: { pod5Paths: ['/path/file.pod5'], bamPath: 123 as any },
                },
            };

            const result = SessionStateManager.validateSession(invalid);

            expect(result.valid).toBe(false);
            expect(result.errors[0]).toContain('bamPath must be string');
        });

        it('should validate fastaPath is string', () => {
            const invalid = {
                ...validSession,
                samples: {
                    Sample1: { pod5Paths: ['/path/file.pod5'], fastaPath: 123 as any },
                },
            };

            const result = SessionStateManager.validateSession(invalid);

            expect(result.valid).toBe(false);
            expect(result.errors[0]).toContain('fastaPath must be string');
        });

        it('should validate plotOptions has required fields', () => {
            const invalid = {
                ...validSession,
                plotOptions: { mode: 'SINGLE' },
            };

            const result = SessionStateManager.validateSession(invalid);

            expect(result.valid).toBe(false);
            expect(result.errors.length).toBeGreaterThan(0);
            expect(result.errors.some((e) => e.includes('normalization'))).toBe(true);
        });

        it('should validate modificationFilters structure', () => {
            const invalid = {
                ...validSession,
                modificationFilters: {
                    minProbability: 'not a number' as any,
                    enabledModTypes: ['5mC'],
                },
            };

            const result = SessionStateManager.validateSession(invalid);

            expect(result.valid).toBe(false);
            expect(result.errors[0]).toContain('minProbability must be number');
        });

        it('should validate modificationFilters enabledModTypes is array', () => {
            const invalid = {
                ...validSession,
                modificationFilters: {
                    minProbability: 0.8,
                    enabledModTypes: 'not an array' as any,
                },
            };

            const result = SessionStateManager.validateSession(invalid);

            expect(result.valid).toBe(false);
            expect(result.errors[0]).toContain('enabledModTypes must be array');
        });

        it('should validate UI expandedSamples is array', () => {
            const invalid = {
                ...validSession,
                ui: {
                    expandedSamples: 'not an array' as any,
                },
            };

            const result = SessionStateManager.validateSession(invalid);

            expect(result.valid).toBe(false);
            expect(result.errors[0]).toContain('expandedSamples must be array');
        });

        it('should validate UI selectedSamplesForComparison is array', () => {
            const invalid = {
                ...validSession,
                ui: {
                    selectedSamplesForComparison: 'not an array' as any,
                },
            };

            const result = SessionStateManager.validateSession(invalid);

            expect(result.valid).toBe(false);
            expect(result.errors[0]).toContain('selectedSamplesForComparison must be array');
        });
    });

    describe('hasUnsavedChanges', () => {
        it('should return true if no saved session exists', async () => {
            mockContext.workspaceState.get.mockReturnValue(undefined);

            const result = await SessionStateManager.hasUnsavedChanges(validSession, mockContext);

            expect(result).toBe(true);
        });

        it('should return false if sessions are identical', async () => {
            mockContext.workspaceState.get.mockReturnValue(validSession);

            const result = await SessionStateManager.hasUnsavedChanges(validSession, mockContext);

            expect(result).toBe(false);
        });

        it('should return true if sessions differ', async () => {
            const savedSession = { ...validSession, sessionName: 'Different Name' };
            mockContext.workspaceState.get.mockReturnValue(savedSession);

            const result = await SessionStateManager.hasUnsavedChanges(validSession, mockContext);

            expect(result).toBe(true);
        });

        it('should ignore timestamp differences', async () => {
            const savedSession = { ...validSession, timestamp: '2023-01-01T00:00:00.000Z' };
            mockContext.workspaceState.get.mockReturnValue(savedSession);

            const currentSession = { ...validSession, timestamp: '2024-01-01T00:00:00.000Z' };

            const result = await SessionStateManager.hasUnsavedChanges(currentSession, mockContext);

            expect(result).toBe(false);
        });
    });

    describe('migrateSession', () => {
        it('should return session unchanged if already current version', () => {
            const result = SessionStateManager.migrateSession(validSession);

            expect(result).toEqual(validSession);
        });

        it('should update version for older sessions', () => {
            const oldSession = { ...validSession, version: '0.9.0' };

            const result = SessionStateManager.migrateSession(oldSession);

            expect(result.version).toBe('1.0.0');
        });
    });
});
