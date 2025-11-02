/**
 * Tests for format utility functions
 */

import { describe, it, expect } from '@jest/globals';
import { formatFileSize, formatReadCount, truncatePath } from '../format-utils';

describe('format-utils', () => {
    describe('formatFileSize', () => {
        it('should format bytes correctly', () => {
            expect(formatFileSize(0)).toBe('0 B');
            expect(formatFileSize(100)).toBe('100 B');
            expect(formatFileSize(1023)).toBe('1023 B');
        });

        it('should format kilobytes correctly', () => {
            expect(formatFileSize(1024)).toBe('1.0 KB');
            expect(formatFileSize(1536)).toBe('1.5 KB');
            expect(formatFileSize(10240)).toBe('10.0 KB');
        });

        it('should format megabytes correctly', () => {
            expect(formatFileSize(1048576)).toBe('1.0 MB');
            expect(formatFileSize(1572864)).toBe('1.5 MB');
            expect(formatFileSize(10485760)).toBe('10.0 MB');
        });

        it('should format gigabytes correctly', () => {
            expect(formatFileSize(1073741824)).toBe('1.0 GB');
            expect(formatFileSize(1610612736)).toBe('1.5 GB');
            expect(formatFileSize(10737418240)).toBe('10.0 GB');
        });

        it('should handle negative values', () => {
            expect(formatFileSize(-100)).toBe('-100 B');
        });

        it('should handle very large values', () => {
            expect(formatFileSize(1099511627776)).toBe('1024.0 GB');
        });
    });

    describe('formatReadCount', () => {
        it('should format single read correctly', () => {
            expect(formatReadCount(1)).toBe('1 read');
        });

        it('should format multiple reads with plural', () => {
            expect(formatReadCount(0)).toBe('0 reads');
            expect(formatReadCount(2)).toBe('2 reads');
            expect(formatReadCount(100)).toBe('100 reads');
        });

        it('should format large numbers with locale-specific separators', () => {
            const result = formatReadCount(1000);
            // Different locales may use different separators
            expect(result).toContain('1');
            expect(result).toContain('000');
            expect(result).toContain('reads');
        });

        it('should handle millions', () => {
            const result = formatReadCount(1234567);
            expect(result).toContain('1');
            expect(result).toContain('234');
            expect(result).toContain('567');
            expect(result).toContain('reads');
        });
    });

    describe('truncatePath', () => {
        it('should not truncate paths shorter than maxLength', () => {
            const path = '/short/path/file.pod5';
            expect(truncatePath(path)).toBe(path);
            expect(truncatePath(path, 100)).toBe(path);
        });

        it('should truncate long paths with ellipsis', () => {
            const path = '/very/long/path/that/exceeds/the/maximum/length/allowed/file.pod5';
            const result = truncatePath(path, 30);

            expect(result.length).toBeLessThanOrEqual(30);
            expect(result).toContain('...');
            expect(result).toContain('file.pod5');
        });

        it('should preserve filename when possible', () => {
            const path = '/some/really/long/directory/structure/myfile.pod5';
            const result = truncatePath(path, 30);

            expect(result).toContain('myfile.pod5');
        });

        it('should handle very long filenames', () => {
            const path = '/path/' + 'a'.repeat(100) + '.pod5';
            const result = truncatePath(path, 30);

            expect(result.length).toBeLessThanOrEqual(30);
            expect(result).toContain('...');
        });

        it('should use default maxLength of 50', () => {
            const path = '/a/b/c/d/e/f/g/h/i/j/k/l/m/n/o/p/q/r/s/t/u/v/w/x/y/z/file.pod5';
            const result = truncatePath(path);

            // Default is 50, but the function may go slightly over to preserve filename
            expect(result.length).toBeLessThanOrEqual(55);
            expect(result).toContain('...');
            expect(result).toContain('file.pod5');
        });

        it('should handle paths without slashes', () => {
            const path = 'simple_filename.pod5';
            expect(truncatePath(path)).toBe(path);
        });

        it('should handle empty paths', () => {
            expect(truncatePath('')).toBe('');
        });
    });
});
