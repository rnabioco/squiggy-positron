/**
 * Formatting utilities for displaying file sizes and other data
 *
 * Consolidates formatting logic used across views.
 */

/**
 * Format bytes into human-readable file size
 *
 * @param bytes - File size in bytes
 * @returns Formatted string (e.g., "1.5 MB")
 */
export function formatFileSize(bytes: number): string {
    if (bytes < 1024) {
        return `${bytes} B`;
    } else if (bytes < 1024 * 1024) {
        return `${(bytes / 1024).toFixed(1)} KB`;
    } else if (bytes < 1024 * 1024 * 1024) {
        return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    } else {
        return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
    }
}

/**
 * Format read count into human-readable string
 *
 * @param count - Number of reads
 * @returns Formatted string (e.g., "1,234 reads")
 */
export function formatReadCount(count: number): string {
    return `${count.toLocaleString()} read${count !== 1 ? 's' : ''}`;
}

/**
 * Truncate file path for display in UI
 *
 * @param filePath - Full file path
 * @param maxLength - Maximum length before truncation (default 50)
 * @returns Truncated path with ellipsis if needed
 */
export function truncatePath(filePath: string, maxLength: number = 50): string {
    if (filePath.length <= maxLength) {
        return filePath;
    }

    const parts = filePath.split('/');
    const filename = parts[parts.length - 1];

    // If filename alone is too long, truncate it
    if (filename.length >= maxLength - 3) {
        return '...' + filename.slice(-(maxLength - 3));
    }

    // Otherwise, show ".../<last few dirs>/<filename>"
    let result = filename;
    for (let i = parts.length - 2; i >= 0; i--) {
        const candidate = parts[i] + '/' + result;
        if (candidate.length + 3 > maxLength) {
            break;
        }
        result = candidate;
    }

    return '.../' + result;
}
