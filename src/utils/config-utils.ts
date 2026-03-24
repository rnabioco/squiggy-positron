/**
 * Workspace Configuration Utilities
 *
 * Provides workspace config updates with fallback to ~/.cache/squiggy/
 * when the workspace directory is read-only (e.g., remote SSH to another user's project).
 */

import * as vscode from 'vscode';
import * as fs from 'fs/promises';
import * as path from 'path';
import * as os from 'os';
import * as crypto from 'crypto';
import { logger } from './logger';

/**
 * Get the fallback config file path for the current workspace.
 * Uses a hash of the first workspace folder URI to create a unique filename.
 */
export function getWorkspaceFallbackPath(): string {
    const folders = vscode.workspace.workspaceFolders;
    const key = folders?.[0]?.uri.toString() ?? 'no-workspace';
    const hash = crypto.createHash('sha256').update(key).digest('hex').slice(0, 16);
    return path.join(os.homedir(), '.cache', 'squiggy', `workspace-${hash}.json`);
}

/**
 * Read the fallback config file, returning an empty object if it doesn't exist.
 */
async function readFallbackConfig(): Promise<Record<string, unknown>> {
    const fallbackPath = getWorkspaceFallbackPath();
    try {
        const json = await fs.readFile(fallbackPath, 'utf-8');
        return JSON.parse(json);
    } catch {
        return {};
    }
}

/**
 * Write to the fallback config file, merging with existing values.
 */
async function writeFallbackConfig(key: string, value: unknown): Promise<void> {
    const fallbackPath = getWorkspaceFallbackPath();
    const dir = path.dirname(fallbackPath);

    await fs.mkdir(dir, { recursive: true });

    const existing = await readFallbackConfig();
    existing[key] = value;

    await fs.writeFile(fallbackPath, JSON.stringify(existing, null, 2), 'utf-8');
}

/**
 * Update a workspace-scoped squiggy configuration value.
 *
 * Tries ConfigurationTarget.Workspace first. On failure (e.g., read-only directory),
 * falls back to writing to ~/.cache/squiggy/workspace-<hash>.json and notifies the user.
 */
export async function updateWorkspaceConfig(
    key: string,
    value: string | boolean | number
): Promise<void> {
    const config = vscode.workspace.getConfiguration('squiggy');
    try {
        await config.update(key, value, vscode.ConfigurationTarget.Workspace);
    } catch (error) {
        const fallbackPath = getWorkspaceFallbackPath();
        logger.warning(
            `Cannot write workspace settings (read-only?), falling back to ${fallbackPath}: ${error}`
        );

        await writeFallbackConfig(`squiggy.${key}`, value);

        vscode.window.showWarningMessage(
            `Squiggy: Workspace settings are read-only. Configuration saved to ${fallbackPath} instead.`
        );
    }
}

/**
 * Get an effective config value, checking workspace config first, then fallback file.
 */
export async function getEffectiveConfig<T>(key: string, defaultValue: T): Promise<T> {
    const config = vscode.workspace.getConfiguration('squiggy');
    const workspaceValue = config.inspect<T>(key);

    // If there's an explicit workspace or workspace folder value, use it
    if (
        workspaceValue?.workspaceValue !== undefined ||
        workspaceValue?.workspaceFolderValue !== undefined
    ) {
        return config.get<T>(key, defaultValue);
    }

    // Check fallback file
    const fallback = await readFallbackConfig();
    const fallbackKey = `squiggy.${key}`;
    if (fallbackKey in fallback) {
        return fallback[fallbackKey] as T;
    }

    return config.get<T>(key, defaultValue);
}
