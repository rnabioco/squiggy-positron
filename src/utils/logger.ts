/**
 * Centralized logging utility for Squiggy extension
 *
 * Provides a consistent logging interface that writes to:
 * 1. VS Code Output Channel (visible in Output panel → "Squiggy")
 * 2. Developer Tools console (for extension development)
 */

import * as vscode from 'vscode';

/**
 * Log levels matching Python's logging levels
 */
export enum LogLevel {
    DEBUG = 'DEBUG',
    INFO = 'INFO',
    WARNING = 'WARNING',
    ERROR = 'ERROR',
}

/**
 * Centralized logger for Squiggy extension
 *
 * Usage:
 *   import { logger } from '../utils/logger';
 *   logger.info('Loading POD5 file...');
 *   logger.error('Failed to load file', error);
 */
class Logger {
    private outputChannel: vscode.OutputChannel | null = null;
    private extensionName = 'Squiggy';

    /**
     * Initialize the logger with an output channel
     * Called once during extension activation
     */
    public initialize(context: vscode.ExtensionContext): void {
        // Create Output Channel (appears in Output panel dropdown)
        this.outputChannel = vscode.window.createOutputChannel(this.extensionName);
        context.subscriptions.push(this.outputChannel);

        this.info('Squiggy extension activated');
    }

    /**
     * Show the output channel (brings Output panel to front with Squiggy selected)
     */
    public show(): void {
        this.outputChannel?.show();
    }

    /**
     * Log a debug message (only in dev tools, not output channel)
     */
    public debug(message: string, ...args: unknown[]): void {
        this.log(LogLevel.DEBUG, message, ...args);
    }

    /**
     * Log an info message
     */
    public info(message: string, ...args: unknown[]): void {
        this.log(LogLevel.INFO, message, ...args);
    }

    /**
     * Log a warning message
     */
    public warning(message: string, ...args: unknown[]): void {
        this.log(LogLevel.WARNING, message, ...args);
    }

    /**
     * Log an error message
     */
    public error(message: string, error?: unknown): void {
        const errorDetails = error instanceof Error ? error.message : String(error);
        const fullMessage = error ? `${message}: ${errorDetails}` : message;

        this.log(LogLevel.ERROR, fullMessage);

        // Include stack trace in output channel for errors
        if (error instanceof Error && error.stack) {
            this.outputChannel?.appendLine(error.stack);
        }
    }

    /**
     * Internal logging implementation
     */
    private log(level: LogLevel, message: string, ...args: unknown[]): void {
        const timestamp = new Date().toISOString().substring(11, 23); // HH:mm:ss.SSS
        const formattedMessage = `[${timestamp}] [${level}] ${message}`;

        // Write to Output Channel (visible in Output panel)
        if (this.outputChannel) {
            this.outputChannel.appendLine(formattedMessage);
            if (args.length > 0) {
                this.outputChannel.appendLine(`  ${JSON.stringify(args)}`);
            }
        }

        // Also write to Developer Tools console for extension debugging
        switch (level) {
            case LogLevel.DEBUG:
                console.debug(`[${this.extensionName}]`, message, ...args);
                break;
            case LogLevel.INFO:
                console.log(`[${this.extensionName}]`, message, ...args);
                break;
            case LogLevel.WARNING:
                console.warn(`[${this.extensionName}]`, message, ...args);
                break;
            case LogLevel.ERROR:
                console.error(`[${this.extensionName}]`, message, ...args);
                break;
        }
    }

    /**
     * Log Python output (from kernel execution)
     * Separate method to distinguish Python vs TypeScript logs
     */
    public python(message: string, level: LogLevel = LogLevel.INFO): void {
        const timestamp = new Date().toISOString().substring(11, 23);
        const formattedMessage = `[${timestamp}] [Python ${level}] ${message}`;

        if (this.outputChannel) {
            this.outputChannel.appendLine(formattedMessage);
        }
    }

    /**
     * Clear the output channel
     */
    public clear(): void {
        this.outputChannel?.clear();
    }
}

// Export singleton instance
export const logger = new Logger();
