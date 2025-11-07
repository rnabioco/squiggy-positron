/**
 * UX Walkthrough Logger
 *
 * Captures and logs user interactions for automated UX walkthroughs
 */

export interface WalkthroughEvent {
    timestamp: number;
    step: number;
    action: string;
    command?: string;
    params?: any;
    result?: 'success' | 'error';
    duration?: number;
    error?: string;
    state?: any;
}

export class WalkthroughLogger {
    private events: WalkthroughEvent[] = [];
    private currentStep = 0;
    private startTime = 0;
    private walkthroughName = '';

    /**
     * Start a new walkthrough session
     */
    start(name: string): void {
        this.walkthroughName = name;
        this.startTime = Date.now();
        this.currentStep = 0;
        this.events = [];
        this.log('START', { name });
    }

    /**
     * Log an action event
     */
    log(action: string, params?: any): void {
        const event: WalkthroughEvent = {
            timestamp: Date.now() - this.startTime,
            step: this.currentStep,
            action,
            params,
        };
        this.events.push(event);
    }

    /**
     * Log a command execution
     */
    async logCommand<T>(
        command: string,
        params: any,
        executor: () => Promise<T>
    ): Promise<T> {
        this.currentStep++;
        const stepStart = Date.now();

        this.log('COMMAND_START', { command, params });

        try {
            const result = await executor();
            const duration = Date.now() - stepStart;

            this.log('COMMAND_SUCCESS', {
                command,
                duration,
                result: typeof result === 'string' ? result.substring(0, 100) : result,
            });

            return result;
        } catch (error) {
            const duration = Date.now() - stepStart;
            const errorMessage = error instanceof Error ? error.message : String(error);

            this.log('COMMAND_ERROR', {
                command,
                duration,
                error: errorMessage,
            });

            throw error;
        }
    }

    /**
     * Log state capture
     */
    logState(description: string, state: any): void {
        this.log('STATE_CAPTURE', { description, state });
    }

    /**
     * Log an assertion/check
     */
    logCheck(description: string, passed: boolean, details?: any): void {
        this.log('CHECK', {
            description,
            passed,
            details,
        });
    }

    /**
     * End the walkthrough session
     */
    end(): void {
        const totalDuration = Date.now() - this.startTime;
        this.log('END', {
            name: this.walkthroughName,
            totalSteps: this.currentStep,
            totalDuration,
        });
    }

    /**
     * Generate formatted report
     */
    generateReport(): string {
        const lines: string[] = [];
        lines.push('='.repeat(80));
        lines.push(`UX Walkthrough Report: ${this.walkthroughName}`);
        lines.push('='.repeat(80));
        lines.push('');

        let currentCommand = '';
        let commandStartTime = 0;

        for (const event of this.events) {
            const time = `[${(event.timestamp / 1000).toFixed(3)}s]`;

            if (event.action === 'START') {
                lines.push(`${time} 🚀 Started: ${event.params?.name}`);
                lines.push('');
            } else if (event.action === 'COMMAND_START') {
                currentCommand = event.params?.command || 'unknown';
                commandStartTime = event.timestamp;
                lines.push(
                    `${time} Step ${event.step}: ${currentCommand}`
                );
                if (event.params?.params && Object.keys(event.params.params).length > 0) {
                    lines.push(`       Parameters: ${JSON.stringify(event.params.params, null, 2).split('\n').join('\n       ')}`);
                }
            } else if (event.action === 'COMMAND_SUCCESS') {
                const duration = event.params?.duration || 0;
                lines.push(`       ✅ Success (${duration}ms)`);
                if (event.params?.result !== undefined && event.params?.result !== null) {
                    const resultStr =
                        typeof event.params.result === 'object'
                            ? JSON.stringify(event.params.result, null, 2)
                            : String(event.params.result);
                    lines.push(`       Result: ${resultStr.substring(0, 200)}${resultStr.length > 200 ? '...' : ''}`);
                }
                lines.push('');
            } else if (event.action === 'COMMAND_ERROR') {
                const duration = event.params?.duration || 0;
                lines.push(`       ❌ Error (${duration}ms)`);
                lines.push(`       ${event.params?.error}`);
                lines.push('');
            } else if (event.action === 'STATE_CAPTURE') {
                lines.push(`${time} 📸 State: ${event.params?.description}`);
                lines.push(`       ${JSON.stringify(event.params?.state, null, 2).split('\n').join('\n       ')}`);
                lines.push('');
            } else if (event.action === 'CHECK') {
                const icon = event.params?.passed ? '✓' : '✗';
                lines.push(
                    `${time} ${icon} Check: ${event.params?.description}`
                );
                if (event.params?.details) {
                    lines.push(`       ${JSON.stringify(event.params.details)}`);
                }
                lines.push('');
            } else if (event.action === 'END') {
                const totalDuration = event.params?.totalDuration || 0;
                lines.push('='.repeat(80));
                lines.push(
                    `✨ Completed: ${event.params?.totalSteps} steps in ${(totalDuration / 1000).toFixed(2)}s`
                );
                lines.push('='.repeat(80));
            } else {
                // Generic event
                lines.push(`${time} ${event.action}: ${JSON.stringify(event.params)}`);
                lines.push('');
            }
        }

        return lines.join('\n');
    }

    /**
     * Export events as JSON
     */
    exportJSON(): string {
        return JSON.stringify(
            {
                name: this.walkthroughName,
                startTime: this.startTime,
                events: this.events,
            },
            null,
            2
        );
    }

    /**
     * Get summary statistics
     */
    getSummary(): {
        name: string;
        totalSteps: number;
        totalDuration: number;
        successCount: number;
        errorCount: number;
        errors: string[];
    } {
        const endEvent = this.events.find((e) => e.action === 'END');
        const successCount = this.events.filter((e) => e.action === 'COMMAND_SUCCESS').length;
        const errorEvents = this.events.filter((e) => e.action === 'COMMAND_ERROR');

        return {
            name: this.walkthroughName,
            totalSteps: this.currentStep,
            totalDuration: endEvent?.params?.totalDuration || 0,
            successCount,
            errorCount: errorEvents.length,
            errors: errorEvents.map((e) => e.params?.error || 'Unknown error'),
        };
    }
}
