/**
 * UX Walkthrough Logger Tests
 */

import { describe, it, expect, beforeEach } from '@jest/globals';
import { WalkthroughLogger } from '../walkthrough-logger';

describe('WalkthroughLogger', () => {
    let logger: WalkthroughLogger;

    beforeEach(() => {
        logger = new WalkthroughLogger();
    });

    describe('Basic Logging', () => {
        it('should start a walkthrough session', () => {
            logger.start('Test Walkthrough');
            const summary = logger.getSummary();

            expect(summary.name).toBe('Test Walkthrough');
            expect(summary.totalSteps).toBe(0);
        });

        it('should log simple events', () => {
            logger.start('Test');
            logger.log('TEST_EVENT', { data: 'test' });

            const report = logger.generateReport();
            expect(report).toContain('TEST_EVENT');
        });

        it('should end a walkthrough session', () => {
            logger.start('Test');
            logger.log('EVENT1');
            logger.end();

            const report = logger.generateReport();
            expect(report).toContain('Completed');
        });
    });

    describe('Command Logging', () => {
        it('should log successful command execution', async () => {
            logger.start('Test');

            await logger.logCommand('test.command', { arg: 'value' }, async () => {
                return 'success';
            });

            const summary = logger.getSummary();
            expect(summary.successCount).toBe(1);
            expect(summary.errorCount).toBe(0);
        });

        it('should log failed command execution', async () => {
            logger.start('Test');

            try {
                await logger.logCommand('test.command', {}, async () => {
                    throw new Error('Test error');
                });
            } catch (error) {
                // Expected
            }

            const summary = logger.getSummary();
            expect(summary.successCount).toBe(0);
            expect(summary.errorCount).toBe(1);
            expect(summary.errors).toContain('Test error');
        });

        it('should capture command duration', async () => {
            logger.start('Test');

            await logger.logCommand('test.command', {}, async () => {
                await new Promise((resolve) => setTimeout(resolve, 50));
                return 'done';
            });

            const report = logger.generateReport();
            expect(report).toMatch(/\d+ms/);
        });
    });

    describe('State and Checks', () => {
        it('should log state captures', () => {
            logger.start('Test');
            logger.logState('Initial State', { items: 5, loaded: true });

            const report = logger.generateReport();
            expect(report).toContain('State: Initial State');
            expect(report).toContain('"items": 5');
        });

        it('should log checks with pass/fail', () => {
            logger.start('Test');
            logger.logCheck('Validation check', true, { expected: 5, actual: 5 });
            logger.logCheck('Failed check', false, { expected: 10, actual: 5 });

            const report = logger.generateReport();
            expect(report).toContain('✓ Check: Validation check');
            expect(report).toContain('✗ Check: Failed check');
        });
    });

    describe('Report Generation', () => {
        it('should generate comprehensive report', async () => {
            logger.start('Full Test');
            logger.log('SETUP', { data: 'test' });

            await logger.logCommand('cmd1', {}, async () => 'result1');
            logger.logState('Mid-point', { progress: 50 });
            await logger.logCommand('cmd2', {}, async () => 'result2');
            logger.logCheck('Final check', true);

            logger.end();

            const report = logger.generateReport();

            expect(report).toContain('Full Test');
            expect(report).toContain('cmd1');
            expect(report).toContain('cmd2');
            expect(report).toContain('Mid-point');
            expect(report).toContain('Final check');
            expect(report).toContain('Completed');
        });

        it('should export JSON', () => {
            logger.start('JSON Test');
            logger.log('EVENT');
            logger.end();

            const json = logger.exportJSON();
            const parsed = JSON.parse(json);

            expect(parsed.name).toBe('JSON Test');
            expect(parsed.events).toBeDefined();
            expect(Array.isArray(parsed.events)).toBe(true);
        });
    });

    describe('Summary Statistics', () => {
        it('should calculate correct summary statistics', async () => {
            logger.start('Stats Test');

            await logger.logCommand('cmd1', {}, async () => 'ok');
            await logger.logCommand('cmd2', {}, async () => 'ok');

            try {
                await logger.logCommand('cmd3', {}, async () => {
                    throw new Error('fail');
                });
            } catch (error) {
                // Expected
            }

            logger.end();

            const summary = logger.getSummary();

            expect(summary.name).toBe('Stats Test');
            expect(summary.totalSteps).toBe(3);
            expect(summary.successCount).toBe(2);
            expect(summary.errorCount).toBe(1);
            expect(summary.errors.length).toBe(1);
        });
    });
});
