/**
 * Mock logger for testing
 */

export const logger = {
    initialize: jest.fn(),
    show: jest.fn(),
    debug: jest.fn(),
    info: jest.fn(),
    warning: jest.fn(),
    error: jest.fn(),
    python: jest.fn(),
    clear: jest.fn(),
    setMinLevel: jest.fn(),
    getMinLevel: jest.fn(() => 'INFO'),
};

export enum LogLevel {
    DEBUG = 'DEBUG',
    INFO = 'INFO',
    WARNING = 'WARNING',
    ERROR = 'ERROR',
}
