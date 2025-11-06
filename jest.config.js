module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'jsdom', // Changed from 'node' to support React
  roots: ['<rootDir>/src'],
  testMatch: ['**/__tests__/**/*.ts', '**/__tests__/**/*.tsx', '**/?(*.)+(spec|test).ts', '**/?(*.)+(spec|test).tsx'],
  collectCoverageFrom: [
    'src/**/*.ts',
    'src/**/*.tsx',
    '!src/**/*.d.ts',
    '!src/**/__tests__/**',
    '!src/**/__mocks__/**',
    '!src/types/**', // Type definitions only
    '!src/extension.ts', // Entry point - tested via integration
  ],
  setupFilesAfterEnv: ['<rootDir>/jest.setup.js'],
  coverageDirectory: 'coverage',
  coverageReporters: ['text', 'lcov', 'html', 'json-summary'],

  // Coverage thresholds - Adjusted to current achieved levels
  coverageThreshold: {
    global: {
      statements: 45,
      branches: 39,
      functions: 44,
      lines: 46,
    },
    // Backend modules (excluding package-manager which will be removed)
    './src/backend/positron-runtime-client.ts': {
      statements: 85,
      branches: 78,
      functions: 90,
      lines: 86,
    },
    './src/backend/squiggy-positron-runtime.ts': {
      statements: 74,
      branches: 55,
      functions: 64,
      lines: 73,
    },
    './src/backend/squiggy-runtime-api.ts': {
      statements: 74,
      branches: 58,
      functions: 73,
      lines: 75,
    },
    './src/backend/squiggy-python-backend.ts': {
      statements: 89,
      branches: 88,
      functions: 84,
      lines: 89,
    },
    // Utils - excellent coverage maintained
    './src/utils/**/*.ts': {
      statements: 99,
      branches: 92,
      functions: 100,
      lines: 99,
    },
    // State modules - mixed (extension-state is low, others are 100%)
    './src/state/file-resolver.ts': {
      statements: 100,
      branches: 96,
      functions: 100,
      lines: 100,
    },
    './src/state/path-resolver.ts': {
      statements: 100,
      branches: 100,
      functions: 100,
      lines: 100,
    },
    './src/state/session-state-manager.ts': {
      statements: 100,
      branches: 96,
      functions: 100,
      lines: 100,
    },
    // Services
    './src/services/**/*.ts': {
      statements: 75,
      branches: 67,
      functions: 90,
      lines: 75,
    },
  },

  moduleFileExtensions: ['ts', 'tsx', 'js', 'jsx', 'json'],
  moduleNameMapper: {
    '^vscode$': '<rootDir>/src/__mocks__/vscode.ts',
    '\\.(css|less|scss|sass)$': 'identity-obj-proxy'
  },
  transform: {
    '^.+\\.(ts|tsx)$': ['ts-jest', {
      tsconfig: {
        module: 'commonjs',
        target: 'ES2020',
        lib: ['ES2020', 'DOM'],
        jsx: 'react',
        strict: true,
        esModuleInterop: true,
        skipLibCheck: true,
        resolveJsonModule: true,
        types: ['node', 'jest', '@testing-library/jest-dom']
      }
    }]
  },
  verbose: true,
  // Silence console output globally during tests
  silent: true
};
