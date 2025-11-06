module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  roots: ['<rootDir>/src'],
  testMatch: ['**/__tests__/**/*.ts', '**/?(*.)+(spec|test).ts'],
  collectCoverageFrom: [
    'src/**/*.ts',
    '!src/**/*.d.ts',
    '!src/**/__tests__/**',
    '!src/**/__mocks__/**',
    '!src/types/**', // Type definitions only
    '!src/extension.ts', // Entry point - tested via integration
  ],
  coverageDirectory: 'coverage',
  coverageReporters: ['text', 'lcov', 'html', 'json-summary'],

  // Coverage thresholds - Phase 3-5 goals
  coverageThreshold: {
    global: {
      statements: 60,
      branches: 55,
      functions: 60,
      lines: 60,
    },
    // Critical modules with higher thresholds
    './src/backend/**/*.ts': {
      statements: 75,
      branches: 70,
      functions: 75,
      lines: 75,
    },
    './src/utils/**/*.ts': {
      statements: 80,
      branches: 75,
      functions: 80,
      lines: 80,
    },
    './src/state/**/*.ts': {
      statements: 75,
      branches: 70,
      functions: 75,
      lines: 75,
    },
    './src/services/**/*.ts': {
      statements: 80,
      branches: 75,
      functions: 80,
      lines: 80,
    },
  },

  moduleFileExtensions: ['ts', 'js', 'json'],
  moduleNameMapper: {
    '^vscode$': '<rootDir>/src/__mocks__/vscode.ts'
  },
  transform: {
    '^.+\\.ts$': ['ts-jest', {
      tsconfig: {
        module: 'commonjs',
        target: 'ES2020',
        lib: ['ES2020'],
        strict: true,
        esModuleInterop: true,
        skipLibCheck: true,
        resolveJsonModule: true,
        types: ['node', 'jest']
      }
    }]
  },
  verbose: true,
  // Silence console output globally during tests
  silent: true
};
