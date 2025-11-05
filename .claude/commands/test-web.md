# description: Run quality checks and test suites (Python + TypeScript) - Web compatible

Run the following quality checks directly (works in Claude Code Web without pixi):

## Quick Check (Recommended)

1. **Format and Lint TypeScript**
   - Run `npm run format` to format TypeScript code
   - Run `npm run lint` to check TypeScript linting

2. **Format and Lint Python**
   - Run `python3.12 -m ruff format squiggy/ tests/` to format Python code
   - Run `python3.12 -m ruff check squiggy/ tests/` to check Python linting

3. **Run Tests**
   - Run `npm test` to run TypeScript tests
   - Run `python3.12 -m pytest tests/ -v` to run Python tests

## Detailed Checks (if needed)

### TypeScript Only
- Run `npm run lint` to check TypeScript linting
- Run `npm run format:check` to verify formatting without changes
- Run `npm test` to run TypeScript tests only
- Run `npm run test:coverage` for coverage report

### Python Only
- Run `python3.12 -m ruff check squiggy/ tests/` to check Python linting
- Run `python3.12 -m ruff format squiggy/ tests/` to format Python code
- Run `python3.12 -m pytest tests/ -v` to run Python tests with verbose output
- Run `python3.12 -m pytest tests/ --cov=squiggy --cov-report=html` for coverage

## Build Check
- Run `npm run compile` to ensure extension builds successfully

Report the results of each step, including:
- Number of linting issues found and fixed (Python and TypeScript)
- Test results (passed/failed/skipped) for both test suites
- Build status
- Any errors or warnings that need attention
- Overall pass/fail status

**Note:** This command uses direct npm and python commands instead of pixi, making it compatible with Claude Code Web environments.
