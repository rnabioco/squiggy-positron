# description: Run quality checks and test suites (Python + TypeScript)

Run the following quality checks using pixi:

## Quick Check (Recommended)

1. Run `pixi run lint` to check Python + TypeScript linting
2. Run `pixi run format` to format Python + TypeScript code
3. Run `pixi run test` to run all tests (Python + TypeScript)

## Detailed Checks (if needed)

### Python Only
- Run `pixi run lint-py` to check Python linting
- Run `pixi run format-py` to format Python code
- Run `pixi run test-py` to run Python tests only

### TypeScript Only
- Run `pixi run lint-ts` to check TypeScript linting
- Run `pixi run format-ts` to format TypeScript code
- Run `pixi run test-ts` to run TypeScript tests only

Report the results of each step, including:
- Number of linting issues found and fixed (Python and TypeScript)
- Test results (passed/failed/skipped) for both test suites
- Any errors or warnings that need attention
- Overall pass/fail status
