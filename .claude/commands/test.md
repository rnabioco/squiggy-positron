# description: Run ruff checks and test suite

Run the following quality checks:

1. Run `ruff check --fix src/ tests/` to check and auto-fix linting issues
2. Run `ruff format src/ tests/ build/squiggy.spec` to format the code
3. Run `uv run pytest tests/ -v` to run the full test suite

Report the results of each step, including:
- Number of linting issues found and fixed
- Test results (passed/failed/skipped)
- Any errors or warnings that need attention
