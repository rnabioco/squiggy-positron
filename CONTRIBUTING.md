# Contributing to Squiggy

Thank you for your interest in contributing to Squiggy! This document provides guidelines for contributors.

## Getting Started

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/rnabioco/squiggy.git
cd squiggy

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# On macOS, also install:
pip install -e ".[macos]"

# Run the application
squiggy -p tests/data/simplex_reads.pod5
```

### Development Workflow

1. **Create a feature branch**: `git checkout -b feature/my-feature`
2. **Make your changes**: Edit code in `src/squiggy/`
3. **Format code**: `ruff format src/ tests/`
4. **Check linting**: `ruff check --fix src/ tests/`
5. **Run tests**: `pytest tests/`
6. **Test manually**: `squiggy -p tests/data/simplex_reads.pod5 -b tests/data/simplex_reads_mapped.bam`
7. **Commit changes**: `git commit -m "feat: add new feature"`
8. **Push and create PR**: `git push origin feature/my-feature`

## Using Claude Code

This project is optimized for use with [Claude Code](https://claude.ai/code). The `CLAUDE.md` file provides detailed guidance for AI-assisted development.

### Tips for Claude Code Users

- Ask Claude to read `CLAUDE.md` first for project context
- Claude knows about the async/Qt architecture and plotting system
- Use Claude for: refactoring, adding features, debugging, writing tests
- Sample commands:
  - "Add a new normalization method to the plotter"
  - "Fix the async loading issue in viewer.py"
  - "Write tests for the reference browser dialog"

## Code Style

- **Python**: PEP 8 via `ruff` (configured in `pyproject.toml`)
- **Line length**: 88 characters (Black-compatible)
- **Type hints**: Use Python 3.8+ compatible type hints
- **Docstrings**: Google-style docstrings for public APIs
- **Async**: Use `@qasync.asyncSlot()` for Qt slots with blocking operations

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=squiggy tests/

# Run specific test file
pytest tests/test_plotting.py

# Run verbose
pytest -v
```

### Sample Data

Test data is located in `tests/data/`:
- `simplex_reads.pod5` - Small POD5 file (~10 reads)
- `simplex_reads_mapped.bam` - BAM file with alignments
- Use these files for manual testing and automated tests

## Pull Request Guidelines

### Before Submitting

- [ ] Code is formatted with `ruff format`
- [ ] Linting passes: `ruff check src/ tests/`
- [ ] All tests pass: `pytest tests/`
- [ ] Manual testing completed with sample data
- [ ] New features have tests
- [ ] Documentation updated if needed

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe how you tested these changes

## Screenshots (if UI changes)
Add screenshots demonstrating the changes
```

## Architecture Overview

### Key Files

- `src/squiggy/main.py` - Entry point and CLI
- `src/squiggy/viewer.py` - Main window (1150 lines)
- `src/squiggy/plotter.py` - Plotting logic
- `src/squiggy/dialogs.py` - Dialog windows (About, Reference Browser)
- `src/squiggy/utils.py` - Utility functions (POD5/BAM I/O)
- `src/squiggy/constants.py` - Configuration and enums
- `src/squiggy/alignment.py` - BAM alignment parsing

### Async Architecture

Squiggy uses `qasync` to integrate Python's `asyncio` with Qt's event loop:
- All blocking I/O (file reading, plotting) runs in background threads via `asyncio.to_thread()`
- UI methods that trigger blocking work are decorated with `@qasync.asyncSlot()`
- This keeps the UI responsive during long operations

### Data Flow

```
POD5 File → pod5.Reader → signal data → SquigglePlotter → plotnine → PNG buffer → QPixmap → QLabel
     ↓
BAM File → pysam → alignment info → base annotations → overlaid on plot
```

## Common Tasks

### Adding a New Dialog

1. Create dialog class in `dialogs.py` inheriting from `QDialog`
2. Add dialog instantiation in `viewer.py`
3. Connect to menu action or button
4. Test with sample data

### Adding a New Plot Type

1. Add enum value to `PlotMode` in `constants.py`
2. Implement plotting method in `plotter.py`
3. Add radio button in `viewer.py` plot options panel
4. Connect to `set_plot_mode()` handler
5. Test with multiple reads

### Modifying File Parsing

1. Update parsing logic in `utils.py`
2. Ensure functions are blocking (not async) - called via `asyncio.to_thread()`
3. Add error handling and validation
4. Write tests in `tests/`

## Troubleshooting

### macOS: App shows "Python" in menu bar
- Install PyObjC: `pip install -e ".[macos]"`
- The app now sets the correct name via `set_macos_app_name()` in `main.py`

### Tests fail with "Sample data not found"
- Ensure you're in the project root when running tests
- Check that `tests/data/*.pod5` files exist
- Tests skip (not fail) if sample data is missing

### Qt Async Issues
- Ensure `@qasync.asyncSlot()` is used for async Qt slots
- Blocking operations must be wrapped in `asyncio.to_thread()`
- Don't call Qt methods from background threads

### PyInstaller Build Issues
- Update `build/squiggy.spec` `hiddenimports` for missing modules
- Ensure data files are listed in `datas` parameter
- Test builds on each platform (macOS, Windows, Linux)

## Questions?

- **Bugs**: Open an issue on GitHub
- **Feature requests**: Open a discussion or issue
- **Development questions**: Use Claude Code with the CLAUDE.md context

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
