# Contributing to Squiggy

Thank you for your interest in contributing to Squiggy! This guide covers everything you need to know about developing and contributing to the project.

## Using Claude Code

Squiggy is optimized for development with [Claude Code](https://claude.ai/code). The repository includes **CLAUDE.md** - a comprehensive project guide that provides:

- Detailed architecture documentation and coding conventions
- Project structure and component descriptions
- Common development tasks and debugging tips
- Context about async/Qt patterns and data flow
- Known issues and gotchas

Contributors using Claude Code will benefit from rich context about the codebase structure, testing strategies, and implementation patterns.

## Quick Contributor Setup

### Prerequisites

Before starting, ensure you have Git LFS installed to download test data files:

```bash
# Install and initialize Git LFS (if not already installed)
git lfs install
```

### Setup Steps

```bash
# Clone and pull LFS files
git clone https://github.com/rnabioco/squiggy.git
cd squiggy
git lfs pull

# Setup environment (using uv - recommended)
uv pip install -e ".[dev]"

# macOS users also install
uv pip install -e ".[macos]"

# Or using pip
pip install -e ".[dev]"
pip install -e ".[macos]"  # macOS only

# Validate setup
python scripts/check_dev_env.py

# Run with sample data
squiggy -p tests/data/yeast_trna_reads.pod5 -b tests/data/yeast_trna_mappings.bam
```

### Start Developing

Once your environment is set up, you can start developing. If using Claude Code, it will automatically use CLAUDE.md for context about the project architecture, conventions, and best practices.

## Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes in `src/squiggy/`
4. Format code: `ruff format src/ tests/`
5. Check linting: `ruff check --fix src/ tests/`
6. Run tests: `uv run pytest tests/`
7. Commit your changes (`git commit -m 'feat: add amazing feature'`)
8. Push to the branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request


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

## Requirements

### Runtime Requirements
- Python 3.12 or higher (for source installation)
- For POD5 files with VBZ compression: `vbz_h5py_plugin` (automatically installed)

### Dependencies
- `numpy>=1.20.0` - Array operations and signal processing
- `pod5>=0.3.0` - POD5 file reading
- `PySide6>=6.5.0` - GUI framework (Qt for Python)
- `bokeh>=3.1.1` - Interactive visualization
- `pysam>=0.20.0` - BAM file reading
- `qasync>=0.24.0` - Async/await support for Qt

## Development Resources

For detailed development information, see:

- [CLAUDE.md](https://github.com/rnabioco/squiggy/blob/main/CLAUDE.md) - Comprehensive development guide
- [GitHub Issues](https://github.com/rnabioco/squiggy/issues) - Bug reports and feature requests
- [GitHub Discussions](https://github.com/rnabioco/squiggy/discussions) - Community discussions

## Key Development Patterns

### Async Programming with qasync

Squiggy uses `qasync` to integrate Python's `asyncio` with Qt's event loop. When adding new features:

- Use `async def` for methods that perform file I/O or heavy computation
- Decorate async slot handlers with `@qasync.asyncSlot()`
- Move blocking operations to separate methods and call with `asyncio.to_thread()`
- Always update UI elements on the main thread after awaiting async operations

### UI Development

- Keep GUI code (`viewer.py`, `dialogs.py`) separate from logic (`plotter_bokeh.py`, `utils.py`)
- Use CollapsibleBox widgets for organizing expandable sections
- Follow the three-panel splitter layout pattern
- Use Okabe-Ito colorblind-friendly palette for data visualization

### Testing

- Tests require sample POD5 files in `tests/data/` (downloaded via Git LFS)
- Use `conftest.py` fixtures for shared test resources
- Tests skip (not fail) if sample data is missing via `pytest.skip()`
- Test both async functionality and UI components

## Code Style

Squiggy follows PEP 8 style guidelines enforced by `ruff`:

```bash
# Format before committing
ruff format src/ tests/

# Auto-fix linting issues
ruff check --fix src/ tests/
```

Line length: 88 characters (Black-compatible)

## Building Executables

### Local Development Builds

To build the macOS application for testing:

```bash
# Install development dependencies (includes PyInstaller)
uv pip install -e ".[dev]"

# Build for macOS (run from build/ directory)
cd build
pyinstaller squiggy.spec

# Output will be in dist/Squiggy.app
```

### Creating a DMG Installer

For creating a distributable DMG package:

```bash
# macOS: Create DMG
brew install create-dmg
cd build
pyinstaller squiggy.spec
create-dmg \
  --volname "Squiggy Installer" \
  --window-pos 200 120 \
  --window-size 800 400 \
  --icon-size 100 \
  --app-drop-link 600 185 \
  Squiggy-macos.dmg \
  dist/Squiggy.app
```

**Note:** The build process is automated via GitHub Actions for releases. See below for the release process.

## Release Process

Releases are automated via GitHub Actions (`.github/workflows/build.yml`):

- **Version Releases**: Triggered on version tags (e.g., `v0.1.0`)
- **Development Builds**: Triggered on pushes to `main` branch
- Builds macOS .dmg installer
- Creates GitHub release with installer attached
- Release artifact: `.dmg` (macOS)

### Creating a Release

1. Update version in `src/squiggy/__init__.py` and `pyproject.toml`
2. Commit changes: `git add src/squiggy/__init__.py pyproject.toml`
3. Commit: `git commit -m "Bump version to 0.1.0"`
4. Create tag: `git tag v0.1.0`
5. Push tag: `git push origin v0.1.0`
6. GitHub Actions will automatically build the macOS installer
7. A new release will be created with the .dmg attached

### Development Builds

Development builds are automatically created on every push to `main` and available under the ["latest" release tag](https://github.com/rnabioco/squiggy/releases/tag/latest).

## Common Development Tasks

### Adding a New Dialog

1. Create dialog class in `dialogs.py` inheriting from `QDialog`
2. Add dialog instantiation in `viewer.py`
3. Connect to menu action or button
4. Test with sample data

### Adding a New Plot Mode

1. Add enum value to `PlotMode` in `constants.py`
2. Implement plotting method in `plotter_bokeh.py`
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
- The app sets the correct name via `set_macos_app_name()` in `main.py`

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
- Test builds on macOS before releasing

## Questions?

- **Bugs**: Open an issue on [GitHub](https://github.com/rnabioco/squiggy/issues)
- **Feature requests**: Open a discussion or issue
- **Development questions**: Use Claude Code with the CLAUDE.md context

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
