# Development with Claude Code

Squiggy is optimized for development with [Claude Code](https://claude.ai/code). The repository includes comprehensive documentation to help AI-assisted development be more effective.

## Project Documentation

The repository includes **CLAUDE.md** - a comprehensive project guide that provides:

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
squiggy -p tests/data/mod_reads.pod5 -b tests/data/mod_mappings.bam
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

See [CONTRIBUTING.md](https://github.com/rnabioco/squiggy/blob/main/CONTRIBUTING.md) for more details on code style, testing, and PR guidelines.

## Requirements

### Runtime Requirements
- Python 3.8 or higher (for source installation)
- For POD5 files with VBZ compression: `vbz_h5py_plugin` (automatically installed)

### Dependencies
- `numpy>=1.20.0` - Array operations
- `pandas>=1.3.0` - Data manipulation
- `pod5>=0.3.0` - POD5 file reading
- `PySide6>=6.5.0` - GUI framework
- `plotnine>=0.12.0` - Plot generation (deprecated, migrating to Bokeh)
- `bokeh>=3.0.0` - Interactive visualization
- `qasync>=0.24.0` - Async/await support for Qt

## Development Resources

For detailed development information, see:

- [CLAUDE.md](https://github.com/rnabioco/squiggy/blob/main/CLAUDE.md) - Comprehensive development guide
- [CONTRIBUTING.md](https://github.com/rnabioco/squiggy/blob/main/CONTRIBUTING.md) - Contribution guidelines
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

To build standalone executables for testing:

```bash
# Install PyInstaller (using uv - recommended)
uv pip install pyinstaller

# Or using pip
pip install pyinstaller

# Build for your current platform (run from build/ directory)
cd build
pyinstaller squiggy.spec

# Output will be in dist/ directory
```

### Platform-Specific Packaging

For creating distributable packages:

```bash
# macOS: Create DMG
brew install create-dmg
cd build
pyinstaller squiggy.spec
create-dmg --volname "Squiggy Installer" --app-drop-link 600 185 Squiggy-macos.dmg dist/Squiggy.app

# Windows: Create ZIP
cd build
pyinstaller squiggy.spec
Compress-Archive -Path dist/Squiggy.exe -DestinationPath Squiggy-windows-x64.zip

# Linux: Create tarball
cd build
pyinstaller squiggy.spec
tar -czf Squiggy-linux-x64.tar.gz -C dist Squiggy
```

**Note:** The build process is automated via GitHub Actions for releases. See below for the release process.

## Release Process

Releases are automated via GitHub Actions (`.github/workflows/build.yml`):

- **Version Releases**: Triggered on version tags (e.g., `v0.1.0`)
- **Development Builds**: Triggered on pushes to `main` branch (macOS only)
- Builds executables for Windows, macOS, and Linux in parallel
- Creates GitHub release with platform-specific installers attached
- Release artifacts: `.zip` (Windows), `.dmg` (macOS), `.tar.gz` (Linux)

### Creating a Release

1. Update version in `src/squiggy/__init__.py` and `pyproject.toml`
2. Commit changes: `git add src/squiggy/__init__.py pyproject.toml`
3. Commit: `git commit -m "Bump version to 0.1.0"`
4. Create tag: `git tag v0.1.0`
5. Push tag: `git push origin v0.1.0`
6. GitHub Actions will automatically build executables for all platforms
7. A new release will be created with all platform binaries attached

### Development Builds

Development builds are automatically created on every push to `main` and available under the ["latest" release tag](https://github.com/rnabioco/squiggy/releases/tag/latest). These builds include only macOS .dmg files for faster CI builds.
