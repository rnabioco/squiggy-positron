# Squiggy Extension - Developer Guide

This guide covers setting up the development environment, understanding the architecture, and contributing to the Squiggy Positron extension.

## Prerequisites

- **Positron IDE** (version 2025.6.0+) - Download from [Positron releases](https://github.com/posit-dev/positron/releases)
- **Git** with Git LFS enabled

**Choose ONE environment manager:**

- **pixi** (recommended for complete environment) - `brew install pixi` or see [pixi installation](https://pixi.sh)
  - Manages both Python + Node.js in one tool
  - Reproducible via `pixi.lock`

- **uv** (recommended for Python-only) - Fast, modern Python package manager
  - Install: `curl -LsSf https://astral.sh/uv/install.sh | sh` (macOS/Linux)
  - Or: `brew install uv` (macOS)
  - See [uv installation](https://github.com/astral-sh/uv)
  - You'll still need to install Node.js separately (via Homebrew, nvm, etc.)

## Initial Setup

### 1. Clone Repository

```bash
git clone https://github.com/rnabioco/squiggy.git
cd squiggy

# Initialize Git LFS for test data
git lfs install
git lfs pull
```

### 2. Install All Dependencies

**Option A: Using pixi (recommended for complete environment)**

```bash
# Install Python + Node.js environments, then npm packages
pixi install && pixi run setup
```

This installs:
- **pixi install**: Python 3.12, Node.js 20+, packages from conda-forge (numpy, pytest, ruff, mkdocs), and PyPI packages (pod5, pysam, bokeh, selenium)
- **pixi run setup**: npm packages (TypeScript, Jest, ESLint, Prettier) via `npm install`
- All dependencies locked via `pixi.lock` and `package-lock.json`

**Option B: Using uv (recommended for Python-only, faster)**

```bash
# Create virtual environment and install Python dependencies
uv venv
source .venv/bin/activate  # macOS/Linux
# OR
.venv\Scripts\activate     # Windows

# Install Python package in editable mode with dev dependencies
uv pip install -e ".[dev,export]"

# Install Node.js packages (requires Node.js installed separately)
npm install
```

> **Why uv?** It's significantly faster than pip and has built-in lockfile support via `uv.lock`
>
> **Why pixi?** It manages both Python AND Node.js in one tool, ensuring version consistency

### 3. Build Extension

```bash
# Compile TypeScript and create .vsix package
pixi run build
```

## Development Workflow

### Running the Extension in Development

1. Open the project in Positron
2. Press `F5` or select `Run → Start Debugging`
3. This launches a new **Extension Development Host** window with your extension loaded
4. The extension recompiles automatically when you save changes (if you ran `npm run watch`)

### Extension Commands

Once the extension is running, open the Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P`) and try:

- `Squiggy: Open POD5 File`
- `Squiggy: Open BAM File`
- `Squiggy: Plot Selected Read(s)`
- `Squiggy: Export Plot`

### Development Commands

**With pixi (4 main commands to remember):**
```bash
pixi run dev      # Watch mode (auto-recompile TypeScript on save)
pixi run test     # Run ALL tests (Python + TypeScript)
pixi run build    # Build .vsix extension package
pixi run docs     # Serve documentation locally (http://127.0.0.1:8000)
```

**With uv + npm:**
```bash
# First time: activate virtual environment
source .venv/bin/activate  # macOS/Linux

npm run watch     # Watch mode (auto-recompile TypeScript on save)
npm test && pytest tests/ -v  # Run ALL tests
npm run package   # Build .vsix extension package
mkdocs serve      # Serve documentation locally
```

**Granular Commands (work with both pixi and uv):**
```bash
# TypeScript only
npm run compile         # Compile TypeScript once
npm test                # Run TypeScript tests only
npm run lint            # Lint TypeScript
npm run format          # Format TypeScript

# Python only (with uv or pixi)
pytest tests/ -v        # Run Python tests only
ruff check squiggy/ tests/    # Lint Python
ruff format squiggy/ tests/   # Format Python

# TypeScript development
npm run test:watch                         # TypeScript tests in watch mode
npm run test:coverage                      # TypeScript coverage report

# Python development
pytest tests/ --cov=squiggy --cov-report=html  # Python coverage
pytest tests/test_io.py -v                     # Run specific test file
```

**Pixi-specific granular commands:**
```bash
pixi run compile    # Compile TypeScript once
pixi run test-ts    # Run TypeScript tests only
pixi run test-py    # Run Python tests only
pixi run lint-ts    # Lint TypeScript
pixi run lint-py    # Lint Python
pixi run lint       # Lint Python + TypeScript
pixi run format-ts  # Format TypeScript
pixi run format-py  # Format Python
pixi run format     # Format Python + TypeScript
```

### Quality Checks

Run all quality checks before submitting a PR:

```bash
# Python
ruff check --fix squiggy/ tests/
ruff format squiggy/ tests/
pytest tests/ -v

# TypeScript
npm run lint:fix
npm run format
npm test
```

Or use the `/test` slash command in Claude Code to run everything.

## Project Structure

```
squiggy-positron-extension/
├── squiggy/                    # Python package (backend)
│   ├── __init__.py             # Public API
│   ├── io.py                   # POD5/BAM file loading
│   ├── plotter.py              # Bokeh plotting
│   ├── alignment.py            # Base annotation extraction
│   ├── normalization.py        # Signal normalization
│   └── utils.py                # Utility functions
│
├── src/                        # TypeScript extension (frontend)
│   ├── extension.ts            # Entry point
│   ├── backend/                # Python communication
│   │   ├── squiggy-positron-runtime.ts  # Positron kernel integration
│   │   └── squiggy-python-backend.ts    # JSON-RPC subprocess fallback
│   ├── views/                  # UI panels
│   │   ├── squiggy-file-panel.ts        # File info panel
│   │   ├── squiggy-read-explorer.ts     # Read list tree view
│   │   ├── squiggy-read-search-view.ts   # Search panel
│   │   ├── squiggy-plot-options-view.ts  # Plot options panel
│   │   └── squiggy-modifications-panel.ts # Modifications panel
│   ├── webview/                # Plot display
│   │   └── squiggy-plot-panel.ts        # Bokeh plot webview
│   ├── types/                  # TypeScript type definitions
│   │   └── squiggy-positron.d.ts       # Positron API types
│   └── __mocks__/              # Test mocks
│       └── vscode.ts           # VSCode API mock
│
├── tests/                      # Python tests
│   ├── test_io.py              # File I/O tests
│   ├── test_plotting.py        # Plotting tests
│   └── data/                   # Test data (POD5/BAM files)
│
├── .github/workflows/          # CI/CD
│   ├── test.yml                # Run tests on PR
│   ├── build.yml               # Build .vsix artifact
│   └── release.yml             # Create releases
│
├── docs/                       # Documentation (MkDocs)
├── jest.config.js              # Jest test configuration
├── tsconfig.json               # TypeScript configuration
├── package.json                # Extension manifest
└── pyproject.toml              # Python package configuration
```

## Architecture Overview

### Data Flow

```
User Action (Positron UI)
    ↓
Extension (TypeScript)
    ↓
Positron Runtime API
    ↓
Active Python Kernel
    ↓
squiggy Package (Python)
    ↓
Bokeh HTML Output
    ↓
Webview Panel (TypeScript)
```

### Key Components

#### 1. Extension Activation (`src/extension.ts`)

Entry point when extension loads:
- Registers commands (`squiggy.openPOD5`, `squiggy.plotRead`, etc.)
- Creates UI panels (sidebar views, webviews)
- Initializes PositronRuntime for kernel communication

#### 2. Python Communication

**PositronRuntime** (`src/backend/squiggy-positron-runtime.ts`):
- Executes Python code in active kernel
- Used when running inside Positron

**PythonBackend** (`src/backend/squiggy-python-backend.ts`):
- JSON-RPC subprocess communication
- Fallback for non-Positron environments (e.g., VSCode)

#### 3. UI Panels

**FilePanelProvider** - Displays POD5/BAM file info
**ReadTreeProvider** - Hierarchical read list (grouped by reference if BAM loaded)
**ReadSearchView** - Search by read ID or reference name
**PlotOptionsView** - Plot configuration (mode, normalization, scaling)
**ModificationsPanelProvider** - Base modification filtering (when BAM has MM/ML tags)

#### 4. Plot Display

**PlotPanel** (`src/webview/squiggy-plot-panel.ts`):
- Webview panel for displaying Bokeh plots
- Receives HTML from Python backend
- Handles export functionality

#### 5. Python Backend

**squiggy Package**:
- `load_pod5()` - Load POD5 file into kernel state
- `load_bam()` - Load BAM file for annotations
- `plot_read()` - Generate Bokeh plot for single read
- Returns Bokeh HTML that's displayed in webview

## Testing

### TypeScript Tests

Located in `src/**/__tests__/`:

- **ReadTreeProvider tests** - Search/filtering logic
- **PythonBackend tests** - JSON-RPC communication mocking

```bash
npm test                 # Run all tests
npm run test:watch       # Watch mode
npm run test:coverage    # Generate coverage report
```

### Python Tests

Located in `tests/`:

```bash
pytest tests/ -v                        # Run all tests
pytest tests/test_io.py -v              # Run specific file
pytest tests/ --cov=squiggy             # With coverage
```

### Manual Testing

Use the sample data in `tests/data/`:

```bash
# In Extension Development Host:
# 1. Open Command Palette
# 2. Squiggy: Open POD5 File
# 3. Select tests/data/yeast_trna_reads.pod5
# 4. Squiggy: Open BAM File
# 5. Select tests/data/yeast_trna_mappings.bam
# 6. Click a read in the Reads panel
```

## Code Style

### Python

- **Formatter**: ruff (`ruff format`)
- **Linter**: ruff (`ruff check`)
- **Line length**: 88 characters (Black-compatible)
- **Imports**: Absolute imports preferred
- **Docstrings**: Google-style

### TypeScript

- **Formatter**: Prettier
- **Linter**: ESLint
- **Line length**: 100 characters
- **Indentation**: 4 spaces
- **Naming**: camelCase for variables/functions, PascalCase for classes

## Common Tasks

### Adding a New Command

1. **Register in `package.json`**:
   ```json
   "contributes": {
       "commands": [{
           "command": "squiggy.myNewCommand",
           "title": "My New Command",
           "category": "Squiggy"
       }]
   }
   ```

2. **Implement in `src/extension.ts`**:
   ```typescript
   const myNewCommand = vscode.commands.registerCommand(
       'squiggy.myNewCommand',
       async () => {
           // Implementation
       }
   );
   context.subscriptions.push(myNewCommand);
   ```

### Adding a New Python Function

1. **Add to `squiggy/__init__.py`** for public API exposure

2. **Implement in appropriate module** (`io.py`, `plotter.py`, etc.)

3. **Add tests** in `tests/`

4. **Call from TypeScript**:
   ```typescript
   await runtime.execute(`
       from squiggy import my_new_function
       result = my_new_function(arg1, arg2)
   `);
   ```

### Debugging

**TypeScript**:
- Set breakpoints in Positron
- Press `F5` to launch debugger
- Use Debug Console to inspect variables

**Python**:
- Add `print()` statements or `breakpoint()`
- Output appears in Extension Host's Output panel
- Or use Positron's debugger with kernel

## Continuous Integration

GitHub Actions runs on every push/PR:

- **test.yml**: Python and TypeScript tests
- **build.yml**: Compile and package .vsix
- **release.yml**: Create GitHub releases (on version tags)

## Release Process

1. **Update version** in `package.json` and `squiggy/__init__.py`

2. **Update NEWS.md** with release notes

3. **Create and push tag**:
   ```bash
   git tag v0.2.0
   git push origin v0.2.0
   ```

4. **GitHub Actions** automatically:
   - Runs tests
   - Builds .vsix
   - Creates GitHub release
   - Attaches .vsix artifact

## Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/my-feature`
3. **Make changes** with tests
4. **Run quality checks**: `/test` slash command or manual
5. **Commit** with conventional commit messages: `feat:`, `fix:`, `docs:`, etc.
6. **Push** and **create Pull Request**

## Resources

- [Positron Extension API](https://github.com/posit-dev/positron/wiki)
- [VSCode Extension API](https://code.visualstudio.com/api) (Positron is VSCode-compatible)
- [Bokeh Documentation](https://docs.bokeh.org/)
- [POD5 File Format](https://github.com/nanoporetech/pod5-file-format)
- [pysam Documentation](https://pysam.readthedocs.io/)

## Troubleshooting

### Extension doesn't activate

- Check Output panel → Extension Host Log
- Verify Python kernel is running
- Try reloading Positron window

### Tests failing

- Ensure Git LFS data is pulled: `git lfs pull`
- Check Python environment has all dependencies: `pixi install`
- Rebuild extension: `pixi run build`

### Can't import squiggy in kernel

- Verify package is installed: `pip show squiggy` or `uv pip show squiggy`
- Check Python path in kernel matches your environment (pixi, uv, or venv)
- Try reinstalling:
  - With pixi: `pixi install` then restart kernel
  - With uv: `uv pip install -e .` then restart kernel
  - With pip: `pip install -e .` then restart kernel

## Getting Help

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and community support
- **Pull Requests**: Code contributions welcome!
