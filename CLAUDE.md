# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Communication Guidelines

When working on this project, maintain clear and neutral communication:
- Use professional, direct language as you would with a mid-level colleague
- Carefully evaluate assertions and suggestions before accepting them
- Respectfully push back when something seems incorrect or unclear
- Avoid overly agreeable or sycophantic responses (e.g., "You're absolutely right")
- Focus on technical accuracy and practical solutions
- Ask clarifying questions when requirements are ambiguous

## Project Overview

Squiggy is a Positron IDE extension for visualizing Oxford Nanopore sequencing data ("squiggle plots") from POD5 files. It integrates directly into the Positron IDE, leveraging the active Python kernel for seamless data exploration.

The extension provides interactive Bokeh visualizations with support for base annotations, modifications, and advanced signal processing - all within the Positron workspace.

## Architecture

### Application Stack

**TypeScript Extension (Frontend)**:
- **VSCode Extension API**: Positron-compatible extension framework
- **Positron Runtime API**: Execute Python code in active kernel
- **Webview API**: Display interactive Bokeh plots
- **TreeView API**: Hierarchical read list with filtering

**Python Package (Backend)**:
- **pod5**: Reading Oxford Nanopore POD5 files
- **pysam**: BAM file parsing for base annotations
- **bokeh**: Interactive visualization library
- **numpy**: Signal processing and normalization

**Development Tools**:
- **TypeScript 5.3+**: Extension code with strict typing
- **Jest + ts-jest**: TypeScript testing framework
- **ESLint**: TypeScript linting
- **Prettier**: Code formatting
- **pytest**: Python testing
- **ruff**: Python linting and formatting

### Project Structure

```
squiggy-positron-extension/
├── squiggy/                           # Python package (backend)
│   ├── __init__.py                    # Public API (load_pod5, load_bam, plot_read)
│   ├── io.py                          # POD5/BAM file loading with kernel state
│   ├── plotter.py                     # Bokeh plot generation
│   ├── alignment.py                   # Base annotation extraction from BAM
│   ├── normalization.py               # Signal normalization (ZNORM, MEDIAN, MAD)
│   └── utils.py                       # Utility functions
│
├── src/                               # TypeScript extension (frontend)
│   ├── extension.ts                   # Entry point, command registration
│   ├── backend/
│   │   ├── squiggy-positronRuntime.ts # Positron kernel communication
│   │   └── squiggy-pythonBackend.ts   # JSON-RPC subprocess fallback
│   ├── views/
│   │   ├── squiggy-filePanel.ts       # File info webview
│   │   ├── squiggy-readExplorer.ts    # Read list TreeView
│   │   ├── squiggy-readSearchView.ts  # Search panel webview
│   │   ├── squiggy-plotOptionsView.ts # Plot options webview
│   │   └── squiggy-modificationsPanel.ts # Modifications filter webview
│   ├── webview/
│   │   └── squiggy-plotPanel.ts       # Bokeh plot display
│   ├── types/
│   │   └── squiggy-positron.d.ts      # Positron API type definitions
│   └── __mocks__/
│       └── vscode.ts                  # VSCode API mock for testing
│
├── tests/                             # Python tests
│   ├── test_io.py                     # File I/O tests
│   ├── test_plotting.py               # Plotting tests
│   └── data/                          # Test data (POD5/BAM files)
│
├── .github/workflows/                 # CI/CD
│   ├── test.yml                       # Run tests on PR/push
│   ├── build.yml                      # Build .vsix artifact
│   ├── release.yml                    # Create GitHub releases
│   └── docs.yml                       # Deploy MkDocs documentation
│
├── docs/                              # Documentation (MkDocs)
│   └── index.md                       # Main documentation page
│
├── package.json                       # Extension manifest and dependencies
├── tsconfig.json                      # TypeScript configuration
├── jest.config.js                     # Jest test configuration
├── .prettierrc.json                   # Prettier formatting config
├── pyproject.toml                     # Python package configuration
├── mkdocs.yml                         # MkDocs configuration
└── docs/
    ├── DEVELOPER.md                   # Developer setup guide
    └── USER_GUIDE.md                  # User documentation
```

## Key Components

### Extension Entry Point (`src/extension.ts`)

Activates when Positron loads:
- Registers commands: `squiggy.openPOD5`, `squiggy.openBAM`, `squiggy.plotRead`, etc.
- Creates sidebar views: Files, Reads, Plot Options, Modifications
- Initializes PositronRuntime for kernel communication
- Sets up webview panels for plots

### Python Communication

**PositronRuntime** (`src/backend/squiggy-positronRuntime.ts`):
- Executes Python code in active Positron kernel
- Primary communication method when running in Positron
- Access to kernel state and variables

**PythonBackend** (`src/backend/squiggy-pythonBackend.ts`):
- JSON-RPC subprocess communication
- Fallback for non-Positron environments (e.g., VSCode)
- Spawns Python process and manages request/response

### UI Panels

**FilePanelProvider** - Webview showing POD5/BAM file metadata
**ReadTreeProvider** - TreeView with hierarchical read list (grouped by reference if BAM loaded)
**ReadSearchView** - Webview for filtering reads by ID or reference name
**PlotOptionsView** - Webview for plot configuration (mode, normalization, x-axis scaling)
**ModificationsPanelProvider** - Webview for base modification filtering (when BAM has MM/ML tags)

All webview panels use HTML/CSS/JavaScript for UI, communicate with extension via `postMessage`.

### Plot Display

**PlotPanel** (`src/webview/squiggy-plotPanel.ts`):
- Webview panel displaying Bokeh HTML plots
- Receives HTML from Python backend via kernel execution
- Handles export to HTML/PNG/SVG formats
- Supports zoom-level export (captures current view)

### Python API

**Public API** (`squiggy/__init__.py`):
```python
from squiggy import load_pod5, load_bam, plot_read, plot_reads

# Load files into kernel state
load_pod5("data.pod5")
load_bam("alignments.bam")

# Generate Bokeh plot HTML
html = plot_read(
    read_id="read_001",
    plot_mode="EVENTALIGN",
    normalization="ZNORM",
    scale_x_by_dwell=False,
    show_mods=True,
    mod_filter={"5mC": 0.8}
)
```

**State Management** (`squiggy/io.py`):
- Maintains global state: `_pod5_reader`, `_bam_file`, `_read_to_reference`
- Lazy loading of POD5 reads
- BAM indexing and reference extraction

**Plotting** (`squiggy/plotter.py`):
- Generates Bokeh figures with interactive tools
- Supports SINGLE and EVENTALIGN plot modes
- Base annotation rendering with color-coded bases
- Modification probability coloring
- Returns HTML via `bokeh.embed.file_html()`

## Development Workflow

### Setup

```bash
# Clone and install
git clone https://github.com/rnabioco/squiggy.git
cd squiggy
git lfs install && git lfs pull

# Install all dependencies
pixi install && pixi run setup  # Python + Node.js + npm packages

# Build extension
pixi run build
```

### Running in Development

1. Open project in Positron
2. Press `F5` to launch Extension Development Host
3. Extension loads automatically
4. Use Command Palette to test commands

### Testing

**Main Commands**:
```bash
pixi run dev       # Watch mode (auto-compile on save)
pixi run test      # Run ALL tests (Python + TypeScript)
pixi run build     # Build extension (.vsix)
pixi run docs      # Serve documentation
```

**Granular Testing**:
```bash
pixi run test-ts   # TypeScript tests only
pixi run test-py   # Python tests only

# Or direct commands:
npm run test:watch                         # TypeScript watch mode
pytest tests/test_io.py -v                 # Specific Python test
pytest tests/ --cov=squiggy --cov-report=html  # Python coverage
```

**Code Quality**:
```bash
pixi run lint      # Lint Python + TypeScript
pixi run format    # Format Python + TypeScript
pixi run lint-py   # Python only
pixi run lint-ts   # TypeScript only
```

Or use the `/test` slash command in Claude Code to run everything.

### Adding Features

**New Command**:
1. Register in `package.json` under `contributes.commands`
2. Implement handler in `src/extension.ts`
3. Call Python via PositronRuntime if needed

**New Python Function**:
1. Add to appropriate module (`io.py`, `plotter.py`, etc.)
2. Export in `squiggy/__init__.py` if public API
3. Add tests in `tests/`
4. Call from TypeScript via `runtime.execute()`

**New UI Panel**:
1. Create provider class in `src/views/`
2. Register in `src/extension.ts`
3. Add to `package.json` under `contributes.views`
4. Implement webview HTML/CSS/JS

## Code Style

### TypeScript

- **Formatter**: Prettier (`.prettierrc.json`)
- **Linter**: ESLint
- **Line Length**: 100 characters
- **Indentation**: 4 spaces
- **Naming**: camelCase for variables/functions, PascalCase for classes
- **Imports**: Organize into groups (stdlib, third-party, local)

### Python

- **Formatter**: ruff format
- **Linter**: ruff check
- **Line Length**: 88 characters (Black-compatible)
- **Imports**: Absolute imports preferred
- **Docstrings**: Google-style
- **Type Hints**: Use where appropriate (Python 3.12+)

## Important Patterns

### Async Execution in Positron

TypeScript extension communicates with Python kernel asynchronously:

```typescript
// Execute Python code in kernel
const result = await runtime.execute(`
    from squiggy import plot_read
    html = plot_read("read_001", plot_mode="SINGLE")
    html
`);

// Result contains the Python expression output
plotPanel.setHtml(result);
```

### Webview Communication

Webview panels communicate via message passing:

```typescript
// Extension to webview
panel.webview.postMessage({
    command: 'updatePlot',
    html: plotHtml
});

// Webview to extension
panel.webview.onDidReceiveMessage((message) => {
    if (message.command === 'export') {
        // Handle export request
    }
});
```

### State Management

Python state persists in kernel:

```python
# Global state in squiggy/io.py
_pod5_reader = None
_bam_file = None
_read_to_reference = {}

def load_pod5(file_path):
    global _pod5_reader
    _pod5_reader = pod5.Reader(file_path)
    # State persists across function calls
```

## Testing Strategy

### Unit Tests

**TypeScript**: Mock VSCode/Positron APIs
- ReadTreeProvider: Test filtering, grouping, search
- PythonBackend: Test JSON-RPC communication
- Located in `src/**/__tests__/`

**Python**: Test core functionality
- File I/O: POD5/BAM loading
- Plotting: Bokeh HTML generation
- Normalization: Signal processing
- Located in `tests/`

### Integration Tests

Test extension ↔ Python communication:
- Load files and verify kernel state
- Generate plots and verify HTML output
- Test error handling and edge cases

### Manual Testing

Use sample data in `tests/data/`:
- `yeast_trna_reads.pod5` - 180 reads
- `yeast_trna_mappings.bam` - Corresponding alignments

## Common Tasks

### Debugging

**TypeScript**:
- Set breakpoints in Positron
- Press `F5` to attach debugger
- Check Extension Host Output panel for logs

**Python**:
- Add `print()` statements or `breakpoint()`
- Output appears in Output panel → Extension Host Log
- Or use Positron's Python debugger with kernel

### Building and Packaging

```bash
# Compile TypeScript
npm run compile

# Package extension as .vsix
npm run package

# Result: squiggy-positron-<version>.vsix
```

### Release Process

1. Update version in `package.json` and `squiggy/__init__.py`
2. Update `NEWS.md` with release notes
3. Create and push tag: `git tag v0.2.0 && git push origin v0.2.0`
4. GitHub Actions automatically:
   - Runs tests
   - Builds .vsix
   - Creates GitHub release with artifact

## Constraints and Gotchas

### Positron API

- Runtime API is specific to Positron (not available in VSCode)
- Fallback to PythonBackend for non-Positron environments
- Kernel must be active for Runtime API to work

### Bokeh Plots

- Generate as standalone HTML with CDN resources
- Use `bokeh.embed.file_html()` with `CDN` resources
- Plots rendered in webview via `setHtml()`
- Interactive features (zoom, pan) work in webview

### POD5 Files

- Require `vbz_h5py_plugin` for VBZ compression (installed with pod5)
- Use context managers: `with pod5.Reader(path) as reader:`
- Read objects are temporary - store IDs, not objects

### BAM Files

- Must be indexed (`.bai` file required)
- Use `pysam.AlignmentFile(..., check_sq=False)` to avoid header issues
- Event-aligned mode requires `mv` tag (move table from dorado/guppy)
- Modifications require `MM`/`ML` tags

### Testing

- TypeScript tests excluded from compilation (see `tsconfig.json`)
- Python tests require Git LFS data: `git lfs pull`
- Sample data in `tests/data/` used for manual testing

## Resources

- [Positron Extension API](https://github.com/posit-dev/positron/wiki)
- [VSCode Extension API](https://code.visualstudio.com/api)
- [Bokeh Documentation](https://docs.bokeh.org/)
- [POD5 Format](https://github.com/nanoporetech/pod5-file-format)
- [pysam Documentation](https://pysam.readthedocs.io/)

## Additional Documentation

- **[DEVELOPER.md](docs/DEVELOPER.md)** - Comprehensive developer guide
- **[USER_GUIDE.md](docs/USER_GUIDE.md)** - User documentation
- **[README.md](README.md)** - Quick start and overview
