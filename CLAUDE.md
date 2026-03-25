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

## Project Organization

### Claude Planning Files

**IMPORTANT:** Claude planning files (e.g., `PLANNING_NOTES.md`, `FEATURE_DESIGN.md`, `IMPLEMENTATION_GUIDE.md`) should NOT be kept at the project root. These files clutter the repository and are not appropriate for release.

**Proper locations for planning files:**
- `docs/guides/` - For planning documents that may be useful for future reference
- Delete - If the planning file is no longer relevant or has been integrated into proper documentation

**Before creating a release:**
- Check for any `.md` files at project root with underscores (e.g., `*_*.md`)
- Move useful planning files to `docs/guides/` or delete them
- The `/release` slash command will automatically check for these files and prompt for action

**Standard files that SHOULD remain at project root:**
- `README.md` - Project overview and quick start
- `CHANGELOG.md` - Version history and release notes
- `LICENSE.md` - Project license
- `CONTRIBUTING.md` - Contribution guidelines
- `CLAUDE.md` - This file (instructions for Claude Code)

## Architecture

### Application Stack

**TypeScript Extension (Frontend)**:
- **VSCode Extension API**: Positron-compatible extension framework
- **Positron Runtime API**: Execute Python code in active kernel
- **Webview API**: Display interactive Bokeh plots
- **React 18**: Webview UI components (reads panel)
- **react-window**: Virtualized list rendering for large datasets
- **webpack**: Module bundler (dual bundle: extension + webview)

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
│   ├── __init__.py                    # Public API (legacy + OO)
│   ├── api.py                         # Object-oriented API for notebooks
│   ├── io.py                          # POD5/BAM file loading with kernel state
│   ├── plot_factory.py                # Factory for creating plot strategies
│   ├── alignment.py                   # Base annotation extraction from BAM
│   ├── constants.py                   # Enums and constants
│   ├── modifications.py               # Base modification parsing
│   ├── normalization.py               # Signal normalization (ZNORM, MEDIAN, MAD)
│   ├── motif.py                       # Sequence motif search
│   ├── utils.py                       # Utility functions
│   │
│   ├── plot_strategies/               # Strategy Pattern for plot generation
│   │   ├── base.py                    # PlotStrategy abstract base class
│   │   ├── single_read.py             # Single read plots
│   │   ├── overlay.py                 # Multiple reads overlaid
│   │   ├── stacked.py                 # Multiple reads stacked
│   │   ├── eventalign.py              # Event-aligned with base annotations
│   │   └── aggregate.py               # Multi-read aggregate statistics
│   │
│   └── rendering/                     # Reusable rendering components
│       ├── theme_manager.py           # Centralized theme management
│       ├── base_annotation_renderer.py # Base annotation rendering
│       └── modification_track_builder.py # Modification track rendering
│
├── src/                               # TypeScript extension (frontend)
│   ├── extension.ts                   # Entry point, command registration
│   ├── backend/
│   │   ├── positron-runtime-client.ts  # Positron kernel communication (low-level)
│   │   ├── squiggy-python-backend.ts   # JSON-RPC subprocess fallback
│   │   ├── squiggy-kernel-manager.ts   # Dedicated Squiggy kernel manager
│   │   ├── runtime-client-interface.ts # RuntimeClient interface
│   │   └── squiggy-runtime-api.ts      # High-level API for file/plot operations
│   ├── views/
│   │   ├── components/                 # React components for reads panel
│   │   │   ├── squiggy-reads-core.tsx  # Main table logic
│   │   │   ├── squiggy-reads-instance.tsx # Webview host
│   │   │   ├── squiggy-read-item.tsx   # Individual read row
│   │   │   ├── squiggy-reference-group.tsx # Grouped by reference
│   │   │   ├── column-resizer.tsx      # Resizable columns
│   │   │   └── webview-entry.tsx       # React entry point
│   │   ├── squiggy-file-panel.ts       # File info webview
│   │   ├── squiggy-reads-view-pane.ts  # Read list React webview
│   │   ├── squiggy-plot-options-view.ts # Plot options webview
│   │   └── squiggy-modifications-panel.ts # Modifications filter webview
│   ├── webview/
│   │   └── squiggy-plot-panel.ts       # Bokeh plot display
│   ├── types/
│   │   ├── squiggy-positron.d.ts       # Positron API type definitions
│   │   └── squiggy-reads-types.ts      # Types for reads data
│   └── __mocks__/
│       └── vscode.ts                   # VSCode API mock for testing
│
├── examples/                          # Example notebooks
│   └── notebook_api_demo.ipynb        # OO API tutorial
│
├── tests/                             # Python tests
│   ├── test_alignment.py              # Alignment tests
│   ├── test_api.py                    # Legacy API tests
│   ├── test_io.py                     # File I/O tests
│   ├── test_modifications.py          # Modification parsing tests
│   ├── test_normalization.py          # Normalization tests
│   ├── test_oo_api.py                 # Object-oriented API tests
│   ├── test_plotting.py               # Plotting tests
│   ├── test_utils.py                  # Utility function tests
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
├── webpack.config.js                  # Webpack bundling configuration
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
- Registers commands: `squiggy.openPOD5`, `squiggy.openBAM`, `squiggy.plotRead`, `squiggy.restartBackgroundKernel`, etc.
- Creates sidebar views: Files, Reads, Plot Options, Modifications
- Initializes SquiggyKernelManager for the dedicated Squiggy kernel
- Sets up status bar item showing kernel state (⭕ → 🔄 → ✅ or ❌)
- Sets up webview panels for plots

### Python Communication

**SquiggyKernelManager** (`src/backend/squiggy-kernel-manager.ts`):
- Manages the dedicated "Squiggy Kernel" session
- State machine: Uninitialized → Starting → Ready | Error
- Provides status bar integration with visual indicators (⭕ → 🔄 → ✅ or ❌)
- Command: `squiggy.restartBackgroundKernel` - Restart kernel with progress UI

**SquiggyRuntimeAPI** (`src/backend/squiggy-runtime-api.ts`):
- High-level API for squiggy operations (loading files, generating plots)
- Built on top of RuntimeClient interface
- Called by FileLoadingService and plot commands

**RuntimeClient Interface** (`src/backend/runtime-client-interface.ts`):
- Common interface for kernel communication
- Provides: `executeSilent()`, `getVariable()`

### Kernel Architecture

**Single Kernel Design**: All extension operations use a dedicated "Squiggy Kernel" session. This simplifies the architecture and eliminates user confusion from having multiple kernel sessions visible in the console selector.

**Key Design Decisions**:
- **No foreground kernel fallback**: The extension requires the dedicated kernel to function
- **Lazy initialization**: Kernel starts on first use (`state.ensureKernel()`)
- **Clear error handling**: If kernel fails to start, user sees clear error message

**API Access Pattern**:
```typescript
async ensureKernel(): Promise<SquiggyRuntimeAPI> {
    if (!this._kernelManager) {
        throw new Error('Squiggy kernel manager not initialized');
    }

    if (this._kernelManager.getState() === SquiggyKernelState.Uninitialized) {
        await this._kernelManager.start();
    }

    if (!this._squiggyAPI) {
        this._squiggyAPI = new SquiggyRuntimeAPI(this._kernelManager);
    }

    return this._squiggyAPI;
}
```

**File Loading and Plot Generation**:
- All file loading (POD5, BAM, FASTA) routes through `state.ensureKernel()`
- All plot generation uses the same dedicated kernel
- State is isolated from user's interactive console


### 🚨 CRITICAL: Positron Extension Integration Patterns

When building Positron extensions that need to access Python data, **NEVER use `print()` to get data from Python to TypeScript**. This pollutes the user's console and defeats the purpose of kernel integration.

#### ✅ CORRECT Pattern: Use Positron's Variable Access API

**For reading Python variables:**
```typescript
// Step 1: Execute code silently to create variables
await positronRuntime.executeSilent(`
import squiggy
_squiggy_reader, _squiggy_read_ids = squiggy.load_pod5('file.pod5')
`);

// Step 2: Read variables directly from kernel memory (NO PRINT!)
const numReads = await positronRuntime.getVariable('len(_squiggy_read_ids)');
const readIds = await positronRuntime.getVariable('_squiggy_read_ids[0:1000]');
```

**Implementation of `getVariable()`:**
```typescript
async getVariable(varName: string): Promise<any> {
    const session = await positron.runtime.getForegroundSession();
    const tempVar = '_temp_' + Math.random().toString(36).substr(2, 9);

    // Convert Python value to JSON in Python
    await this.executeSilent(`
import json
${tempVar} = json.dumps(${varName})
`);

    // Read the JSON string via getSessionVariables
    const [[variable]] = await positron.runtime.getSessionVariables(
        session.metadata.sessionId,
        [[tempVar]]
    );

    await this.executeSilent(`del ${tempVar}`);

    // Parse: Python string repr -> JSON string -> JavaScript value
    const cleaned = variable.display_value.replace(/^['"]|['"]$/g, '');
    return JSON.parse(cleaned);
}
```

**Implementation of `executeSilent()`:**
```typescript
async executeSilent(code: string): Promise<void> {
    await this.executeCode(
        code,
        false,  // focus=false
        true,   // allowIncomplete
        positron.RuntimeCodeExecutionMode.Silent  // ✅ NO console output!
    );
}
```

#### ❌ WRONG Pattern: Using print() (Causes Console Pollution)

```typescript
// DON'T DO THIS - pollutes user's console
await executeCode(`
import squiggy
_squiggy_reader, _squiggy_read_ids = squiggy.load_pod5('file.pod5')
print(len(_squiggy_read_ids))  # ❌ Shows in console!
`, RuntimeCodeExecutionMode.Interactive);
```

Even `Silent` mode with `print()` doesn't work - the output still appears.

#### Key Positron APIs

**`positron.runtime.getForegroundSession()`**:
- Returns the active Python/R session
- Needed to get `sessionId` for variable access

**`positron.runtime.getSessionVariables(sessionId, accessKeys)`**:
- Reads variable values directly from kernel memory
- Returns `RuntimeVariable[]` with `display_value` field
- Used by Variables pane - same pattern we follow

**`positron.runtime.executeCode(languageId, code, focus, allowIncomplete, mode, errorBehavior, observer)`**:
- `mode=RuntimeCodeExecutionMode.Silent` - No code echo, no history
- `mode=RuntimeCodeExecutionMode.Interactive` - Shows in console (avoid for data queries)
- `focus=false` - Prevents console from getting focus

#### Benefits of This Pattern

✅ **Zero console pollution** - user's console stays clean for interactive work
✅ **Direct memory access** - no serialization overhead from print()
✅ **Kernel variables preserved** - variables like `_squiggy_read_ids` available in console
✅ **Lazy loading** - scalable for millions of reads
✅ **Follows Positron conventions** - same pattern used by Variables pane and Data Explorer

#### Type Definitions Required

Add to `src/types/squiggy-positron.d.ts`:
```typescript
export interface BaseLanguageRuntimeSession {
    metadata: { sessionId: string; sessionName: string; sessionMode: string };
    runtimeMetadata: { languageId: string; /* ... */ };
}

export interface RuntimeVariable {
    display_value: string;  // Python repr of value
    display_type: string;
    /* ... */
}

export namespace runtime {
    export function getForegroundSession(): Thenable<BaseLanguageRuntimeSession | undefined>;
    export function getSessionVariables(
        sessionId: string,
        accessKeys?: Array<Array<string>>
    ): Thenable<Array<Array<RuntimeVariable>>>;
}
```

**Reference**: Positron's own extensions use this pattern. See `positron/extensions/positron-connections` and how the Variables pane queries kernel state.

### UI Panels

**FilePanelProvider** - Webview showing POD5/BAM file metadata
**ReadsViewPane** - React webview with virtualized multi-column table (Read ID, Length, Quality, Reference, Position). Uses react-window for performance with large datasets. Search functionality integrated into UI. Grouped by reference when BAM loaded.
**PlotOptionsView** - Webview for plot configuration (mode, normalization, x-axis scaling)
**ModificationsPanelProvider** - Webview for base modification filtering (when BAM has MM/ML tags)

All webview panels communicate with extension via `postMessage`.

**🚨 IMPORTANT: React-First UI Development**

When building new UI panels or refactoring existing ones:
- **ALWAYS prefer React** for interactive panels with complex state or data display
- Use React for: tables, lists, forms, filters, multi-step wizards, dynamic content
- React provides better maintainability, testability, and type safety
- Only use plain HTML/CSS for: static content panels, simple displays, embedded Bokeh plots

**React Panel Pattern** (see `ReadsViewPane` as reference):
```typescript
// 1. Create React components in src/views/components/
//    - my-panel-core.tsx (main logic)
//    - my-panel-instance.tsx (webview host)
//    - webview-entry.tsx (React entry point)

// 2. Provider class creates webview and bundles React
class MyPanelProvider implements vscode.WebviewViewProvider {
    resolveWebviewView(webviewView: vscode.WebviewView) {
        webviewView.webview.html = this.getHtmlForWebview(webviewView.webview);
        // Set up postMessage communication
    }
}

// 3. Webpack bundles React components separately
// See webpack.config.js for webview bundle configuration
```

**Benefits of React over custom HTML**:
- Component reusability and composition
- Type-safe props and state management
- Efficient virtual DOM updates
- Rich ecosystem (react-window for virtualization, etc.)
- Jest testing with @testing-library/react
- Better developer experience with TSX syntax

### Plot Display

**PlotPanel** (`src/webview/squiggy-plot-panel.ts`):
- Webview panel displaying Bokeh HTML plots
- Receives HTML from Python backend via kernel execution
- Handles export to HTML/PNG/SVG formats
- Supports zoom-level export (captures current view)

### Python API

**Public API** (`squiggy/__init__.py`):
```python
from squiggy import load_pod5, load_bam, plot_read, plot_reads, close_pod5, close_bam
from squiggy.io import squiggy_kernel

# Load files into kernel state (populates squiggy_kernel)
load_pod5("data.pod5")
load_bam("alignments.bam")

# Check session state
print(squiggy_kernel)
# <SquiggyKernel: POD5: data.pod5 (1,234 reads) | BAM: alignments.bam (1,234 reads)>

# Generate Bokeh plot HTML
html = plot_read(
    read_id="read_001",
    plot_mode="EVENTALIGN",
    normalization="ZNORM",
    scale_x_by_dwell=False,
    show_mods=True,
    mod_filter={"5mC": 0.8}
)

# Cleanup
close_pod5()  # Close POD5 reader
close_bam()   # Clear BAM state
squiggy_kernel.close_all()  # Or close everything via session
```

**State Management** (`squiggy/io.py`):
- **NEW**: Consolidated `SquiggyKernel` object (`squiggy_kernel`) - single variable in Variables pane
- **Legacy**: Individual globals: `_current_pod5_reader`, `_current_bam_path`, `_current_read_ids`
- Lazy loading of POD5 reads
- BAM indexing and reference extraction
- New cleanup functions: `close_bam()` for BAM state cleanup

**Plotting** (Strategy Pattern - `squiggy/plot_factory.py` + `squiggy/plot_strategies/`):
- Uses Strategy Pattern via PlotFactory to generate plots
- Supports 7 plot modes: SINGLE, OVERLAY, STACKED, EVENTALIGN, AGGREGATE, DELTA, SIGNAL_OVERLAY_COMPARISON
- Reusable rendering components in `squiggy/rendering/`:
  - ThemeManager: Centralized theme configuration (light/dark mode)
  - BaseAnnotationRenderer: Color-coded base annotations
  - ModificationTrackBuilder: Modification probability tracks
- Each strategy returns HTML via `bokeh.embed.file_html()`

## Development Workflow

### Setup

```bash
# Clone and install
git clone https://github.com/rnabioco/squiggy-positron.git
cd squiggy-positron
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
1. **Decide on implementation approach**:
   - **React** (preferred): For interactive content, forms, data tables, complex state
   - **Plain HTML**: Only for static content or embedded visualizations (e.g., Bokeh plots)
2. **If using React** (see `ReadsViewPane` as reference):
   - Create React components in `src/views/components/`:
     - `my-panel-core.tsx` - Main component logic
     - `my-panel-instance.tsx` - Webview host wrapper
     - `webview-entry.tsx` - Entry point with React 18 root
   - Create provider class in `src/views/my-panel-provider.ts`
   - Add webpack entry in `webpack.config.js` for webview bundle
   - Use TypeScript interfaces in `src/types/` for message passing
3. **If using plain HTML** (legacy panels, use sparingly):
   - Create provider class in `src/views/`
   - Generate HTML string in provider's `getHtmlForWebview()` method
   - Include inline CSS/JS or reference bundled assets
4. **Register panel**:
   - Add to `package.json` under `contributes.views` or `contributes.viewsContainers`
   - Register provider in `src/extension.ts` activation function
   - Set up `postMessage` communication between extension and webview

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

**Session-Based State** (Recommended):

Python state is consolidated in a `SquiggyKernel` object for cleaner UX:

```python
# Global session instance in squiggy/io.py
from squiggy.io import squiggy_kernel

# Load files - automatically populates session
import squiggy
squiggy.load_pod5('data.pod5')
squiggy.load_bam('alignments.bam')

# Session is visible in Variables pane as single object
print(squiggy_kernel)
# <SquiggyKernel: POD5: data.pod5 (1,234 reads) | BAM: alignments.bam (1,234 reads)>

# Access session attributes
squiggy_kernel.reader      # POD5 reader
squiggy_kernel.read_ids    # List of read IDs
squiggy_kernel.bam_path    # BAM file path
squiggy_kernel.bam_info    # BAM metadata dict
squiggy_kernel.ref_mapping # Reference to read IDs mapping

# Cleanup
squiggy_kernel.close_pod5()  # Close POD5 only
squiggy_kernel.close_bam()   # Clear BAM only
squiggy_kernel.close_all()   # Close everything
```

**Legacy Global State** (Deprecated but still supported):

For backward compatibility, individual global variables are still maintained:

```python
# Legacy globals in squiggy/io.py
_current_pod5_reader = None
_current_bam_path = None
_current_read_ids = []

def load_pod5(file_path):
    global _current_pod5_reader
    _current_pod5_reader = pod5.Reader(file_path)
    # State persists across function calls
```

**Extension Usage**:

The TypeScript extension uses the session object for cleaner kernel state:

```typescript
// Load POD5 - populates squiggy_kernel in kernel
await squiggyAPI.loadPOD5(filePath);

// Access data via session
const readIds = await client.getVariable('squiggy_kernel.read_ids[:100]');
const bamInfo = await client.getVariable('squiggy_kernel.bam_info');
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

Use sample data in `squiggy/data/`:
- `ecoli_trna_wt_reads.pod5` / `ecoli_trna_tb_reads.pod5` - E. coli tRNA (WT vs TruB)
- `ecoli_trna_wt_mappings.bam` / `ecoli_trna_tb_mappings.bam` - Corresponding alignments

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
2. Update `CHANGELOG.md` with release notes
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
- Sample data in `squiggy/data/` used for demos and testing

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
