# Task: Refactor Squiggy as Positron Extension

## Description
Refactor Squiggy from a standalone Qt/PySide6 desktop application into a Positron IDE extension with TypeScript frontend and Python backend.

## Objective
Create a VS Code/Positron-compatible extension that preserves all core Squiggy features while integrating into the Positron data science workflow.

## Architecture Overview

**Clean Separation of Concerns:**
- **TypeScript Extension**: ALL UI via Positron/VS Code APIs
  - File pickers, dialogs, notifications
  - TreeView for read list
  - Custom webview panels for Bokeh plots
  - Search panels and controls
  - Settings and configuration
- **Python Backend**: PURE data processing (NO UI, NO Qt/PySide6)
  - POD5/BAM file reading
  - Signal processing and normalization
  - Bokeh HTML generation (returns string)
  - Search algorithms
  - Returns JSON data only
- **Communication**: JSON-RPC over subprocess stdin/stdout
- **Dependencies**: pixi for Python, npm for TypeScript

**Key Features to Preserve:**
- Interactive Bokeh plots with zoom/pan/hover
- Multi-read overlay (SINGLE, OVERLAY, STACKED modes)
- Event-aligned mode with base annotations
- Search capabilities (Read ID, genomic region, sequence motif)

**What Gets Removed:**
- All PySide6/Qt imports and code
- qasync (no Qt event loop needed)
- QWebEngineView, QMainWindow, QWidget, etc.
- Qt dialogs, file pickers, message boxes
- All UI layout code in Python

## Implementation Phases

### Phase 1: Project Scaffolding ✓
- [x] Create git worktree
- [ ] Initialize npm project with TypeScript
- [ ] Setup pixi configuration for Python environment
- [ ] Create extension project structure
- [ ] Configure build tools (webpack/esbuild)

### Phase 2: Python Backend Integration
- [ ] Design JSON-RPC API
- [ ] Copy/adapt existing Python modules (plotter.py, utils.py, alignment.py, normalization.py)
- [ ] Remove Qt dependencies from Python code
- [ ] Create Python subprocess server
- [ ] Test backend communication

### Phase 3: Core UI Components
- [ ] Custom webview panel for Bokeh plots
- [ ] Read list TreeView (sidebar)
- [ ] Search panel UI
- [ ] File open commands

### Phase 4: Feature Implementation
- [ ] POD5 file loading
- [ ] BAM file loading
- [ ] Plot generation (all modes)
- [ ] Signal normalization
- [ ] Search functionality (ID, region, sequence)
- [ ] Export functionality (HTML, PNG, SVG)

### Phase 5: Positron-Specific Enhancements
- [ ] Integrate @posit-dev/positron APIs
- [ ] Use Positron plot viewer
- [ ] Configuration settings
- [ ] Extension marketplace metadata

### Phase 6: Testing & Distribution
- [ ] Unit tests (Jest for TypeScript)
- [ ] Integration tests (VS Code test runner)
- [ ] Package as .vsix
- [ ] Publish to Open VSX

## Files to Create

**Extension Structure:**
```
squiggy-positron/
├── package.json              # Extension manifest
├── tsconfig.json             # TypeScript config
├── pixi.toml                 # Python dependency management
├── src/
│   ├── extension.ts          # Main entry point
│   ├── webview/
│   │   ├── plotPanel.ts      # Custom webview for plots
│   │   └── plotPanel.html    # Webview template
│   ├── python/               # Python backend
│   │   ├── server.py         # JSON-RPC server
│   │   ├── plotter.py        # Reused from Squiggy
│   │   ├── utils.py          # Reused from Squiggy
│   │   ├── alignment.py      # Reused from Squiggy
│   │   └── normalization.py  # Reused from Squiggy
│   ├── views/
│   │   ├── readExplorer.ts   # TreeView for reads
│   │   └── searchPanel.ts    # Search controls
│   └── commands/
│       ├── openFile.ts       # File operations
│       ├── plotRead.ts       # Plot generation
│       └── exportPlot.ts     # Export functionality
└── .vscodeignore
```

## Success Criteria
- [ ] Extension installs and activates in Positron
- [ ] Can open POD5 files and display read list
- [ ] Can generate interactive Bokeh plots in webview
- [ ] All four plot modes work (SINGLE, OVERLAY, STACKED, EVENTALIGN)
- [ ] Signal normalization options functional
- [ ] Search functionality works (ID, region, sequence)
- [ ] Export to HTML/PNG/SVG successful
- [ ] No Qt dependencies in final package
- [ ] Single .vsix file distributable

## Key Technical Decisions

1. **Communication Pattern**: Subprocess + JSON-RPC (simpler than full LSP)
2. **Bokeh Rendering**: Static HTML generation (already works perfectly!)
3. **Dependency Management**: pixi instead of uv
4. **TypeScript**: Full rewrite (not hybrid)
5. **Python Code Reuse**: ~80% of existing logic preserved

## Notes
- Branch: feature/positron-extension
- Base: main
- Worktree: /Users/jayhesselberth/devel/rnabioco/squiggy-feature-positron-extension
- Main repo preserved for desktop app users

## Next Steps
1. Initialize npm project and install VS Code extension dependencies
2. Setup pixi configuration with Python 3.12
3. Create basic extension.ts with activation event
4. Copy Python modules from src/squiggy/ and remove Qt code
5. Implement JSON-RPC server skeleton
