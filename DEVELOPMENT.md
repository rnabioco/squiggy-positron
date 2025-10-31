# Squiggy Extension Development Guide

## Testing the Extension in Positron

### The Problem

Positron's Extension Development Host cannot open the same folder as your main development window. This creates a challenge when testing extensions that depend on Python packages installed in the project's `.venv`.

### The Solution: Test Workspace

We use a separate test workspace that symlinks to the project's resources.

## Setup (One-Time)

The test workspace is located at `/tmp/squiggy-extension-test/` and contains:

```bash
/tmp/squiggy-extension-test/
├── .venv/        # Symlink to project .venv
├── data/         # Symlink to tests/data/
└── README.md     # Documentation
```

**To recreate the test workspace** (if deleted):

```bash
# Create workspace directory
mkdir -p /tmp/squiggy-extension-test

# Symlink .venv
ln -s /Users/jayhesselberth/devel/rnabioco/squiggy-feature-positron-extension/.venv /tmp/squiggy-extension-test/.venv

# Symlink test data
ln -s /Users/jayhesselberth/devel/rnabioco/squiggy-feature-positron-extension/tests/data /tmp/squiggy-extension-test/data
```

## Testing Workflow

### 1. Launch Extension Development Host

In your main Positron window (with the extension project open):
- Press **F5** (or Run → Start Debugging)
- A new "Extension Development Host" window will open

### 2. Open Test Workspace

In the Extension Development Host window:
- **File → Open Folder**
- Navigate to: `/tmp/squiggy-extension-test`
- Click **Open**

### 3. Verify Python Interpreter

Check the bottom-right corner:
- Should show: **"Python 3.12.8 (.venv)"** or similar
- If not: Command Palette (Cmd+Shift+P) → "Python: Select Interpreter"
  - Select: `/tmp/squiggy-extension-test/.venv/bin/python`

### 4. Start Python Console

- View → Console (or Cmd+Shift+Y)
- Click **"+"** to start a new console
- Verify it works:
  ```python
  import squiggy
  print("Success!")
  ```

### 5. Test Extension Commands

- Command Palette (Cmd+Shift+P)
- **"Squiggy: Open POD5 File"**
- Select: `data/yeast_trna_reads.pod5` (180 reads)
- Verify read list appears in sidebar

### 6. Test Plotting

- Select a read from the list in the Squiggy sidebar
- Command Palette → **"Squiggy: Plot Selected Read(s)"**
- Verify Bokeh plot appears in webview panel

## Development Cycle

1. Make code changes in main window
2. Code auto-compiles (or run `npm run compile`)
3. In Extension Development Host: **Developer: Reload Window** (Cmd+R)
4. Test changes

## Troubleshooting

### "ModuleNotFoundError: No module named 'squiggy'"

The extension automatically adds the project to `sys.path`, but if this fails:

```python
import sys
sys.path.insert(0, '/Users/jayhesselberth/devel/rnabioco/squiggy-feature-positron-extension')
import squiggy
```

### Wrong Python Interpreter

If the console starts with "Python 3.14.0 (Global)":
1. Close the console
2. Command Palette → "Python: Select Interpreter"
3. Choose the one showing `.venv` or browse to `/tmp/squiggy-extension-test/.venv/bin/python`
4. Start a new console

### Test Workspace Deleted

Re-run the setup commands above to recreate the symlinks.

## Files You'll Edit

When developing, you'll primarily edit these files in the **main window**:

### Python Package (root `squiggy/`)
- `squiggy/__init__.py` - Public API
- `squiggy/io.py` - File loading
- `squiggy/plotter.py` - Bokeh plotting logic

### TypeScript Extension (`src/`)
- `src/extension.ts` - Main extension logic
- `src/backend/positronRuntime.ts` - Kernel integration
- `src/views/readExplorer.ts` - TreeView provider
- `src/webview/plotPanel.ts` - Plot display

### Configuration
- `package.json` - Extension manifest (commands, views, settings)
- `tsconfig.json` - TypeScript compilation
- `.vscode/launch.json` - Debug configuration

## Clean Up

To remove the test workspace:
```bash
rm -rf /tmp/squiggy-extension-test
```

Note: This doesn't delete any real files - only the symlinks.
