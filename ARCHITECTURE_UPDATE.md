# Architecture Update: Jupyter Kernel Integration

## Decision: Use Positron Runtime API (Jupyter Kernel) as Primary Method

After reconsidering the architecture, we're pivoting to use Positron's Jupyter kernel integration as the primary method of Python communication, with JSON-RPC subprocess as a fallback.

## Why This Change?

### Problems with JSON-RPC Subprocess:
1. **Isolated from user workflow** - separate Python process can't share data
2. **Duplicates runtime** - two Python processes running
3. **Not Positron-native** - doesn't leverage Positron's data science features
4. **Can't use console** - users can't inspect loaded data or call functions

### Benefits of Jupyter Kernel:
1. **Shared state** - POD5 readers, signal data visible in Variables pane
2. **Console integration** - users can call `squiggy` functions directly
3. **Notebook workflow** - can load data in notebook, visualize in extension
4. **Native Positron** - uses `positron.runtime.executeCode()` API
5. **Better UX** - see progress in console, inspect intermediate results

## New Architecture

```
┌─────────────────────────────────────────┐
│  TypeScript Extension                   │
│  ├─ Commands (open POD5, plot, etc)     │
│  ├─ TreeView (reads list)               │
│  └─ Webview (Bokeh plots)               │
└──────────────┬──────────────────────────┘
               │
               ▼
    ┌──────────────────────┐
    │ Positron Runtime API │  ← Primary method
    │ (Jupyter Kernel)     │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │  Python Kernel       │
    │  ├─ pod5, pysam      │
    │  ├─ squiggy package  │
    │  └─ bokeh            │
    └──────────────────────┘

    Fallback (if no kernel):
    ┌──────────────────────┐
    │ JSON-RPC Subprocess  │  ← Fallback for VS Code
    └──────────────────────┘
```

## Implementation Plan

### 1. Create Python Package (not just backend)

Instead of a JSON-RPC server, create an installable `squiggy` package that works in Jupyter:

```python
# User installs: pip install squiggy
# Or extension auto-installs to active kernel

from squiggy import load_pod5, plot_read

# Load file
reader, read_ids = load_pod5('file.pod5')

# Generate plot (returns Bokeh HTML)
html = plot_read(reader, 'read_001', mode='SINGLE')
```

### 2. Extension Calls Python via Kernel

```typescript
import * as positron from '@posit-dev/positron';

async function openPOD5(filePath: string) {
    const api = await positron.tryAcquirePositronApi();

    if (api) {
        // Execute in Jupyter kernel
        await api.runtime.executeCode(`
from squiggy import load_pod5
_squiggy_reader, _squiggy_read_ids = load_pod5('${filePath}')
print(f"Loaded {len(_squiggy_read_ids)} reads")
        `, 'python');

        // Get read IDs from kernel
        const readIds = await api.runtime.getVariable('_squiggy_read_ids');
        return readIds;
    } else {
        // Fallback to subprocess
        return await jsonRpcBackend.call('open_pod5', {file_path: filePath});
    }
}
```

### 3. User Workflow Example

```python
# In Positron Console:
>>> import squiggy
>>> import pod5

# Load POD5 manually
>>> reader = pod5.Reader('yeast_trna.pod5')
>>> reads = list(reader.reads())
>>> len(reads)
180

# Extension can see this in Variables pane!
# Click "Plot" in extension sidebar → uses existing reader

# Or use extension functions directly:
>>> html = squiggy.plot_read(reader, 'read_001', mode='EVENTALIGN')
>>> # Extension shows in webview automatically
```

## Python Package Structure

```
squiggy/  (pip installable package)
├── __init__.py
├── core/
│   ├── io.py           # load_pod5(), load_bam()
│   ├── plotter.py      # plot_read(), plot_overlay()
│   ├── alignment.py
│   ├── normalization.py
│   └── constants.py
└── utils/
    └── export.py       # export_plot()
```

## Benefits of This Approach

1. **Extension enhances console** - not replaces it
2. **Users can script workflows** - load data programmatically
3. **Debuggable** - see what's happening in console
4. **Extensible** - users can modify/extend functions
5. **Variables pane integration** - inspect POD5 readers, signals
6. **Still works in VS Code** - via subprocess fallback

## Migration from Current Code

### Keep:
- ✅ TypeScript extension structure (commands, views, webview)
- ✅ Python core modules (plotter, utils, alignment, normalization)
- ✅ Bokeh HTML generation

### Change:
- ❌ JSON-RPC server → ✅ Installable Python package
- ❌ Subprocess communication → ✅ Positron runtime API
- ❌ API handlers → ✅ Public Python functions

### Add:
- ✅ Positron runtime integration
- ✅ Python package `setup.py` / `pyproject.toml`
- ✅ Kernel variable inspection
- ✅ Console output integration

## Next Steps

1. Restructure Python code as installable package
2. Update TypeScript to use `positron.runtime` API
3. Add kernel detection and fallback logic
4. Test with Positron console integration
5. Document user workflows (console + extension)
