# Squiggy Positron Extension

Positron extension for visualizing Oxford Nanopore sequencing data (squiggle plots) from POD5 files.

## 🚀 Quick Start for Developers

**Testing the extension?** See **[DEVELOPMENT.md](DEVELOPMENT.md)** for the complete testing workflow including the required test workspace setup.

## Architecture

- **TypeScript Frontend**: VS Code extension with TreeView and Webview panels
- **Python Backend**: Pure data processing (POD5/BAM files, Bokeh plotting)
- **Communication**: JSON-RPC over subprocess stdin/stdout
- **Dependencies**: pixi for Python, npm for TypeScript

## Setup

### Prerequisites

1. **Node.js 18+** and **npm**
2. **Python 3.12**
3. **pixi** (or uv as alternative)

Install pixi:
```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

### Installation

```bash
# 1. Install Python dependencies with pixi
pixi install

# Or alternatively with uv:
# uv pip install -e ".[dev]"

# 2. Install Node.js dependencies
npm install

# 3. Compile TypeScript
npm run compile
```

## Development

### Running the Extension

Press `F5` in VS Code to launch Extension Development Host with the extension loaded.

### Testing Python Backend

```bash
# Test JSON-RPC server
python test_server.py

# Or manually:
pixi run python src/python/server.py
# Then send JSON-RPC requests via stdin
```

### Building the Extension

```bash
# Package as .vsix
npm run package

# Output: squiggy-positron-0.1.0.vsix
```

## Project Structure

```
├── src/
│   ├── extension.ts              # TypeScript entry point
│   ├── webview/
│   │   └── plotPanel.ts          # Bokeh plot webview
│   ├── views/
│   │   ├── readExplorer.ts       # TreeView for reads
│   │   └── searchPanel.ts        # Search controls
│   ├── commands/
│   │   ├── openFile.ts           # File operations
│   │   ├── plotRead.ts           # Plot generation
│   │   └── exportPlot.ts         # Export functionality
│   ├── python/
│   │   ├── server.py             # JSON-RPC server
│   │   ├── core/                 # Data processing (from Squiggy)
│   │   │   ├── plotter.py
│   │   │   ├── utils.py
│   │   │   ├── alignment.py
│   │   │   ├── normalization.py
│   │   │   └── constants.py
│   │   └── api/                  # API handlers
│   │       ├── file_ops.py
│   │       ├── plot_ops.py
│   │       └── search_ops.py
├── package.json                  # Extension manifest
├── tsconfig.json                 # TypeScript config
├── pixi.toml                     # Python dependencies
└── test_server.py                # Backend test script
```

## Python Backend API

See [BACKEND_API.md](BACKEND_API.md) for complete JSON-RPC API documentation.

### Example Request

```json
{
  "jsonrpc": "2.0",
  "method": "open_pod5",
  "params": {"file_path": "/path/to/file.pod5"},
  "id": 1
}
```

### Example Response

```json
{
  "jsonrpc": "2.0",
  "result": {
    "file_path": "/path/to/file.pod5",
    "num_reads": 180,
    "read_ids": ["read_001", "read_002", ...]
  },
  "id": 1
}
```

## Key Differences from Qt App

| Feature | Qt App | Positron Extension |
|---------|--------|-------------------|
| UI Framework | PySide6 | VS Code APIs (TypeScript) |
| Plot Display | QWebEngineView | Custom Webview Panel |
| File Dialogs | Qt Native | VS Code File Picker |
| Read List | QTreeWidget | TreeView Provider |
| Console | N/A | Built-in Positron Console |
| Notebooks | N/A | Native Jupyter Support |
| Distribution | Platform binaries | Single .vsix file |

## Contributing

See [TASK.md](TASK.md) for development roadmap and implementation phases.

## License

MIT (same as original Squiggy desktop app)
