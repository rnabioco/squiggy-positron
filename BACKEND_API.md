# Squiggy Python Backend API

## Overview

The Python backend is a **pure data processing layer** with **zero UI code**. It communicates with the TypeScript extension via JSON-RPC over stdin/stdout.

## Communication Protocol

- **Transport**: stdin/stdout (subprocess)
- **Format**: JSON-RPC 2.0
- **Lifecycle**: Python process spawned on extension activation, kept alive for session

## API Methods

### File Operations

#### `open_pod5(file_path: str) -> dict`
Opens a POD5 file and extracts metadata.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "open_pod5",
  "params": {"file_path": "/path/to/file.pod5"},
  "id": 1
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "result": {
    "file_path": "/path/to/file.pod5",
    "file_size_mb": 125.4,
    "num_reads": 4823,
    "sample_rate": 4000,
    "read_ids": ["read_id_1", "read_id_2", ...]
  },
  "id": 1
}
```

#### `open_bam(file_path: str) -> dict`
Opens a BAM file with alignments.

**Response:**
```json
{
  "result": {
    "file_path": "/path/to/file.bam",
    "num_reads": 4823,
    "references": [
      {"name": "chr1", "length": 248956422, "read_count": 1234},
      {"name": "chr2", "length": 242193529, "read_count": 987}
    ]
  }
}
```

### Plot Generation

#### `generate_plot(params: dict) -> str`
Generates Bokeh HTML plot for specified reads.

**Request:**
```json
{
  "method": "generate_plot",
  "params": {
    "read_ids": ["read_id_1", "read_id_2"],
    "mode": "OVERLAY",  // SINGLE, OVERLAY, STACKED, EVENTALIGN
    "normalization": "ZNORM",  // NONE, ZNORM, MEDIAN, MAD
    "options": {
      "downsample": true,
      "downsample_threshold": 100000,
      "show_dwell_time": false,
      "position_label_interval": 100,
      "show_base_annotations": true
    }
  }
}
```

**Response:**
```json
{
  "result": {
    "html": "<html>... full Bokeh HTML ...</html>",
    "metadata": {
      "num_reads": 2,
      "total_samples": 150000,
      "downsampled": true
    }
  }
}
```

### Search Operations

#### `search_region(region: str) -> list[str]`
Searches for reads in a genomic region (requires BAM).

**Request:**
```json
{
  "method": "search_region",
  "params": {"region": "chr1:10000-20000"}
}
```

**Response:**
```json
{
  "result": {
    "read_ids": ["read_id_1", "read_id_2", ...],
    "count": 15
  }
}
```

#### `search_sequence(sequence: str, include_reverse: bool) -> list[dict]`
Searches reference sequences for DNA motif.

**Request:**
```json
{
  "method": "search_sequence",
  "params": {
    "sequence": "ATCGATCG",
    "include_reverse": true
  }
}
```

**Response:**
```json
{
  "result": {
    "matches": [
      {
        "read_id": "read_id_1",
        "reference": "chr1",
        "position": 12345,
        "strand": "+",
        "context": "...AATCGATCGTA..."
      }
    ],
    "count": 23
  }
}
```

### Export Operations

#### `export_plot(format: str, path: str, options: dict) -> dict`
Exports current plot to file.

**Request:**
```json
{
  "method": "export_plot",
  "params": {
    "format": "PNG",  // HTML, PNG, SVG
    "output_path": "/path/to/output.png",
    "bokeh_html": "<html>... current plot HTML ...",
    "options": {
      "width": 1200,
      "height": 800,
      "zoom_ranges": {
        "x": [1000, 5000],
        "y": [-3, 3]
      }
    }
  }
}
```

**Response:**
```json
{
  "result": {
    "success": true,
    "output_path": "/path/to/output.png",
    "file_size_kb": 245
  }
}
```

### Utility Methods

#### `get_read_data(read_id: str) -> dict`
Gets raw signal data and metadata for a read.

**Response:**
```json
{
  "result": {
    "read_id": "read_id_1",
    "signal": [120.5, 121.2, ...],  // Array of signal values
    "sample_rate": 4000,
    "duration_seconds": 2.5,
    "num_samples": 10000
  }
}
```

#### `get_references() -> list[dict]`
Gets list of reference sequences from BAM file.

**Response:**
```json
{
  "result": [
    {"name": "chr1", "length": 248956422, "read_count": 1234},
    {"name": "chr2", "length": 242193529, "read_count": 987}
  ]
}
```

## Error Handling

All errors return JSON-RPC error responses:

```json
{
  "jsonrpc": "2.0",
  "error": {
    "code": -32000,
    "message": "File not found: /path/to/file.pod5",
    "data": {
      "exception_type": "FileNotFoundError",
      "traceback": "..."
    }
  },
  "id": 1
}
```

## Python Backend Structure

```
src/python/
├── server.py           # JSON-RPC server (stdin/stdout)
├── core/
│   ├── plotter.py      # Bokeh plot generation (from Squiggy)
│   ├── utils.py        # File I/O (from Squiggy)
│   ├── alignment.py    # BAM parsing (from Squiggy)
│   └── normalization.py # Signal processing (from Squiggy)
└── api/
    ├── file_ops.py     # File operation handlers
    ├── plot_ops.py     # Plot generation handlers
    └── search_ops.py   # Search handlers
```

## Key Changes from Original Squiggy

1. **Remove all Qt imports**: No PySide6, QWidget, QDialog, etc.
2. **Remove qasync**: No async/await Qt integration needed
3. **Pure functions**: All methods take params, return data (no UI side effects)
4. **Error as data**: Don't show dialogs, return error info as JSON
5. **Synchronous**: Backend methods can be sync or async, handled by server

## Example Python Backend Method

```python
# Before (Qt version - BAD)
@qasync.asyncSlot()
async def open_pod5_file(self):
    file_path, _ = QFileDialog.getOpenFileName(...)
    if file_path:
        self.pod5_path = file_path
        await self.load_read_ids()
        self.statusBar().showMessage(f"Loaded {len(self.read_ids)} reads")

# After (JSON-RPC version - GOOD)
def open_pod5(file_path: str) -> dict:
    """Pure function, no UI, returns data"""
    with pod5.Reader(file_path) as reader:
        read_ids = [str(read.read_id) for read in reader.reads()]
        file_size = os.path.getsize(file_path) / (1024 * 1024)

        return {
            "file_path": file_path,
            "file_size_mb": round(file_size, 2),
            "num_reads": len(read_ids),
            "read_ids": read_ids
        }
```
