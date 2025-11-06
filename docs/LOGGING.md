# Squiggy Logging Guide

This guide explains where Squiggy logs appear in Positron and how to access them for debugging.

## Overview

Squiggy uses two separate logging systems:

1. **TypeScript Extension Logs** - Extension-level operations (UI, commands, file handling)
2. **Python Backend Logs** - Data processing, plotting, file I/O

## TypeScript Extension Logs

### Where They Appear

Extension logs appear in the **Output Panel** under the "Squiggy" channel.

### How to Access

**Option 1: Via Command Palette**
1. Open Command Palette (`Cmd+Shift+P` / `Ctrl+Shift+P`)
2. Run `Squiggy: Show Extension Logs`

**Option 2: Via Output Panel**
1. Open Output panel (`Cmd+Shift+U` / `Ctrl+Shift+U`)
2. Select "Squiggy" from the dropdown menu

**Option 3: Click "Show Logs" on error notifications**
When an error occurs, click the "Show Logs" button in the error notification.

### What's Logged

- File loading operations (POD5, BAM, FASTA)
- Plot generation requests
- Extension activation/deactivation
- Error messages with stack traces
- Warnings and debug info

### Example Output

```
[14:23:45.123] [INFO] Squiggy extension activated
[14:23:47.456] [INFO] Loading POD5 file: /data/reads.pod5
[14:23:48.789] [ERROR] Error while loading POD5 file: FileNotFoundError
```

## Python Backend Logs

### Where They Appear

Python logs appear in **two places**:

1. **Python Console** (default) - Where your interactive Python code runs
2. **Output Panel → Squiggy** (when configured) - Dedicated extension output

### Default Behavior (Python Console)

By default, Python's `logging` module sends output to `stderr`, which Positron displays in the Python Console. This can pollute your interactive workspace.

**Example in Python Console:**
```python
>>> import squiggy
>>> squiggy.load_pod5('nonexistent.pod5')
2025-01-15 14:23:48 - squiggy.io - ERROR - POD5 file not found at path: /data/nonexistent.pod5
Traceback (most recent call last):
  ...
FileNotFoundError: Failed to open pod5 file at: /data/nonexistent.pod5
```

### Controlling Python Log Level

Set the `SQUIGGY_LOG_LEVEL` environment variable to control verbosity:

**In Positron:**
```python
import os
os.environ['SQUIGGY_LOG_LEVEL'] = 'DEBUG'  # Show all logs
os.environ['SQUIGGY_LOG_LEVEL'] = 'INFO'   # Show info and above
os.environ['SQUIGGY_LOG_LEVEL'] = 'WARNING'  # Default: only warnings and errors
os.environ['SQUIGGY_LOG_LEVEL'] = 'ERROR'  # Only errors
```

**Before starting Python:**
```bash
export SQUIGGY_LOG_LEVEL=DEBUG
positron
```

### Log Levels

- **DEBUG**: Detailed diagnostic info (e.g., "Checking cache for POD5 index")
- **INFO**: Confirmation of successful operations (e.g., "Loaded 1,234 reads")
- **WARNING**: Something unexpected but not critical (e.g., "No alignment found for read")
- **ERROR**: Operation failed (e.g., "File not found", "Invalid parameter")

### What's Logged

Python logs include:

- File loading operations with file paths
- Read counts and metadata
- Validation errors with helpful suggestions
- Cache hits/misses
- Alignment extraction failures
- Modification parsing warnings

### Example Python Logs

```python
# INFO level
2025-01-15 14:23:48 - squiggy.io - INFO - Loaded POD5: reads.pod5 (1,234 reads)

# ERROR level
2025-01-15 14:24:12 - squiggy.io - ERROR - POD5 file not found at path: /data/missing.pod5

# WARNING level
2025-01-15 14:25:33 - squiggy.alignment - WARNING - Error reading BAM file for read 'read_001': FileNotFoundError
```

## Redirecting Python Logs to Output Channel (Advanced)

If you prefer Python logs to appear in the extension's Output Channel instead of polluting the Python Console, you can configure a custom logging handler:

```python
import logging
import sys

# Get the squiggy logger
squiggy_logger = logging.getLogger('squiggy')

# Remove default stderr handler
for handler in squiggy_logger.handlers[:]:
    squiggy_logger.removeHandler(handler)

# Add a file handler instead (write to file, extension can monitor)
file_handler = logging.FileHandler('/tmp/squiggy.log')
file_handler.setFormatter(
    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
)
squiggy_logger.addHandler(file_handler)
squiggy_logger.setLevel(logging.INFO)
```

**Note**: The extension does not currently monitor Python log files, but this approach prevents console pollution. Future versions may implement automatic log routing.

## Best Practices

### For Users

1. **Set appropriate log level**: Use `WARNING` (default) for normal use, `INFO` for troubleshooting
2. **Check Extension Logs first**: TypeScript errors (file not found, kernel issues) appear here
3. **Check Python Console for Python errors**: Data processing errors appear here
4. **Enable DEBUG only when needed**: Verbose logging can impact performance

### For Developers

1. **Use structured logging**: Include context like file paths, read IDs, sample names
2. **Log before raising exceptions**: Helps debugging when exceptions are caught
3. **Use appropriate log levels**:
   - `DEBUG`: Diagnostic info
   - `INFO`: Successful operations
   - `WARNING`: Recoverable issues
   - `ERROR`: Operation failures

## Troubleshooting

### "I don't see any logs in the Output panel"

The Output panel only shows **TypeScript extension logs**. Python logs appear in the **Python Console** by default.

### "Python logs are cluttering my interactive console"

Set log level to ERROR: `os.environ['SQUIGGY_LOG_LEVEL'] = 'ERROR'`

Or configure a file handler (see "Redirecting Python Logs" above).

### "I want to see more detailed logs"

Set log level to DEBUG:
```python
import os
os.environ['SQUIGGY_LOG_LEVEL'] = 'DEBUG'

# Restart Python console or reload squiggy
import importlib
import squiggy
importlib.reload(squiggy)
```

### "How do I share logs for a bug report?"

1. Set `SQUIGGY_LOG_LEVEL=DEBUG`
2. Reproduce the issue
3. Copy logs from:
   - Output panel → Squiggy (TypeScript logs)
   - Python Console (Python logs)
4. Include both in your bug report

## Future Improvements

Planned logging enhancements:

1. **Unified Output Channel**: Route Python logs to extension Output Channel automatically
2. **Log file monitoring**: Extension monitors Python log files and displays in Output panel
3. **Log filtering**: UI controls to filter logs by level or module
4. **Export logs**: One-click export of full logs for bug reports

## See Also

- [Developer Guide](DEVELOPER.md) - Extension architecture
- [User Guide](USER_GUIDE.md) - Using Squiggy
- [Issue #116](https://github.com/rnabioco/squiggy-positron/issues/116) - Logging improvements tracking issue
