# Squiggy Logging Guide

This guide explains where Squiggy logs appear in Positron and how to access them for debugging.

## Overview

Squiggy uses a single logging system for TypeScript extension operations (UI, commands, file handling). Python backend operations do not produce logs - errors are propagated as exceptions that are caught and displayed by the TypeScript extension.

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
- Error messages with stack traces from both TypeScript and Python
- Warnings and debug info

### Example Output

```
[14:23:45.123] [INFO] Squiggy extension activated
[14:23:47.456] [INFO] Loading POD5 file: /data/reads.pod5
[14:23:48.789] [ERROR] Error while loading POD5 file: FileNotFoundError: File not found at path: /data/reads.pod5
```

## Python Backend Error Handling

### No Console Pollution

Python backend code does **not** log to the console. Instead:

1. Python exceptions are raised normally (e.g., `FileNotFoundError`, `ValueError`)
2. The TypeScript extension catches these exceptions via the Positron kernel
3. Errors are displayed in the extension's Output Channel with full context
4. User's Python Console remains clean for interactive work

### Example Error Flow

**Python code:**
```python
>>> import squiggy
>>> squiggy.load_pod5('nonexistent.pod5')
```

**What happens:**
1. Python raises: `FileNotFoundError: POD5 file not found at path: /data/nonexistent.pod5`
2. TypeScript catches exception via kernel
3. Error appears in Output Panel → Squiggy:
   ```
   [14:23:48.789] [ERROR] Failed to load POD5 file
   FileNotFoundError: POD5 file not found at path: /data/nonexistent.pod5
   ```
4. Python Console shows only the Python exception (standard Python behavior)

### Why This Approach?

**Benefits:**
- **Clean Python Console**: No logging clutter in interactive workspace
- **Centralized Logs**: All Squiggy logs (TypeScript + Python errors) in one place
- **Better UX**: Users can focus on their data analysis without distraction
- **Standard Python**: Python exceptions work as expected

## Best Practices

### For Users

1. **Check Extension Logs for errors**: TypeScript and Python errors appear in Output → Squiggy
2. **Python Console is for your code**: Only your interactive Python code and standard Python exceptions appear here
3. **Use "Show Logs" button**: Click it on error notifications for quick access to logs

### For Developers

1. **Use exceptions, not logging**: Raise informative exceptions in Python code (e.g., `ValueError`, `FileNotFoundError`)
2. **Log in TypeScript**: Use `outputChannel.appendLine()` in TypeScript extension code
3. **Include context in exceptions**: Error messages should include file paths, read IDs, etc.
4. **Structured error handling**: Catch Python exceptions in TypeScript and log with context

## Troubleshooting

### "I don't see any logs in the Output panel"

Make sure you've selected "Squiggy" from the dropdown in the Output panel (`Cmd+Shift+U` / `Ctrl+Shift+U`).

### "Where do Python errors appear?"

Python errors appear in **two places**:
1. **Output Panel → Squiggy** (extension log) - Full context and details
2. **Python Console** - Standard Python exception traceback

### "I want more detailed logs"

TypeScript extension logging is controlled in the extension code. For development, you can modify `src/extension.ts` to increase log verbosity.

### "How do I share logs for a bug report?"

1. Reproduce the issue
2. Open Output panel → Squiggy
3. Copy all logs
4. Include in your bug report along with:
   - Python exception traceback (if applicable)
   - Steps to reproduce
   - File sizes and types (POD5, BAM, FASTA)

## Architecture

### TypeScript Extension Logging

The extension creates an `OutputChannel` on activation:

```typescript
const outputChannel = vscode.window.createOutputChannel('Squiggy');
outputChannel.appendLine('[INFO] Extension activated');
```

### Python Error Propagation

Python backend uses standard exception handling:

```python
def load_pod5(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"POD5 file not found at path: {file_path}")
```

TypeScript catches and logs these exceptions:

```typescript
try {
    await runtime.execute(`squiggy.load_pod5('${filePath}')`);
} catch (error) {
    outputChannel.appendLine(`[ERROR] Failed to load POD5: ${error.message}`);
    vscode.window.showErrorMessage('Failed to load POD5 file', 'Show Logs');
}
```

## See Also

- [Developer Guide](DEVELOPER.md) - Extension architecture
- [User Guide](USER_GUIDE.md) - Using Squiggy
