# Squiggy Quick Reference

Quick reference for common development tasks.

## Environment Setup

```bash
# Initial setup
pip install -e ".[dev]"           # Core + dev dependencies
pip install -e ".[macos]"         # macOS: fix "Python" in menu bar
python scripts/check_dev_env.py   # Validate setup
```

## Running the App

```bash
squiggy                           # Launch GUI
squiggy -p file.pod5              # Pre-load POD5 file
squiggy -p file.pod5 -b file.bam  # Pre-load both files
python -m squiggy.main            # Run from source
```

## Development Commands

```bash
# Code quality
ruff format src/ tests/           # Format code
ruff check --fix src/ tests/      # Lint and auto-fix
ruff check src/ tests/            # Check without fixing

# Testing
pytest                            # Run all tests
pytest -v                         # Verbose output
pytest --cov=squiggy tests/       # With coverage
pytest tests/test_plotting.py     # Specific test file

# Documentation
mkdocs serve                      # Live preview
mkdocs build                      # Build docs
mkdocs gh-deploy                  # Deploy to GitHub Pages
```

## File Locations

```
src/squiggy/
├── main.py          → Entry point, CLI args, async init
├── viewer.py        → Main window, UI layout, event handlers
├── plotter.py       → Plotting logic (plotnine)
├── dialogs.py       → About, Reference Browser dialogs
├── utils.py         → File I/O, POD5/BAM parsing
├── constants.py     → Enums, config values
├── alignment.py     → BAM alignment parsing
└── normalization.py → Signal normalization methods

tests/
├── conftest.py      → pytest fixtures
├── test_*.py        → Test modules
└── data/            → Sample POD5/BAM files

docs/                → MkDocs documentation
build/               → PyInstaller spec, icons
scripts/             → Dev tools
```

## Common Patterns

### Adding a UI Element

```python
# In viewer.py init_ui()
self.my_button = QPushButton("Label")
self.my_button.clicked.connect(self.on_button_click)
layout.addWidget(self.my_button)

# Handler with async operations
@qasync.asyncSlot()
async def on_button_click(self):
    self.statusBar().showMessage("Working...")
    result = await asyncio.to_thread(self._blocking_work)
    self.statusBar().showMessage("Done!")
```

### Adding a Plot Mode

```python
# 1. Add to constants.py
class PlotMode(Enum):
    NEW_MODE = "new_mode"

# 2. Add radio button in viewer.py create_plot_options_content()
self.mode_new = QRadioButton("New Mode")
self.mode_new.toggled.connect(
    lambda checked: self.set_plot_mode(PlotMode.NEW_MODE) if checked else None
)

# 3. Implement in plotter.py
@staticmethod
def plot_new_mode(data, ...):
    # Create plotnine plot
    return plot
```

### Working with POD5 Files

```python
# Always use context manager
with pod5.Reader(pod5_file) as reader:
    for read in reader.reads():
        read_id = str(read.read_id)
        signal = read.signal  # numpy array
        sample_rate = read.run_info.sample_rate

# In async context
def _read_blocking(pod5_file):
    with pod5.Reader(pod5_file) as reader:
        return [str(r.read_id) for r in reader.reads()]

async def load_reads(self):
    reads = await asyncio.to_thread(_read_blocking, self.pod5_file)
```

### Working with BAM Files

```python
# Use pysam with check_sq=False
with pysam.AlignmentFile(bam_file, "rb", check_sq=False) as bam:
    for alignment in bam.fetch(until_eof=True):
        read_id = alignment.query_name
        sequence = alignment.query_sequence

        # Get move table (signal-to-base mapping)
        if alignment.has_tag("mv"):
            move_table = alignment.get_tag("mv")
```

## Testing with Sample Data

```bash
# Files in tests/data/
tests/data/simplex_reads.pod5        # ~10 reads
tests/data/simplex_reads_mapped.bam  # Aligned reads

# Run app with sample data
squiggy -p tests/data/simplex_reads.pod5 -b tests/data/simplex_reads_mapped.bam

# Test reference browser:
# 1. Load both files above
# 2. Switch to "Reference Region" search mode
# 3. Click "Browse References..." button
```

## Async/Qt Patterns

```python
# Async slot (for Qt signals)
@qasync.asyncSlot()
async def handler(self):
    await self.do_async_work()

# Run blocking work in thread pool
result = await asyncio.to_thread(blocking_function, args)

# Update UI (must be on main thread)
self.label.setText("Updated")  # OK - called from async slot
```

## Debugging Tips

- **Qt issues**: Check signal/slot connections, ensure proper parent widgets
- **Async issues**: Verify `@qasync.asyncSlot()` decorator, use `asyncio.to_thread()` for blocking ops
- **Plot rendering**: Check plotnine plot object, ensure BytesIO buffer is reset before reading
- **BAM parsing**: Use `check_sq=False`, verify index file exists (.bai)
- **macOS menu bar**: Install PyObjC: `pip install -e ".[macos]"`

## Build Executables

```bash
# Local build
cd build
pyinstaller squiggy.spec

# Output locations:
# - macOS: dist/Squiggy.app
# - Windows: dist/Squiggy.exe
# - Linux: dist/Squiggy

# Test build
./dist/Squiggy.app/Contents/MacOS/Squiggy  # macOS
./dist/Squiggy                              # Linux
dist\Squiggy.exe                            # Windows
```

## Git Workflow

```bash
git checkout -b feature/my-feature
# Make changes
ruff format src/ tests/
ruff check --fix src/ tests/
pytest
git add .
git commit -m "feat: add new feature"
git push origin feature/my-feature
# Open PR on GitHub
```

## Useful Claude Code Prompts

- "Add a new normalization method called [name]"
- "Create a test for the reference browser dialog"
- "Fix the async loading issue when opening large POD5 files"
- "Add a menu item to export the current plot"
- "Refactor the plotting code to support [new feature]"
- "Why is the UI freezing when loading files?"

## Key Constraints

- **Python 3.8+ compatible** (no walrus operators, etc.)
- **Line length**: 88 chars
- **Always format with ruff** before committing
- **Use type hints** for function signatures
- **Async operations** must use `@qasync.asyncSlot()` + `asyncio.to_thread()`
- **Read-only access** to POD5/BAM during iteration
